"""
ai/detection_pipeline.py — TAE object detection pipeline (v2)
==============================================================
Six-stage pipeline for `object_detection` queries.

YOLO-World API notes (zsxkib/yolo-world model schema confirmed):
  Input:  input_media (URI), class_names (str), score_thr (float),
          nms_thr (float), max_num_boxes (int)
  Output: single image URI — NOT structured JSON
  Boxes:  emitted to stdout → available in pred.logs

Submission strategy: ALL tiles submitted at once (non-blocking
predictions.create), then polled together every 2 s in one shared
loop. No ThreadPoolExecutor blocking per prediction.

CLIP pre-filter: easy/medium queries select top-30% tiles by
CLIP cosine similarity before YOLO. Hard queries use all tiles.
Imported from core.services to avoid circular imports with main.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import math

logger = logging.getLogger("TAE.Tracker")

from core.app_state import _state
from config import Settings
settings = Settings()

# ── Constants ─────────────────────────────────────────────────────────────────
 
_STATIC_GATE_M   =   5.0   # base gate for stationary objects (m)
_MAX_COLD_SPEED  =  30.0   # m/s — max speed assumed before velocity is known
                            # (≈108 km/h covers any ground vehicle)
_VEL_ALPHA       =   0.6   # EMA weight for velocity update (0=inertia, 1=instant)
_M_PER_DEG_LAT   = 111_320.0
 


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    det_id:      str
    parent_path: str
    tile_x:      int
    tile_y:      int
    tile_w:      int
    tile_h:      int
    bbox_tile:   list     # [x1,y1,x2,y2] tile-local
    bbox_frame:  list     # [x1,y1,x2,y2] full-frame absolute
    label:       str
    confidence:  float
    lat:         float = 0.0
    lon:         float = 0.0
    mask:        object = field(default=None, repr=False)
    masked_crop: object = field(default=None, repr=False)
    mask_area:   int    = 0
    fill_ratio:  float  = 0.0
    vlm_report:  dict   = field(default_factory=dict)
    confirmed:   bool   = False
    track_id:    object = None
    fp_nw_lat:   float  = 0.0
    fp_nw_lon:   float  = 0.0
    fp_ne_lat:   float  = 0.0
    fp_ne_lon:   float  = 0.0
    fp_se_lat:   float  = 0.0
    fp_se_lon:   float  = 0.0
    fp_sw_lat:   float  = 0.0
    fp_sw_lon:   float  = 0.0


@dataclass
class Track:
    track_id:   str
    detections: list = field(default_factory=list)
    label:      str  = ""
    color:      str  = "#4ade80"

    @property
    def lat(self):
        # Latest observed latitude (most recent detection in temporal order).
        return self.detections[-1].lat if self.detections else 0.0

    @property
    def lon(self):
        # Latest observed longitude (most recent detection in temporal order).
        return self.detections[-1].lon if self.detections else 0.0

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        # Ordered (lat, lon) pairs — used by _save_tracks() for polylines.
        return [(d.lat, d.lon) for d in self.detections]
    
    @property
    def best(self):
        return max(self.detections, key=lambda d: d.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile_img(tile):
    parent = tile.get("parent_path", "")
    if not parent or not Path(parent).exists():
        return None
    img = cv2.imread(parent)
    if img is None:
        return None
    x, y, w, h = tile["tile_x"], tile["tile_y"], tile["tile_w"], tile["tile_h"]
    return img[y: y + h, x: x + w]

 
def _m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Detector
# ─────────────────────────────────────────────────────────────────────────────

def _parse_detector_output(output, tile):
    """
    Parse Grounding DINO structured output.

    Confirmed format (from live run):
      output["detections"] = [
          {"bbox": [x1, y1, x2, y2], "confidence": 0.83, "label": "cow"},
          ...
      ]
    bbox coordinates are absolute pixels in the tile (not normalised 0-1).
    """
    if not output or not isinstance(output, dict):
        logger.debug("Detector tile(%d,%d): empty/non-dict output",
                     tile.get("tile_x", 0), tile.get("tile_y", 0))
        return []

    raw_dets = output.get("detections", [])
    if not raw_dets:
        return []

    tx, ty = tile["tile_x"], tile["tile_y"]
    tw, th = tile["tile_w"], tile["tile_h"]
    dets   = []

    for item in raw_dets:
        try:
            label = str(item.get("label", "object"))
            score = float(item.get("confidence", 0.0))
            box   = item.get("bbox", [])   # [x1, y1, x2, y2] absolute pixels

            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Safety: if values look normalised (all <= 1.0), scale to tile dims
            if max(x1, y1, x2, y2) <= 1:
                x1 = int(x1 * tw); y1 = int(y1 * th)
                x2 = int(x2 * tw); y2 = int(y2 * th)

            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(Detection(
                det_id      = uuid.uuid4().hex[:10],
                parent_path = tile["parent_path"],
                tile_x      = tx, tile_y = ty,
                tile_w      = tw, tile_h  = th,
                bbox_tile   = [x1, y1, x2, y2],
                bbox_frame  = [tx+x1, ty+y1, tx+x2, ty+y2],
                label       = label,
                confidence  = score,
                fp_nw_lat   = tile.get("fp_nw_lat", 0.0),
                fp_nw_lon   = tile.get("fp_nw_lon", 0.0),
                fp_ne_lat   = tile.get("fp_ne_lat", 0.0),
                fp_ne_lon   = tile.get("fp_ne_lon", 0.0),
                fp_se_lat   = tile.get("fp_se_lat", 0.0),
                fp_se_lon   = tile.get("fp_se_lon", 0.0),
                fp_sw_lat   = tile.get("fp_sw_lat", 0.0),
                fp_sw_lon   = tile.get("fp_sw_lon", 0.0),
            ))
        except Exception as exc:
            logger.debug("Bad detection item %s: %s", item, exc)

    return dets


def _run_detector_on_tile(tile, query_str, box_threshold, client, model_version):
    """Run Grounding DINO on one tile via client.run() (handles upload + poll)."""
    tile_img = _load_tile_img(tile)
    if tile_img is None:
        return []
    ok, buf = cv2.imencode(".jpg", tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return []
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(buf.tobytes())
        tmp.close()
        with open(tmp.name, "rb") as fh:
            output = client.run(
                model_version,
                input={
                    "image":          fh,
                    "query":          query_str,
                    "box_threshold":  box_threshold,
                    "text_threshold": box_threshold,
                },
            )
        return _parse_detector_output(output, tile)
    except Exception as exc:
        logger.warning(
            "Detector failed tile(%d,%d): %s",
            tile.get("tile_x", 0), tile.get("tile_y", 0), exc,
        )
        return []
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def run_detector_stage(tiles, classes, confidence, api_key, model_version, timeout_s=180):
    """
    Run Grounding DINO on all tiles in parallel using ThreadPoolExecutor.

    Uses client.run() (not predictions.create) because client.run() handles
    file upload to Replicate's file store automatically. predictions.create()
    requires a URL, not a file handle, and fails with NoneType on file objects.

    Input fields (Grounding DINO schema):
      image          — tile image file object
      query          — comma-separated class names: "cow, horse, sheep"
      box_threshold  — detection confidence threshold
      text_threshold — text matching threshold (same value as box_threshold)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import replicate

    if not api_key:
        logger.warning("REPLICATE_API_KEY not set — detector stage skipped")
        return []

    client    = replicate.Client(api_token=api_key)
    query_str = ", ".join(classes)   # Grounding DINO: comma-separated

    logger.info(
        "Detector stage | %d tiles | query='%s' | threshold=%.3f",
        len(tiles), query_str, confidence,
    )

    raw = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                _run_detector_on_tile,
                tile, query_str, confidence, client, model_version,
            ): tile
            for tile in tiles
        }
        for future in as_completed(futures, timeout=timeout_s):
            tile = futures[future]
            try:
                dets = future.result()
                if dets:
                    logger.info(
                        "Detector | %s tile(%d,%d) → %d det(s)",
                        Path(tile.get("parent_path", "")).name,
                        tile.get("tile_x", 0), tile.get("tile_y", 0),
                        len(dets),
                    )
                raw.extend(dets)
            except Exception as exc:
                logger.warning("Tile future failed: %s", exc)

    logger.info("Detector stage: %d raw detection(s)", len(raw))
    return raw


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter       = (ix2 - ix1) * (iy2 - iy1)
    area_a      = (a[2] - a[0]) * (a[3] - a[1])
    area_b      = (b[2] - b[0]) * (b[3] - b[1])
    standard    = inter / max(1, area_a + area_b - inter)
    # Containment: if the SMALLER box is almost entirely inside the LARGER
    # box they represent the same object on an overlapping tile boundary.
    # Standard IoU misses this when one box is much smaller than the other.
    containment = inter / max(1, min(area_a, area_b))
    return max(standard, containment)


def _nms_frame(dets, iou_threshold):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept, suppressed = [], set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(d)
        for j in range(i + 1, len(dets)):
            if j not in suppressed and _iou(d.bbox_frame, dets[j].bbox_frame) > iou_threshold:
                suppressed.add(j)
    return kept


def cross_tile_nms(raw, iou_threshold=0.50):
    by_frame = defaultdict(list)
    for d in raw:
        by_frame[d.parent_path].append(d)
    kept = []
    for fp, dets in by_frame.items():
        after = _nms_frame(dets, iou_threshold)
        if len(dets) != len(after):
            logger.info("NMS | %s: %d → %d", Path(fp).name, len(dets), len(after))
        kept.extend(after)
    logger.info("NMS: %d → %d", len(raw), len(kept))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — SAM refinement + shape filter
# ─────────────────────────────────────────────────────────────────────────────

def _fill_ratio(mask, bbox):
    x1, y1, x2, y2 = bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return float(mask[y1:y2, x1:x2].sum()) / bbox_area


def _shape_ok(mask, bbox, priors):
    """
    With center-point SAM, masks are smaller than GDINO bboxes.
    The only reliable check: is the mask actually inside the bbox?
    (Rules out background masks that cover most of the tile.)
    """
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w)); x2 = max(x1+1, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(y1+1, min(y2, h))

    mask_total  = int(mask.sum())
    if mask_total == 0:
        return False

    # Containment: what fraction of the SAM mask lies inside the GDINO bbox?
    overlap = int(mask[y1:y2, x1:x2].sum())
    containment = overlap / mask_total

    # Background masks extend all over the tile → low containment in any single bbox
    if containment < 0.15:
        logger.info("Shape reject CONTAINMENT: %.2f (mask outside bbox)", containment)
        return False

    return True


def sam_refine_stage(candidates, priors, actual_alt_m, sam_segmentor):

    # If no SAM segmentor available, skip refinement and shape filtering steps
    if sam_segmentor is None:
        # Fast path: just cut bbox crops, no API calls
        for det in candidates:
            tile_img = _load_tile_img({
                "parent_path": det.parent_path,
                "tile_x": det.tile_x, "tile_y": det.tile_y,
                "tile_w": det.tile_w, "tile_h": det.tile_h,
            })
            if tile_img is not None:
                x1, y1, x2, y2 = [max(0, v) for v in det.bbox_tile]
                det.masked_crop = tile_img[y1:y2, x1:x2].copy()
        return candidates   # no filtering without SAM

    from ai.intent import scale_shape_priors
    scaled        = scale_shape_priors(priors, actual_alt_m)
    logger.info("Scaled ShapePriors at %.0fm: area=[%d,%d] fill=[%.2f,%.2f] aspect=[%.2f,%.2f]",
            actual_alt_m,
            scaled.min_area_px, scaled.max_area_px,
            scaled.min_fill, scaled.max_fill,
            scaled.min_aspect, scaled.max_aspect)
    
    sam_available = sam_segmentor is not None and hasattr(sam_segmentor, "predict_box")
    
    if not sam_available:
        if sam_segmentor is None:
            reason = "segmentor is None — check REPLICATE_API_KEY and SAM_REPLICATE_MODEL in config"
        else:
            reason = "predict_box method missing — segmentor.py needs update"
        logger.info("SAM unavailable (%s) — bbox crops used directly", reason)

    refined = []
    for det in candidates:
        tile_rec = {
            "parent_path": det.parent_path,
            "tile_x": det.tile_x, "tile_y": det.tile_y,
            "tile_w": det.tile_w, "tile_h": det.tile_h,
        }
        tile_img = _load_tile_img(tile_rec)
        if tile_img is None:
            continue
        x1, y1, x2, y2 = det.bbox_tile
        h_t, w_t = tile_img.shape[:2]
        x1 = max(0, min(x1, w_t - 1)); x2 = max(x1 + 1, min(x2, w_t))
        y1 = max(0, min(y1, h_t - 1)); y2 = max(y1 + 1, min(y2, h_t))
        bbox = [x1, y1, x2, y2]

        if sam_available:
            try:
                mask = sam_segmentor.predict_box(tile_img, bbox)
            except Exception as exc:
                logger.debug("SAM failed: %s", exc)
                mask = None
            if mask is not None:
                if not _shape_ok(mask, bbox, scaled):
                    logger.info("Shape filter rejected %s in %s",
                                det.label, Path(det.parent_path).name)
                    continue
                mc = tile_img.copy()
                mc[~mask] = [128, 128, 128]
                det.mask        = mask
                det.masked_crop = mc[y1:y2, x1:x2]
                det.mask_area   = int(mask.sum())
                det.fill_ratio  = _fill_ratio(mask, bbox)
            else:
                det.masked_crop = tile_img[y1:y2, x1:x2].copy()
        else:
            det.masked_crop = tile_img[y1:y2, x1:x2].copy()

        refined.append(det)

    logger.info("SAM: %d → %d", len(candidates), len(refined))
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — VLM verification
# ─────────────────────────────────────────────────────────────────────────────

def vlm_verify_stage(candidates, params, original_query, analyst):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import defaultdict

    # Group candidates by (parent_path, tile_x, tile_y) — same tile
    tile_groups: dict[tuple, list[Detection]] = defaultdict(list)
    for det in candidates:
        key = (det.parent_path, det.tile_x, det.tile_y)
        tile_groups[key].append(det)

    logger.info("VLM batch: %d candidates across %d tiles",
                len(candidates), len(tile_groups))

    def _verify_tile(tile_key, tile_dets):
        parent, tx, ty = tile_key
        tile_rec = {
            "parent_path": parent,
            "tile_x": tx, "tile_y": ty,
            "tile_w": tile_dets[0].tile_w,
            "tile_h": tile_dets[0].tile_h,
        }
        tile_img = _load_tile_img(tile_rec)
        if tile_img is None:
            return []

        batch_input = [
            {"label": d.label, "bbox": d.bbox_tile}
            for d in tile_dets
        ]
        results = analyst.verify_detections_batch(
            tile_img       = tile_img,
            detections     = batch_input,
            criteria       = params.vlm_verification_criteria,
            report_fields  = params.vlm_reporting_fields,
            original_query = original_query,
            colour_hint    = params.colour_hint,
            size_qualifier = params.size_qualifier,
        )
        confirmed = []
        for det, result in zip(tile_dets, results):
            if result.get("confirmed"):
                det.confirmed = True
                det.vlm_report = result.get("report", {})
                # Prefer VLM's own label over GDINO's; fall back to GDINO label
                vlm_label = result.get("detected_label", "").strip()
                if vlm_label:
                    det.label = vlm_label
                confirmed.append(det)
                logger.info("VLM confirmed %s in %s tile(%d,%d)",
                            det.label, Path(parent).name, tx, ty)
            else:
                reason = result.get("reason", "")
                logger.info("VLM rejected %s in %s tile(%d,%d): %s",
                            det.label, Path(parent).name, tx, ty, reason)
                # Save rejected crop for debugging
                try:
                    paths = _state.get("mission_paths")
                    if paths and det.masked_crop is not None:
                        rej_dir = Path(paths.detections) / "rejected"
                        rej_dir.mkdir(parents=True, exist_ok=True)
                        safe = det.label.replace(" ", "_")[:20]
                        fname = f"{safe}_{det.det_id[:8]}.jpg"
                        cv2.imwrite(str(rej_dir / fname), det.masked_crop)
                except Exception:
                    pass
        return confirmed

    confirmed = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_verify_tile, key, dets): key
            for key, dets in tile_groups.items()
        }
        for future in as_completed(futures):
            confirmed.extend(future.result())

    logger.info("VLM: %d candidates → %d confirmed", len(candidates), len(confirmed))
    return confirmed

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Geo-location
# ─────────────────────────────────────────────────────────────────────────────

def _bilinear(nw, ne, sw, se, u, v):
    top_lat = nw[0] + u * (ne[0] - nw[0])
    top_lon = nw[1] + u * (ne[1] - nw[1])
    bot_lat = sw[0] + u * (se[0] - sw[0])
    bot_lon = sw[1] + u * (se[1] - sw[1])
    return (top_lat + v * (bot_lat - top_lat),
            top_lon + v * (bot_lon - top_lon))


def geolocate_detection(det):
    tw, th = max(1, det.tile_w), max(1, det.tile_h)
    if det.mask is not None:
        ys, xs = np.where(det.mask)
        if len(xs):
            cx = float(xs.mean()) / tw
            cy = float(ys.mean()) / th
        else:
            cx = ((det.bbox_tile[0] + det.bbox_tile[2]) / 2) / tw
            cy = ((det.bbox_tile[1] + det.bbox_tile[3]) / 2) / th
    else:
        cx = ((det.bbox_tile[0] + det.bbox_tile[2]) / 2) / tw
        cy = ((det.bbox_tile[1] + det.bbox_tile[3]) / 2) / th
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    return _bilinear(
        (det.fp_nw_lat, det.fp_nw_lon), (det.fp_ne_lat, det.fp_ne_lon),
        (det.fp_sw_lat, det.fp_sw_lon), (det.fp_se_lat, det.fp_se_lon),
        cx, cy,
    )


def geolocate_stage(confirmed):
    for det in confirmed:
        det.lat, det.lon = geolocate_detection(det)
    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TrackState:
    """
    Internal mutable state for one live track.
    Converted to the public Track at the end of track_stage().
    """
    track_id : str
    label    : str
    color    : str
    dets     : list        # Detection objects in time order
    last_lat : float
    last_lon : float
    last_ts  : float       # milliseconds
 
    # Velocity in degrees/second (updated via EMA after each association)
    vel_lat  : float = 0.0
    vel_lon  : float = 0.0
 
    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, ts_ms: float) -> tuple[float, float]:
        """Constant-velocity position prediction at timestamp ts_ms."""
        dt = (ts_ms - self.last_ts) / 1000.0
        return (
            self.last_lat + self.vel_lat * dt,
            self.last_lon + self.vel_lon * dt,
        )
 
    # ── Adaptive association gate ──────────────────────────────────────────────
    def gate_m(self, dt_s: float) -> float:
        """
        Returns the maximum acceptable distance (metres) between the predicted
        position and a candidate detection.
 
        Cold-start (< 2 hits): use MAX_COLD_SPEED to cover the first interval
            regardless of direction, since velocity is unknown.
        Warm (≥ 2 hits): scale the gate with the estimated speed + 50% margin.
        """
        if len(self.dets) < 2:
            return _STATIC_GATE_M + _MAX_COLD_SPEED * abs(dt_s)
        speed = math.hypot(
            self.vel_lat * _M_PER_DEG_LAT,
            self.vel_lon * _m_per_deg_lon(self.last_lat),
        )  # m/s
        return _STATIC_GATE_M + speed * abs(dt_s) * 1.5
 
    # ── Update after a successful association ──────────────────────────────────
    def update(self, det, ts_ms: float) -> None:
        dt = (ts_ms - self.last_ts) / 1000.0
        if dt > 0:
            new_vl = (det.lat - self.last_lat) / dt
            new_vn = (det.lon - self.last_lon) / dt
            # Exponential moving average — dampens outlier jumps
            self.vel_lat = _VEL_ALPHA * new_vl + (1 - _VEL_ALPHA) * self.vel_lat
            self.vel_lon = _VEL_ALPHA * new_vn + (1 - _VEL_ALPHA) * self.vel_lon
        self.last_lat = det.lat
        self.last_lon = det.lon
        self.last_ts  = ts_ms
        self.dets.append(det)
 
    # ── Estimated ground speed ─────────────────────────────────────────────────
    @property
    def speed_ms(self) -> float:
        return math.hypot(
            self.vel_lat * _M_PER_DEG_LAT,
            self.vel_lon * _m_per_deg_lon(self.last_lat),
        )
 
 
def _frame_ts_ms(det) -> float:
    """
    Look up the sampled-frame timestamp for a detection.
 
    main.py stores {filename: timestamp_ms} in _state["frame_timestamps"]
    when video frames are extracted.  For still images with no timestamp we
    fall back to a large sentinel that preserves the file-sort order.
    """
    # Import here to avoid circular imports at module load time
    from core.app_state import _state
    name = Path(det.parent_path).name
    ts   = _state.get("frame_timestamps", {}).get(name)
    if ts is not None:
        return float(ts)
    # Still image without a video timestamp — use filename sort order as proxy.
    # All stills get ts = 0; they are grouped by parent_path anyway.
    return 0.0
 
 
def track_stage(confirmed: list, color: str = "#4ade80") -> list:
    """
    Velocity-aware multi-object tracker (replaces the old static geo-NMS).
 
    Algorithm
    ---------
    1. Attach a timestamp to every detection from _state["frame_timestamps"].
    2. Within each label group, sort detections by timestamp.
    3. For each detection (in time order), score it against every open track:
         predicted_pos = last_pos + velocity × Δt
         score         = haversine(predicted_pos, det_pos)
         gate          = STATIC_GATE + speed × Δt × 1.5   (warm)
                       = STATIC_GATE + MAX_COLD_SPEED × Δt  (cold-start)
       Associate to the closest track within its gate (greedy nearest-neighbour).
    4. Unmatched detections start new tracks.
    5. After all detections are processed, convert _TrackState → Track.
 
    Why this works for moving vehicles
    -----------------------------------
    A 6 m/s car moves 60 m between 10-second samples.  On the first match
    (cold-start, velocity unknown) the gate is 5 + 30×10 = 305 m — wide enough
    to catch the car.  After the first match the velocity estimate is ~6 m/s;
    subsequent gates are 5 + 6×10×1.5 = 95 m, naturally tightening around the
    expected displacement and rejecting false positives.
 
    For stationary cars the estimated velocity → 0 so the gate collapses to
    the original 5 m threshold — no regression for the parked-vehicle case.
    """
    if not confirmed:
        return []
 
    # Import here to avoid circular-import issues at module level
    from core.geo import haversine_m
 
    # Attach timestamps; detections from the same frame share the same ts
    for det in confirmed:
        det._ts_ms = _frame_ts_ms(det)
 
    by_label: dict[str, list] = defaultdict(list)
    for det in confirmed:
        by_label[det.label.lower()].append(det)
 
    all_tracks: list = []
 
    for label, dets in by_label.items():
        # Process in temporal order so velocity estimates are causal
        dets.sort(key=lambda d: (d._ts_ms, d.parent_path))
 
        open_tracks: list[_TrackState] = []
 
        for det in dets:
            ts = det._ts_ms
 
            best_st   : _TrackState | None = None
            best_dist : float              = float("inf")
 
            for st in open_tracks:
                # Skip if this detection comes from the same frame as the
                # track's last hit (same physical instant → different object)
                if any(h.parent_path == det.parent_path for h in st.dets):
                    continue
 
                dt_s      = (ts - st.last_ts) / 1000.0
                pred_lat, pred_lon = st.predict(ts)
                dist      = haversine_m(pred_lat, pred_lon, det.lat, det.lon)
                gate      = st.gate_m(dt_s)
 
                if dist < gate and dist < best_dist:
                    best_dist = dist
                    best_st   = st
 
            if best_st is not None:
                best_st.update(det, ts)
                det.track_id = best_st.track_id
            else:
                new_st = _TrackState(
                    track_id = uuid.uuid4().hex[:8],
                    label    = label,
                    color    = color,
                    dets     = [det],
                    last_lat = det.lat,
                    last_lon = det.lon,
                    last_ts  = ts,
                )
                det.track_id = new_st.track_id
                open_tracks.append(new_st)
 
        # Convert _TrackState → public Track dataclass (API unchanged)
        for st in open_tracks:
            # Import the Track class from the same module
            t = Track(
                track_id   = st.track_id,
                detections = st.dets,
                label      = label,
                color      = color,
            )
            # Attach velocity metadata for logging / future use
            t._speed_ms = st.speed_ms
            all_tracks.append(t)
 
    logger.info(
        "Tracking: %d confirmed → %d track(s)",
        len(confirmed), len(all_tracks),
    )
    for t in all_tracks:
        n = len(t.detections)
        spd = getattr(t, "_speed_ms", 0.0)
        logger.info(
            "  Track %s | %s | %d frame(s) | %.1f m/s",
            t.track_id, t.label, n, spd,
        )
 
    return all_tracks
 
 

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_pipeline(
    params,
    all_tiles,
    original_query,
    analyst,
    sam_segmentor,
    api_key,
    model_version,
    timeout_s = 180,
    actual_alt_m=100.0,
    color="#4ade80",
    **_kwargs,
):
    logger.info(
        "Detection pipeline | query=%r | classes=%s | confidence=%.3f | "
        "tiles=%d | alt=%.0fm",
        original_query, params.yolo_classes, params.yolo_confidence,
        len(all_tiles), actual_alt_m,
    )

    if not all_tiles:
        logger.warning("No tiles — aborted")
        return []

    # CLIP pre-filter (easy/medium only)
    if params.expected_difficulty != "hard" and len(all_tiles) > 20:
        try:
            from core.services import get_search_lib
            lib   = get_search_lib()
            q_str = "aerial drone nadir overhead view: " + " ".join(params.yolo_classes)
            q_vec = np.array(lib.encode_text(q_str), dtype=np.float32)
            q_vec /= np.linalg.norm(q_vec) + 1e-9
            vecs   = np.array([t["vector"] for t in all_tiles], dtype=np.float32)
            norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            scores = (vecs / norms) @ q_vec
            top_n  = max(20, int(len(all_tiles) * 0.30))
            idx    = np.argsort(scores)[::-1][:top_n]
            tiles  = [all_tiles[i] for i in idx]
            logger.info("CLIP pre-filter (%s): %d → %d tiles",
                        params.expected_difficulty, len(all_tiles), len(tiles))
        except Exception as exc:
            logger.warning("CLIP pre-filter error (%s) — using all tiles", exc)
            tiles = all_tiles
    else:
        tiles = all_tiles

    # Stage 1
    raw = run_detector_stage(
        tiles=tiles,
        classes=params.yolo_classes,
        confidence=params.yolo_confidence,
        api_key=api_key,
        model_version=model_version,
        timeout_s=timeout_s,
    )
    if not raw:
        logger.info("No detections from detector — pipeline ends")
        return []

    # Stage 2
    candidates = cross_tile_nms(raw)    
    candidates = [
        d for d in candidates
        if (d.bbox_tile[2] - d.bbox_tile[0]) >= settings.DETECTION_MIN_BBOX_PX
        and (d.bbox_tile[3] - d.bbox_tile[1]) >= settings.DETECTION_MIN_BBOX_PX
    ]
    logger.info("Min-size filter: kept %d candidates", len(candidates))
    if not candidates:
        return []    

    # Stage 3
    candidates = sam_refine_stage(candidates, params.shape_priors, actual_alt_m, sam_segmentor)
    if not candidates:
        return []

    # Stage 4
    confirmed = vlm_verify_stage(candidates, params, original_query, analyst)
    if not confirmed:
        return []

    # Stage 5
    confirmed = geolocate_stage(confirmed)

    # Stage 6
    tracks = track_stage(confirmed, color=color)

    logger.info("Pipeline: %d tracks, %d detections", len(tracks), len(confirmed))
    return tracks