"""
core/tracker.py — Object tracking for TAE (Phase D).

Two-stage pipeline
------------------
Stage 1 │ YOLO-World (Replicate API) — open-vocabulary dense detector.
        │ Runs on EVERY sampled frame in temporal order.
        │ All 73 frames are submitted in parallel via ThreadPoolExecutor
        │ so total wall time is ~10-20 s (Replicate concurrency limit)
        │ instead of 73 × 2 s = 2.5 min sequential.

Stage 2 │ SORT (Simple Online and Realtime Tracking) — IoU-based assignment.
        │ No deep re-ID, no Kalman filter.  Works well for low-speed objects
        │ at 0.5 fps sampling because bboxes barely move between frames.

GPS projection
--------------
Each detection bbox center is projected to WGS84 using the frame's
4-corner footprint (already stored in LanceDB) via bilinear interpolation —
the same path used by the existing VLM pipeline.

Persistence
-----------
Tracks are saved as tracks.json inside the mission's maps/ directory.
_build_map() in core/analysis.py reads this file and renders PolyLines.

Replicate model
---------------
Set REPLICATE_YOLO_WORLD_MODEL in your .env or override in Settings.
Default model: https://replicate.com/zsxkib/yolo-world
The model must accept:
  image      (file URL or data URI)
  text       (comma-separated class names)
  confidence (float 0-1)
And return a list of dicts with xmin/ymin/xmax/ymax/name/confidence.
"""

import json
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger("TAE.Tracker")

# ── Replicate model ID — override in .env as REPLICATE_YOLO_WORLD_MODEL ───────
YOLO_WORLD_MODEL = os.environ.get(
    "REPLICATE_YOLO_WORLD_MODEL",
    "zsxkib/yolo-world:latest",
)

# ── SORT hyperparameters ──────────────────────────────────────────────────────
IOU_THRESHOLD  = 0.25  # minimum IoU to associate detection → track
MAX_AGE        = 3     # frames a track survives without a detection before dying
MIN_HITS       = 1     # minimum confirmed detections before a track is reported
MAX_WORKERS    = 10    # parallel Replicate calls


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single YOLO-World detection on one frame."""
    frame_name:   str
    frame_idx:    int
    timestamp_ms: int
    bbox:         list      # [xmin, ymin, xmax, ymax] in full-frame pixels
    confidence:   float
    label:        str
    lat:          float     # GPS of bbox centre (from footprint projection)
    lon:          float


@dataclass
class Track:
    """An object tracked across multiple frames."""
    id:           str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:        str       = ""
    color:        str       = "#60a5fa"
    hits:         int       = 0     # total confirmed detections
    age:          int       = 0     # frames since last detection
    detections:   list      = field(default_factory=list)
    last_bbox:    list      = field(default_factory=list)

    def to_dict(self) -> dict:
        trajectory = [
            {"lat": d["lat"], "lon": d["lon"], "timestamp_ms": d["timestamp_ms"]}
            for d in self.detections
        ]
        return {
            "id":          self.id,
            "label":       self.label,
            "color":       self.color,
            "frame_count": len(self.detections),
            "detections":  self.detections,
            "trajectory":  trajectory,
        }


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(boxA: list, boxB: list) -> float:
    """
    Intersection over Union for two bboxes [xmin, ymin, xmax, ymax].

    Returns a value in [0, 1]:
      0.0 → no overlap at all
      1.0 → identical bboxes
    Two bboxes from consecutive frames with IoU > IOU_THRESHOLD are
    considered the same object.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(tracks: list[Track], detections: list[Detection]) -> np.ndarray:
    """Build the N_tracks × N_detections IoU cost matrix."""
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            mat[i, j] = compute_iou(t.last_bbox, d.bbox)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign(tracks: list[Track], dets: list[Detection]):
    """
    Greedy Hungarian assignment of detections to active tracks.
    Returns (matched pairs, unmatched_det_indices, unmatched_track_indices).
    """
    if not tracks or not dets:
        return [], list(range(len(dets))), list(range(len(tracks)))

    try:
        from scipy.optimize import linear_sum_assignment
        iou_mat = _iou_matrix(tracks, dets)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched = [
            (r, c) for r, c in zip(row_ind, col_ind)
            if iou_mat[r, c] >= IOU_THRESHOLD
        ]
        matched_r = {r for r, _ in matched}
        matched_c = {c for _, c in matched}
    except ImportError:
        # Fallback: greedy matching without scipy
        matched, matched_r, matched_c = [], set(), set()
        iou_mat = _iou_matrix(tracks, dets)
        # Sort by descending IoU
        pairs = sorted(
            [(i, j, iou_mat[i, j]) for i in range(len(tracks))
             for j in range(len(dets))],
            key=lambda x: -x[2]
        )
        for i, j, iou in pairs:
            if iou < IOU_THRESHOLD:
                break
            if i not in matched_r and j not in matched_c:
                matched.append((i, j))
                matched_r.add(i)
                matched_c.add(j)

    unmatched_dets   = [j for j in range(len(dets))   if j not in matched_c]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_r]
    return matched, unmatched_dets, unmatched_tracks


# ─────────────────────────────────────────────────────────────────────────────
# Replicate YOLO-World detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_one_frame(
    frame_path: Path,
    frame_name: str,
    frame_idx:  int,
    timestamp_ms: int,
    classes:    list[str],
    confidence: float,
    footprint:  dict,           # from SpatialEngine.compute_footprint()
    frame_w:    int,
    frame_h:    int,
) -> list[Detection]:
    """
    Run YOLO-World on a single frame via Replicate API.
    Converts output bboxes to GPS using the frame footprint.
    """
    # Guard against missing API token — fails fast with a clear message
    import os as _os
    if not _os.environ.get("REPLICATE_API_TOKEN"):
        logger.warning(
            "REPLICATE_API_TOKEN not set — YOLO-World skipped for %s", frame_name
        )
        return []

    try:
        import replicate

        with open(frame_path, "rb") as fh:
            output = replicate.run(
                YOLO_WORLD_MODEL,
                input={
                    "image":      fh,
                    "text":       ", ".join(classes),
                    "confidence": confidence,
                },
            )

        # Normalise output — Replicate models vary in output format
        raw_dets = []
        if isinstance(output, list):
            raw_dets = output
        elif isinstance(output, dict):
            raw_dets = output.get("detections", output.get("boxes", []))
        elif isinstance(output, str):
            try:
                raw_dets = json.loads(output)
            except Exception:
                logger.warning("Unparseable YOLO-World output for %s: %s",
                               frame_name, output[:200])
                return []

        dets = []
        for rd in raw_dets:
            # Support both "name"/"label" and "xmin"/"x1" field names
            label = rd.get("name") or rd.get("label") or rd.get("class", "object")
            conf  = float(rd.get("confidence", rd.get("score", 1.0)))
            bbox  = _parse_bbox(rd, frame_w, frame_h)
            if not bbox:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            lat, lon = _pixel_to_gps(cx, cy, frame_w, frame_h, footprint)
            dets.append(Detection(
                frame_name   = frame_name,
                frame_idx    = frame_idx,
                timestamp_ms = timestamp_ms,
                bbox         = bbox,
                confidence   = conf,
                label        = label,
                lat          = lat,
                lon          = lon,
            ))
        logger.info("YOLO-World | %s → %d detection(s)", frame_name, len(dets))
        return dets

    except Exception as e:
        logger.error("YOLO-World failed for %s: %s", frame_name, e)
        return []


def _parse_bbox(raw: dict, frame_w: int, frame_h: int) -> list | None:
    """Normalise bbox dict to [xmin, ymin, xmax, ymax] in pixels."""
    try:
        # Pixel coords (most common)
        if "xmin" in raw:
            return [int(raw["xmin"]), int(raw["ymin"]),
                    int(raw["xmax"]), int(raw["ymax"])]
        if "x1" in raw:
            return [int(raw["x1"]), int(raw["y1"]),
                    int(raw["x2"]), int(raw["y2"])]
        # Normalised [0,1] coords
        if "x_center" in raw:
            x, y   = raw["x_center"] * frame_w, raw["y_center"] * frame_h
            w2, h2 = raw["width"] * frame_w / 2, raw["height"] * frame_h / 2
            return [int(x - w2), int(y - h2), int(x + w2), int(y + h2)]
        # XYXY normalised
        if "bbox" in raw and len(raw["bbox"]) == 4:
            b = raw["bbox"]
            if max(b) <= 1.0:
                return [int(b[0]*frame_w), int(b[1]*frame_h),
                        int(b[2]*frame_w), int(b[3]*frame_h)]
            return [int(v) for v in b]
    except Exception:
        pass
    return None


def _pixel_to_gps(
    cx: float, cy: float,
    frame_w: int, frame_h: int,
    footprint: dict,
) -> tuple[float, float]:
    """
    Convert a full-frame pixel to WGS84 using bilinear interpolation
    across the 4 GPS corners of the frame footprint.
    """
    u = cx / frame_w          # 0 = left,  1 = right
    v = cy / frame_h          # 0 = top,   1 = bottom
    nw = footprint["nw"]      # (lat, lon)
    ne = footprint["ne"]
    se = footprint["se"]
    sw = footprint["sw"]
    lat = ((1-v) * ((1-u)*nw[0] + u*ne[0]) +
                v * ((1-u)*sw[0] + u*se[0]))
    lon = ((1-v) * ((1-u)*nw[1] + u*ne[1]) +
                v * ((1-u)*sw[1] + u*se[1]))
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# SORT tracker
# ─────────────────────────────────────────────────────────────────────────────

# Palette for track colours — cycles through available colours
_TRACK_COLORS = [
    "#60a5fa", "#f59e0b", "#a78bfa", "#34d399",
    "#f472b6", "#fb923c", "#e879f9", "#4ade80",
]


class SORTTracker:
    """
    Minimal SORT implementation for TAE.

    At each frame, receives a list of Detections and returns the current
    set of active Tracks with updated positions.
    """

    def __init__(self) -> None:
        self._tracks: list[Track] = []
        self._next_color_idx = 0

    def update(self, dets: list[Detection]) -> list[Track]:
        """Process one frame's detections and return updated active tracks."""
        # Age all existing tracks
        for t in self._tracks:
            t.age += 1

        active = [t for t in self._tracks if t.age <= MAX_AGE]
        matched, unmatched_dets, _ = _assign(active, dets)

        # Update matched tracks
        for t_idx, d_idx in matched:
            t = active[t_idx]
            d = dets[d_idx]
            t.hits    += 1
            t.age      = 0
            t.last_bbox = d.bbox
            t.detections.append({
                "frame_name":   d.frame_name,
                "frame_idx":    d.frame_idx,
                "timestamp_ms": d.timestamp_ms,
                "bbox":         d.bbox,
                "confidence":   round(d.confidence, 3),
                "lat":          round(d.lat, 7),
                "lon":          round(d.lon, 7),
            })

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            d = dets[d_idx]
            color = _TRACK_COLORS[self._next_color_idx % len(_TRACK_COLORS)]
            self._next_color_idx += 1
            trk = Track(label=d.label, color=color, hits=1, age=0,
                        last_bbox=d.bbox)
            trk.detections.append({
                "frame_name":   d.frame_name,
                "frame_idx":    d.frame_idx,
                "timestamp_ms": d.timestamp_ms,
                "bbox":         d.bbox,
                "confidence":   round(d.confidence, 3),
                "lat":          round(d.lat, 7),
                "lon":          round(d.lon, 7),
            })
            active.append(trk)

        self._tracks = active
        return [t for t in active if t.hits >= MIN_HITS]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Streaming entry point (Phase E)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level SORT tracker that persists across individual frame arrivals
# during live streaming.  Reset when a new stream starts.
_stream_tracker: SORTTracker | None = None
_stream_classes: list[str] = []


def init_stream_tracker(query: str) -> None:
    """
    Create a fresh SORT tracker for a new live stream.
    Called once by StreamManager when streaming starts.
    """
    global _stream_tracker, _stream_classes
    _stream_tracker = SORTTracker()
    _stream_classes = _query_to_classes(query)
    logger.info("Stream tracker initialised | classes=%s", _stream_classes)


def track_one_frame(
    frame_path:   Path,
    frame_name:   str,
    frame_idx:    int,
    timestamp_ms: int,
    footprint:    dict,
    frame_w:      int,
    frame_h:      int,
    tracks_file:  Path,
    confidence:   float = 0.15,
) -> list[Track]:
    """
    Streaming entry point: detect on ONE incoming frame and update the
    persistent SORT tracker.  Saves the current tracks to disk and returns
    the active track list so the caller can trigger a map update.

    Called by StreamManager._process_frames() every FRAME_INTERVAL_S.
    Replicate latency (~1-3 s) is within the 2 s frame interval — if it
    occasionally exceeds it, the next frame is simply delayed, not dropped.
    """
    global _stream_tracker, _stream_classes
    if _stream_tracker is None:
        logger.warning("track_one_frame called before init_stream_tracker — skipping")
        return []

    # Stage 1: single YOLO-World call (no parallelism needed — one frame)
    dets = _detect_one_frame(
        frame_path, frame_name, frame_idx, timestamp_ms,
        _stream_classes, confidence, footprint, frame_w, frame_h,
    )

    # Stage 2: incremental SORT update — same algorithm as batch case
    active = _stream_tracker.update(dets)

    # Stage 3: persist current track state
    confirmed = [t for t in active if t.hits >= MIN_HITS]
    save_tracks(confirmed, tracks_file)

    return confirmed


def reset_stream_tracker() -> None:
    """Called by StreamManager.stop() to clean up."""
    global _stream_tracker, _stream_classes
    _stream_tracker = None
    _stream_classes = []


# ─────────────────────────────────────────────────────────────────────────────
# Batch entry point (video upload)
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(
    frames: list[dict],         # sorted list of frame dicts from LanceDB
    spatial,                    # SpatialEngine instance
    query: str,                 # natural-language query → YOLO-World classes
    confidence: float = 0.15,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Track]:
    """
    Full tracking pipeline for a mission's sampled frames.

    Parameters
    ----------
    frames      : list of LanceDB row dicts, one per unique parent frame,
                  sorted by frame_idx / timestamp ascending.
                  Each row must have: parent_path, lat, lon, alt_m,
                  gimbal_yaw, img_w_px, img_h_px (and fp_* footprint cols).
    spatial     : SpatialEngine (for compute_footprint).
    query       : The natural-language query.  Split into class tokens for
                  YOLO-World (e.g. "find vehicles and people" → ["vehicle","person"]).
    confidence  : YOLO-World confidence threshold.
    on_progress : optional callback(done, total) for UI progress.

    Returns
    -------
    list of confirmed Track objects (hits >= MIN_HITS).
    """
    # Parse query into class names
    classes = _query_to_classes(query)
    logger.info("Tracking | classes=%s | frames=%d", classes, len(frames))

    # ── Stage 1: parallel dense detection ─────────────────────────────────────
    total = len(frames)
    frame_dets: dict[int, list[Detection]] = {}   # frame_idx → detections

    def _detect_frame(frame_row: dict) -> tuple[int, list[Detection]]:
        fp_path    = Path(frame_row["parent_path"])
        frame_name = fp_path.name
        frame_idx  = frame_row.get("frame_idx", 0)
        ts_ms      = frame_row.get("timestamp_ms", 0)
        lat        = frame_row["lat"]
        lon        = frame_row["lon"]
        alt_m      = float(frame_row.get("alt_m", 80.0))
        yaw_deg    = float(frame_row.get("gimbal_yaw", 0.0))
        img_w      = int(frame_row.get("img_w_px", 1920))
        img_h      = int(frame_row.get("img_h_px", 1080))

        footprint = spatial.compute_footprint(
            lat, lon, alt_m, yaw_deg, img_w, img_h
        )
        dets = _detect_one_frame(
            fp_path, frame_name, frame_idx, ts_ms,
            classes, confidence, footprint, img_w, img_h,
        )
        return frame_idx, dets

    done_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_detect_frame, fr): fr for fr in frames}
        for fut in as_completed(futures):
            fi, dets = fut.result()
            frame_dets[fi] = dets
            done_count += 1
            if on_progress:
                on_progress(done_count, total)

    # ── Stage 2: SORT association in temporal order ────────────────────────────
    tracker  = SORTTracker()
    finished: list[Track] = []

    for fi in sorted(frame_dets.keys()):
        dets    = frame_dets[fi]
        active  = tracker.update(dets)

    finished = [t for t in tracker._tracks if t.hits >= MIN_HITS]
    # Filter tracks with at least 2 GPS points (otherwise no polyline)
    finished = [t for t in finished if len(t.detections) >= 1]

    logger.info("Tracking complete: %d track(s)", len(finished))
    return finished


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_tracks(tracks: list[Track], tracks_file: Path) -> None:
    tracks_file.parent.mkdir(parents=True, exist_ok=True)
    data = [t.to_dict() for t in tracks]
    tracks_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved %d track(s) to %s", len(tracks), tracks_file)


def load_tracks(tracks_file: Path) -> list[dict]:
    if not tracks_file.exists():
        return []
    try:
        return json.loads(tracks_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load tracks: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Query parsing
# ─────────────────────────────────────────────────────────────────────────────

# Action verbs and filler words to strip before sending to YOLO-World
_STRIP_PREFIXES = {
    "find", "detect", "track", "locate", "show", "identify",
    "count", "get", "look", "search",
}

def _query_to_classes(query: str) -> list[str]:
    """
    Convert a natural-language query to YOLO-World text prompt(s).

    YOLO-World is open-vocabulary: it accepts natural language phrases as
    class descriptions, not just COCO class names.  So:

      "find white cars and large rocks"  →  ["white car", "large rock"]
      "find a house with a red roof"     →  ["house with a red roof"]
      "detect all vehicles"              →  ["vehicle"]

    The full query is passed through to preserve attribute context
    (colour, size, etc.) that VLM would otherwise need to re-check.
    We strip only leading action verbs like "find"/"detect".

    The list is split on "and" / "," to give YOLO-World separate prompts
    when the user asks for multiple object types — YOLO-World handles each
    prompt independently and returns merged results.
    """
    q = query.strip().lower()

    # Strip leading action verbs
    for prefix in _STRIP_PREFIXES:
        if q.startswith(prefix + " "):
            q = q[len(prefix):].lstrip()
            break
    # Also strip filler "all", "a", "an", "the"
    for filler in ("all ", "a ", "an ", "the "):
        if q.startswith(filler):
            q = q[len(filler):]

    # Split on "and" and commas
    import re as _re
    parts = [p.strip() for p in _re.split(r",| and | or ", q) if p.strip()]

    if not parts:
        return [query.strip()]

    # Deduplicate while preserving order
    seen, result = set(), []
    for p in parts:
        if p not in seen and len(p) >= 2:
            seen.add(p)
            result.append(p)

    return result[:6]   # YOLO-World handles up to ~6 independent prompts well