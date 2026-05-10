"""
core/ingestion.py
-----------------
Theater ingestion pipeline — two stages:

Stage 1 — CLIP tile encoding  (object_search)
  Always runs.  Slices each frame into 640px tiles, encodes with CLIP,
  stores vectors + footprints in LanceDB.  Identical to the original pipeline.

Stage 2 — SAM2 frame segmentation  (anomaly_detection)
  Runs only when "anomaly_detection" is in the mission's allowed_intents
  AND a segmentation model is available.  Runs SAM2 on the full frame
  (not tiles) to produce region masks, encodes each region with CLIP,
  stores results in the mission's SegmentStore (SQLite).

Both stages are idempotent: frames already processed are skipped.
"""

import logging
import traceback
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("TAE.Ingestion")

TILE_SIZE    = 640
TILE_OVERLAP = 64


# ── Footprint helpers (Stage 1) ───────────────────────────────────────────────

def _bilinear(nw, ne, sw, se, u: float, v: float) -> tuple:
    top_lat = nw[0] + u * (ne[0] - nw[0])
    top_lon = nw[1] + u * (ne[1] - nw[1])
    bot_lat = sw[0] + u * (se[0] - sw[0])
    bot_lon = sw[1] + u * (se[1] - sw[1])
    return (
        top_lat + v * (bot_lat - top_lat),
        top_lon + v * (bot_lon - top_lon),
    )


def _tile_footprint(frame_fp: dict, tx: int, ty: int,
                    tw: int, th: int, fw: int, fh: int) -> dict:
    nw, ne = frame_fp["nw"], frame_fp["ne"]
    sw, se = frame_fp["sw"], frame_fp["se"]
    u0, v0 = tx / fw,         ty / fh
    u1, v1 = (tx + tw) / fw,  (ty + th) / fh
    return {
        "nw":        _bilinear(nw, ne, sw, se, u0, v0),
        "ne":        _bilinear(nw, ne, sw, se, u1, v0),
        "se":        _bilinear(nw, ne, sw, se, u1, v1),
        "sw":        _bilinear(nw, ne, sw, se, u0, v1),
        "gsd_cm_px": frame_fp["gsd_cm_px"],
    }


def _iter_tiles(img: np.ndarray, tile_size: int, overlap: int):
    h, w   = img.shape[:2]
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            yield img[y:y2, x:x2], x, y, x2 - x, y2 - y


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — CLIP tile encoding
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    sim,
    spatial,
    search_lib,
    db,
    tile_size:    int           = TILE_SIZE,
    tile_overlap: int           = TILE_OVERLAP,
    on_frame      = None,
) -> tuple[int, int]:
    """
    Stage 1: slice every frame into tiles, encode with CLIP, write to LanceDB.

    Returns (tiles_ok, frames_failed).
    Idempotent: frames already in DB (matched by parent_path) are skipped.
    """
    logger.info(
        f"=== Stage 1 start | tile_size={tile_size}px overlap={tile_overlap}px ==="
    )

    # Build set of already-indexed parent paths — DB is the source of truth
    already_indexed: set[str] = set()
    if db.table is not None:
        try:
            df = db.table.to_pandas()
            already_indexed = set(df["parent_path"].unique())
        except Exception:
            pass

    tiles_ok       = 0
    frames_failed  = 0
    frames_skipped = 0

    for frame_idx, (img_cv2, telemetry) in enumerate(sim):
        frame_name  = Path(telemetry["full_path"]).name
        parent_path = Path(telemetry["full_path"])

        if on_frame:
            on_frame(frame_idx, frame_name)

        if str(parent_path) in already_indexed:
            logger.info(f"[{frame_idx+1}] {frame_name} — already indexed, skipping")
            frames_skipped += 1
            continue

        try:
            fh, fw = img_cv2.shape[:2]
            frame_fp = spatial.compute_footprint(
                lat           = telemetry["lat"],
                lon           = telemetry["lon"],
                alt_m         = telemetry["z"],
                gimbal_yaw_deg= telemetry.get("gimbal_yaw", 0.0),
                img_w_px      = fw,
                img_h_px      = fh,
            )

            rows: list[dict] = []
            for tile_img, tx, ty, tw, th in _iter_tiles(img_cv2, tile_size, tile_overlap):
                vector   = search_lib.encode_image(tile_img)
                tile_fp  = _tile_footprint(frame_fp, tx, ty, tw, th, fw, fh)
                tile_fname = f"{parent_path.stem}_t{tx}_{ty}.jpg"

                rows.append({
                    "vector":      vector.tolist() if hasattr(vector, "tolist") else vector,
                    "image_path":  tile_fname,
                    "parent_path": str(parent_path),
                    "tile_x":      int(tx),
                    "tile_y":      int(ty),
                    "tile_w":      int(tw),
                    "tile_h":      int(th),
                    "lat":         float(telemetry["lat"]),
                    "lon":         float(telemetry["lon"]),
                    "alt_m":       float(telemetry["z"]),
                    "gimbal_yaw":  float(telemetry.get("gimbal_yaw", 0.0)),
                    "gsd_cm_px":   float(tile_fp["gsd_cm_px"]),
                    "fp_nw_lat":   float(tile_fp["nw"][0]),
                    "fp_nw_lon":   float(tile_fp["nw"][1]),
                    "fp_ne_lat":   float(tile_fp["ne"][0]),
                    "fp_ne_lon":   float(tile_fp["ne"][1]),
                    "fp_se_lat":   float(tile_fp["se"][0]),
                    "fp_se_lon":   float(tile_fp["se"][1]),
                    "fp_sw_lat":   float(tile_fp["sw"][0]),
                    "fp_sw_lon":   float(tile_fp["sw"][1]),
                })

            db.add_observations_batch(rows)
            tiles_ok += len(rows)
            logger.info(
                f"[{frame_idx+1}] {frame_name} | ✓ {len(rows)} tiles → LanceDB"
            )

        except Exception:
            logger.error(
                f"[{frame_idx+1}] {frame_name} FAILED:\n{traceback.format_exc()}"
            )
            frames_failed += 1

    logger.info(
        f"=== Stage 1 complete | {tiles_ok} tiles | "
        f"{frames_skipped} skipped | {frames_failed} failed ==="
    )
    return tiles_ok, frames_failed


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — SAM2 frame segmentation
# ─────────────────────────────────────────────────────────────────────────────

def run_sam2_segmentation(
    sim,
    segmentor,
    segment_store,
    search_lib,
    on_frame = None,
) -> tuple[int, int]:
    """
    Stage 2: segment each frame with SAM2, CLIP-encode every region,
    store in the mission's SegmentStore.

    Operates on full frames (not tiles) for:
      - Better mask quality (full context for SAM2)
      - No boundary truncation at tile edges
      - Region CLIP vectors computed from full-frame crops

    Returns (segments_written, frames_failed).
    Idempotent: frames already in segment_store are skipped.

    Parameters
    ----------
    sim           : SimD3Environment  — same sim object used in Stage 1
    segmentor     : SAM2Segmentor     — wraps SAM2 or SAM
    segment_store : SegmentStore      — mission's SQLite segment DB
    search_lib    : SearchLibrarian   — CLIP model (reused from Stage 1)
    on_frame      : optional callback(frame_idx, frame_name) for progress
    """
    if not segmentor.is_available():
        logger.warning(
            "Stage 2 skipped — no segmentation model available. "
            "Configure SAM2_CHECKPOINT or SAM_CHECKPOINT in settings."
        )
        return 0, 0

    logger.info("=== Stage 2 start | SAM2 frame segmentation ===")

    segments_written = 0
    frames_failed    = 0

    for frame_idx, (img_cv2, telemetry) in enumerate(sim):
        frame_path = str(telemetry["full_path"])
        frame_name = Path(frame_path).name

        if on_frame:
            on_frame(frame_idx, frame_name)

        # Idempotency: skip frames already segmented
        if segment_store.has_frame(frame_path):
            logger.info(
                f"[{frame_idx+1}] {frame_name} — segments already stored, skipping"
            )
            continue

        try:
            # Generate region masks on the full frame
            regions = segmentor.generate(img_cv2)

            if not regions:
                logger.info(
                    f"[{frame_idx+1}] {frame_name} | "
                    f"no regions survived filtering"
                )
                # Insert a sentinel to mark this frame as processed
                # so it won't be retried on the next ingest run.
                segment_store.insert_segments(frame_path, [])
                continue

            # CLIP-encode each region crop
            segments_to_store: list[dict] = []
            for region in regions:
                crop = region["crop"]   # BGR numpy array from segmentor
                if crop.size == 0:
                    continue
                clip_vec = search_lib.encode_image(crop)
                segments_to_store.append({
                    "bbox":        region["bbox"],
                    "area":        region["area"],
                    "clip_vector": clip_vec,
                })

            n = segment_store.insert_segments(frame_path, segments_to_store)
            segments_written += n
            logger.info(
                f"[{frame_idx+1}] {frame_name} | "
                f"✓ {n} segments → SegmentStore"
            )

        except Exception:
            logger.error(
                f"[{frame_idx+1}] {frame_name} Stage 2 FAILED:\n"
                f"{traceback.format_exc()}"
            )
            frames_failed += 1

    logger.info(
        f"=== Stage 2 complete | {segments_written} segments | "
        f"{frames_failed} frame(s) failed ==="
    )
    return segments_written, frames_failed
