"""
core/ingestion.py
-----------------
Theater ingestion pipeline: frame → tiles → CLIP vectors + footprints → LanceDB.

Imported by both main.py (CLI) and tae_app.py (web UI).
"""

import logging
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger("TAE.Ingestion")

TILE_SIZE    = 640
TILE_OVERLAP = 64


# ── Footprint helpers ─────────────────────────────────────────────────────────

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
    u0, v0 = tx / fw,        ty / fh
    u1, v1 = (tx + tw) / fw, (ty + th) / fh
    return {
        "nw":        _bilinear(nw, ne, sw, se, u0, v0),
        "ne":        _bilinear(nw, ne, sw, se, u1, v0),
        "se":        _bilinear(nw, ne, sw, se, u1, v1),
        "sw":        _bilinear(nw, ne, sw, se, u0, v1),
        "gsd_cm_px": frame_fp["gsd_cm_px"],
    }


def _iter_tiles(img: np.ndarray, tile_size: int, overlap: int):
    h, w = img.shape[:2]
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            yield img[y:y2, x:x2], x, y, x2 - x, y2 - y


# ── Public API ────────────────────────────────────────────────────────────────

def run_ingestion(sim, spatial, search_lib, db,
                  tile_dir: Path | None = None,
                  tile_size: int = TILE_SIZE,
                  tile_overlap: int = TILE_OVERLAP) -> tuple:
    """
    Iterate over all frames in *sim*, slice into tiles, encode with CLIP,
    compute footprints, write to *db*.

    Tiles are NOT written to disk — they are reconstructed on demand at query
    time by cropping the parent frame using the stored pixel offsets.
    image_path in the DB is a synthetic name used only as a display label.

    Returns (tiles_ok, frames_failed).
    """
    logger.info(
        f"=== Ingestion start | tile_size={tile_size}px overlap={tile_overlap}px ==="
    )

    # ── Build set of already-indexed parent paths from DB ────────────────────
    # This is the authoritative dedup check: a file may exist on disk but
    # not be in the DB (failed run), so we always use the DB as truth.
    already_indexed: set[str] = set()
    try:
        if db.row_count() > 0:
            df = db.table.to_pandas()
            already_indexed = set(df["parent_path"].tolist())
            logger.info(
                f"{len(already_indexed)} parent frame(s) already in DB — "
                f"will skip if re-uploaded."
            )
    except Exception as e:
        logger.warning(f"Could not read existing parent paths: {e}. Will ingest all.")

    tiles_ok = frames_failed = frames_skipped = 0
    frame_idx = -1

    for frame_idx, (img_cv2, telemetry) in enumerate(sim):
        frame_name = Path(telemetry.get("full_path", f"frame_{frame_idx}")).name
        parent_path = Path(telemetry["full_path"])

        # ── Skip if already indexed ───────────────────────────────────────────
        if str(parent_path) in already_indexed:
            logger.info(f"[{frame_idx+1}] {frame_name} | already indexed — skipping.")
            frames_skipped += 1
            continue

        # Tile output dir: alongside the parent frame if not specified
        out_dir = tile_dir if tile_dir else parent_path.parent

        try:
            fh, fw = img_cv2.shape[:2]
            telemetry["img_w_px"] = fw
            telemetry["img_h_px"] = fh

            frame_fp = spatial.compute_footprint(
                lat=telemetry["lat"],
                lon=telemetry["lon"],
                alt_m=telemetry["z"],
                gimbal_yaw_deg=telemetry.get("gimbal_yaw", 0.0),
                img_w_px=fw,
                img_h_px=fh,
            )

            tile_list = list(_iter_tiles(img_cv2, tile_size, tile_overlap))
            logger.info(
                f"[{frame_idx+1}] {frame_name} | "
                f"{fw}×{fh}px | "
                f"alt {telemetry['z']:.1f}m | "
                f"GSD {frame_fp['gsd_cm_px']:.1f} cm/px | "
                f"{len(tile_list)} tiles"
            )

            rows = []
            for tile_img, tx, ty, tw, th in tile_list:
                vector  = search_lib.encode_image(tile_img)
                tile_fp = _tile_footprint(frame_fp, tx, ty, tw, th, fw, fh)

                # Synthetic name only — tile is never written to disk.
                # Reconstruction at query time: crop parent_path at (tile_x, tile_y, tile_w, tile_h)
                tile_fname = f"{parent_path.stem}_t{tx}_{ty}.jpg"

                rows.append({
                    "vector":      vector.tolist() if hasattr(vector, "tolist") else vector,
                    "image_path":  tile_fname,  # display label only, not a real path
                    "parent_path": str(telemetry["full_path"]),
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
            logger.info(f"[{frame_idx+1}] {frame_name} | ✓ {len(rows)} tiles written to DB")

        except Exception:
            logger.error(f"[{frame_idx+1}] {frame_name} FAILED:\n{traceback.format_exc()}")
            frames_failed += 1

    total_frames = frame_idx + 1  # includes skipped frames
    logger.info(
        f"=== Ingestion complete | "
        f"{tiles_ok} tiles | "
        f"{total_frames - frames_failed - frames_skipped}/{total_frames} frames newly indexed | "
        f"{frames_skipped} skipped (already in DB) | "
        f"{frames_failed} failed ==="
    )
    return tiles_ok, frames_failed