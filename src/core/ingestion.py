"""
core/ingestion.py
-----------------
Theater ingestion pipeline: frame → tiles → CLIP vectors + footprints → LanceDB.

Each frame is sliced into overlapping square tiles. Each tile gets its own
CLIP embedding and a ground footprint computed by bilinear interpolation of
the parent frame's four WGS84 corners.

Imported by both main.py (CLI) and tae_app.py (web UI).
"""

import logging
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("TAE.Ingestion")

# ── Tile settings ─────────────────────────────────────────────────────────────
TILE_SIZE    = 640   # pixels — square tiles fed to CLIP
TILE_OVERLAP = 64    # pixels — overlap so objects on tile edges aren't lost


# ── Footprint helpers ─────────────────────────────────────────────────────────

def _bilinear(nw, ne, sw, se, u: float, v: float) -> tuple:
    """
    Bilinear interpolation over a quad defined by four (lat, lon) corners.
    u in [0,1] = West->East,  v in [0,1] = North->South
    Corner convention matches SpatialEngine.compute_footprint():
        nw=(0,0)  ne=(1,0)  sw=(0,1)  se=(1,1)
    """
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
    """
    Derive the ground footprint of a tile by bilinear interpolation of the
    parent frame's corner coordinates.
    """
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


# ── Tile generator ─────────────────────────────────────────────────────────────

def _iter_tiles(img: np.ndarray, tile_size: int = TILE_SIZE,
                overlap: int = TILE_OVERLAP):
    """
    Yield (tile_img, tx, ty, tw, th) for every tile in a sliding-window grid.
    The last tile in each row/column is clamped to the image boundary.
    """
    h, w = img.shape[:2]
    stride = tile_size - overlap
    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))

    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            tile = img[y:y2, x:x2]
            yield tile, x, y, x2 - x, y2 - y


# ── Public API ────────────────────────────────────────────────────────────────

def run_ingestion(sim, spatial, search_lib, db,
                  tile_size: int = TILE_SIZE,
                  tile_overlap: int = TILE_OVERLAP) -> tuple:
    """
    Iterate over all frames in *sim*, slice each into tiles, encode with CLIP,
    compute per-tile footprints, and write to *db*.

    Returns
    -------
    (tiles_ok, frames_failed) : int tuple
    """
    logger.info(
        f"Starting ingestion — tile_size={tile_size}px, overlap={tile_overlap}px"
    )
    tiles_ok = frames_failed = frame_idx = 0

    with tempfile.TemporaryDirectory(prefix="tae_tiles_") as tmp_dir:
        tmp = Path(tmp_dir)

        for frame_idx, (img_cv2, telemetry) in enumerate(sim):
            frame_name = Path(telemetry.get("full_path", f"frame_{frame_idx}")).name

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

                rows = []
                for tile_img, tx, ty, tw, th in _iter_tiles(img_cv2, tile_size, tile_overlap):
                    vector   = search_lib.encode_image(tile_img)
                    tile_fp  = _tile_footprint(frame_fp, tx, ty, tw, th, fw, fh)

                    # Persist tile image so the VLM can read it by path
                    tile_fname = f"{Path(frame_name).stem}_t{tx}_{ty}.jpg"
                    tile_path  = tmp / tile_fname
                    cv2.imwrite(str(tile_path), tile_img)

                    rows.append({
                        "vector":      vector.tolist() if hasattr(vector, "tolist") else vector,
                        "image_path":  str(tile_path),
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

                if (frame_idx + 1) % 10 == 0:
                    logger.info(
                        f"Indexed {frame_idx + 1} frames | "
                        f"{tiles_ok} tiles total | "
                        f"GSD: {frame_fp['gsd_cm_px']:.1f} cm/px"
                    )

            except Exception:
                logger.error(f"Frame skipped ({frame_name}):\n{traceback.format_exc()}")
                frames_failed += 1

    logger.info(
        f"Ingestion complete — {tiles_ok} tiles from "
        f"{frame_idx + 1 - frames_failed} frames "
        f"({frames_failed} frames failed)."
    )
    return tiles_ok, frames_failed