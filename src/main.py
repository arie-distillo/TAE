import logging
import cv2
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

from ai import analyst
from config import settings
from core.sim_provider import SimD3Environment
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TAE-Core")


# ---------------------------------------------------------------------------
# Phase 1: Ingestion
# ---------------------------------------------------------------------------

def run_ingestion(sim, spatial, search_lib, db):
    logger.info("Starting Fresh Ingestion Phase...")
    total_tiles = 0

    for frame_idx, (img_cv2, telemetry) in enumerate(sim):
        img_h, img_w = img_cv2.shape[:2]

        frame_footprint = spatial.compute_footprint(
            lat=telemetry['lat'], lon=telemetry['lon'],
            alt_m=telemetry['z'], gimbal_yaw_deg=telemetry['gimbal_yaw'],
            img_w_px=img_w, img_h_px=img_h,
        )

        # Collect all tiles for this frame first
        tiles = list(spatial.tile_image(img_cv2))  # [(tile_img, x, y, w, h), ...]

        # FIX 1: Single batched CLIP forward pass for all tiles in the frame
        tile_imgs = [t[0] for t in tiles]
        vectors = search_lib.encode_image_batch(tile_imgs)  # shape (N, 512)

        # FIX 3: Accumulate all rows, write once per frame
        rows = []
        for (tile_img, x_off, y_off, tile_w, tile_h), vector in zip(tiles, vectors):
            tile_footprint = spatial.compute_tile_footprint(
                x_off=x_off, y_off=y_off,
                tile_w=tile_w, tile_h=tile_h,
                img_w=img_w, img_h=img_h,
                frame_footprint=frame_footprint,
            )
            # FIX 2: No disk write — store parent path + offsets only
            rows.append({
                "vector":      vector.tolist(),
                "image_path":  str(telemetry['full_path']),  # parent path, not tile
                "parent_path": str(telemetry['full_path']),
                "tile_x":      int(x_off),
                "tile_y":      int(y_off),
                "tile_w":      int(tile_w),
                "tile_h":      int(tile_h),
                "lat":         float(telemetry['lat']),
                "lon":         float(telemetry['lon']),
                "alt_m":       float(telemetry['z']),
                "gimbal_yaw":  float(telemetry.get('gimbal_yaw', 0.0)),
                "gsd_cm_px":   float(tile_footprint['gsd_cm_px']),
                "fp_nw_lat":   float(tile_footprint['nw'][0]),
                "fp_nw_lon":   float(tile_footprint['nw'][1]),
                "fp_ne_lat":   float(tile_footprint['ne'][0]),
                "fp_ne_lon":   float(tile_footprint['ne'][1]),
                "fp_se_lat":   float(tile_footprint['se'][0]),
                "fp_se_lon":   float(tile_footprint['se'][1]),
                "fp_sw_lat":   float(tile_footprint['sw'][0]),
                "fp_sw_lon":   float(tile_footprint['sw'][1]),
            })

        db.add_observations_batch(rows)
        total_tiles += len(rows)

        if (frame_idx + 1) % 10 == 0:
            logger.info(
                f"Ingested {frame_idx + 1} frames | {total_tiles} tiles | "
                f"GSD: {frame_footprint['gsd_cm_px']} cm/px"
            )

    logger.info(f"Ingestion complete — {total_tiles} tiles indexed.")


# ---------------------------------------------------------------------------
# Phase 2: Search
# ---------------------------------------------------------------------------

def run_search(user_request, search_lib, db, limit=5) -> list:
    """
    Encodes the user query as a CLIP vector and retrieves the top N tiles.
    Tile-level retrieval gives much higher precision than frame-level because
    the target object occupies a larger fraction of the tile embedding space.
    """
    logger.info(f"Querying Vector DB: '{user_request}' (top {limit} tiles)")
    query_vec = search_lib.encode_text(user_request)
    candidates = db.semantic_search(query_vec, limit=limit)
    logger.info(f"Retrieved {len(candidates)} candidate tiles.")
    return candidates


# ---------------------------------------------------------------------------
# Phase 3: VLM Grounding
# ---------------------------------------------------------------------------

def run_vlm_analysis(candidates, user_request, analyst) -> dict:
    """
    Sends each candidate tile independently to the VLM for grounding.
    Returns merged { report, targets } across all tiles.
    """
    intel = analyst.analyze_multiple_views(candidates, user_request)
    logger.info(f"VLM report: {intel.get('report', 'No report returned')}")
    return intel


# ---------------------------------------------------------------------------
# Phase 4: Render
# ---------------------------------------------------------------------------

def _load_tile(candidate: dict) -> np.ndarray | None:
    """
    Reconstructs a tile by cropping its parent frame.
    Called at query time — tiles are never stored on disk.
    """
    img = cv2.imread(candidate['parent_path'])
    if img is None:
        logger.warning(f"Cannot read parent frame: {candidate['parent_path']}")
        return None
    x = candidate['tile_x']
    y = candidate['tile_y']
    w = candidate['tile_w']
    h = candidate['tile_h']
    return img[y:y+h, x:x+w]

def render_results(candidates, intel):
    """
    Draws VLM bounding boxes on candidate tiles and saves annotated images.
    Bboxes are in tile pixel coordinates — no frame-level translation needed.
    Also logs the geo-coordinates of each detected target.
    """
    os.makedirs("tactical_results", exist_ok=True)
    all_targets = intel.get('targets', [])

    for cand in candidates:
        fname = Path(cand['image_path']).name   # tile filename for target matching
        img = _load_tile(cand)                  # ← reconstruct from parent frame

        if img is None:
            logger.warning(f"Could not read tile: {cand['image_path']}")
            continue

        h, w = img.shape[:2]
        frame_targets = [t for t in all_targets
                         if t['filename'].lower() == fname.lower()]

        if frame_targets:
            for target in frame_targets:
                # Bboxes are in tile pixel coords — render directly
                xmin, ymin, xmax, ymax = [int(v) for v in target['bbox']]
                p1, p2 = (xmin, ymin), (xmax, ymax)
                cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
                label = f"TARGET ({target.get('confidence', '?')})"
                cv2.putText(img, label, (p1[0], max(p1[1] - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Log geo-coordinates of bbox center
                cx_frac = ((xmin + xmax) / 2) / w
                cy_frac = ((ymin + ymax) / 2) / h
                nw = (cand['fp_nw_lat'], cand['fp_nw_lon'])
                ne = (cand['fp_ne_lat'], cand['fp_ne_lon'])
                se = (cand['fp_se_lat'], cand['fp_se_lon'])
                sw = (cand['fp_sw_lat'], cand['fp_sw_lon'])
                target_lat = ((1 - cy_frac) * ((1 - cx_frac) * nw[0] + cx_frac * ne[0]) +
                                   cy_frac  * ((1 - cx_frac) * sw[0] + cx_frac * se[0]))
                target_lon = ((1 - cy_frac) * ((1 - cx_frac) * nw[1] + cx_frac * ne[1]) +
                                   cy_frac  * ((1 - cx_frac) * sw[1] + cx_frac * se[1]))
                logger.info(
                    f"Target geo: {target_lat:.6f}, {target_lon:.6f} | "
                    f"tile: {fname} | confidence: {target.get('confidence', '?')}"
                )

            color, label = (0, 255, 0), "VLM GROUNDED"
        else:
            cv2.drawMarker(img, (w // 2, h // 2), (0, 165, 255),
                           cv2.MARKER_CROSS, 60, 3)
            color, label = (0, 165, 255), "NO DETECTION"

        # Status bar
        fp_info = (f"NW({cand['fp_nw_lat']:.5f},{cand['fp_nw_lon']:.5f}) "
                   f"SE({cand['fp_se_lat']:.5f},{cand['fp_se_lon']:.5f})")
        status = f"{label} | {fp_info} | GSD:{cand['gsd_cm_px']:.1f}cm/px"
        cv2.rectangle(img, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        out_path = f"tactical_results/grounded_{fname}"
        cv2.imwrite(out_path, img)
        logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ingestion_needed(db) -> bool:
    """True if the index is missing or empty."""
    try:
        return db.row_count() == 0
    except Exception as e:
        logger.warning(f"Could not check index state: {e}. Will re-ingest.")
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("=== MISSION START: Tactical Awareness Engine ===")

    sim        = SimD3Environment(settings.SIM_METADATA_FILE)
    spatial    = SpatialEngine(settings.SENSOR_WIDTH_MM,
                               settings.SENSOR_HEIGHT_MM,
                               settings.FOCAL_LENGTH_MM)
    search_lib = SearchLibrarian(settings.CLIP_MODEL)
    db         = TacticalDatabase()
    analyst    = TacticalAnalyst(settings.AI_PROVIDER,
                                 settings.VLM_MODEL,
                                 settings.OPENROUTER_API_KEY)

    db.initialize_table(vector_dim=512)

    if check_ingestion_needed(db):
        run_ingestion(sim, spatial, search_lib, db)
    else:
        logger.info(f"Persistent index found ({db.row_count()} tiles). Skipping ingestion.")

    user_request = "Find a helipad marked with H"
    candidates   = run_search(user_request, search_lib, db, limit=5)
    intel        = run_vlm_analysis(candidates, user_request, analyst)

    render_results(candidates, intel)

    print(f"\n{'=' * 60}")
    print("TACTICAL REPORT")
    print('=' * 60)
    print(intel.get('report', 'No data'))
    print('=' * 60)

    logger.info("=== MISSION COMPLETE ===")


if __name__ == "__main__":
    main()
