import logging
import cv2
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import time

from ai import analyst
from config import settings
from core.sim_provider import SimD3Environment
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst
from core.object_detection import merge_detections, ObjectInstance, Detection

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
        t0 = time.perf_counter()
        tiles = list(spatial.tile_image(img_cv2))  # [(tile_img, x, y, w, h), ...]
        t1 = time.perf_counter()

        # FIX 1: Single batched CLIP forward pass for all tiles in the frame
        tile_imgs = [t[0] for t in tiles]
        vectors = search_lib.encode_image_batch(tile_imgs)  # shape (N, 512)
        t2 = time.perf_counter()

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
        t3 = time.perf_counter()
        
        db.add_observations_batch(rows)
        t4 = time.perf_counter()
        total_tiles += len(rows)

        if frame_idx == 0:
            logger.info(
                f"Frame 0 timing: tiling={t1-t0:.2f}s | "
                f"CLIP={t2-t1:.2f}s | rows={t3-t2:.2f}s | DB={t4-t3:.2f}s"
            )

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

def render_results(instances: list[ObjectInstance]):
    os.makedirs("tactical_results", exist_ok=True)

    for inst in instances:
        best = inst.best   # highest-confidence detection

        # Render the best view
        tile_img = _load_tile(best.candidate)
        if tile_img is None:
            continue

        h, w = tile_img.shape[:2]
        xmin, ymin, xmax, ymax = [int(v) for v in best.bbox]
        cv2.rectangle(tile_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        tag = f"OBJ#{inst.instance_id} ({best.confidence:.2f})"
        if inst.is_multiangle:
            tag += f" [{len(inst.detections)} angles]"
        cv2.putText(tile_img, tag, (xmin, max(ymin-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Also mark secondary angles with a different colour
        for secondary in inst.detections[1:]:
            sec_img = _load_tile(secondary.candidate)
            if sec_img is None:
                continue
            sx1, sy1, sx2, sy2 = [int(v) for v in secondary.bbox]
            cv2.rectangle(sec_img, (sx1, sy1), (sx2, sy2), (255, 165, 0), 3)
            cv2.putText(sec_img, f"OBJ#{inst.instance_id} angle#{inst.detections.index(secondary)+1}",
                        (sx1, max(sy1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            fname = Path(secondary.filename).stem
            cv2.imwrite(f"tactical_results/obj{inst.instance_id}_angle{inst.detections.index(secondary)+1}_{fname}.jpg", sec_img)

        status = (f"OBJ#{inst.instance_id} | "
                  f"LAT:{best.lat:.6f} LON:{best.lon:.6f} | "
                  f"conf:{best.confidence:.2f} | "
                  f"{'MULTI-ANGLE x'+str(len(inst.detections)) if inst.is_multiangle else 'SINGLE VIEW'}")
        cv2.rectangle(tile_img, (0, h-50), (w, h), (0, 0, 0), -1)
        cv2.putText(tile_img, status, (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        fname = Path(best.filename).stem
        cv2.imwrite(f"tactical_results/obj{inst.instance_id}_best_{fname}.jpg", tile_img)
        logger.info(
            f"Object #{inst.instance_id} | "
            f"{best.lat:.6f},{best.lon:.6f} | "
            f"conf:{best.confidence:.2f} | "
            f"{'multi-angle: '+str(len(inst.detections))+' views' if inst.is_multiangle else 'single view'}"
        )

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

    db.initialize_table(vector_dim=settings.CLIP_DIM)

    if check_ingestion_needed(db):
        run_ingestion(sim, spatial, search_lib, db)
    else:
        logger.info(f"Persistent index found ({db.row_count()} tiles). Skipping ingestion.")

    user_request = "Find a helipad marked with H"
    candidates   = run_search(user_request, search_lib, db, limit=5)
    intel        = run_vlm_analysis(candidates, user_request, analyst)

    # Geo-NMS + Multi-Angle Persistence
    instances = merge_detections(intel.get('targets', []), candidates)
    logger.info(
        f"Detections: {sum(len(i.detections) for i in instances)} raw → "
        f"{len(instances)} unique objects | "
        f"{sum(1 for i in instances if i.is_multiangle)} multi-angle"
    )

    render_results(instances)

    print(f"\n{'=' * 60}")
    print("TACTICAL REPORT")
    print('=' * 60)
    print(intel.get('report', 'No data'))
    print('=' * 60)

    logger.info("=== MISSION COMPLETE ===")


if __name__ == "__main__":
    main()
