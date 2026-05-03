import logging
import cv2
import os
import sys
from pathlib import Path
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


def run_ingestion(sim, spatial, search_lib, db):
    """
    Phase 1: Theater Ingestion.
    Iterates over all simulated frames, computes CLIP vectors and footprints,
    and writes everything to LanceDB.
    """
    logger.info("Starting Fresh Ingestion Phase...")

    for i, (img_cv2, telemetry) in enumerate(sim):
        # CLIP embedding
        vector = search_lib.encode_image(img_cv2)

        # Footprint — requires gimbal_yaw and image dimensions from telemetry,
        # which are now present after the ingest_telemetry.py fix
        footprint = spatial.compute_footprint(
            lat=telemetry['lat'],
            lon=telemetry['lon'],
            alt_m=telemetry['z'],
            gimbal_yaw_deg=telemetry['gimbal_yaw'],
            img_w_px=telemetry['img_w_px'],
            img_h_px=telemetry['img_h_px']
        )

        db.add_observation(vector, telemetry, footprint)

        if (i + 1) % 10 == 0:
            logger.info(
                f"Indexed {i + 1} frames | "
                f"GSD: {footprint['gsd_cm_px']:.1f} cm/px | "
                f"Coverage: {footprint['ground_w_m']:.1f} x {footprint['ground_h_m']:.1f} m"
            )

    logger.info("Ingestion complete.")


def run_search(user_request, search_lib, db):
    """
    Phase 2: Semantic Search.
    Encodes the user query as a CLIP vector and retrieves top candidates from LanceDB.
    Returns the raw candidate list.
    """
    logger.info(f"Querying Vector DB: '{user_request}'")
    query_vec = search_lib.encode_text(user_request)
    candidates = db.semantic_search(query_vec, limit=3)
    logger.info(f"Retrieved {len(candidates)} candidates.")
    return candidates


def run_vlm_analysis(candidates, user_request, analyst):
    """
    Phase 3: VLM Grounding.
    Sends top candidate images to the VLM for precise bbox grounding.
    Returns the full intel dict: { report, targets }.
    """
    paths = [c['image_path'] for c in candidates]
    logger.info(f"Sending {len(paths)} images to VLM for grounding...")
    intel = analyst.analyze_multiple_views(paths, user_request)
    logger.info(f"VLM report: {intel.get('report', 'No report returned')}")
    return intel


def render_results(candidates, intel):
    """
    Phase 4: Annotated Output.
    Draws VLM bounding boxes (or fallback markers) on each candidate image
    and writes annotated files to tactical_results/.
    """
    os.makedirs("tactical_results", exist_ok=True)
    all_targets = intel.get('targets', [])

    for cand in candidates:
        img_path = cand['image_path']
        fname = Path(img_path).name
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        h, w, _ = img.shape
        frame_targets = [t for t in all_targets if t['filename'].lower() == fname.lower()]

        if frame_targets:
            for target in frame_targets:
                # Convention: [xmin, ymin, xmax, ymax], normalized 0-1000
                xmin, ymin, xmax, ymax = target['bbox']
                p1 = (int(xmin * w / 1000), int(ymin * h / 1000))
                p2 = (int(xmax * w / 1000), int(ymax * h / 1000))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 5)
                cv2.putText(img, "TARGET", (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            color, label = (0, 255, 0), "VLM GROUNDED"
        else:
            # No targets found for this frame — draw center fallback marker
            cv2.drawMarker(img, (w // 2, h // 2), (0, 165, 255),
                           cv2.MARKER_CROSS, 100, 5)
            color, label = (0, 165, 255), "CENTER FALLBACK"

        # Tactical status bar at bottom
        fp_info = (
            f"NW({cand['fp_nw_lat']:.5f},{cand['fp_nw_lon']:.5f}) "
            f"SE({cand['fp_se_lat']:.5f},{cand['fp_se_lon']:.5f})"
        )
        status = f"{label} | {fp_info} | GSD: {cand['gsd_cm_px']:.1f} cm/px"
        cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(img, status, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out_path = f"tactical_results/grounded_{fname}"
        cv2.imwrite(out_path, img)
        logger.info(f"Saved: {out_path}")


def check_ingestion_needed(db):
    """
    Returns True if the theater_index table doesn't exist or is empty.
    NOTE: database.py must use mode='create' (not 'overwrite') for this to work.
    """
    try:
        if db.table_name not in db.db.table_names():
            return True
        return db.db.open_table(db.table_name).count_rows() == 0
    except Exception as e:
        logger.warning(f"Could not check index state: {e}. Will re-ingest.")
        return True


def main():
    logger.info("=== MISSION START: Tactical Awareness Engine ===")

    # --- Component Initialization ---
    sim         = SimD3Environment(settings.SIM_METADATA_FILE)
    spatial     = SpatialEngine(settings.SENSOR_WIDTH_MM, settings.SENSOR_HEIGHT_MM,
                                settings.FOCAL_LENGTH_MM)
    search_lib  = SearchLibrarian(settings.CLIP_MODEL)
    db          = TacticalDatabase()
    analyst     = TacticalAnalyst(settings.AI_PROVIDER, settings.VLM_MODEL,
                                  settings.OPENROUTER_API_KEY)

    # --- Phase 1: Ingestion (skipped if index already exists) ---
    db.initialize_table(vector_dim=512)

    if check_ingestion_needed(db):
        run_ingestion(sim, spatial, search_lib, db)
    else:
        row_count = db.db.open_table(db.table_name).count_rows()
        logger.info(f"Persistent index found ({row_count} frames). Skipping ingestion.")

    # --- Phase 2 & 3: Search + VLM Grounding ---
    user_request = "Find a helipad marked with H"
    candidates = run_search(user_request, search_lib, db)
    intel      = run_vlm_analysis(candidates, user_request, analyst)

    # --- Phase 4: Render Annotated Outputs ---
    render_results(candidates, intel)

    # --- Final Report ---
    print(f"\n{'='*60}")
    print(f"TACTICAL REPORT")
    print(f"{'='*60}")
    print(intel.get('report', 'No data'))
    print(f"{'='*60}\n")

    logger.info("=== MISSION COMPLETE ===")


if __name__ == "__main__":
    main()