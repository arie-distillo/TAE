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

# Restoration of detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TAE-Core")

def main():
    logger.info("=== MISSION START: Grounded Tactical Reasoning ===")

    # Component Init
    sim = SimD3Environment(settings.SIM_METADATA_FILE)
    spatial = SpatialEngine(settings.SENSOR_WIDTH_MM, settings.FOCAL_LENGTH_MM)
    search_lib = SearchLibrarian(settings.CLIP_MODEL)
    db = TacticalDatabase()
    db.initialize_table(vector_dim=512)
    analyst = TacticalAnalyst(settings.AI_PROVIDER, settings.VLM_MODEL, settings.OPENROUTER_API_KEY)

    # 1. THEATER INGESTION
    try:
        table_names = db.db.table_names()
        if "theater_index" in table_names:
            table = db.db.open_table("theater_index")
            if table.count_rows() > 0:
                logger.info(f"Persistent index found ({table.count_rows()} frames). Skipping ingestion.")
                ingest_needed = False
            else:
                ingest_needed = True
        else:
            ingest_needed = True
    except:
        ingest_needed = True

    if ingest_needed:
        logger.info("Starting Fresh Ingestion Phase...")
        for i, (img_cv2, telemetry) in enumerate(sim):
            vector = search_lib.encode_image(img_cv2)
            db.add_observation(vector, telemetry['full_path'], telemetry, 0)
            if (i+1) % 10 == 0: logger.info(f"Indexed {i+1} frames...")

    # 2. Search & Analyze
    user_request = "Find a helipad marked with H"
    logger.info(f"Querying Vector DB: {user_request}")
    
    query_vec = search_lib.encode_text(user_request)
    top_candidates = db.semantic_search(query_vec, limit=3)
    
    paths = [c['image_path'] for c in top_candidates]
    intel = analyst.analyze_multiple_views(paths, user_request)

    # 3. Grounding & Multi-Target Bounding Box Overlay
    os.makedirs("tactical_results", exist_ok=True)
    all_vlm_targets = intel.get('targets', [])

    for i, cand in enumerate(top_candidates):
        img_path = cand['image_path']
        fname = Path(img_path).name
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        # Filter all targets for this specific filename
        frame_targets = [t for t in all_vlm_targets if t['filename'].lower() == fname.lower()]
        
        if frame_targets:
            for target_data in frame_targets:
                # FIX: VLM is outputting [xmin, ymin, xmax, ymax] 
                # (Verified because values > 3040 height are appearing in indices 0 and 2)
                v_xmin, v_ymin, v_xmax, v_ymax = target_data['bbox']
                
                is_pixels = max(v_xmin, v_ymin, v_xmax, v_ymax) > 1001
                
                if is_pixels:
                    p1 = (int(v_xmin), int(v_ymin))
                    p2 = (int(v_xmax), int(v_ymax))
                else:
                    p1 = (int(v_xmin * w / 1000), int(v_ymin * h / 1000))
                    p2 = (int(v_xmax * w / 1000), int(v_ymax * h / 1000))

                # Draw the Bounding Box
                cv2.rectangle(img, p1, p2, (0, 255, 0), 5)
                cv2.putText(img, "TARGET", (p1[0], p1[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            color, label = (0, 255, 0), "VLM GROUNDED"
        else:
            # Fallback marker only if NO targets found for this image
            color, label = (0, 165, 255), "CENTER FALLBACK"
            cv2.drawMarker(img, (w//2, h//2), color, cv2.MARKER_CROSS, 100, 5)

        # Tactical Status Bar
        status_text = f"{label} | LAT: {cand['lat']:.6f} LON: {cand['lon']:.6f}"
        cv2.rectangle(img, (0, h-80), (w, h), (0,0,0), -1)
        cv2.putText(img, status_text, (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        cv2.imwrite(f"tactical_results/grounded_{fname}", img)

    print(f"\nTACTICAL REPORT:\n{intel.get('report', 'No data')}\n")
    logger.info("=== MISSION COMPLETE ===")

if __name__ == "__main__":
    main()