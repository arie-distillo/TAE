import logging
import sys
import os
import cv2
from pathlib import Path
from config import settings
from core.sim_provider import SimD3Environment
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tae_operation.log")
    ]
)
logger = logging.getLogger("TAE-Core")

def main():
    logger.info("=== TAE TACTICAL ENGINE: MISSION START ===")

    # 1. SETUP COMPONENTS
    try:
        sim = SimD3Environment(settings.SIM_METADATA_FILE)
        spatial = SpatialEngine(sensor_width=settings.SENSOR_WIDTH_MM, focal_length=settings.FOCAL_LENGTH_MM)
        search_lib = SearchLibrarian(settings.CLIP_MODEL)
        
        db = TacticalDatabase()
        # Initialize table with fixed 512-dim schema
        db.initialize_table(vector_dim=512)

        analyst = TacticalAnalyst(
            provider=settings.AI_PROVIDER,
            model_name=settings.VLM_MODEL,
            api_key=settings.OPENROUTER_API_KEY
        )
        logger.info("All components initialized successfully.")
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. THEATER PERSISTENCE CHECK (SKIP INGESTION IF INDEXED)
    # We check if the table exists and has rows to avoid redundant CLIP encoding
    try:
        table = db.db.open_table("theater_index")
        row_count = table.count_rows()
    except:
        row_count = 0

    if row_count > 0:
        logger.info(f"Existing theater index found with {row_count} frames. Skipping ingestion.")
    else:
        logger.info("Starting Fresh Ingestion Phase...")
        frame_count = 0
        for img_cv2, telemetry in sim:
            try:
                frame_count += 1
                # A. Semantic Encoding
                vector = search_lib.encode_image(img_cv2)

                # B. Spatial Projection
                projection = spatial.get_projection_data(telemetry, img_cv2.shape[1])
                
                # C. Indexing
                db.add_observation(
                    vector=vector,
                    image_path=telemetry['full_path'],
                    telemetry=telemetry,
                    footprint=projection['ground_width_m']
                )
                if frame_count % 10 == 0:
                    logger.info(f"Indexed {frame_count} frames...")
            except Exception as e:
                logger.error(f"Error indexing frame {frame_count}: {e}")
        logger.info(f"Ingestion Complete. {frame_count} frames indexed.")

    # 3. TACTICAL QUERY & SEARCH
    user_request = "Find a helipad marked with H"
    logger.info(f"Executing Semantic Search: '{user_request}'")
    
    query_vec = search_lib.encode_text(user_request)
    top_candidates = db.semantic_search(query_vec, limit=3)

    if not top_candidates:
        logger.warning("No targets found for the current query.")
        return

    # 4. VISUAL VALIDATION & DECK PREPARATION
    logger.info("Generating visual validation frames for candidates...")
    os.makedirs("tactical_results", exist_ok=True)
    candidate_paths = []

    print("\n" + "="*60)
    print("TARGET DECK RETRIEVED")
    print("="*60)

    for i, cand in enumerate(top_candidates):
        img_path = cand['image_path']
        candidate_paths.append(img_path)
        
        # Load image for overlaying coordinates
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            # Draw Target Crosshair
            cv2.drawMarker(img, (w//2, h//2), (0, 255, 0), cv2.MARKER_CROSS, 100, 5)
            
            # Burn Coordinates into image
            overlay_text = f"LAT: {cand['lat']:.6f} LON: {cand['lon']:.6f}"
            cv2.rectangle(img, (10, h-60), (w-10, h-10), (0,0,0), -1)
            cv2.putText(img, overlay_text, (30, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            val_path = f"tactical_results/cand_{i+1}_{Path(img_path).name}"
            cv2.imwrite(val_path, img)

        print(f"[{i+1}] Frame: {Path(img_path).name}")
        print(f"    Location: {cand['lat']:.6f}, {cand['lon']:.6f}")
        print(f"    Confidence: {1 - cand.get('_distance', 0):.4f}")
        print(f"    Validation Image: {val_path}")

    # 5. MULTI-ANGLE PERSISTENCE (MAP) ANALYSIS
    logger.info("Requesting Synthesized Tactical Report...")
    report = analyst.analyze_multiple_views(candidate_paths, user_request)
    
    print("\n" + "="*60)
    print("SYNTHESIZED TACTICAL INTELLIGENCE")
    print("="*60)
    print(report)
    print("="*60 + "\n")

    logger.info("=== TAE MISSION COMPLETE ===")

if __name__ == "__main__":
    main()