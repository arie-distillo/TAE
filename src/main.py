from core.sim_provider import SimD3Environment # Reusing the generator-friendly version
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst
from config import settings

def main():
    # 1. Setup components
    sim = SimD3Environment(settings.SIM_METADATA_FILE)
    spatial = SpatialEngine(sensor_width=settings.SENSOR_WIDTH_MM, focal_length=settings.FOCAL_LENGTH_MM)
    search_lib = SearchLibrarian(settings.CLIP_MODEL)

    db = TacticalDatabase()
    # CLIP ViT-B/32 produces 512-dimensional vectors
    db.initialize_table(vector_dim=512)

    print("--- TAE TACTICAL INGESTION START ---")

    for img_cv2, telemetry in sim:
        # A. Semantic Encoding (CLIP)
        vector = search_lib.encode_image(img_cv2)

        # B. Spatial Projection
        # Projects the image center and scale onto the baseline map
        projection = spatial.get_projection_data(telemetry, img_cv2.shape[1])
        
        # C. Indexing
        db.add_observation(
            vector=vector,
            image_path=telemetry['full_path'],
            telemetry=telemetry,
            footprint=projection['ground_width_m'] # This fuels the map scaling
        )

    print(f"--- Ingestion Complete. Theater indexed in LanceDB ---")

    # 2. Example Tactical Query
    user_request = "Find a helipad marked with H"
    query_vec = search_lib.encode_text(user_request) # Assume encode_text added to search.py
    
    candidates = db.semantic_search(query_vec, limit=1)
    
    if candidates:
        print(f"Top candidate found at {candidates[0]['lat']}, {candidates[0]['lon']}")
        # D. High-Level Reasoning (VLM)
        analyst = TacticalAnalyst(model_name=settings.VLM_MODEL)

        top_candidates = db.semantic_search(query_vec, limit=3)
        candidate_paths = [c['image_path'] for c in top_candidates]
        report = analyst.analyze_multiple_views(candidate_paths, user_request)
        print(f"Tactical Report: {report}")

if __name__ == "__main__":
    main()