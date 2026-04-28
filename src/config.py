from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Path to the generated JSON file
    SIM_METADATA_FILE: str = "./data/raw/sim_env_1/pose_metadata.json"
    
    # Storage and Baseline
    DB_PATH: str = "./data/processed/tae_vectors.lancedb"
    MAP_IMAGE: str = "./data/maps/baseline_satellite.png"
    
    # Model Configs
    CLIP_MODEL: str = "ViT-B/32"
    VLM_MODEL: str = "moondream" # "qwen2.5-vl"

    # DJI Mini 2 / Mavic Air defaults
    SENSOR_WIDTH_MM: float = 6.3
    FOCAL_LENGTH_MM: float = 4.5

settings = Settings()