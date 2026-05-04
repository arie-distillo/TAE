from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This tells Pydantic to look for these keys in a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Define the keys here. If they exist in .env, they get populated.
    AI_PROVIDER: str = "openrouter"
    OPENROUTER_API_KEY: str | None = None  # Loaded from .env
    VLM_MODEL: str = "qwen/qwen-vl-plus"

    # Path to the generated JSON file
    SIM_METADATA_FILE: str = "./data/raw/sim_env_1/pose_metadata.json"
    
    # Storage and Baseline
    DB_PATH: str = "./data/processed/tae_vectors.lancedb"
    MAP_IMAGE: str = "./data/maps/baseline_satellite.png"
    
    # Tile configuration
    TILE_SIZE:       int   = 640  # Tile size in pixels (e.g., 512x512)
    TILE_OVERLAP:    float = 0.2   # 20% overlap
    
    # Model Configs
    #CLIP_MODEL: str = "ViT-B/32"
    CLIP_MODEL: str = "RN50"
    CLIP_DIM:   int = 1024   # RN50=1024, ViT-B/32=512, ViT-L/14=768

    # DJI Mini / Mavic Air defaults
    SENSOR_WIDTH_MM: float = 6.3
    FOCAL_LENGTH_MM: float = 4.5
    SENSOR_HEIGHT_MM: float = 4.7 # DJI Mini 2, Mavic Air, etc. Adjust if using a different drone.
    SENSOR_WIDTH_MM: float = 6.3


settings = Settings()