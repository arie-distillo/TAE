from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # This tells Pydantic to look for these keys in a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Define the keys here. If they exist in .env, they get populated.
    AI_PROVIDER: str = "openrouter"
    OPENROUTER_API_KEY: str | None = None  # Loaded from .env
    VLM_MODEL: str = "qwen/qwen-2.5-vl-72b-instruct"

    # Path to the generated JSON file
    SIM_METADATA_FILE: str = "./data/raw/sim_env_1/pose_metadata.json"
    
    # Storage and Baseline
    DB_PATH: str = "./data/processed/tae_vectors.lancedb"
    MAP_IMAGE: str = "./data/maps/baseline_satellite.png"
    
    # Model Configs
    CLIP_MODEL: str = "ViT-B/32"

    # DJI Mini 2 / Mavic Air defaults
    SENSOR_WIDTH_MM: float = 6.3
    FOCAL_LENGTH_MM: float = 4.5

settings = Settings()