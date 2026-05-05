import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to the project root (parent of src/ where config.py lives)
# This resolves correctly regardless of which directory you run from.
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── AI ────────────────────────────────────────────────────────────────────
    AI_PROVIDER:        str        = "openrouter"
    OPENROUTER_API_KEY: str | None = None
    VLM_MODEL:          str        = "qwen/qwen2.5-vl-72b-instruct"
    CLIP_MODEL:         str        = "ViT-B/32"
    CLIP_DIM:           int        = 512

    # ── Persistent storage root ───────────────────────────────────────────────
    # Default: <project_root>/data — resolved relative to this file, not CWD.
    # On Railway: set DATA_DIR=/data (volume mount point).
    # In .env: use an absolute path, e.g. DATA_DIR=/home/user/myproject/data
    DATA_DIR: str = _DEFAULT_DATA_DIR

    # Sub-paths — auto-derived from DATA_DIR unless overridden.
    VECTOR_DB_PATH:   str = ""   # LanceDB index
    UPLOAD_PATH:      str = ""   # original drone frames (permanent)
    DETECTIONS_PATH:  str = ""   # annotated tile/frame images
    MAP_PATH:         str = ""   # generated map.html
    SIM_METADATA_FILE: str = ""  # pose_metadata.json (main.py / CLI only)

    # ── Camera (DJI Mini 2 defaults) ─────────────────────────────────────────
    SENSOR_WIDTH_MM:  float = 6.3
    SENSOR_HEIGHT_MM: float = 4.7
    FOCAL_LENGTH_MM:  float = 4.5

    def model_post_init(self, __context):
        """Resolve empty path fields relative to DATA_DIR."""
        defaults = {
            "VECTOR_DB_PATH":   "lancedb",
            "UPLOAD_PATH":      "uploads",
            "DETECTIONS_PATH":  "detections",
            "MAP_PATH":         "maps",
            "SIM_METADATA_FILE": "uploads/pose_metadata.json",
        }
        for field, subdir in defaults.items():
            if not getattr(self, field):
                object.__setattr__(
                    self, field, os.path.join(self.DATA_DIR, subdir)
                )


settings = Settings()