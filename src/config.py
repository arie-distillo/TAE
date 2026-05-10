import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to the project root (parent of src/ where config.py lives)
_PROJECT_ROOT    = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── AI — VLM ─────────────────────────────────────────────────────────────
    AI_PROVIDER:        str        = "openrouter"
    OPENROUTER_API_KEY: str | None = None
    VLM_MODEL:          str        = "qwen/qwen2.5-vl-72b-instruct"
    INTENT_MODEL:       str | None = "anthropic/claude-haiku-4-5"

    # ── AI — CLIP ────────────────────────────────────────────────────────────
    CLIP_MODEL: str = "ViT-B/32"
    CLIP_DIM:   int = 512

    # ── AI — Segmentation (SAM2 via Replicate) ───────────────────────────────
    # Calls Meta SAM-2 on the Replicate cloud platform.
    # No local GPU or model weights required.
    # Leave REPLICATE_API_KEY empty to disable Stage 2 entirely.
    #
    # Get your key at: https://replicate.com/account/api-tokens
    # Install client:  pip install replicate
    REPLICATE_API_KEY:   str = ""
    SAM_REPLICATE_MODEL: str = (
        "meta/sam-2:"
        "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
    )
    SAM_MAX_DIM: int = 1024   # resize long edge before upload (0 = disabled)

    # ── Persistent storage root ───────────────────────────────────────────────
    # Default: <project_root>/data  — resolved relative to this file, not CWD.
    # On Railway: set DATA_DIR=/data (volume mount point).
    DATA_DIR: str = _DEFAULT_DATA_DIR

    # Sub-paths — auto-derived from DATA_DIR unless overridden in .env
    VECTOR_DB_PATH:    str = ""   # LanceDB tile index
    UPLOAD_PATH:       str = ""   # original drone frames (permanent)
    DETECTIONS_PATH:   str = ""   # annotated tile/frame images
    MAP_PATH:          str = ""   # generated map.html
    MISSIONS_DB_PATH:  str = ""   # global missions catalogue (SQLite)
    SEGMENTS_DIR:      str = ""   # parent dir for per-mission segments.db files
    SIM_METADATA_FILE: str = ""   # pose_metadata.json

    # ── Camera (DJI Mini 2 defaults) ─────────────────────────────────────────
    SENSOR_WIDTH_MM:  float = 6.3
    SENSOR_HEIGHT_MM: float = 4.7
    FOCAL_LENGTH_MM:  float = 4.5

    def model_post_init(self, __context):
        """Resolve empty path fields relative to DATA_DIR."""
        defaults = {
            "VECTOR_DB_PATH":    "lancedb",
            "UPLOAD_PATH":       "uploads",
            "DETECTIONS_PATH":   "detections",
            "MAP_PATH":          "maps",
            "MISSIONS_DB_PATH":  "missions.db",
            "SEGMENTS_DIR":      "segments",
            "SIM_METADATA_FILE": "uploads/pose_metadata.json",
        }
        for field, subdir in defaults.items():
            if not getattr(self, field):
                object.__setattr__(
                    self, field, os.path.join(self.DATA_DIR, subdir)
                )


settings = Settings()