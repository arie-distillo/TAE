import os
from pathlib import Path
from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to the project root (parent of src/ where config.py lives)
_PROJECT_ROOT    = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


# ─────────────────────────────────────────────────────────────────────────────
# Per-mission filesystem paths  (new — multi-mission support)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MissionPaths:
    """All filesystem paths scoped to a single mission."""
    root:         Path
    uploads:      Path   # original drone frames + pose_metadata.json
    lancedb:      Path   # LanceDB vector index
    segments:     Path   # segments.db + visualisation images
    detections:   Path   # reserved for future use
    maps:         Path   # generated map.html
    sim_metadata: Path   # uploads/pose_metadata.json

    @classmethod
    def for_mission(cls, data_dir: str, mission_id: str) -> "MissionPaths":
        root = Path(data_dir) / mission_id
        return cls(
            root         = root,
            uploads      = root / "uploads",
            lancedb      = root / "lancedb",
            segments     = root / "segments",
            detections   = root / "detections",
            maps         = root / "maps",
            sim_metadata = root / "uploads" / "pose_metadata.json",
        )

    def makedirs(self) -> None:
        for p in (self.uploads, self.lancedb, self.segments,
                  self.detections, self.maps):
            p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Application settings  (unchanged from original — all fields preserved)
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # silently ignore any .env keys not declared below
    )

    # ── AI - provider ─────────────────────────────────────────────────────────────
    AI_PROVIDER:        str        = "openrouter"
    INTENT_MODEL:       str | None = "anthropic/claude-haiku-4-5"
    OPENROUTER_API_KEY: str | None = None


    # ── AI - VLM ─────────────────────────────────────────────────────────────
    VLM_MODEL:          str        = "qwen/qwen2.5-vl-72b-instruct"
    # When True, send the full parent frame to VLM for candidate validation instead
    # of one tile at a time.  The frame image carries all candidates from all tiles
    # of that frame in a single call, annotated with numbered bboxes.
    # Response schema is compact: confirmed → {index, confirmed, confidence, reason}
    #                             rejected  → {index, confirmed, confidence}
    VLM_FULL_FRAME_MODE: bool = True

    # ── AI — CLIP ────────────────────────────────────────────────────────────
    CLIP_MODEL: str = "ViT-B/32"
    CLIP_DIM:   int = 512

    # ── AI — Object detection ───────────────────────────────────────────────
    DETECTION_MIN_BBOX_PX: int = 30   # minimum detection bounding box side length in pixels
    DETECTION_CONFIDENCE: dict[str, float] = {
        "easy":   0.13,   # large, visually distinctive objects (buildings, vehicles)
        "medium": 0.13,   # medium objects with moderate camouflage (cows in grass)
        "hard":   0.03,   # small or highly camouflaged (people, stones, prone animals)
    }
    # Maximum G-DINO candidates forwarded to VLM per tile (sorted by confidence desc).
    # Prevents VLM token overflow on dense tiles; 8 matches verify_detections_batch capacity.
    MAX_CANDIDATES_PER_TILE: int = 8

    # ── AI — Anomaly detection ────────────────────────────────────────────────
    ANOMALY_SCORE_MARGIN: float = 0.05

    # ── AI — Segmentation (SAM2 via Replicate) ───────────────────────────────
    REPLICATE_API_KEY:   str = ""
    SAM_REPLICATE_MODEL: str = (
        "meta/sam-2:"
        "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
    )
    SAM_MAX_DIM: int = 1024

    DETECTOR_REPLICATE_MODEL: str = (
        "adirik/grounding-dino:"
        "efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
    )
    DETECTOR_TIMEOUT_S: int = 180   # max seconds to wait for all predictions to complete

    # Approximately 22 metres at mid-latitudes (0.0002° × 111,000 m/° ≈ 22 m). 
    # Two detections from different frames that are within 22 metres of each other in geo-coordinates get assigned to the same track. 
    # At typical survey altitudes (80–150 m AGL) with overlapping tiles, the same physical object will appear in adjacent frames within a few metres, so 22 m gives comfortable headroom without merging genuinely separate objects. 
    # Halve it to 0.0001 (~11 m) for dense urban surveys with closely spaced objects. 
    TRACKER_GEO_PROXIMITY_DEG: float = 0.0002

    # ── Persistent storage root ───────────────────────────────────────────────
    DATA_DIR: str = _DEFAULT_DATA_DIR

    # Sub-paths — auto-derived from DATA_DIR in model_post_init unless
    # explicitly overridden in .env.  Still used by Phase-1 compat paths
    # (MissionManager.get_or_create_active) and any code not yet migrated
    # to per-mission MissionPaths.
    VECTOR_DB_PATH:    str = ""
    UPLOAD_PATH:       str = ""
    DETECTIONS_PATH:   str = ""
    MAP_PATH:          str = ""
    MISSIONS_DB_PATH:  str = ""
    SEGMENTS_DIR:      str = ""
    SIM_METADATA_FILE: str = ""

    # ── Camera (DJI Mini 2 defaults) ─────────────────────────────────────────
    SENSOR_WIDTH_MM:  float = 6.3
    SENSOR_HEIGHT_MM: float = 4.7
    FOCAL_LENGTH_MM:  float = 4.5

    # ── Video ingestion ──────────────────────────────────────────────────────
    VIDEO_SAMPLE_INTERVAL_SEC: float = 2.0   # fixed interval (used when adaptive=False)
    VIDEO_ADAPTIVE_SAMPLING:   bool  = True  # compute interval from SRT altitude+speed
    VIDEO_TARGET_OVERLAP_PCT:  float = 60.0  # target ground overlap between samples

    def model_post_init(self, __context):
        """Resolve empty sub-path fields relative to DATA_DIR."""
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