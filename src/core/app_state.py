"""
core/app_state.py — Shared mutable application state.

Single source of truth for the runtime state dict, image caches, and the
domain constants that every layer needs.  No FastHTML imports, no HTTP, no
business logic — pure data.

Every other module that needs to read state imports from here.
Only app_state.py and the orchestration functions in main.py write to _state.
"""

# ── Query marker colours (cycled per query) ───────────────────────────────────
_MARKER_COLORS: list[str] = [
    "#4ade80", "#60a5fa", "#f59e0b", "#f472b6",
    "#a78bfa", "#34d399", "#fb923c", "#e879f9",
]

# ── CLIP domain prefix ────────────────────────────────────────────────────────
# CLIP was trained on ground-level photos; prepending this context string
# shifts text embeddings toward overhead/drone imagery representations.
_CLIP_AERIAL_CTX: str = "aerial drone nadir overhead view:"

# ── In-memory image caches ────────────────────────────────────────────────────
# key → raw JPEG bytes served by /tile_img/{key} and /frame_img/{det_id}
_frame_img_cache: dict[str, bytes] = {}
_tile_img_cache:  dict[str, bytes] = {}

# ── Runtime application state ─────────────────────────────────────────────────
_state: dict = {
    # mission context
    "mission_id":      None,
    "mission_name":    "",
    "mission_paths":   None,   # MissionPaths | None

    # operational
    "map_center":      [32.08, 34.78],
    "map_zoom":        14,
    "detections":      {},
    "ingested":        False,
    "frame_count":     0,
    "tile_count":      0,
    "query_color_idx": 0,
    "show_coverage":   True,    # hull shown by default
    "ingest_progress": "",
    "ingesting":       False,
    "ingest_msg":      None,

    # video playback
    "video_files":      [],   # [filename, ...] in upload order
    "frame_timestamps": {},   # {frame_name: timestamp_ms}
    "last_query":       None,
}