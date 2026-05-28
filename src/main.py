import re
"""
tae_app.py – TAE Tactical Awareness Engine · FastHTML Web Interface
Run from the src/ directory:  python main.py
"""

import logging
import os
import sys
import json
import uuid
import shutil
import time
from pathlib import Path
from datetime import datetime
from types import GeneratorType

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import folium
from fasthtml.common import *
from monsterui.all import *
from starlette.requests import Request

from config import settings, MissionPaths
from core.mission import MissionManager, Mission
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.clip import SearchLibrarian
from ai.vlm import TacticalAnalyst
from tools.ingest_telemetry import TAESimGenerator
from core.video import SRTParser, VideoSampler, AdaptiveSampler
from core.streaming import StreamManager
from core.app_state import (
    _state, _frame_img_cache, _tile_img_cache,
    _MARKER_COLORS, _CLIP_AERIAL_CTX,
)
from core.services import (
    _build_map, _optimal_zoom, _recenter_on_detections,
    _load_tile, _annotate_and_save, _tile_to_static_url,
    _extract_video_frames, _save_detections, _load_detections,
    init as _init_services, 
)
from ai.intent import (
    IntentClassifier, ObjectDetectionParams, AnomalyDetectionParams,
    ClassifiedQuery,
)
from ai.detection_pipeline import run_detection_pipeline, Track
from ui.styles import _CSS, _JS


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _uv in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uv).setLevel(logging.INFO)
for _wf in ("watchfiles", "watchfiles.main"):
    logging.getLogger(_wf).setLevel(logging.WARNING)
logger = logging.getLogger("TAE-UI")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons
# ─────────────────────────────────────────────────────────────────────────────
mission_mgr = MissionManager(settings.MISSIONS_DB_PATH)

spatial = SpatialEngine(
    settings.SENSOR_WIDTH_MM,
    settings.SENSOR_HEIGHT_MM,
    settings.FOCAL_LENGTH_MM,
)
db      = TacticalDatabase()
analyst = TacticalAnalyst(
    settings.AI_PROVIDER,
    settings.VLM_MODEL,
    settings.OPENROUTER_API_KEY,
)
intent_clf = IntentClassifier(
    api_key    = settings.OPENROUTER_API_KEY,
    model_name = getattr(settings, "INTENT_MODEL", None),
)
_search_lib: SearchLibrarian | None = None

# Video telemetry helpers (singletons — stateless, safe to reuse)
_srt_parser    = SRTParser()
_video_sampler = VideoSampler()

# Live streaming manager (Phase D)
stream_mgr = StreamManager()

def _main_get_search_lib():          # rename locally to be unambiguous
    global _search_lib
    if _search_lib is None:
        logger.info("Loading CLIP model…")
        _search_lib = SearchLibrarian(settings.CLIP_MODEL)
    return _search_lib

_init_services(db, analyst, _main_get_search_lib, spatial)   # ← pass the local one

# ─────────────────────────────────────────────────────────────────────────────
# Per-query state (reset on mission switch)
# ─────────────────────────────────────────────────────────────────────────────
# CLIP was trained on ground-level photos; prepending this context string
# shifts text embeddings toward overhead/drone imagery representations.
# _state, caches, and constants imported from core.app_state


# ─────────────────────────────────────────────────────────────────────────────
# Mission activation
# ─────────────────────────────────────────────────────────────────────────────


def _mission_folder(mission_name: str, mission_id: str) -> str:
    """
    Resolve the filesystem folder name for a mission.

    Strategy (backward-compatible):
      1. If DATA_DIR/{mission_id}/ already exists → use the legacy ID folder.
         This covers missions created before the name-based folder scheme.
      2. Otherwise → use {sanitized_name}_{id[:8]} (new scheme).
    """
    # Legacy check — data already on disk under the bare ID
    legacy = Path(settings.DATA_DIR) / mission_id
    if legacy.exists():
        return mission_id
    slug = re.sub(r"[^\w\-]", "_", mission_name.strip()).lower()
    slug = re.sub(r"_+", "_", slug).strip("_")[:32] or "mission"
    return f"{slug}_{mission_id[:8]}"

def _activate_mission(mission: Mission) -> None:
    """Switch all server state to a different mission."""
    paths = MissionPaths.for_mission(settings.DATA_DIR, _mission_folder(mission.name, mission.id))
    paths.makedirs()

    _frame_img_cache.clear()
    _tile_img_cache.clear()

    _state.update({
        "mission_id":      mission.id,
        "mission_name":    mission.name,
        "mission_paths":   paths,
        "detections":      {},
        "ingested":        False,
        "frame_count":     0,
        "tile_count":      0,
        "map_center":      [32.08, 34.78],
        "map_zoom":        14,
        "show_coverage":   True,
        "query_color_idx": 0,
        "ingesting":       False,
        "ingest_msg":      None,
        "ingest_progress": "",
        "video_files":     [],
        "frame_timestamps": {},
        "last_query":       None,
    })

    db.reconnect(str(paths.lancedb))
    db.initialize_table(vector_dim=settings.CLIP_DIM)
    _restore_state()

    mission_mgr.touch(mission.id)
    mission_mgr.set_last_active(mission.id)
    logger.info(f"Activated mission '{mission.name}' ({mission.id})")


def _restore_state() -> None:
    """Restore ingested-index state from the active mission's LanceDB."""
    try:
        count = db.row_count()
        if count > 0:
            _state["ingested"]   = True
            _state["tile_count"] = count
            # Restore frame count and map centre from the tile index
            try:
                df = db.table.to_pandas()[["parent_path", "lat", "lon"]]
                frame_count = df["parent_path"].nunique()
                _state["frame_count"] = frame_count
                # Re-centre map on the centroid of all indexed frames
                lats = df.groupby("parent_path")["lat"].first().tolist()
                lons = df.groupby("parent_path")["lon"].first().tolist()
                if lats and lons:
                    import statistics as _st
                    _state["map_center"] = [_st.median(lats), _st.median(lons)]
                    _state["map_zoom"]   = 16
            except Exception as inner:
                logger.warning(f"Could not restore frame count / map centre: {inner}")
            logger.info(
                f"Restored index — {count} tiles, "
                f"{_state['frame_count']} frames."
            )
            _build_map()
        else:
            logger.info("DB empty — waiting for upload.")
    except Exception as e:
        logger.warning(f"Could not restore DB state: {e}")

    # ── Restore video files and timestamps from disk ──────────────────────
    try:
        paths = _state.get("mission_paths")
        if paths and paths.uploads.exists():
            video_exts = {".mp4", ".mov", ".avi", ".mkv"}
            vids = sorted(
                [f.name for f in paths.uploads.iterdir()
                 if f.suffix.lower() in video_exts],
                key=lambda n: (paths.uploads / n).stat().st_mtime,
            )
            if vids:
                _state["video_files"] = vids
                logger.info("Restored video file(s): %s", vids)
            # Restore frame timestamps from persisted JSON
            ts_file = paths.uploads / "frame_timestamps.json"
            if ts_file.exists():
                _state["frame_timestamps"] = json.loads(
                    ts_file.read_text(encoding="utf-8")
                )
                logger.info(
                    "Restored %d frame timestamps",
                    len(_state["frame_timestamps"]),
                )
    except Exception as e:
        logger.warning("Could not restore video state: %s", e)

    # ── Restore detections from disk — zero API calls on startup ───────────
    try:
        paths = _state.get("mission_paths")
        if paths and _state.get("ingested"):
            dets = _load_detections(paths.detections)
            if dets:
                _state["detections"]      = dets
                _state["detections_ready"] = True
                logger.info("Restored %d detection(s) from disk", len(dets))
                _build_map()   # re-render with restored detections + tracks
            else:
                logger.info(
                    "No saved detections — run a query to populate them"
                )
    except Exception as e:
        logger.warning("Could not restore detections from disk: %s", e)


def _startup() -> None:
    last_id  = mission_mgr.get_last_active()
    missions = mission_mgr.list_active()
    target: Mission | None = None
    if last_id:
        target = next((m for m in missions if m.id == last_id), None)
    if target is None and missions:
        target = missions[0]
    if target:
        _activate_mission(target)

# Map/image helpers imported from core.analysis

# ─────────────────────────────────────────────────────────────────────────────
# Chat components
# ─────────────────────────────────────────────────────────────────────────────

def _msg(content: str, role: str = "sys") -> FT:
    ts = datetime.now().strftime("%H:%M")
    return Div(
        Span(ts, cls="msg-time"),
        Div(content, cls="msg-bubble"),
        cls=f"msg {role}",
    )


def _msg_html(html_content: str, role: str = "sys") -> FT:
    ts = datetime.now().strftime("%H:%M")
    return Div(
        Span(ts, cls="msg-time"),
        Div(NotStr(html_content), cls="msg-bubble"),
        cls=f"msg {role}",
    )


def _status_badge() -> FT:
    if _state["ingested"]:
        dot_cls = "sdot active"
        txt     = f"Index ready · {_state['frame_count']} frames"
    else:
        dot_cls = "sdot"
        txt     = "No index · Upload images to begin"
    return Span(
        Span(cls=dot_cls),
        txt,
        id="tae-status",
        cls="status-badge",
        hx_swap_oob="true",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
# _CSS and _JS imported from core.styles


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
_MAX_UPLOAD_MB = 500

app, rt = fast_app(
    max_upload_size=_MAX_UPLOAD_MB * 1024 * 1024,
    hdrs=(
        Theme.slate.headers(),
        Link(rel="stylesheet",
             href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
        _CSS,
        _JS,
    ),
    pico=False,
    bodykw={"style": "margin:0;background:#080910;overflow:hidden"},
)


# ─────────────────────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_mission_date(created_at: str) -> str:
    """Return human-readable date string for a mission's created_at (ISO-8601 string)."""
    from datetime import date, datetime as _dt
    try:
        d = _dt.fromisoformat(created_at).date()
    except (ValueError, TypeError):
        return ""
    today = date.today()
    if d == today:
        return "today"
    if (today - d).days == 1:
        return "yesterday"
    return d.strftime("%b %d")


def _navbar(ingested: bool = False, frame_count: int = 0, mission=None) -> FT:
    # Resolve mission info from state (more reliable than parameter for live display)
    mid          = _state.get("mission_id", "")
    active_name  = _state.get("mission_name") or (mission.name if mission else "No mission")
    missions     = mission_mgr.list_active()

    # Mission dropdown items
    m_items = []
    for m in missions:
        is_active = (m.id == mid)
        m_items.append(Div(
            Span(cls="sdot" + (" active" if is_active else ""), style="flex-shrink:0"),
            Div(
                Span(m.name, cls="m-item-name"),
                Span(_fmt_mission_date(m.created_at), cls="m-item-date"),
                style="display:flex;flex-direction:column;min-width:0",
            ),
            cls="m-item" + (" m-item-active" if is_active else ""),
            hx_post=f"/missions/{m.id}/activate",
            hx_swap="none",
            **{"hx-on::after-request": "location.reload()"},
        ))

    # "Show archived" row + "New mission" row
    m_items.append(Div(
        Div(style="flex:1;height:1px;background:var(--border)"),
        style="padding:4px 12px",
    ))
    m_items.append(Div(
        I(cls="fas fa-archive", style="color:var(--muted);font-size:10px"),
        Span("Show archived", cls="m-item-name"),
        cls="m-item",
        hx_get="/missions/archived",
        hx_target="#mission-drop-list",
        hx_swap="beforeend",
    ))
    m_items.append(Div(
        I(cls="fas fa-plus", style="color:var(--accent);font-size:10px"),
        Span("New mission", cls="m-item-name", style="color:var(--accent)"),
        cls="m-item",
        onclick="openDrawer('new')",
    ))

    mission_selector = Div(
        # Trigger button — shows active mission name
        Button(
            I(cls="fas fa-map-marker-alt", style="font-size:10px;color:var(--blue)"),
            Span(active_name, cls="m-active-name", id="active-mission-name"),
            I(cls="fas fa-chevron-down", style="font-size:8px;margin-left:4px;color:var(--muted)"),
            cls="mission-btn",
            onclick="toggleMissionDrop(event)",
        ),
        # Settings cog for active mission
        Button(
            I(cls="fas fa-cog"),
            cls="mission-cog",
            title="Mission settings",
            onclick=f"openDrawer('{mid}')" if mid else "openDrawer('new')",
        ),
        # Dropdown list
        Div(
            Div(*m_items, id="mission-drop-list"),
            cls="mission-drop",
            id="mission-drop",
        ),
        cls="mission-selector",
    )

    if ingested:
        dot_cls    = "sdot active"
        status_txt = f"Index ready · {frame_count} frames"
    else:
        dot_cls    = "sdot"
        status_txt = "No index · Upload images to begin"

    return Div(
        Span("TAE", cls="brand"),
        Span("Intelligence", cls="brand-sub"),
        Div(cls="sep"),
        mission_selector,
        Div(cls="sep"),
        Button(
            I(cls="fas fa-satellite-dish", style="font-size:11px"),
            " Feed",
            cls="upload-btn",
            onclick="openFeedPanel()",
        ),
        Span(
            Span(cls="spinner"),
            Span("Processing…", cls="pulse-txt"),
            id="upload-ind",
            cls="htmx-indicator",
        ),
        Div(cls="sep"),
        _panel_toolbar(),
        Span(
            Span(cls=dot_cls),
            status_txt,
            id="tae-status",
            cls="status-badge",
            hx_swap_oob="true",
        ),
        cls="tae-nav",
    )
 
 

def _settings_drawer_content(mission: Mission | None, is_new: bool = False) -> FT:
    """Inner content of the settings drawer — shared by edit and create modes."""
    name_val       = "" if is_new else (mission.name       if mission else "")
    definition_val = "" if is_new else (mission.definition if mission else "")
    intents        = mission.allowed_intents if (mission and not is_new) else ["object_detection"]
    mid      = mission.id if mission else ""
    title    = "New mission" if is_new else name_val

    save_route  = "/missions" if is_new else f"/missions/{mid}"
    save_method = "hx_post" if is_new else "hx_patch"
    save_label  = "Create" if is_new else "Save"
    after_save  = "location.reload()" if is_new else "closeDrawer(); document.querySelector('.drawer-title').textContent=document.getElementById('drawer-name').value;"

    danger = [] if is_new else [
        Div(cls="drawer-sep"),
        Div(
            Span("Danger zone", cls="drawer-label"),
            Button("Archive mission", cls="arch-btn",
                   hx_post=f"/missions/{mid}/archive",
                   hx_swap="none",
                   **{"hx-on::after-request": "location.reload()"}),
            Button("Delete mission…", cls="del-btn",
                   onclick="toggleDeleteConfirm()"),
            Div(
                Span(f'Type "{name_val}" to confirm:', cls="del-hint"),
                Input(id="del-confirm-input", name="del-confirm-input",
                      cls="del-confirm-input", placeholder=name_val),
                Button("Permanently delete", cls="del-go",
                       hx_post=f"/missions/{mid}/delete",
                       hx_include="#del-confirm-input",
                       hx_swap="none",
                       **{"hx-on::after-request": "location.reload()"}),
                cls="del-confirm",
                id="del-confirm-box",
            ),
            cls="drawer-section",
        ),
    ]

    return (
        Div(
            I(cls="fas fa-cog", style="font-size:12px;color:var(--muted)"),
            Span(title, cls="drawer-title"),
            Button("✕", cls="drawer-close", onclick="closeDrawer()"),
            cls="drawer-header",
        ),
        Div(
            Span("Name", cls="drawer-label"),
            Input(value=name_val, id="drawer-name", cls="drawer-input",
                  placeholder="e.g. Alpha Site — May 12", name="name"),
            cls="drawer-section",
        ),
        Div(cls="drawer-sep"),
        Div(
            Span("Mission definition", cls="drawer-label"),
            Textarea(
                definition_val,
                id="drawer-definition",
                name="definition",
                cls="drawer-textarea",
                placeholder=(
                    "e.g. Find all isolated trees and large stones\n"
                    "Applied automatically to every frame on ingestion."
                ),
                rows="4",
            ),
            Span(
                "When set, TAE auto-analyses each ingested frame against this goal.",
                cls="drawer-hint",
            ),
            cls="drawer-section",
        ),
        Div(cls="drawer-sep"),
        Div(
            Span("Intents", cls="drawer-label"),
            Label(
                Input(type="checkbox", name="intents", value="object_detection",
                      checked=("object_detection" in intents) or None),
                Span("Object detection"),
                cls="intent-row",
            ),
            Label(
                Input(type="checkbox", name="intents", value="anomaly_detection",
                      checked=("anomaly_detection" in intents) or None),
                Span("Anomaly detection"),
                cls="intent-row",
            ),
            Button(
                save_label, cls="drawer-save-btn",
                **{save_method: save_route,
                   "hx_include": "#drawer-name,#drawer-definition,[name='intents']",
                   "hx_swap": "none",
                   "hx-on::after-request": after_save},
            ),
            cls="drawer-section",
        ),
        *danger,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic panel factory
# ─────────────────────────────────────────────────────────────────────────────
def _panel(
    panel_id: str,
    icon_cls: str,          # e.g. "pi-map fas fa-map-marked-alt"
    title: str,
    body: FT | list,
    foot: FT | None = None,
    *,
    panel_icon_cls: str = "pi-map",  # colour variant
    extra_cls: str = "",
    has_resize: bool = True,
    collapsed: bool = False,
) -> FT:
    """
    Renders a draggable / collapsible / resizable floating panel.
 
    Structure
    ─────────
    .tae-panel#panel_id
      .panel-header   ← drag handle
        .panel-icon
        .panel-title
        .panel-controls
          [collapse]  [close]
      .panel-body     ← scrollable content
        *body
      .panel-foot?    ← optional sticky footer (e.g. chat input row)
      .panel-resize   ← bottom-right corner drag
    """
    collapsed_cls = " is-collapsed" if collapsed else ""
    chev_cls = "fas fa-chevron-down" if collapsed else "fas fa-chevron-up"
 
    header = Div(
        Div(I(cls=icon_cls), cls=f"panel-icon {panel_icon_cls}"),
        Span(title, cls="panel-title"),
        Div(
            Button(
                I(cls=chev_cls),
                **{"data-collapse": "1"},
                cls="panel-btn",
                title="Collapse / expand",
            ),
            cls="panel-controls",
        ),
        cls="panel-header",
    )
 
    children = [header]
 
    if isinstance(body, (list, tuple, GeneratorType)):
        body_div = Div(*body, cls="panel-body")
    else:
        body_div = Div(body, cls="panel-body")
    children.append(body_div)
 
    if foot:
        children.append(Div(foot, cls="panel-foot"))
 
    if has_resize:
        children.append(Div(cls="panel-resize"))
 
    cls = f"tae-panel{collapsed_cls} {extra_cls}".strip()
    return Div(*children, id=panel_id, cls=cls)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Map panel  (replaces .map-wrap + .img-panel in the old layout)
# ─────────────────────────────────────────────────────────────────────────────
def _map_panel() -> FT:
    body = Div(
        Iframe(src="/map", id="tae-map-frame"),
        cls="panel-body map-panel-body",
    )
    return _panel(
        panel_id="tae-map-panel",
        icon_cls="fas fa-map-marked-alt",
        panel_icon_cls="pi-map",
        title="Tactical Map",
        body=body,
        has_resize=True,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Chat panel  (replaces old _chat_panel)
# ─────────────────────────────────────────────────────────────────────────────
def _chat_panel(collapsed: bool = False) -> FT:
    """
    The query-theater chat widget.
    This is the panel that gets swapped by HTMX on /toggle_chat —
    NOTE: with the new JS collapse the /toggle_chat route is no longer
    needed for collapse, but is kept for back-compat with existing hx_get calls.
    """
    from datetime import datetime
 
    def _msg(content, role="sys"):
        ts = datetime.now().strftime("%H:%M")
        return Div(
            Span(ts, cls="msg-time"),
            Div(content, cls="msg-bubble"),
            cls=f"msg {role}",
        )
 
    msgs_area = Div(
        _msg("TAE ready. Upload imagery to build the theater index, "
             "then query in natural language.", "sys"),
        id="tae-msgs",
        cls="chat-msgs",
    )
 
    input_row = Div(
        Div(cls="chat-divider"),
        Div(
            Form(
                Input(
                    placeholder="e.g. 'white pickup truck near building'",
                    id="tae-query-input",
                    name="q",
                    autocomplete="off",
                    **{"hx-on:keydown":
                       "if(event.key==='Enter'){event.preventDefault();"
                       "htmx.trigger(this.closest('form'),'submit');}"},
                ),
                Button("Send", cls="send-btn",
                       hx_post="/query",
                       hx_target="#tae-msgs",
                       hx_swap="beforeend",
                       hx_include="#tae-query-input",
                       **{"hx-on::after-request": "scrollChat();"}),
                cls="chat-input-row",
                **{"hx-on:submit": "event.preventDefault();"},
            ),
        ),
    )
 
    return _panel(
        panel_id="tae-chat-panel",
        icon_cls="fas fa-crosshairs",
        panel_icon_cls="pi-chat",
        title="Chat",
        body=msgs_area,
        foot=input_row,
        has_resize=True,
        collapsed=collapsed,
        extra_cls="",
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Video panel
# ─────────────────────────────────────────────────────────────────────────────
def _video_panel() -> FT:
    body = [
        # #video-panel is the HTMX target used by /video_panel route AND the
        # upload-route Script() that auto-activates the panel after video upload.
        Div(
            Div("No video uploaded yet — click Feed to upload.",
                cls="video-empty"),
            id="video-panel",
            cls="video-panel-inner",
            hx_get="/video_panel",
            hx_trigger="load",
            hx_swap="innerHTML",
        ),
    ]
    return _panel(
        panel_id="tae-video-panel",
        icon_cls="fas fa-video",
        panel_icon_cls="pi-video",
        title="Video",
        body=body,
        has_resize=True,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Frame / detection panel  (replaces old .img-panel + _image_panel_empty)
# ─────────────────────────────────────────────────────────────────────────────
def _image_panel_empty():
    """
    Kept for HTMX targets in existing routes — returns the inner content
    that gets swapped into #tae-frame-panel .panel-body.
    """
    return (
        Div(
            I(cls="fas fa-search-location"),
            P("Click a map marker to inspect detected objects.", cls="empty-txt"),
            cls="empty-state",
        ),
    )
 
 
def _frame_panel(hidden: bool = True) -> FT:
    body = Div(
        *_image_panel_empty(),
        id="tae-imgpanel",
        cls="frame-panel-body",
    )
    style = "display:none;" if hidden else ""
    p = _panel(
        panel_id="tae-frame-panel",
        icon_cls="fas fa-search-location",
        panel_icon_cls="pi-frame",
        title="Frame",
        body=body,
        has_resize=True,
    )
    # Inject inline style for initial hide
    if hidden:
        p.attrs["style"] = style
    return p
 
 


# ─────────────────────────────────────────────────────────────────────────────
# Detection panel  — independent list of all detections
# ─────────────────────────────────────────────────────────────────────────────
def _det_panel() -> FT:
    body = Div(
        Div(
            I(cls="fas fa-bullseye"),
            P("No detections yet — run a query.", cls="empty-txt"),
            cls="empty-state",
        ),
        id="det-panel-list",
        cls="det-panel-body",
        hx_get="/detections_panel_content",
        hx_trigger="load",
        hx_swap="innerHTML",
    )
    return _panel(
        panel_id="tae-det-panel",
        icon_cls="fas fa-bullseye",
        panel_icon_cls="pi-det",
        title="Detections",
        body=body,
        has_resize=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Toolbar shortcut pills (show/hide panels)
# ─────────────────────────────────────────────────────────────────────────────
def _panel_toolbar() -> FT:
    def _pill(label, panel_id, icon):
        return Button(
            I(cls=f"fas fa-{icon}"),
            f" {label}",
            cls="nav-pill-btn",
            onclick=f"showPanel('{panel_id}')",
        )
 
    return Div(
        _pill("Map",    "tae-map-panel",   "map-marked-alt"),
        _pill("Chat",   "tae-chat-panel",  "crosshairs"),
        _pill("Video",  "tae-video-panel", "video"),
        _pill("Frame",  "tae-frame-panel",  "search-location"),
        _pill("Detect", "tae-det-panel",    "bullseye"),
        cls="nav-pill",
    )
 

def _image_panel_empty() -> tuple:
    return (
        Div(
            Span("Detection View"),
            Span("×", cls="panel-close", onclick="closeImages()"),
            cls="panel-header",
        ),
        Div(
            P("Click a map marker to inspect detected objects.", cls="empty-txt"),
            cls="empty-state",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — pages
# ─────────────────────────────────────────────────────────────────────────────

def _index_page(build_map_fn, state: dict) -> tuple:
    """
    Call this from your index() route:
 
        @rt("/")
        def index():
            _build_map()
            return _index_page(_build_map, _state)
 
    Parameters
    ----------
    build_map_fn : callable — your existing _build_map() function
    state        : the global _state dict
    """
    mission = state.get("mission")
 
    return (
        Title("TAE · Tactical Awareness Engine"),
 
        # Navbar with panel toolbar
        _navbar(
            ingested=state.get("ingested", False),
            frame_count=state.get("frame_count", 0),
            mission=mission,
        ),
 
        # Workspace: dot-grid canvas + floating panels
        Div(
            Canvas(id="bg-canvas"),
            _map_panel(),
            _chat_panel(),
            _video_panel(),
            _frame_panel(hidden=True),
            _det_panel(),
            cls="tae-workspace",
        ),
        # Settings drawer — slides in from the right
        Div(cls="drawer-overlay", id="drawer-overlay", onclick="closeDrawer()"),
        Div(
            Div(id="drawer-content"),
            cls="drawer-panel",
            id="drawer-panel",
        ),
    )
 
 
@rt("/")
def index():
    _build_map()
    return _index_page(_build_map, _state)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Phase C: Video playback
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Phase D: Live streaming
# ─────────────────────────────────────────────────────────────────────────────



@rt("/feed_drawer")
def feed_drawer():
    """Settings-drawer content: file upload + live stream input."""
    lat0, lon0 = _state["map_center"]
    has_stream  = stream_mgr.running
    stream_style = (
        "width:100%;padding:7px;border-radius:5px;cursor:pointer;"
        "font-size:12px;font-family:var(--font-mono);font-weight:700;"
        + ("background:var(--blue-dim);color:var(--blue);border:1px solid var(--blue)"
           if not has_stream else
           "background:#7f1d1d;color:#fca5a5;border:1px solid #f87171")
    )
    return (
        Div(
            I(cls="fas fa-satellite-dish", style="font-size:12px;color:var(--muted)"),
            Span("Feed", cls="drawer-title"),
            Button("✕", cls="drawer-close", onclick="closeDrawer()"),
            cls="drawer-header",
        ),
        Div(
            Span("Upload files",
                 style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                       "text-transform:uppercase;display:block;margin-bottom:8px"),
            Label(
                Div(
                    I(cls="fas fa-cloud-upload-alt",
                      style="font-size:22px;color:var(--blue);margin-bottom:6px;display:block"),
                    Div("Drop files here or click to browse",
                        style="font-size:11px;color:var(--muted);line-height:1.6"),
                    Div(".mp4 + .srt  ·  .jpg / .jpeg",
                        style="font-size:9px;color:var(--border);margin-top:2px"),
                    style="text-align:center;padding:18px",
                ),
                Input(
                    type="file", name="files", multiple=True,
                    accept=".jpg,.jpeg,.png,.mp4,.mov,.avi,.mkv,.srt,.SRT",
                    hx_post="/upload",
                    hx_target="#tae-msgs",
                    hx_swap="beforeend",
                    hx_encoding="multipart/form-data",
                    hx_indicator="#upload-ind",
                    style="display:none",
                    **{"hx-on::after-request": "scrollChat(); refreshMap(); closeDrawer();"},
                ),
                style=(
                    "display:block;cursor:pointer;border:1.5px dashed var(--border);"
                    "border-radius:8px;transition:border-color .15s;"
                    "margin-bottom:12px"
                ),
            ),
            Div(style="height:1px;background:var(--border);margin:0 0 12px"),
            Span("Or connect a live stream",
                 style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                       "text-transform:uppercase;display:block;margin-bottom:8px"),
            Div(
                Span("RTSP / RTMP URL",
                     style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                           "text-transform:uppercase"),
                Input(id="stream-url", type="text",
                      placeholder="rtsp://192.168.1.1:554/live",
                      style="width:100%;background:var(--bg3);border:1px solid var(--border);"
                            "border-radius:5px;padding:6px 9px;color:var(--text);"
                            "font-family:var(--font-mono);font-size:11px;outline:none;margin-top:4px"),
                style="margin-bottom:8px",
            ),
            Div(
                Span("GPS anchor",
                     style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                           "text-transform:uppercase"),
                Div(
                    Input(id="stream-lat", type="number", step="any",
                          value=str(round(lat0, 5)), placeholder="Lat",
                          style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                "border-radius:5px;padding:6px 8px;color:var(--text);"
                                "font-family:var(--font-mono);font-size:11px;outline:none"),
                    Input(id="stream-lon", type="number", step="any",
                          value=str(round(lon0, 5)), placeholder="Lon",
                          style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                "border-radius:5px;padding:6px 8px;color:var(--text);"
                                "font-family:var(--font-mono);font-size:11px;outline:none"),
                    style="display:flex;gap:6px;margin-top:4px",
                ),
                style="margin-bottom:10px",
            ),
            Button(
                "▶  Start stream" if not has_stream else "■  Stop stream",
                onclick="startStream(event)" if not has_stream else "stopStream()",
                style=stream_style,
            ),
            Span("ffmpeg must be in PATH",
                 style="font-size:9px;color:var(--muted);margin-top:6px;display:block"),
            cls="drawer-section",
        ),
    )

@rt("/feed_panel")
def feed_panel():
    """Feed panel — unified file upload + live stream input."""
    lat0, lon0 = _state["map_center"]
    has_stream = stream_mgr.running
    stream_btn_style = (
        "width:100%;padding:7px;border-radius:5px;cursor:pointer;"
        "font-size:12px;font-family:var(--font-mono);font-weight:700;"
        + ("background:var(--blue-dim);color:var(--blue);border:1px solid var(--blue)"
           if not has_stream else
           "background:#7f1d1d;color:#fca5a5;border:1px solid #f87171")
    )
    upload_label_style = (
        "display:block;cursor:pointer;border:1.5px dashed var(--border);"
        "border-radius:8px;transition:border-color .15s"
    )
    return (
        Div(
            Span("📡 Feed",
                 style="font-family:var(--font-head);font-size:13px;font-weight:700;"
                       "color:var(--blue);letter-spacing:.06em"),
            Span("×", cls="video-panel-close", onclick="closeFeedPanel()"),
            cls="video-panel-header",
        ),
        Div(
            Div("Upload files",
                style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                      "text-transform:uppercase;margin-bottom:8px"),
            Label(
                Div(
                    I(cls="fas fa-cloud-upload-alt",
                      style="font-size:22px;color:var(--blue);margin-bottom:6px"),
                    Div("Drop files here or click to browse",
                        style="font-size:11px;color:var(--muted);line-height:1.6"),
                    Div(".mp4 + .srt  ·  .jpg / .jpeg",
                        style="font-size:9px;color:var(--border);margin-top:2px"),
                    style="display:flex;flex-direction:column;align-items:center;padding:18px",
                ),
                Input(
                    type="file", name="files", multiple=True,
                    accept=".jpg,.jpeg,.png,.mp4,.mov,.avi,.mkv,.srt,.SRT",
                    hx_post="/upload",
                    hx_target="#tae-msgs",
                    hx_swap="beforeend",
                    hx_encoding="multipart/form-data",
                    hx_indicator="#upload-ind",
                    style="display:none",
                    **{"hx-on::after-request": "scrollChat(); refreshMap(); closeFeedPanel();"},
                ),
                style=upload_label_style,
            ),
            style="padding:14px 14px 10px",
        ),
        Div(
            Div(style="flex:1;height:1px;background:var(--border)"),
            Span("or live stream",
                 style="font-size:9px;color:var(--muted);padding:0 10px;white-space:nowrap"),
            Div(style="flex:1;height:1px;background:var(--border)"),
            style="display:flex;align-items:center;padding:0 14px;margin-bottom:10px",
        ),
        Div(
            Div(
                Span("Stream URL (RTSP / RTMP)",
                     style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                           "text-transform:uppercase"),
                Input(id="stream-url", type="text",
                      placeholder="rtsp://192.168.1.1:554/live",
                      style="width:100%;background:var(--bg3);border:1px solid var(--border);"
                            "border-radius:5px;padding:6px 9px;color:var(--text);"
                            "font-family:var(--font-mono);font-size:11px;outline:none;margin-top:4px"),
                style="margin-bottom:8px",
            ),
            Div(
                Span("GPS anchor",
                     style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                           "text-transform:uppercase"),
                Div(
                    Input(id="stream-lat", type="number", step="any",
                          value=str(round(lat0, 5)), placeholder="Lat",
                          style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                "border-radius:5px;padding:6px 8px;color:var(--text);"
                                "font-family:var(--font-mono);font-size:11px;outline:none"),
                    Input(id="stream-lon", type="number", step="any",
                          value=str(round(lon0, 5)), placeholder="Lon",
                          style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                "border-radius:5px;padding:6px 8px;color:var(--text);"
                                "font-family:var(--font-mono);font-size:11px;outline:none"),
                    style="display:flex;gap:6px;margin-top:4px",
                ),
                style="margin-bottom:10px",
            ),
            Button(
                "▶  Start stream" if not has_stream else "■  Stop stream",
                onclick="startStream(event)" if not has_stream else "stopStream()",
                style=stream_btn_style,
            ),
            Span("ffmpeg must be in PATH",
                 style="font-size:9px;color:var(--muted);margin-top:6px;display:block"),
            style="padding:0 14px 14px",
        ),
    )

@rt("/stream/panel")
def stream_panel():
    """HTMX: returns the stream panel — either setup form or live player."""
    st = stream_mgr.status()
    paths = _state.get("mission_paths")
    lat0  = _state["map_center"][0]
    lon0  = _state["map_center"][1]

    header = Div(
        Span("🔴 Live Stream",
             style="font-family:var(--font-head);font-size:13px;font-weight:700;"
                   "color:#f87171;letter-spacing:.06em"),
        Span("×", cls="video-panel-close", onclick="closeVideoPanel()"),
        cls="video-panel-header",
    )

    if st["running"]:
        # ── Live player ───────────────────────────────────────────────────────
        return (
            header,
            Div(
                NotStr("""
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<video id="tae-live-video" controls autoplay muted
  style="width:100%;display:block;max-height:280px;object-fit:contain;background:#000">
</video>
<script>
(function(){
  const v = document.getElementById('tae-live-video');
  if (Hls.isSupported()) {
    const h = new Hls({lowLatencyMode:true});
    h.loadSource('/hls/stream.m3u8');
    h.attachMedia(v);
  } else if (v.canPlayType('application/vnd.apple.mpegurl')) {
    v.src = '/hls/stream.m3u8';
  }
})();
</script>"""),
                cls="video-wrap",
            ),
            Div(
                Div(
                    Span("● LIVE", style="color:#f87171;font-size:10px;font-weight:700"),
                    Span("", id="stream-frame-count",
                         style="color:var(--muted);font-size:9px"),
                    style="display:flex;justify-content:space-between;padding:10px 14px 6px",
                ),
                Button(
                    "■ Stop stream",
                    onclick="stopStream()",
                    style="margin:0 14px 12px;padding:6px 12px;background:#7f1d1d;"
                          "color:#fca5a5;border:1px solid #f87171;border-radius:5px;"
                          "cursor:pointer;font-size:11px;font-family:var(--font-mono);width:calc(100% - 28px)",
                ),
                cls="stream-status",
            ),
        )
    else:
        # ── Setup form ────────────────────────────────────────────────────────
        return (
            header,
            Div(
                Div(
                    Span("Stream URL (RTSP / RTMP / file path)", style="font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase"),
                    Input(id="stream-url", type="text", placeholder="rtsp://192.168.1.1:554/live",
                          style="width:100%;background:var(--bg3);border:1px solid var(--border);"
                                "border-radius:5px;padding:6px 9px;color:var(--text);"
                                "font-family:var(--font-mono);font-size:11px;outline:none;margin-top:4px"),
                    style="margin-bottom:10px",
                ),
                Div(
                    Span("Anchor GPS (used when stream has no telemetry)", style="font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase"),
                    Div(
                        Input(id="stream-lat", type="number", step="any",
                              value=str(round(lat0, 5)), placeholder="Latitude",
                              style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                    "border-radius:5px;padding:6px 9px;color:var(--text);"
                                    "font-family:var(--font-mono);font-size:11px;outline:none"),
                        Input(id="stream-lon", type="number", step="any",
                              value=str(round(lon0, 5)), placeholder="Longitude",
                              style="flex:1;background:var(--bg3);border:1px solid var(--border);"
                                    "border-radius:5px;padding:6px 9px;color:var(--text);"
                                    "font-family:var(--font-mono);font-size:11px;outline:none"),
                        style="display:flex;gap:6px;margin-top:4px",
                    ),
                    style="margin-bottom:12px",
                ),
                Button(
                    "▶ Start stream",
                    onclick="startStream(event)",
                    style="width:100%;padding:8px;background:var(--blue-dim);"
                          "color:var(--blue);border:1px solid var(--blue);"
                          "border-radius:5px;cursor:pointer;font-size:12px;"
                          "font-family:var(--font-mono);font-weight:700",
                ),
                Span(
                    "ffmpeg must be installed and reachable in PATH.",
                    style="font-size:9px;color:var(--muted);margin-top:8px;display:block",
                ),
                style="padding:14px",
            ),
        )


@rt("/stream/start", methods=["POST"])
async def stream_start(request: Request):
    """Start the live stream. Body: {url, lat, lon}"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
        url  = (body.get("url") or "").strip()
        lat  = float(body.get("lat") or _state["map_center"][0])
        lon  = float(body.get("lon") or _state["map_center"][1])
        if not url:
            return JSONResponse({"ok": False, "error": "URL required"})

        paths = _state.get("mission_paths")
        if not paths:
            return JSONResponse({"ok": False, "error": "No active mission"})

        hls_dir    = paths.uploads.parent / "hls"
        frames_dir = paths.uploads / "live_frames"

        stream_mgr.start(url, lat, lon, hls_dir, frames_dir)
        _state["video_files"] = list(_state.get("video_files", []))  # keep existing
        logger.info("Stream started: url=%s lat=%s lon=%s", url, lat, lon)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error("Stream start error: %s", e)
        return JSONResponse({"ok": False, "error": str(e)})


@rt("/stream/stop", methods=["POST"])
def stream_stop():
    """Stop the live stream."""
    from starlette.responses import JSONResponse
    stream_mgr.stop()
    return JSONResponse({"ok": True})


@rt("/stream/status")
def stream_status():
    """Poll endpoint for stream state."""
    from starlette.responses import JSONResponse
    return JSONResponse(stream_mgr.status())


@rt("/hls/{filename}")
def serve_hls(filename: str):
    """Serve HLS playlist and segment files."""
    from starlette.responses import Response
    paths = _state.get("mission_paths")
    if not paths:
        return Response("No active mission", status_code=404)
    hls_dir  = paths.uploads.parent / "hls"
    hls_file = hls_dir / Path(filename).name   # sanitise
    if not hls_file.exists():
        return Response("Not found", status_code=404)
    suffix = hls_file.suffix.lower()
    mime = {".m3u8": "application/vnd.apple.mpegurl",
            ".ts":   "video/mp2t"}.get(suffix, "application/octet-stream")
    return FileResponse(str(hls_file), media_type=mime,
                        headers={"Cache-Control": "no-cache"})


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Phase D: Object tracking
# ─────────────────────────────────────────────────────────────────────────────

@rt("/track", methods=["POST"])
async def track(request: Request):
    """
    Dense object tracking over all mission frames.
    Body: {query: "track all vehicles", confidence: 0.15}
    Fires as a background task; progress streamed via /track_progress.
    """
    from starlette.responses import JSONResponse
    body  = await request.json()
    query = (body.get("query") or "").strip()
    conf  = float(body.get("confidence", 0.15))

    if not query:
        return JSONResponse({"ok": False, "error": "query required"})

    paths = _state.get("mission_paths")
    if not paths:
        return JSONResponse({"ok": False, "error": "No active mission"})

    if not _state.get("ingested"):
        return JSONResponse({"ok": False, "error": "Ingest video frames first"})

    _state["tracking"]      = True
    _state["track_progress"] = 0
    _state["track_total"]    = 0
    _state["track_query"]    = query

    import threading
    threading.Thread(
        target=_tracking_background,
        args=(query, conf, paths),
        daemon=True,
    ).start()

    return JSONResponse({"ok": True, "message": f"Tracking started for: {query}"})


def _tracking_background(query: str, confidence: float, paths) -> None:
    """Tracking is now integrated into run_detection_pipeline — this stub kept for
    backward-compat with the /start_tracking route."""
    logger.info(
        "Tracking is handled automatically by run_detection_pipeline. "
        "Re-run a query to detect and track objects across all frames."
    )
    return




@rt("/track_progress")
def track_progress():
    """Poll endpoint for tracking progress."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "running":   _state.get("tracking", False),
        "done":      _state.get("track_progress", 0),
        "total":     _state.get("track_total", 0),
        "n_tracks":  _state.get("track_done"),
        "error":     _state.get("track_error"),
        "query":     _state.get("track_query", ""),
    })


@rt("/track_clear", methods=["POST"])
def track_clear():
    """Remove tracks for the active mission and rebuild the map."""
    from starlette.responses import JSONResponse
    paths = _state.get("mission_paths")
    if paths:
        tf = paths.maps / "tracks.json"
        if tf.exists():
            tf.unlink()
    _build_map()
    for k in ("tracking", "track_progress", "track_total",
              "track_done", "track_error", "track_query"):
        _state.pop(k, None)
    return JSONResponse({"ok": True})

@rt("/serve_video")
def serve_video(filename: str = ""):
    """
    Range-aware video serving via query param: /serve_video?filename=foo.mp4
    NOTE: do not name the param 'f' — FastHTML uses 'f' internally in _handle().
    """
    logger.info("serve_video called: filename=%s", filename)
    if not filename:
        from starlette.responses import Response
        return Response("Missing filename", status_code=400)
    paths = _state.get("mission_paths")
    if paths is None:
        logger.warning("serve_video: mission_paths is None")
        from starlette.responses import Response
        return Response("No active mission", status_code=404)
    video_file = paths.uploads / Path(filename).name   # sanitise — no path traversal
    logger.info("serve_video: resolved=%s exists=%s", video_file, video_file.exists())
    if not video_file.exists():
        from starlette.responses import Response
        return Response("Video not found", status_code=404)
    suffix = video_file.suffix.lower()
    mime = {"mp4": "video/mp4", "mov": "video/quicktime",
            "avi": "video/x-msvideo", "mkv": "video/x-matroska"}.get(suffix[1:], "video/mp4")
    return FileResponse(str(video_file), media_type=mime,
                        headers={"Accept-Ranges": "bytes"})




@rt("/detections_ready")
def detections_ready():
    """Lightweight poll: returns True once background replay has populated detections."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "ready":  _state.get("detections_ready", len(_state["detections"]) > 0),
        "count":  len(_state["detections"]),
    })

@rt("/video_ping")
def video_ping():
    """Diagnostic: confirms the video route layer is reachable."""
    from starlette.responses import JSONResponse
    logger.info("video_ping hit")
    return JSONResponse({"ok": True, "videos": _state.get("video_files", [])})

@rt("/video_panel")
def video_panel_content():
    """HTMX: returns the video panel inner HTML for the active mission."""
    video_files = _state.get("video_files", [])
    paths       = _state.get("mission_paths")

    if not video_files or paths is None:
        return (
            Div(
                Span("Video Playback",
                     style="font-family:var(--font-head);font-size:13px;font-weight:700;"
                           "color:var(--blue);letter-spacing:.06em"),
                Span("x", cls="video-panel-close", onclick="closeVideoPanel()"),
                cls="video-panel-header",
            ),
            Div("No video uploaded for this mission. "
                "Upload a .MP4 alongside the .SRT file.",
                cls="video-empty"),
        )

    vfile   = video_files[0]
    vpath   = paths.uploads / vfile
    dur_msg = ""
    try:
        import cv2 as _cv
        cap  = _cv.VideoCapture(str(vpath))
        fps_v = cap.get(_cv.CAP_PROP_FPS) or 30
        nf    = cap.get(_cv.CAP_PROP_FRAME_COUNT) or 0
        secs  = int(nf / fps_v)
        dur_msg = f"{secs//60}:{secs%60:02d}"
        cap.release()
    except Exception:
        pass


    return (
        Div(
            Span(f"Video — {vfile[:26]}",
                 style="font-family:var(--font-head);font-size:13px;font-weight:700;"
                       "color:var(--blue);letter-spacing:.06em;overflow:hidden;"
                       "text-overflow:ellipsis;white-space:nowrap"),
            Span("x", cls="video-panel-close", onclick="closeVideoPanel()"),
            cls="video-panel-header",
        ),
        Div(
            NotStr(
                f'<video id="tae-video" controls preload="metadata"'
                f' src="/serve_video?filename={vfile}"'
                f' onloadedmetadata="onVideoMeta()"'
                f' ontimeupdate="onVideoTime()"'
                f' style="width:100%;display:block;max-height:280px;'
                f'object-fit:contain;background:#000">'
                f'Your browser does not support HTML5 video.</video>'
            ),
            Canvas(id="bbox-canvas", cls="bbox-overlay"),
            cls="video-wrap",
        ),
        Div(
            Div(
                Span("Timeline",
                     style="color:var(--muted);font-size:9px;letter-spacing:.1em;"
                           "text-transform:uppercase"),
                Span(dur_msg, id="vtime",
                     style="color:var(--muted);font-size:9px"),
                style="display:flex;justify-content:space-between;margin-bottom:5px",
            ),
            Canvas(
                id="timeline-canvas",
                cls="timeline-canvas",
                onclick="seekVideo(event)",
                style="width:100%;height:36px",
            ),
            cls="timeline-wrap",
        ),
    )


@rt("/video_detections")
def video_detections():
    """JSON: detections enriched with video timestamps for the timeline."""
    from starlette.responses import JSONResponse
    frame_ts = _state.get("frame_timestamps", {})
    result   = []
    for det_id, det in _state["detections"].items():
        source = det.get("source", "")
        ts_ms  = frame_ts.get(source)
        if ts_ms is None:
            parent = det.get("parent_path", "")
            ts_ms  = frame_ts.get(Path(parent).name)
        result.append({
            "id":           det_id,
            "timestamp_ms": ts_ms,
            "lat":          det.get("lat"),
            "lon":          det.get("lon"),
            "color":        det.get("color", "#4ade80"),
            "confirmed":    det.get("confirmed", False),
            "label":        det.get("label", "")[:60],
            "query":        det.get("query", "")[:60],
            "bbox":         det.get("bbox"),
            "tile_x":       det.get("tile_x", 0),
            "tile_y":       det.get("tile_y", 0),
            "tile_w":       det.get("tile_w", 640),
            "tile_h":       det.get("tile_h", 640),
        })
    return JSONResponse(result)



@rt("/detections_panel_content")
def detections_panel_content():
    """HTMX: renders the detection list for the Detection panel."""
    dets     = _state.get("detections", {})
    frame_ts = _state.get("frame_timestamps", {})

    if not dets:
        return Div(
            I(cls="fas fa-bullseye"),
            P("No detections yet — run a query.", cls="empty-txt"),
            cls="empty-state",
        )

    rows = []
    for det_id, det in dets.items():
        label     = det.get("label", "Unknown")[:52]
        color     = det.get("color", "#4ade80")
        lat       = det.get("lat", 0)
        lon       = det.get("lon", 0)
        confirmed = det.get("confirmed", False)
        source    = det.get("source", "")
        ts_ms     = frame_ts.get(source)
        if ts_ms is None:
            ts_ms = frame_ts.get(Path(det.get("parent_path", "")).name)

        ts_str = ""
        if ts_ms is not None:
            s = int(ts_ms / 1000)
            ts_str = f"{s // 60}:{s % 60:02d}"

        dot = "●" if confirmed else "○"
        onclick = (
            f"selectDetection('{det_id}',{lat},{lon},"
            + (str(ts_ms) if ts_ms is not None else "null")
            + ")"
        )

        rows.append(Div(
            Span(dot, style=f"color:{color};font-size:14px;flex-shrink:0;line-height:1"),
            Div(
                Div(label, cls="det-row-label"),
                Div(ts_str or source[:24], cls="det-row-meta"),
                style="flex:1;min-width:0",
            ),
            id=f"det-row-{det_id}",
            cls="det-row",
            onclick=onclick,
        ))

    return Div(*rows, cls="det-rows-wrap")


@rt("/set_map_focus")
def set_map_focus(det_id: str = ""):
    """Recentre map on a specific detection; called by selectDetection()."""
    from starlette.responses import JSONResponse
    det = _state.get("detections", {}).get(det_id)
    if det:
        _state["map_center"] = [det["lat"], det["lon"]]
        _state["map_zoom"]   = 18
        _build_map()
    return JSONResponse({"ok": bool(det)})

@rt("/map")
def serve_map():
    paths = _state.get("mission_paths")
    if paths is None:
        from starlette.responses import Response
        return Response("<html><body>No active mission</body></html>",
                        media_type="text/html")
    map_file = paths.maps / "map.html"
    if not map_file.exists():
        _build_map()
    return FileResponse(str(map_file), media_type="text/html")


@rt("/tile_img/{key}")
def serve_tile_img(key: str):
    from starlette.responses import Response
    data = _tile_img_cache.get(key)
    if data is None:
        return Response("Tile not in cache.", status_code=404)
    return Response(content=data, media_type="image/jpeg")


@rt("/frame_img/{det_id}")
def serve_frame_img(det_id: str):
    from starlette.responses import Response
    data = _frame_img_cache.get(det_id)
    if data is None:
        return Response("Frame not rendered yet.", status_code=404)
    return Response(content=data, media_type="image/jpeg")


@rt("/segments/{filename}")
def serve_segment(filename: str):
    """Serve segment visualisation images from the active mission's segments dir."""
    from starlette.responses import Response
    paths = _state.get("mission_paths")
    if paths is None:
        return Response("No active mission.", status_code=404)
    seg_file = paths.segments / filename
    if not seg_file.exists():
        return Response("Not found.", status_code=404)
    suffix = seg_file.suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    return Response(content=seg_file.read_bytes(), media_type=mime)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — missions
# ─────────────────────────────────────────────────────────────────────────────

@rt("/missions/{mission_id}/drawer")
def mission_settings_drawer(mission_id: str):
    """Return settings drawer content for an existing mission (or new-mission form)."""
    if mission_id == "new":
        return _settings_drawer_content(None, is_new=True)
    m = mission_mgr.get(mission_id)
    if not m:
        return P("Mission not found.", style="padding:16px;color:var(--danger)")
    return _settings_drawer_content(m, is_new=False)


@rt("/missions/new/drawer")
def mission_new_drawer():
    """Return settings drawer content for creating a new mission."""
    return _settings_drawer_content(None, is_new=True)


@rt("/missions/{mission_id}/activate", methods=["POST"])
def mission_activate(mission_id: str):
    m = mission_mgr.get(mission_id)
    if m:
        _activate_mission(m)
    return ""   # caller does location.reload()


@rt("/missions", methods=["POST"])
async def mission_create(request: Request):
    form    = await request.form()
    name    = (form.get("name") or "Unnamed").strip()
    intents = form.getlist("intents")
    mid   = uuid.uuid4().hex
    paths = MissionPaths.for_mission(settings.DATA_DIR, _mission_folder(name, mid))
    definition = (form.get("definition") or "").strip()
    m = mission_mgr.create(
        name             = name,
        upload_path      = str(paths.uploads),
        lancedb_path     = str(paths.lancedb),
        segments_db_path = str(paths.segments / "segments.db"),
        allowed_intents  = intents or ["object_detection"],
        mission_id       = mid,
    )
    if definition:
        mission_mgr.update(m.id, definition=definition)
    _activate_mission(m)
    return ""   # caller does location.reload()


@rt("/missions/{mission_id}", methods=["PATCH"])
async def mission_update(mission_id: str, request: Request):
    form    = await request.form()
    name    = (form.get("name") or "").strip() or None
    intents = form.getlist("intents") or None
    definition = (form.get("definition") or "").strip() or None
    mission_mgr.update(mission_id, name=name, allowed_intents=intents, definition=definition)
    # Keep live state in sync if this is the active mission
    if mission_id == _state.get("mission_id") and definition is not None:
        m = mission_mgr.get(mission_id)
        if m:
            _state["mission_name"] = m.name
    # Update live display name if it's the active mission
    if mission_id == _state.get("mission_id"):
        if name:
            _state["mission_name"] = name
    return ""


@rt("/missions/{mission_id}/archive", methods=["POST"])
def mission_archive(mission_id: str):
    # If archiving the active mission, fall back to another active one
    if mission_id == _state.get("mission_id"):
        mission_mgr.archive(mission_id)
        others = mission_mgr.list_active()
        if others:
            _activate_mission(others[0])
    else:
        mission_mgr.archive(mission_id)
    return ""


@rt("/missions/{mission_id}/delete", methods=["POST"])
async def mission_delete(mission_id: str, request: Request):
    form         = await request.form()
    confirm_name = (form.get("del-confirm-input") or "").strip()
    m = mission_mgr.get(mission_id)
    if not m:
        return ""
    if confirm_name != m.name:
        logger.warning("Delete rejected: typed %r != %r", confirm_name, m.name)
        return ""
    # Wipe data directory — name-based folder scheme
    folder    = _mission_folder(m.name, mission_id)
    data_root = Path(settings.DATA_DIR) / folder
    if data_root.exists():
        shutil.rmtree(str(data_root))
    # Also wipe legacy bare-ID path if present
    legacy = Path(settings.DATA_DIR) / mission_id
    if legacy.exists() and legacy != data_root:
        shutil.rmtree(str(legacy))
    # If it was active, switch before deleting the DB record
    if mission_id == _state.get("mission_id"):
        mission_mgr.delete(mission_id)
        others = mission_mgr.list_active()
        if others:
            _activate_mission(others[0])
        else:
            # edge-case: no missions left → create Default
            _mid   = uuid.uuid4().hex
            _paths = MissionPaths.for_mission(settings.DATA_DIR, _mission_folder("Default", _mid))
            new_m = mission_mgr.create(
                name             = "Default",
                upload_path      = str(_paths.uploads),
                lancedb_path     = str(_paths.lancedb),
                segments_db_path = str(_paths.segments / "segments.db"),
                allowed_intents  = ["object_detection"],
                mission_id       = _mid,
            )
            _activate_mission(new_m)
    else:
        mission_mgr.delete(mission_id)
    return ""


@rt("/missions/archived")
def missions_archived():
    """Append archived mission items into the dropdown."""
    archived = mission_mgr.list_archived()
    items = []
    for m in archived:
        items.append(
            Div(
                Span(cls="sdot", style="background:var(--muted)"),
                Span(m.name, cls="m-item-name", style="color:var(--muted)"),
                Span("archived", cls="m-item-date"),
                cls="m-item",
                hx_post=f"/missions/{m.id}/activate",
                hx_swap="none",
                **{"hx-on::after-request": "location.reload()"},
            )
        )
    return *items,


# ─────────────────────────────────────────────────────────────────────────────
# Routes — ingestion
# ─────────────────────────────────────────────────────────────────────────────
def _ingest_background(saved_images: list[str], meta_file: Path, meta: dict):
    from core.sim_provider import SimD3Environment
    from core.ingestion import run_ingestion
    try:
        db.initialize_table(vector_dim=settings.CLIP_DIM)
        sim   = SimD3Environment(str(meta_file))
        lib   = _main_get_search_lib()
        total = len(sim.frame_names)
        logger.info("Starting CLIP ingestion…")

        def _on_frame(idx: int, name: str):
            _state["ingest_progress"] = f"frame {idx} of {total} — {name}"

        tiles_ok, frames_failed = run_ingestion(sim, spatial, lib, db,
                                                on_frame=_on_frame)
        logger.info(f"Ingestion done — {tiles_ok} tiles, {frames_failed} failed.")

        try:
            df           = db.table.to_pandas()
            total_frames = df["parent_path"].nunique()
            lats = df.groupby("parent_path")["lat"].first().tolist()
            lons = df.groupby("parent_path")["lon"].first().tolist()
            if lats and lons:
                import statistics as _st
                _state["map_center"] = [_st.median(lats), _st.median(lons)]
                _state["map_zoom"]   = 16
        except Exception:
            total_frames = _state["frame_count"]

        _state["frame_count"] = total_frames
        _state["ingested"]    = True
        _state["tile_count"]  = tiles_ok
        _state["detections"]  = {}
        _build_map()

        # ── Auto-analysis against mission definition ──────────────────────────
        # Runs AFTER full ingestion — wrapped in its own try/except so a
        # failure here never masks a successful ingestion in the UI message.
        auto_summary = ""
        try:
            mid = _state.get("mission_id")
            if mid:
                m = mission_mgr.get(mid)
                if m and m.definition:
                    _state["ingest_progress"] = (
                        f"Auto-analyzing: '{m.definition[:50]}'..."
                    )
                    color = _MARKER_COLORS[
                        _state["query_color_idx"] % len(_MARKER_COLORS)
                    ]
                    _state["query_color_idx"] += 1
                    # Same parameters as a manual query — CLIP handles
                    # scanning the full index, VLM sees only the top matches.
                    from ai.detection_pipeline import run_detection_pipeline
                    _classified = intent_clf.classify(m.definition)
                    _tracks = run_detection_pipeline(
                        params           = _classified.params,
                        all_tiles        = db.get_all_tiles(),
                        original_query   = m.definition,
                        analyst          = analyst,
                        sam_segmentor    = None,
                        api_key          = getattr(settings, "REPLICATE_API_KEY", ""),
                        model_version    = getattr(settings, "DETECTOR_REPLICATE_MODEL", ""),
                        timeout_s     = getattr(settings, "DETECTOR_TIMEOUT_S", 180),
                        actual_alt_m     = _state.get("mean_alt_m", 100.0),
                        color            = color,
                    )
                    mission_mgr.update(mid, scene_context=m.definition)
                    _state["last_query"] = m.definition
                    _state["detections_ready"] = True

                    # Persist tracks → _state["detections"], rebuild map
                    _new_det_ids = []
                    for _track in _tracks:
                        _det_id = uuid.uuid4().hex[:10]
                        _best   = _track.best
                        _img_urls = []
                        for _det in _track.detections:
                            _u = _annotate_and_save(
                                {"parent_path": _det.parent_path,
                                "tile_x": _det.tile_x, "tile_y": _det.tile_y,
                                "tile_w": _det.tile_w, "tile_h": _det.tile_h},
                                _det.bbox_tile, m.definition[:20]
                            )
                            if _u:
                                _img_urls.append(_u)
                        _state["detections"][_det_id] = {
                            "lat": _track.lat, "lon": _track.lon,
                            "label": _best.label, "color": color,
                            "confirmed": True, "img_urls": _img_urls,
                            "is_multiangle": len(_track.detections) > 1,
                            "source_count": len(_track.detections),
                            "bbox": _best.bbox_tile,
                            "source": Path(_best.parent_path).name,
                            "parent_path": _best.parent_path,
                            "tile_x": _best.tile_x, "tile_y": _best.tile_y,
                            "tile_w": _best.tile_w, "tile_h": _best.tile_h,
                            "vlm_report": _best.vlm_report,
                            "track_id": _track.track_id,
                            "gsd":          "—",
                        }
                        _new_det_ids.append(_det_id)

                    if _new_det_ids:
                        _recenter_on_detections(_new_det_ids)
                        _build_map()
                        _paths = _state.get("mission_paths")
                        if _paths:
                            _save_detections(_paths.detections)

                    snip = m.definition[:40] + (
                        "..." if len(m.definition) > 40 else ""
                    )
                    auto_summary = (
                        f"Auto-analysis: {len(_tracks)} object(s) found for '{snip}'."
                        if _tracks
                        else f"Auto-analysis complete — no objects confirmed for '{snip}'."
                    )
                    logger.info(auto_summary)
        except Exception as ae:
            logger.error("Auto-analysis failed (ingestion was successful): %s", ae)
            auto_summary = "Auto-analysis failed — you can still query manually."

        # Tracking is now integrated into run_detection_pipeline (query-time).
        # No separate post-ingestion tracking pass needed.

        _state["ingest_msg"] = (
            f"Indexed {tiles_ok} tiles across {total_frames} frame(s). "
            + (auto_summary if auto_summary else "Theater is ready — query below.")
        )
        _state["ingesting"]  = False   # signal poll to deliver the message
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}", exc_info=True)
        _state["ingesting"]  = False
        _state["ingest_msg"] = f"Ingestion failed: {e}"


@rt("/upload", methods=["POST"])
async def upload(request: Request):
    from starlette.requests import ClientDisconnect
    paths = _state.get("mission_paths")
    if paths is None:
        return _msg("⚠️  No active mission — create one first.", "sys")

    try:
        form  = await request.form()
        files = form.getlist("files")
    except ClientDisconnect:
        return _msg(
            "⚠️  Upload failed: connection dropped. "
            "Try fewer images at once (10 max recommended).", "sys"
        )

    if not files or all(not f.filename for f in files):
        return _msg("⚠️  No files received.", "sys")

    # ── Save files, separate by type ────────────────────────────────────────
    saved_images: list[str] = []   # all JPEG paths entering the pipeline
    video_files:  list[Path] = []  # raw video files
    srt_files:    dict[str, Path] = {}  # stem.lower() → .SRT path
    meta: dict = {}  # pre-populated from SRT for video frames

    for f in files:
        if not f.filename:
            continue
        fname  = Path(f.filename).name
        suffix = Path(fname).suffix.lower()
        dest   = paths.uploads / fname
        dest.write_bytes(await f.read())
        if suffix == ".srt":
            srt_files[Path(fname).stem.lower()] = dest
            logger.info(f"SRT sidecar saved: {fname}")
        elif suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            video_files.append(dest)
        elif suffix in {".jpg", ".jpeg", ".png"}:
            saved_images.append(str(dest))

    # ── Process video files with telemetry-aware sampler ─────────────────────
    for video_path in video_files:
        # Find matching SRT (uploaded in same batch or already on disk)
        srt_path = srt_files.get(video_path.stem.lower())
        if not srt_path:
            for ext in (".SRT", ".srt"):
                candidate = video_path.with_suffix(ext)
                if candidate.exists():
                    srt_path = candidate
                    break

        srt_frames = None
        if srt_path:
            try:
                srt_frames = _srt_parser.parse(srt_path)
                logger.info(
                    f"SRT loaded for '{video_path.name}': "
                    f"{len(srt_frames)} telemetry frames"
                )
            except Exception as e:
                logger.warning(f"SRT parse failed for {srt_path}: {e}")
        else:
            logger.warning(
                f"No .SRT sidecar found for '{video_path.name}' — "
                f"video frames will have no telemetry."
            )

        try:
            frame_pairs = _video_sampler.sample_file(
                video_path         = video_path,
                out_dir            = paths.uploads,
                srt_frames         = srt_frames,
                interval_sec       = getattr(settings, "VIDEO_SAMPLE_INTERVAL_SEC", 2.0),
                adaptive           = getattr(settings, "VIDEO_ADAPTIVE_SAMPLING", True),
                target_overlap_pct = getattr(settings, "VIDEO_TARGET_OVERLAP_PCT", 60.0),
                sensor_w_mm        = settings.SENSOR_WIDTH_MM,
                focal_mm           = settings.FOCAL_LENGTH_MM,
            )
        except Exception as e:
            logger.error(f"Video sampling failed for {video_path}: {e}")
            continue

        # Track video file for playback panel
        if video_path.name not in _state["video_files"]:
            _state["video_files"].append(video_path.name)

        for jpeg_path, srt_frame in frame_pairs:
            saved_images.append(str(jpeg_path))
            if srt_frame is not None:
                # Store frame timestamp for timeline
                _state["frame_timestamps"][jpeg_path.name] = (
                    srt_frame.timestamp_ms
                )
                # Peek at image dimensions for the meta entry
                img = cv2.imread(str(jpeg_path))
                if img is not None:
                    h, w = img.shape[:2]
                    meta[jpeg_path.name] = _srt_parser.to_meta_entry(
                        srt_frame, jpeg_path, w, h
                    )

        # Persist timestamps so they survive server restarts
        if _state["frame_timestamps"]:
            ts_file = paths.uploads / "frame_timestamps.json"
            try:
                ts_file.write_text(
                    json.dumps(_state["frame_timestamps"]), encoding="utf-8"
                )
            except Exception as e:
                logger.warning("Could not save frame_timestamps.json: %s", e)

    if not saved_images:
        return _msg("⚠️  No processable files found in upload.", "sys")

    # ── Extract XMP metadata from still images not already covered by SRT ─────
    gen = TAESimGenerator(str(paths.uploads), str(paths.uploads))
    for img_path in saved_images:
        name = Path(img_path).name
        if name in meta:
            continue   # already populated from SRT
        try:
            entry = gen._extract_dji_data(Path(img_path))
            if entry:
                meta[name] = entry
        except Exception as e:
            logger.warning(f"XMP extraction failed for {img_path}: {e}")

    if not meta:
        return _msg(
            "⚠️  No telemetry found. For video, upload the .SRT sidecar file "
            "alongside the .MP4. For images, ensure DJI XMP metadata is present.",
            "sys"
        )

    meta_file = paths.sim_metadata
    meta_file.write_text(json.dumps(meta))

    lats = [v["lat"] for v in meta.values() if v.get("lat") not in (None, 0.0)]
    lons = [v["lon"] for v in meta.values() if v.get("lon") not in (None, 0.0)]
    if lats and lons:
        import statistics as _st
        _state["map_center"] = [_st.median(lats), _st.median(lons)]
        _state["map_zoom"]   = 16

    # Extract mean AGL altitude for shape-prior scaling in the detection pipeline
    alts = [v.get("alt_m") or v.get("z") or 0 for v in meta.values()]
    alts = [a for a in alts if isinstance(a, (int, float)) and a > 0]
    if alts:
        _state["mean_alt_m"] = sum(alts) / len(alts)
        logger.info("Mean AGL altitude: %.1f m", _state["mean_alt_m"])

    from starlette.background import BackgroundTasks
    _state["ingesting"]   = True
    _state["ingest_msg"]  = None
    _state["frame_count"] = len(meta)
    tasks = BackgroundTasks()
    tasks.add_task(_ingest_background, saved_images, meta_file, meta)

    # If video files are now present, activate the video panel immediately.
    # The panel div is already in the DOM but was rendered without "open"
    # (because video_files was empty at page load time).
    video_activation = Script(
        "const vp=document.getElementById('video-panel');"
        "if(vp && !vp.classList.contains('open')){"
        "  vp.classList.add('open');"
        "  htmx.ajax('GET','/video_panel',"
        "    {target:'#video-panel',swap:'innerHTML'});"
        "}"
    ) if _state.get("video_files") else ""

    return (
        _msg(f"{len(saved_images)} image(s) received. Indexing in background…", "sys"),
        Div(
            Span("Indexing... ", cls="pulse-txt"),
            hx_get="/upload_progress",
            hx_trigger="every 2s",
            hx_target="this",
            hx_swap="outerHTML",
        ),
        _status_badge(),
        video_activation,
    ), tasks


@rt("/upload_progress")
def upload_progress():
    if _state.get("ingesting"):
        progress = _state.get("ingest_progress", "")
        label    = f"Indexing {progress}…" if progress else "Indexing…"
        return Div(
            Span(label, cls="pulse-txt"),
            hx_get="/upload_progress",
            hx_trigger="every 2s",
            hx_swap="outerHTML",
        )
    msg = _state.pop("ingest_msg", None)
    if msg:
        refresh_script = Script("""
            refreshMap();
            htmx.ajax('GET', '/detections_panel_content',
                {target: '#detections-panel-content', swap: 'innerHTML'});
            htmx.ajax('GET', '/video_panel',
                {target: '#video-panel', swap: 'innerHTML'});
        """)
        video_act = Script(
            "const vp=document.getElementById('video-panel');"
            "if(vp && !vp.classList.contains('open')){"
            "  vp.classList.add('open');"
            "}"
        ) if _state.get("video_files") else ""
        return _msg(msg, "sys"), _status_badge(), refresh_script, video_act
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# SAM2 segmentor (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_segmentor = None

def _get_segmentor():
    """Lazy-load SAM2Segmentor — avoids heavy model init at startup."""
    global _segmentor
    if _segmentor is None:
        from ai.segmentor import SAM2Segmentor
        _segmentor = SAM2Segmentor(
            api_key       = getattr(settings, "REPLICATE_API_KEY", ""),
            model_version = getattr(settings, "SAM_REPLICATE_MODEL", ""),
            max_dim       = getattr(settings, "SAM_MAX_DIM", 1024),
        )
    return _segmentor


# ─────────────────────────────────────────────────────────────────────────────
# Active mission helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_active_mission():
    """Return the Mission object for the current mission_id, or None."""
    mid = _state.get("mission_id")
    if not mid:
        return None
    try:
        return mission_mgr.get(mid)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection query handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_anomaly_query(message, params, color, user_bubble, mission):
    """
    Handle anomaly_detection intent:
    CLIP scene retrieval → SAM segments from SegmentStore → CLIP scoring → VLM verify.
    """
    try:
        from ai.anomaly import VocabularyBuilder, score_segments
        from core.segment_store import SegmentStore

        paths = _state.get("mission_paths")
        if not paths:
            return user_bubble, _msg("⚠️  No active mission.", "sys")

        seg_db_path = getattr(paths, "segments_db_path",
                              getattr(paths, "segments_db", None))
        if not seg_db_path or not Path(str(seg_db_path)).exists():
            return (
                user_bubble,
                _msg(
                    "⚠️  No segment index for this mission. "
                    "Enable 'anomaly_detection' intent before uploading imagery "
                    "so SAM2 segments are computed at ingest time.",
                    "sys",
                ),
            )

        lib   = _main_get_search_lib()
        vocab = VocabularyBuilder(lib)
        store = SegmentStore(str(seg_db_path))

        # Retrieve candidate frames via CLIP, then score their segments
        clip_query = f"aerial drone nadir overhead view: {message}"
        q_vec      = lib.encode_text(clip_query)
        candidates = db.semantic_search(q_vec, limit=20, frames_to_return=8)

        if not candidates:
            return user_bubble, _msg("No candidate frames found for anomaly search.", "sys")

        scored = score_segments(
            store       = store,
            frame_paths = [c["parent_path"] for c in candidates],
            vocab       = vocab,
            query       = message,
            top_n       = 5,
        )

        if not scored:
            return (
                user_bubble,
                _msg(f"No anomalies detected for: <i>{message}</i>", "sys"),
            )

        new_det_ids: list[str] = []
        for seg in scored:
            # VLM verify
            tile_img = seg.crop  # numpy array
            if tile_img is None:
                continue

            verify = analyst.verify_detection(
                image          = tile_img,
                criteria       = params.vlm_verification_criteria,
                report_fields  = params.vlm_reporting_fields,
                original_query = message,
            )
            if not verify.get("confirmed"):
                continue

            det_id = uuid.uuid4().hex[:10]
            # Geo-locate: use segment bbox centre within its parent frame
            cand_match = next(
                (c for c in candidates if c["parent_path"] == seg.frame_path), None
            )
            lat = cand_match["lat"] if cand_match else 0.0
            lon = cand_match["lon"] if cand_match else 0.0

            _state["detections"][det_id] = {
                "lat":       lat,
                "lon":       lon,
                "label":     message,
                "color":     color,
                "confirmed": True,
                "img_urls":  [],
                "gsd":       "—",
                "bbox":      seg.bbox,
                "source":    Path(seg.frame_path).name,
                "parent_path": seg.frame_path,
                "tile_x":    seg.bbox[0] if seg.bbox else 0,
                "tile_y":    seg.bbox[1] if seg.bbox else 0,
                "tile_w":    (seg.bbox[2] - seg.bbox[0]) if seg.bbox else 640,
                "tile_h":    (seg.bbox[3] - seg.bbox[1]) if seg.bbox else 640,
                "vlm_report": verify.get("report", {}),
            }
            new_det_ids.append(det_id)

        _recenter_on_detections(new_det_ids)
        _build_map()

        dot = f'<span style="color:{color};font-size:13px">&#9679;</span>'
        if new_det_ids:
            reply = (
                f"{dot} {len(new_det_ids)} anomaly/anomalies confirmed. "
                f"Query: <i>{message}</i>"
            )
        else:
            reply = (
                f"{dot} No confirmed anomalies for: <i>{message}</i>. "
                f"CLIP found candidates but VLM did not confirm."
            )
        return user_bubble, _msg_html(reply)

    except Exception as exc:
        logger.error("Anomaly query failed: %s", exc, exc_info=True)
        return user_bubble, _msg(f"⚠️  Anomaly detection error: {exc}", "sys")

# ─────────────────────────────────────────────────────────────────────────────
# Routes — query
# ─────────────────────────────────────────────────────────────────────────────

@rt("/query", methods=["POST"])
async def query(message: str):  # noqa — signature only for illustration
    from ai.detection_pipeline import run_detection_pipeline
 
    if not message.strip():
        return ""
 
    user_bubble = _msg(message, "user")
 
    if not _state["ingested"] and db.row_count() == 0:
        return user_bubble, _msg("⚠️  No imagery indexed yet. Upload images first.", "sys")
 
    _state["ingested"]   = True
    _state["last_query"] = message
    mid = _state.get("mission_id")
    if mid:
        mission_mgr.update(mid, scene_context=message)
 
    color = _MARKER_COLORS[_state["query_color_idx"] % len(_MARKER_COLORS)]
    _state["query_color_idx"] += 1
 
    classified = intent_clf.classify(message)
    intent     = classified.params.intent
    logger.info(
        "Intent: %s | conf=%.2f | %s", intent, classified.confidence, classified.reasoning
    )
 
    mission = _state.get("mission") or _get_active_mission()
    if mission and not mission.allows(intent):
        allowed = ", ".join(mission.allowed_intents)
        return (
            user_bubble,
            _msg(
                f"⚠️ Intent <b>{intent}</b> not enabled for this mission.<br>"
                f"Allowed: <b>{allowed}</b>.",
                "sys",
            ),
        )
 
    # ── anomaly_detection ─────────────────────────────────────────────────────
    if intent == "anomaly_detection":
        return _handle_anomaly_query(
            message, classified.params, color, user_bubble, mission
        )
 
    # ── object_detection — new v2 6-stage pipeline ────────────────────────────
    params: ObjectDetectionParams = classified.params
 
    all_tiles = db.get_all_tiles()
    if not all_tiles:
        return user_bubble, _msg("⚠️  No tiles in index. Upload imagery first.", "sys")
 
    actual_alt_m = _state.get("mean_alt_m", 100.0)
 
    tracks = run_detection_pipeline(
        params           = params,
        all_tiles        = all_tiles,
        original_query   = message,
        analyst          = analyst,
        sam_segmentor = _get_segmentor() if getattr(settings, "REPLICATE_API_KEY", "") else None,
        api_key          = getattr(settings, "REPLICATE_API_KEY", ""),
        model_version    = getattr(settings, "DETECTOR_REPLICATE_MODEL", ""),
        timeout_s        = getattr(settings, "DETECTOR_TIMEOUT_S", 180),
        actual_alt_m     = actual_alt_m,
        color            = color,
    )
 
    new_det_ids: list[str] = []
    for track in tracks:
        det_id = uuid.uuid4().hex[:10]
        best   = track.best
 
        img_urls = []
        for det in track.detections:
            tile_candidate = {
                "parent_path": det.parent_path,
                "tile_x": det.tile_x, "tile_y": det.tile_y,
                "tile_w": det.tile_w, "tile_h": det.tile_h,
            }
            url = _annotate_and_save(tile_candidate, det.bbox_tile, best.label)
            if url:
                img_urls.append(url)
 
        report_summary = ""
        if best.vlm_report:
            parts = [f"{k}: {v}" for k, v in best.vlm_report.items() if v]
            report_summary = " · ".join(parts[:3])
 
        _state["detections"][det_id] = {
            "lat":           track.lat,
            "lon":           track.lon,
            "label":         best.label,
            "color":         color,
            "confirmed":     True,
            "img_urls":      img_urls,
            "is_multiangle": len(track.detections) > 1,
            "source_count":  len(track.detections),
            "gsd":           "-",
            "bbox":          best.bbox_tile,
            "source":        Path(best.parent_path).name,
            "parent_path":   best.parent_path,
            "tile_x":        best.tile_x,
            "tile_y":        best.tile_y,
            "tile_w":        best.tile_w,
            "tile_h":        best.tile_h,
            "vlm_report":    best.vlm_report,
            "track_id":      track.track_id,
        }
        new_det_ids.append(det_id)
        n_frames = len(set(d.parent_path for d in track.detections))
        logger.info(
            "Track %s | %s | LAT %.6f LON %.6f | %d frame(s)",
            track.track_id, track.label, track.lat, track.lon, n_frames,
        )
 
    _recenter_on_detections(new_det_ids)
    _build_map()
 
    paths = _state.get("mission_paths")
    if paths:
        _save_detections(paths.detections)
 
    dot_solid  = f'<span style="color:{color};font-size:13px">&#9679;</span>'
    dot_hollow = f'<span style="color:{color};font-size:13px">&#9675;</span>'
 
    if tracks:
        n        = len(tracks)
        n_frames = len(set(d.parent_path for t in tracks for d in t.detections))
        reply    = (
            f"{dot_solid} {n} instance{'s' if n > 1 else ''} confirmed "
            f"across {n_frames} frame(s). Query: <i>{message}</i>"
        )
    else:
        reply = (
            f"{dot_hollow} YOLO-World + VLM found no confirmed detections for: "
            f"<i>{message}</i>. "
            f"Try a more specific query or check that imagery is uploaded."
        )
 
    return user_bubble, _msg_html(reply)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — detection view
# ─────────────────────────────────────────────────────────────────────────────

@rt("/toggle_chat")
def toggle_chat():
    return _chat_panel(collapsed=False)


@rt("/toggle_coverage")
def toggle_coverage():
    try:
        _state["show_coverage"] = not _state["show_coverage"]
        _build_map()
        state = "shown" if _state["show_coverage"] else "hidden"
        return _msg(f"Coverage polygon {state} (Ctrl+P to toggle).", "sys"), _status_badge()
    except Exception as e:
        logger.warning(f"toggle_coverage error: {e}")
        return ""


@rt("/images/{det_id}")
def images(det_id: str):
    det = _state["detections"].get(det_id)
    if not det:
        return (

            Div(P("Detection not found.", style="color:var(--danger);padding:20px"),
                cls="empty-state"),
        )
    fv_widget = frame_view(det_id, mode="tile")
    meta_div  = Div(
        Div(det["label"][:60], cls="label"),
        f"LAT {det['lat']:.6f}  ·  LON {det['lon']:.6f}",
        Br(),
        f"GSD {det.get('gsd', '—')} cm/px  ·  Source: {det['source']}",
        cls="det-meta", style="margin:0 12px 12px",
    )
    return (

        Div(fv_widget, meta_div, cls="det-card"),
    )


@rt("/frame_view/{det_id}")
def frame_view(det_id: str, mode: str = "tile"):
    det = _state["detections"].get(det_id)
    if not det:
        return P("Detection not found.", style="color:var(--danger);padding:20px")

    parent_path = det.get("parent_path")
    tx, ty = det.get("tile_x", 0), det.get("tile_y", 0)
    tw, th = det.get("tile_w", 0), det.get("tile_h", 0)
    bbox   = det.get("bbox")

    if mode == "frame" and parent_path:
        img = cv2.imread(parent_path)
        if img is not None:
            TILE_CLR = (180, 0, 255)
            overlay  = img.copy()
            cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), TILE_CLR, -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), (0,0,0), 20)
            cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), TILE_CLR, 12)
            cv2.putText(img, "TILE", (tx+8, ty+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 8)
            cv2.putText(img, "TILE", (tx+8, ty+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, TILE_CLR, 3)
            if bbox and len(bbox) == 4:
                bx1 = tx+int(bbox[0]); by1 = ty+int(bbox[1])
                bx2 = tx+int(bbox[2]); by2 = ty+int(bbox[3])
                cv2.rectangle(img, (bx1,by1), (bx2,by2), (0,0,0), 10)
                cv2.rectangle(img, (bx1,by1), (bx2,by2), (74,222,128), 6)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return P("Failed to encode frame.", style="color:var(--danger);padding:12px")
            _frame_img_cache[det_id] = buf.tobytes()
            img_url = f"/frame_img/{det_id}"
            other_mode, other_label = "tile", "Tile view"
        else:
            return P("Could not load parent frame.", style="color:var(--danger);padding:12px")
    else:
        stored = det["img_urls"][0] if det.get("img_urls") else None
        tile_data = None
        if stored and stored.startswith("/tile_img/"):
            key = stored.split("/")[-1]
            tile_data = _tile_img_cache.get(key)
            img_url   = stored if tile_data else None
        else:
            img_url = None
        if not img_url:
            img_url = _tile_to_static_url(det) or ""
            if img_url and img_url.startswith("/tile_img/"):
                tile_data = _tile_img_cache.get(img_url.split("/")[-1])

        # Annotate tile with bounding box server-side
        if tile_data and bbox and len(bbox) == 4:
            import numpy as np
            arr = np.frombuffer(tile_data, np.uint8)
            tile_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if tile_img is not None:
                bx1, by1 = int(bbox[0]), int(bbox[1])
                bx2, by2 = int(bbox[2]), int(bbox[3])
                cv2.rectangle(tile_img, (bx1, by1), (bx2, by2), (0, 0, 0), 6)
                cv2.rectangle(tile_img, (bx1, by1), (bx2, by2), (74, 222, 128), 3)
                ok2, buf = cv2.imencode(".jpg", tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok2:
                    cache_key = det_id + "_tile"
                    _frame_img_cache[cache_key] = buf.tobytes()
                    img_url = f"/frame_img/{cache_key}"

        other_mode, other_label = "frame", "Full frame"

    toggle_btn = Button(
        other_label,
        hx_get=f"/frame_view/{det_id}?mode={other_mode}",
        hx_target=f"#fv-{det_id}",
        hx_swap="outerHTML",
        style=("margin:8px 12px;padding:5px 14px;"
               "background:var(--bg4);border:1px solid var(--border);"
               "border-radius:5px;cursor:pointer;font-size:11px;"
               "font-family:var(--font-mono);color:var(--text);"),
    )
    return Div(
        toggle_btn,
        Img(src=img_url, style="width:100%;display:block", loading="lazy") if img_url else "",
        id=f"fv-{det_id}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Boot
# ─────────────────────────────────────────────────────────────────────────────

# Wire stream callbacks
def _stream_on_frames(paths, lat, lon):
    """Index a batch of new live frames into LanceDB."""
    import cv2 as _cv2
    paths_obj = _state.get('mission_paths')
    if not paths_obj:
        return
    gen = TAESimGenerator(str(paths_obj.uploads), str(paths_obj.uploads))
    meta = {}
    for p in paths:
        img = _cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            meta[p.name] = {
                "full_path":    str(p),
                "lat":          lat,
                "lon":          lon,
                "z":            80.0,
                "gimbal_pitch": -90.0,
                "gimbal_yaw":   0.0,
                "gimbal_roll":  0.0,
                "img_w_px":     w,
                "img_h_px":     h,
            }
    if not meta:
        return
    meta_file = paths_obj.uploads / 'live_meta.json'
    meta_file.write_text(json.dumps(meta))
    sim = TAESimGenerator(str(paths_obj.uploads), str(paths_obj.uploads))
    sim.pose_metadata = meta
    from core.ingestion import run_ingestion
    tiles_ok, _ = run_ingestion(sim, spatial, _main_get_search_lib(), db)
    _state['frame_count'] = _state.get('frame_count', 0) + len(paths)
    _state['tile_count']  = _state.get('tile_count',  0) + tiles_ok
    _state['ingested']    = True
    logger.info('Stream: indexed %d tiles from %d frames', tiles_ok, len(paths))

def _stream_on_analyse():
    """Trigger auto-analysis against mission definition."""
    mid = _state.get('mission_id')
    if not mid:
        return
    m = mission_mgr.get(mid)
    if not (m and m.definition):
        return
    color = _MARKER_COLORS[_state['query_color_idx'] % len(_MARKER_COLORS)]
    _state['query_color_idx'] += 1
    from ai.detection_pipeline import run_detection_pipeline
    _classified = intent_clf.classify(m.definition)
    run_detection_pipeline(
        params         = _classified.params,
        all_tiles      = db.get_all_tiles(),
        original_query = m.definition,
        analyst        = analyst,
        sam_segmentor  = None,
        api_key        = getattr(settings, "REPLICATE_API_KEY", ""),
        model_version  = getattr(settings, "DETECTOR_REPLICATE_MODEL", ""),
        timeout_s     = getattr(settings, "DETECTOR_TIMEOUT_S", 180),
        actual_alt_m   = _state.get("mean_alt_m", 100.0),
        color          = color,
    )
    logger.info('Stream auto-analysis complete')

stream_mgr.init(_stream_on_frames, _stream_on_analyse)

_startup()
_PORT = int(os.environ.get("PORT", 8000))
logger.info(f"Starting TAE on port {_PORT}")
serve(host="0.0.0.0", port=_PORT)