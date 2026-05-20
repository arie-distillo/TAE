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
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst
from tools.ingest_telemetry import TAESimGenerator
from core.video import SRTParser, VideoSampler, AdaptiveSampler
from core.app_state import (
    _state, _frame_img_cache, _tile_img_cache,
    _MARKER_COLORS, _CLIP_AERIAL_CTX,
)
from core.analysis import (
    _build_map, _optimal_zoom, _recenter_on_detections,
    _load_tile, _annotate_and_save, _tile_to_static_url,
    _extract_video_frames, _execute_analysis,
    init as _init_analysis,
)
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
_search_lib: SearchLibrarian | None = None

# Video telemetry helpers (singletons — stateless, safe to reuse)
_srt_parser    = SRTParser()
_video_sampler = VideoSampler()


def get_search_lib() -> SearchLibrarian:
    global _search_lib
    if _search_lib is None:
        logger.info("Loading CLIP model – this may take a moment…")
        _search_lib = SearchLibrarian(settings.CLIP_MODEL)
    return _search_lib


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
    bodykw={"style": "margin:0"},
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


def _navbar() -> FT:
    missions = mission_mgr.list_active()
    archived = mission_mgr.list_archived()
    cur_id   = _state.get("mission_id", "")
    cur_name = _state.get("mission_name", "Default")

    # Mission dropdown items
    items = []
    for m in missions:
        items.append(
            Div(
                Span(cls=f"sdot {'active' if m.id == cur_id else ''}"),
                Span(m.name, cls="m-item-name"),
                Span(_fmt_mission_date(m.created_at), cls="m-item-date"),
                cls=f"m-item {'cur' if m.id == cur_id else ''}",
                hx_post=f"/missions/{m.id}/activate",
                hx_swap="none",
                **{"hx-on::after-request": "location.reload()"},
            )
        )
    items.append(Div(cls="m-sep"))
    items.append(
        Div(
            "＋  New mission",
            cls="m-new",
            hx_get="/missions/new/drawer",
            hx_target="#settings-drawer",
            hx_swap="innerHTML",
            onclick="openDrawer()",
        )
    )
    if archived:
        items.append(
            Div(f"▸  Show archived ({len(archived)})", cls="m-archived",
                hx_get="/missions/archived",
                hx_target="#m-dropdown",
                hx_swap="beforeend")
        )

    return Div(
        # ── Mission chip + dropdown ──────────────────────────────────────────
        Div(
            Div(
                Span(cls="sdot active"),
                Span(cur_name, cls="m-chip-name", id="mission-chip-name"),
                Span("▾", cls="m-chip-chev"),
                cls="m-chip",
                onclick="toggleMissionDropdown()",
            ),
            Div(*items, cls="m-dropdown", id="m-dropdown"),
            cls="m-chip-wrap",
        ),

        # ── Settings icon ────────────────────────────────────────────────────
        Button(
            I(cls="fas fa-cog"),
            cls="cfg-btn",
            title="Mission settings",
            hx_get=f"/missions/{cur_id}/drawer",
            hx_target="#settings-drawer",
            hx_swap="innerHTML",
            onclick="openDrawer()",
        ),

        Div(cls="sep"),
        Span("TAE", cls="brand"),
        Div(cls="sep"),

        # ── Upload ───────────────────────────────────────────────────────────
        Label(
            I(cls="fas fa-cloud-upload-alt", style="font-size:11px"),
            " Upload",
            Input(
                type="file", name="files", multiple=True,
                accept=".jpg,.jpeg,.png,.mp4,.mov,.avi,.mkv,.srt,.SRT",
                hx_post="/upload",
                hx_target="#tae-msgs",
                hx_swap="beforeend",
                hx_encoding="multipart/form-data",
                hx_indicator="#upload-ind",
                **{"hx-on::after-request": "scrollChat(); refreshMap();"},
            ),
            cls="upload-btn",
        ),
        Span(
            Span(cls="spinner"),
            Span("Processing…", cls="pulse-txt"),
            id="upload-ind",
            cls="htmx-indicator",
        ),

        Button(
            I(cls="fas fa-film", style="font-size:11px"),
            " Video",
            cls="video-btn",
            id="video-btn",
            title="Video playback panel",
            # HTMX loads panel content; onclick only toggles CSS classes
            hx_get="/video_panel",
            hx_target="#video-panel",
            hx_swap="innerHTML",
            **{"hx-on::after-request":
               "document.getElementById('video-panel').classList.add('open');"
               "document.getElementById('tae-imgpanel').classList.remove('open');"
               "document.getElementById('video-btn').classList.add('active');"},
        ),
        _status_badge(),
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
                Input(id="del-confirm-input", cls="del-confirm-input",
                      placeholder=name_val),
                Button("Permanently delete", cls="del-go",
                       hx_delete=f"/missions/{mid}",
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


def _chat_panel(collapsed: bool = False) -> FT:
    body_style = "" if not collapsed else "display:none"
    chev_cls   = "fas fa-chevron-up" if not collapsed else "fas fa-chevron-down"

    return Div(
        Div(
            Div(
                Span("🎯  Query Theater", cls="chat-head-label"),
                I(cls=f"{chev_cls} chat-head-chevron"),
                cls="chat-head",
                hx_get="/toggle_chat",
                hx_target="#tae-chat-panel",
                hx_swap="outerHTML",
            ),
            Div(
                _msg("TAE ready. Upload imagery to build the theater index, "
                     "then query in natural language.", "sys"),
                id="tae-msgs",
                cls="chat-msgs",
                style=body_style,
            ),
            Div(cls="chat-divider", style=body_style),
            Div(
                Form(
                    Input(placeholder='e.g. "Find a helipad marked H"',
                          name="message", id="tae-query-input", autocomplete="off"),
                    Button("→", type="submit", cls="send-btn"),
                    hx_post="/query",
                    hx_target="#tae-msgs",
                    hx_swap="beforeend",
                    hx_indicator="#query-ind",
                    **{"hx-on::after-request":
                       "document.getElementById('tae-query-input').value='';"
                       "scrollChat(); refreshMap();"},
                ),
                Span(Span(cls="spinner"), Span("Querying…", cls="pulse-txt"),
                     id="query-ind", cls="htmx-indicator",
                     style="padding:0 12px 8px"),
                cls="chat-input-row",
                style=body_style,
            ),
            cls="chat-box",
        ),
        id="tae-chat-panel",
        cls="chat-wrap",
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

@rt("/")
def index():
    _build_map()
    return (
        Title("TAE · Tactical Awareness Engine"),
        _navbar(),
        Div(
            Div(id="video-panel", cls="video-panel"),
            Div(Iframe(src="/map", id="tae-map-frame"), cls="map-wrap"),
            Div(*_image_panel_empty(), id="tae-imgpanel", cls="img-panel"),
            Div(id="settings-drawer", cls="settings-drawer"),
            cls="tae-main",
        ),
        _chat_panel(),
    )



# ─────────────────────────────────────────────────────────────────────────────
# Routes — Phase C: Video playback
# ─────────────────────────────────────────────────────────────────────────────

@rt("/video/{filename}")
async def serve_video(filename: str, request: Request):
    """Range-aware video file serving — required for browser seek support."""
    paths = _state.get("mission_paths")
    if paths is None:
        from starlette.responses import Response
        return Response("No active mission", status_code=404)
    video_file = paths.uploads / filename
    logger.info("serve_video: looking for %s", video_file)
    if not video_file.exists():
        logger.warning("serve_video: 404 — file not at %s", video_file)
        from starlette.responses import Response
        return Response("Video not found", status_code=404)
    suffix = video_file.suffix.lower()
    mime = {"mp4": "video/mp4", "mov": "video/quicktime",
            "avi": "video/x-msvideo", "mkv": "video/x-matroska"}.get(suffix[1:], "video/mp4")
    return FileResponse(str(video_file), media_type=mime,
                        headers={"Accept-Ranges": "bytes"})



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

    n_dets = len(_state.get("detections", {}))

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
                f' src="/video/{vfile}"'
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
        Div(
            Div(
                Span(f"{n_dets} detection(s)",
                     style="font-size:9px;color:var(--muted);letter-spacing:.1em;"
                           "text-transform:uppercase"),
                style="padding:8px 0 4px",
            ),
            Div("", id="video-det-list",
                style="font-size:10px;color:var(--muted)"),
            cls="video-detlist",
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
            "color":        det.get("color", "#4ade80"),
            "confirmed":    det.get("confirmed", False),
            "label":        det.get("label", "")[:40],
            "bbox":         det.get("bbox"),
            "tile_x":       det.get("tile_x", 0),
            "tile_y":       det.get("tile_y", 0),
            "tile_w":       det.get("tile_w", 640),
            "tile_h":       det.get("tile_h", 640),
        })
    return JSONResponse(result)

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


@rt("/missions/{mission_id}", methods=["DELETE"])
async def mission_delete(mission_id: str, request: Request):
    form         = await request.form()
    confirm_name = (form.get("del-confirm-input") or "").strip()
    m = mission_mgr.get(mission_id)
    if not m or confirm_name != m.name:
        return ""   # silently reject wrong confirmation
    # Wipe data directory
    data_root = Path(settings.DATA_DIR) / mission_id
    if data_root.exists():
        shutil.rmtree(str(data_root))
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
        lib   = get_search_lib()
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
                    result = _execute_analysis(
                        m.definition,
                        color,
                        search_limit     = 20,
                        frames_to_return = 8,
                    )
                    snip = m.definition[:40] + (
                        "..." if len(m.definition) > 40 else ""
                    )
                    auto_summary = (
                        f"Auto-analysis: {result['n_confirmed']} object(s) "
                        f"found for '{snip}'."
                        if result["n_confirmed"]
                        else f"Auto-analysis complete — no objects confirmed "
                             f"for '{snip}'."
                    )
                    logger.info(auto_summary)
        except Exception as ae:
            logger.error("Auto-analysis failed (ingestion was successful): %s", ae)
            auto_summary = "Auto-analysis failed — you can still query manually."

        _state["ingest_msg"] = (
            f"Indexed {tiles_ok} tiles across {total_frames} frame(s). "
            + (auto_summary if auto_summary else "Theater is ready — query below.")
        )
        _state["ingesting"]  = False   # signal poll to deliver the message
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
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

    from starlette.background import BackgroundTasks
    _state["ingesting"]   = True
    _state["ingest_msg"]  = None
    _state["frame_count"] = len(meta)
    tasks = BackgroundTasks()
    tasks.add_task(_ingest_background, saved_images, meta_file, meta)
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
        return _msg(msg, "sys"), _status_badge(), Script("refreshMap();")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Routes — query
# ─────────────────────────────────────────────────────────────────────────────


@rt("/query", methods=["POST"])
async def query(message: str):
    if not message.strip():
        return ""

    user_bubble = _msg(message, "user")

    if not _state["ingested"] and db.row_count() == 0:
        return user_bubble, _msg("⚠️  No imagery indexed yet. Upload images first.", "sys")

    _state["ingested"] = True

    color = _MARKER_COLORS[_state["query_color_idx"] % len(_MARKER_COLORS)]
    _state["query_color_idx"] += 1

    result = _execute_analysis(message, color, search_limit=20, frames_to_return=8)

    if not result["det_ids"]:
        return user_bubble, _msg("No matching frames found in the index.", "sys")

    raw_report = result["report"]
    dot_solid  = f'<span style="color:{color};font-size:13px">&#9679;</span>'
    dot_hollow = f'<span style="color:{color};font-size:13px">&#9675;</span>'

    lines = []
    if raw_report and result["n_confirmed"]:
        for p in raw_report.split(" | "):
            p = p.strip()
            if ": " in p:
                p = p.split(": ", 1)[1]
            if p:
                lines.append(f"{dot_solid} {p}")
    if result["n_unconfirmed"]:
        lines.append(
            f"{dot_hollow} {result['n_unconfirmed']} additional frame(s) matched "
            f"semantically but not confirmed by VLM — click hollow markers to inspect."
        )
    if not lines:
        lines = [raw_report or "No objects matching the query were found."]

    return user_bubble, _msg_html("<br>".join(lines))


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
            Div(Span("Detection View"),
                Span("×", cls="panel-close", onclick="closeImages()"),
                cls="panel-header"),
            Div(P("Detection not found.", style="color:var(--danger);padding:20px"),
                cls="empty-state"),
        )
    fv_widget = frame_view(det_id, mode="tile")
    meta_div  = Div(
        Div(det["label"][:60], cls="label"),
        f"LAT {det['lat']:.6f}  ·  LON {det['lon']:.6f}",
        Br(),
        f"GSD {det['gsd']} cm/px  ·  Source: {det['source']}",
        cls="det-meta", style="margin:0 12px 12px",
    )
    return (
        Div(Span(f"📍 {det['label'][:36]}"),
            Span("×", cls="panel-close", onclick="closeImages()"),
            cls="panel-header"),
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
        if stored and stored.startswith("/tile_img/"):
            key     = stored.split("/")[-1]
            img_url = stored if key in _tile_img_cache else None
        else:
            img_url = None
        if not img_url:
            img_url = _tile_to_static_url(det) or ""
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
# Wire injected dependencies into the analysis service layer
_init_analysis(db, analyst, get_search_lib)

_startup()
_PORT = int(os.environ.get("PORT", 8000))
logger.info(f"Starting TAE on port {_PORT}")
serve(host="0.0.0.0", port=_PORT)