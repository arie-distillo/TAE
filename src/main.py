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
_CLIP_AERIAL_CTX = "aerial drone nadir overhead view:"

_MARKER_COLORS = [
    "#4ade80", "#60a5fa", "#f59e0b", "#f472b6",
    "#a78bfa", "#34d399", "#fb923c", "#e879f9",
]

_frame_img_cache: dict[str, bytes] = {}
_tile_img_cache:  dict[str, bytes] = {}

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
    "show_coverage":   False,
    "ingest_progress": "",
    "ingesting":       False,
    "ingest_msg":      None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Mission activation
# ─────────────────────────────────────────────────────────────────────────────

def _activate_mission(mission: Mission) -> None:
    """Switch all server state to a different mission."""
    paths = MissionPaths.for_mission(settings.DATA_DIR, mission.id)
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
        "show_coverage":   False,
        "query_color_idx": 0,
        "ingesting":       False,
        "ingest_msg":      None,
        "ingest_progress": "",
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


# ─────────────────────────────────────────────────────────────────────────────
# Map helpers
# ─────────────────────────────────────────────────────────────────────────────
_MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoiYXJpZWdlbmtpbiIsImEiOiJjaWg2bWR1djIwMDBodmdtM2YxeXR5bzYzIn0.UvReMoVzH7szfW_1xszBhg",
)
_DARK_TILES = (
    "https://api.mapbox.com/styles/v1/mapbox/navigation-night-v1"
    f"/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={_MAPBOX_TOKEN}"
)


def _optimal_zoom(lat_span: float, lon_span: float) -> int:
    span = max(lat_span, lon_span)
    if span <= 0:
        return 17
    thresholds = [
        (0.002, 18), (0.005, 17), (0.01, 16), (0.02, 15),
        (0.05,  14), (0.1,   13), (0.2,  12), (0.5,  11),
        (1.0,   10), (2.0,    9), (5.0,   8),
    ]
    for deg, z in thresholds:
        if span <= deg:
            return z
    return 7


def _recenter_on_detections(det_ids: list[str]) -> None:
    lats = [_state["detections"][d]["lat"] for d in det_ids if d in _state["detections"]]
    lons = [_state["detections"][d]["lon"] for d in det_ids if d in _state["detections"]]
    if not lats:
        return
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    zoom = _optimal_zoom((max(lats) - min(lats)) * 1.3,
                         (max(lons) - min(lons)) * 1.3)
    _state["map_center"] = [center_lat, center_lon]
    _state["map_zoom"]   = zoom


def _build_map() -> None:
    """Regenerate map.html for the active mission."""
    paths = _state.get("mission_paths")
    if paths is None:
        return

    lat, lon = _state["map_center"]
    m = folium.Map(
        location=[lat, lon],
        zoom_start=_state["map_zoom"],
        tiles=_DARK_TILES,
        attr="Mapbox",
        control_scale=False,
        zoomControl=True,
    )

    for det_id, det in _state["detections"].items():
        color     = det.get("color", "#4ade80")
        confirmed = det.get("confirmed", True)
        dot_html = (
            f'<div style="width:18px;height:18px;'
            f'background:{"" if not confirmed else color};'
            f'border-radius:50%;border:2.5px solid {color};'
            f'box-shadow:0 0 {"8" if confirmed else "6"}px {color};'
            f'cursor:pointer"></div>'
        )
        icon = folium.DivIcon(html=dot_html, icon_size=(18, 18), icon_anchor=(9, 9))
        popup_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;'
            f'min-width:210px;padding:6px 2px">'
            f'<div style="color:{color};font-weight:700;font-size:13px;margin-bottom:5px">'
            f'{"●" if confirmed else "○"} {det["label"][:40]}</div>'
            f'<div style="font-size:11px;line-height:1.7;color:#334155">'
            f'LAT &nbsp;{det["lat"]:.6f}<br>'
            f'LON &nbsp;{det["lon"]:.6f}<br>'
            f'GSD &nbsp;{det.get("gsd","—")} cm/px</div>'
            f'<button onclick="window.parent.postMessage('
            f'{{type:\'show_images\',id:\'{det_id}\'}},'
            f'\'*\')" '
            f'style="margin-top:9px;padding:5px 14px;background:{color};color:#1a1b26;'
            f'border:none;border-radius:5px;cursor:pointer;font-weight:700;font-size:12px">'
            f'📸 View Images</button></div>'
        )
        folium.Marker(
            location=[det["lat"], det["lon"]],
            popup=folium.Popup(popup_html, max_width=240),
            icon=icon,
        ).add_to(m)

    if _state.get("show_coverage") and db.table is not None:
        try:
            df = db.table.to_pandas()[
                ["parent_path",
                 "fp_nw_lat", "fp_nw_lon", "fp_ne_lat", "fp_ne_lon",
                 "fp_se_lat", "fp_se_lon", "fp_sw_lat", "fp_sw_lon"]
            ].drop_duplicates(subset=["parent_path"])

            pts = []
            for _, r in df.iterrows():
                pts += [
                    (r.fp_nw_lat, r.fp_nw_lon), (r.fp_ne_lat, r.fp_ne_lon),
                    (r.fp_se_lat, r.fp_se_lon), (r.fp_sw_lat, r.fp_sw_lon),
                ]

            # Outlier filter: remove corners from geographically distant sessions
            # (e.g. still images from area A + video from area B in same LanceDB).
            # Median is robust — 90% of tiles in area B pulls median into area B.
            if len(pts) >= 6:
                import statistics as _stats
                med_lat = _stats.median(p[0] for p in pts)
                med_lon = _stats.median(p[1] for p in pts)
                MAX_DEG = 0.15   # ~16 km radius — generous for a single survey
                pts_clean = [
                    p for p in pts
                    if abs(p[0] - med_lat) < MAX_DEG
                    and abs(p[1] - med_lon) < MAX_DEG
                ]
                if len(pts_clean) >= 3:
                    removed = len(pts) - len(pts_clean)
                    if removed:
                        logger.info(
                            f"Hull outlier filter: dropped {removed} corners "
                            f"({len(pts_clean)} of {len(pts)} kept)"
                        )
                    pts = pts_clean

            def _convex_hull(points):
                pts_s = sorted(set(map(tuple, points)))
                if len(pts_s) < 3:
                    return pts_s
                def _cross(O, A, B):
                    return (A[0]-O[0])*(B[1]-O[1]) - (A[1]-O[1])*(B[0]-O[0])
                lower = []
                for p in pts_s:
                    while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
                        lower.pop()
                    lower.append(p)
                upper = []
                for p in reversed(pts_s):
                    while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
                        upper.pop()
                    upper.append(p)
                return lower[:-1] + upper[:-1]

            hull_pts = _convex_hull(pts)
            folium.Polygon(
                locations=hull_pts,
                color="#7aa2f7", weight=2,
                fill=True, fill_color="#7aa2f7", fill_opacity=0.15,
            ).add_to(m)
        except Exception as e:
            import traceback as _tb
            logger.error(f"Coverage polygon error: {e}\n{_tb.format_exc()}")

    map_file = paths.maps / "map.html"
    m.save(str(map_file))


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile(candidate: dict):
    parent = candidate.get("parent_path", "")
    if not parent or not Path(parent).exists():
        logger.error(f"_load_tile: parent frame not found: {parent}")
        return None
    img = cv2.imread(parent)
    if img is None:
        logger.error(f"_load_tile: cv2.imread returned None for: {parent}")
        return None
    x, y = candidate["tile_x"], candidate["tile_y"]
    w, h = candidate["tile_w"], candidate["tile_h"]
    return img[y:y+h, x:x+w]


def _annotate_and_save(candidate: dict, bbox_px: list | None, label: str) -> str | None:
    img = _load_tile(candidate)
    if img is None:
        return None
    if bbox_px and len(bbox_px) == 4:
        xmin, ymin, xmax, ymax = [int(v) for v in bbox_px]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (74, 222, 128), 4)
        cv2.putText(img, label[:24], (xmin, max(ymin - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (74, 222, 128), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    key = uuid.uuid4().hex[:16]
    _tile_img_cache[key] = buf.tobytes()
    return f"/tile_img/{key}"


def _tile_to_static_url(candidate: dict) -> str | None:
    return _annotate_and_save(candidate, None, "")


def _extract_video_frames(video_path: str, out_dir: Path, fps: float = 1.0) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(src_fps / fps))
    stem     = Path(video_path).stem
    paths, fi, saved = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % interval == 0:
            out = out_dir / f"{stem}_f{saved:05d}.jpg"
            cv2.imwrite(str(out), frame)
            paths.append(str(out))
            saved += 1
        fi += 1
    cap.release()
    logger.info(f"Extracted {saved} frames from {video_path}")
    return paths


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
_CSS = Style("""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg0: #0d0e1a;
  --bg1: #12131f;
  --bg2: #181929;
  --bg3: #1e1f32;
  --bg4: #252637;
  --border: #2a2c45;
  --accent: #4ade80;
  --accent-dim: rgba(74,222,128,.15);
  --blue: #7aa2f7;
  --blue-dim: rgba(122,162,247,.12);
  --text: #c0caf5;
  --muted: #6272a4;
  --danger: #f7768e;
  --danger-dim: rgba(247,118,142,.12);
  --font-mono: 'JetBrains Mono', monospace;
  --font-head: 'Rajdhani', sans-serif;
}

html, body { height:100%; background: var(--bg0); color: var(--text); font-family: var(--font-mono); overflow: hidden; }

/* ── Navbar ─────────────────────────────────────────────────────────────── */
.tae-nav {
  position: fixed; top:0; left:0; right:0; z-index:200;
  height: 52px;
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 14px;
  gap: 10px;
}

/* ── Mission chip ────────────────────────────────────────────────────────── */
.m-chip-wrap { position: relative; }

.m-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 5px 10px;
  font-size: 10px;
  cursor: pointer;
  min-width: 150px;
  max-width: 220px;
  user-select: none;
  transition: border-color .15s, background .15s;
}
.m-chip:hover, .m-chip.open { border-color: var(--blue); background: var(--blue-dim); }
.m-chip-name { font-weight: 600; color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.m-chip-chev { font-size: 8px; color: var(--muted); transition: transform .15s; flex-shrink: 0; }
.m-chip.open .m-chip-chev { transform: rotate(180deg); }

/* ── Mission dropdown ────────────────────────────────────────────────────── */
.m-dropdown {
  display: none;
  position: absolute;
  top: calc(100% + 5px);
  left: 0;
  width: 272px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 8px 28px rgba(0,0,0,.55);
  overflow: hidden;
  z-index: 300;
}
.m-dropdown.open { display: block; }

.m-item {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 9px 12px;
  font-size: 10px;
  cursor: pointer;
  transition: background .1s;
}
.m-item:hover { background: var(--bg3); }
.m-item.cur { background: var(--accent-dim); }
.m-item-name { flex: 1; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.m-item-date { color: var(--muted); font-size: 9px; white-space: nowrap; }

.m-sep { height: 1px; background: var(--border); margin: 2px 0; }

.m-new {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  font-size: 10px;
  color: var(--accent);
  cursor: pointer;
  font-weight: 600;
}
.m-new:hover { background: var(--accent-dim); }

.m-archived {
  padding: 7px 12px;
  font-size: 9px;
  color: var(--muted);
  cursor: pointer;
}
.m-archived:hover { color: var(--text); }

/* ── Settings (config) button ────────────────────────────────────────────── */
.cfg-btn {
  background: none;
  border: none;
  color: var(--muted);
  cursor: pointer;
  font-size: 15px;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 5px;
  transition: color .15s, background .15s;
  flex-shrink: 0;
}
.cfg-btn:hover, .cfg-btn.active { color: var(--text); background: var(--bg4); }

/* ── Brand ───────────────────────────────────────────────────────────────── */
.brand {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .12em;
  text-transform: uppercase;
  flex-shrink: 0;
}

.sep { width:1px; height:28px; background: var(--border); margin: 0 2px; flex-shrink:0; }

/* ── Upload button ───────────────────────────────────────────────────────── */
.upload-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg4);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 13px;
  cursor: pointer;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text);
  transition: border-color .18s, background .18s;
}
.upload-btn:hover { border-color: var(--blue); background: var(--blue-dim); }
.upload-btn input[type="file"] { display:none; }

/* ── Status badge ────────────────────────────────────────────────────────── */
.status-badge {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-left: auto;
  font-size: 11px;
  color: var(--muted);
}
.sdot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--danger);
  flex-shrink: 0;
}
.sdot.active {
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
}

/* ── Main layout ─────────────────────────────────────────────────────────── */
.tae-main {
  display: flex;
  height: calc(100vh - 52px);
  margin-top: 52px;
}

/* ── Map ─────────────────────────────────────────────────────────────────── */
.map-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
}
.map-wrap iframe {
  width: 100%; height: 100%;
  border: none; display: block;
}

/* ── Settings drawer ─────────────────────────────────────────────────────── */
.settings-drawer {
  width: 300px;
  min-width: 300px;
  background: var(--bg1);
  border-left: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow-y: auto;
  transition: width .22s ease;
}
.settings-drawer.open { display: flex; }
.settings-drawer::-webkit-scrollbar { width: 4px; }
.settings-drawer::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.drawer-header {
  position: sticky; top: 0; z-index: 1;
  background: var(--bg1);
  padding: 13px 15px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.drawer-title { font-size: 11px; font-weight: 600; color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.drawer-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 15px; line-height: 1; padding: 2px 4px; }
.drawer-close:hover { color: var(--text); }

.drawer-section { padding: 13px 15px; }
.drawer-label { font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 7px; }
.drawer-sep { height: 1px; background: var(--border); }

.drawer-input {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 10px;
  outline: none;
}
.drawer-input:focus { border-color: var(--blue); }

.drawer-textarea {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 7px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 10px;
  outline: none;
  resize: vertical;
  min-height: 72px;
  line-height: 1.6;
}
.drawer-textarea:focus { border-color: var(--blue); }
.drawer-hint { font-size: 9px; color: var(--muted); margin-top: 4px; line-height: 1.5; }

.intent-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  cursor: pointer;
  font-size: 10px;
  color: var(--text);
  border: none !important;
  box-shadow: none !important;
  background: none !important;
  padding: 0 !important;
}
.intent-row input[type="checkbox"] {
  accent-color: var(--accent);
  cursor: pointer;
  width: 15px !important;
  height: 15px !important;
  min-width: 15px;
  flex-shrink: 0;
  margin: 0;
  padding: 0;
  border: none;
  background: none;
  box-shadow: none;
  appearance: auto;
  -webkit-appearance: checkbox;
}

.drawer-save-btn {
  margin-top: 10px;
  background: var(--accent-dim);
  border: 1px solid var(--accent);
  border-radius: 5px;
  padding: 6px 12px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--accent);
  cursor: pointer;
  font-weight: 600;
}
.drawer-save-btn:hover { background: var(--accent); color: var(--bg0); }

.arch-btn {
  width: 100%;
  background: none;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 11px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  cursor: pointer;
  text-align: left;
  margin-bottom: 7px;
}
.arch-btn:hover { border-color: var(--text); color: var(--text); }

.del-btn {
  width: 100%;
  background: none;
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 11px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--danger);
  cursor: pointer;
  text-align: left;
}
.del-btn:hover { border-color: var(--danger); background: var(--danger-dim); }

.del-confirm { margin-top: 8px; display: none; }
.del-confirm.show { display: block; }
.del-hint { font-size: 9px; color: var(--muted); margin-bottom: 5px; }
.del-confirm-input {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--danger);
  border-radius: 5px;
  padding: 6px 9px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 9px;
  outline: none;
  margin-bottom: 6px;
}
.del-go {
  width: 100%;
  background: var(--danger-dim);
  border: 1px solid var(--danger);
  border-radius: 5px;
  padding: 6px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--danger);
  cursor: pointer;
  font-weight: 600;
}
.del-go:hover { background: var(--danger); color: var(--bg0); }

/* ── Image panel (right side) ────────────────────────────────────────────── */
.img-panel {
  width: 380px;
  min-width: 380px;
  background: var(--bg2);
  border-left: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow-y: auto;
}
.img-panel.open { display: flex; }
.img-panel::-webkit-scrollbar { width: 4px; }
.img-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.panel-header {
  position: sticky; top: 0; z-index: 1;
  background: var(--bg2);
  padding: 13px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--blue);
  letter-spacing: .06em;
  text-transform: uppercase;
}
.panel-close {
  cursor: pointer; font-size: 20px; line-height: 1;
  color: var(--muted); transition: color .15s;
}
.panel-close:hover { color: var(--text); }

.det-card {
  margin: 12px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.det-meta {
  padding: 8px 12px;
  font-size: 10px;
  color: var(--muted);
  background: var(--bg3);
  line-height: 1.7;
}
.det-meta .label {
  color: var(--accent); font-weight: 600; font-size: 11px;
  margin-bottom: 3px; text-transform: uppercase; letter-spacing: .05em;
}

/* ── Chat ────────────────────────────────────────────────────────────────── */
.chat-wrap {
  position: fixed;
  bottom: 20px; left: 20px;
  z-index: 100;
  width: 370px;
}
.chat-box {
  background: rgba(18,19,31,.97);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,.7);
  backdrop-filter: blur(14px);
}
.chat-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  cursor: pointer; user-select: none;
}
.chat-head-label {
  font-family: var(--font-head); font-size: 12px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--blue);
}
.chat-head-chevron { color: var(--muted); font-size: 11px; transition: transform .2s; }
.chat-msgs {
  height: 260px; overflow-y: auto; padding: 12px 14px;
}
.chat-msgs::-webkit-scrollbar { width: 4px; }
.chat-msgs::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.chat-divider { height: 1px; background: var(--border); }
.chat-input-row { padding: 10px 12px; }
.chat-input-row form { display: flex; gap: 8px; }
.chat-input-row input {
  flex: 1; background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; color: var(--text);
  font-family: var(--font-mono); font-size: 11px; outline: none;
}
.chat-input-row input:focus { border-color: var(--blue); }
.send-btn {
  background: var(--accent-dim); border: 1px solid var(--accent);
  border-radius: 6px; padding: 8px 14px; cursor: pointer;
  color: var(--accent); font-size: 14px; font-weight: 700;
}
.send-btn:hover { background: var(--accent); color: var(--bg0); }

.msg {
  display: flex; flex-direction: column; margin-bottom: 10px;
}
.msg.user { align-items: flex-end; }
.msg.sys  { align-items: flex-start; }
.msg-time { font-size: 9px; color: var(--muted); margin-bottom: 3px; }
.msg-bubble {
  max-width: 92%; padding: 8px 11px; border-radius: 8px;
  font-size: 11px; line-height: 1.6;
  background: var(--bg3); color: var(--text);
}
.msg.user .msg-bubble { background: var(--blue-dim); border: 1px solid var(--blue); }

/* ── Misc ────────────────────────────────────────────────────────────────── */
.htmx-indicator { display: none; }
.htmx-request .htmx-indicator { display: flex; align-items: center; gap: 7px; }
@keyframes spin  { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.5; } }
.spinner {
  width: 14px; height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
.pulse-txt { font-size: 11px; color: var(--accent); animation: pulse 1.5s ease-in-out infinite; }

.empty-state {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:10px; padding:30px; text-align:center;
}
.empty-txt { font-size:12px; color:var(--muted); line-height:1.7; }
""")


# ─────────────────────────────────────────────────────────────────────────────
# JavaScript
# ─────────────────────────────────────────────────────────────────────────────
_JS = Script("""
function scrollChat() {
    const el = document.getElementById('tae-msgs');
    if (el) el.scrollTop = el.scrollHeight;
}

function refreshMap() {
    const f = document.getElementById('tae-map-frame');
    if (f) f.src = '/map?' + Date.now();
}

// ── Mission dropdown toggle ───────────────────────────────────────────────
function toggleMissionDropdown() {
    const chip = document.querySelector('.m-chip');
    const dd   = document.getElementById('m-dropdown');
    const open = dd.classList.toggle('open');
    chip.classList.toggle('open', open);
}

document.addEventListener('click', function(e) {
    if (!e.target.closest('.m-chip-wrap')) {
        const dd   = document.getElementById('m-dropdown');
        const chip = document.querySelector('.m-chip');
        if (dd)   dd.classList.remove('open');
        if (chip) chip.classList.remove('open');
    }
});

// ── Settings drawer ───────────────────────────────────────────────────────
function openDrawer() {
    const dd   = document.getElementById('m-dropdown');
    const chip = document.querySelector('.m-chip');
    if (dd)   dd.classList.remove('open');
    if (chip) chip.classList.remove('open');
    document.getElementById('tae-imgpanel').classList.remove('open');
    document.getElementById('settings-drawer').classList.add('open');
    document.querySelector('.cfg-btn').classList.add('active');
}

function closeDrawer() {
    document.getElementById('settings-drawer').classList.remove('open');
    document.querySelector('.cfg-btn').classList.remove('active');
}

function toggleDeleteConfirm() {
    document.getElementById('del-confirm-box').classList.toggle('show');
}

// ── Map / image panel messages ────────────────────────────────────────────
window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'show_images') return;
    closeDrawer();
    htmx.ajax('GET', '/images/' + e.data.id, {
        target: '#tae-imgpanel', swap: 'innerHTML'
    });
    document.getElementById('tae-imgpanel').classList.add('open');
});

function closeImages() {
    document.getElementById('tae-imgpanel').classList.remove('open');
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        htmx.ajax('GET', '/toggle_coverage', {
            target: '#tae-msgs', swap: 'beforeend'
        });
        setTimeout(() => { scrollChat(); refreshMap(); }, 400);
    }
    if (e.key === 'Escape') { closeDrawer(); }
});
""")


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
            Div(Iframe(src="/map", id="tae-map-frame"), cls="map-wrap"),
            Div(*_image_panel_empty(), id="tae-imgpanel", cls="img-panel"),
            Div(id="settings-drawer", cls="settings-drawer"),
            cls="tae-main",
        ),
        _chat_panel(),
    )


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
    paths = MissionPaths.for_mission(settings.DATA_DIR, mid)
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
            _paths = MissionPaths.for_mission(settings.DATA_DIR, _mid)
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
        _state["ingesting"]   = False
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

        for jpeg_path, srt_frame in frame_pairs:
            saved_images.append(str(jpeg_path))
            if srt_frame is not None:
                # Peek at image dimensions for the meta entry
                img = cv2.imread(str(jpeg_path))
                if img is not None:
                    h, w = img.shape[:2]
                    meta[jpeg_path.name] = _srt_parser.to_meta_entry(
                        srt_frame, jpeg_path, w, h
                    )

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


def _execute_analysis(
    message:         str,
    color:           str,
    search_limit:    int = 20,
    frames_to_return: int = 8,
) -> dict:
    """
    Single shared analysis pipeline used by both the interactive query route
    and the post-ingestion auto-analysis hook.

    Encodes the query with aerial CLIP context, retrieves the best-matching
    tiles (one per unique parent frame), runs the VLM, builds detections in
    _state, recentres the map, and returns a summary dict.

    Returns
    -------
    {
      "det_ids":       list[str]   — new detection IDs added this call
      "n_confirmed":   int         — VLM-confirmed detections
      "n_unconfirmed": int         — CLIP candidates not confirmed by VLM
      "report":        str         — raw VLM report string
      "color":         str         — marker colour used
    }
    """
    lib        = get_search_lib()
    clip_query = f"{_CLIP_AERIAL_CTX} {message}"
    q_vec      = lib.encode_text(clip_query)
    logger.info("CLIP query: '%s'", clip_query)

    candidates = db.semantic_search(
        q_vec,
        limit            = search_limit,
        frames_to_return = frames_to_return,
    )
    if not candidates:
        return {"det_ids": [], "n_confirmed": 0, "n_unconfirmed": 0,
                "report": "", "color": color}

    logger.info("%d candidate tile(s) → VLM", len(candidates))
    intel   = analyst.analyze_multiple_views(candidates, message)
    targets = intel.get("targets", [])
    new_det_ids: list[str] = []

    for cand in candidates:
        tile_name     = Path(cand["image_path"]).name
        parent_name   = Path(cand["parent_path"]).name
        frame_targets = [
            t for t in targets
            if Path(t["filename"]).name.lower() == tile_name.lower()
        ]
        if frame_targets:
            for t in frame_targets:
                det_id  = uuid.uuid4().hex[:10]
                ann_url = _annotate_and_save(cand, t.get("bbox"), message[:20])
                _state["detections"][det_id] = {
                    "lat":         cand["lat"],
                    "lon":         cand["lon"],
                    "label":       message,
                    "color":       color,
                    "confirmed":   True,
                    "img_urls":    [ann_url] if ann_url else [_tile_to_static_url(cand)],
                    "gsd":         f"{cand.get('gsd_cm_px', 0):.1f}",
                    "bbox":        t.get("bbox"),
                    "source":      parent_name,
                    "parent_path": cand["parent_path"],
                    "tile_x":      cand["tile_x"],
                    "tile_y":      cand["tile_y"],
                    "tile_w":      cand["tile_w"],
                    "tile_h":      cand["tile_h"],
                }
                new_det_ids.append(det_id)
        else:
            det_id  = uuid.uuid4().hex[:10]
            img_url = _tile_to_static_url(cand)
            _state["detections"][det_id] = {
                "lat":         cand["lat"],
                "lon":         cand["lon"],
                "label":       f"Candidate: {message[:40]}",
                "color":       color,
                "confirmed":   False,
                "img_urls":    [img_url] if img_url else [],
                "gsd":         f"{cand.get('gsd_cm_px', 0):.1f}",
                "bbox":        None,
                "source":      parent_name,
                "parent_path": cand["parent_path"],
                "tile_x":      cand["tile_x"],
                "tile_y":      cand["tile_y"],
                "tile_w":      cand["tile_w"],
                "tile_h":      cand["tile_h"],
            }
            new_det_ids.append(det_id)

    _recenter_on_detections(new_det_ids)
    _build_map()

    n_confirmed   = sum(1 for d in _state["detections"].values()
                        if d.get("confirmed") and d.get("color") == color)
    n_unconfirmed = sum(1 for d in _state["detections"].values()
                        if not d.get("confirmed") and d.get("color") == color)
    return {
        "det_ids":       new_det_ids,
        "n_confirmed":   n_confirmed,
        "n_unconfirmed": n_unconfirmed,
        "report":        intel.get("report", ""),
        "color":         color,
    }

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
_startup()
_PORT = int(os.environ.get("PORT", 8000))
logger.info(f"Starting TAE on port {_PORT}")
serve(host="0.0.0.0", port=_PORT)