"""
tae_app.py – TAE Tactical Awareness Engine · FastHTML Web Interface
Run from the src/ directory:  python tae_app.py
"""

import logging
import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

# tools/ is a sibling of src/, so add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import folium
from fasthtml.common import *
from monsterui.all import *
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

from config import settings
from core.spatial import SpatialEngine
from core.database import TacticalDatabase
from ai.search import SearchLibrarian
from ai.analyst import TacticalAnalyst
from ai.intent import IntentClassifier, ObjectSearchParams, AnomalyDetectionParams, MovingObjectParams
from tools.ingest_telemetry import TAESimGenerator
from core.geo import tile_center_geo
from core.object_detection import merge_detections

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
# force=True ensures this wins over uvicorn's own logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
# Re-set uvicorn loggers to INFO so they don't suppress ours
for _uv in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uv).setLevel(logging.INFO)

logger = logging.getLogger("TAE-UI")

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
# All persistent paths come from settings (backed by Railway volume via DATA_DIR).
UPLOAD_PATH  = Path(settings.UPLOAD_PATH)      # original frames — permanent
DETECTIONS_PATH     = Path(settings.DETECTIONS_PATH)  # annotated detection images
MAP_PATH     = Path(settings.MAP_PATH)         # generated map.html



for _d in [UPLOAD_PATH, DETECTIONS_PATH, MAP_PATH]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Global app state
# ─────────────────────────────────────────────────────────────────────────────
# Colors cycle per query — chat icon and map marker always match
_MARKER_COLORS = [
    "#4ade80",  # green
    "#60a5fa",  # blue
    "#f59e0b",  # amber
    "#f472b6",  # pink
    "#a78bfa",  # purple
    "#34d399",  # emerald
    "#fb923c",  # orange
    "#e879f9",  # fuchsia
]

# ── CLIP aerial domain context ────────────────────────────────────────────────
# CLIP was trained on ground-level internet photos. UAV nadir imagery is
# underrepresented, so text embeddings for queries like "white car" skew toward
# street-level perspectives. Prepending this context string to every CLIP query
# shifts the text embedding toward overhead / drone imagery representations.
# The VLM prompt is unaffected — it already has its own UAV system context.
_CLIP_AERIAL_CTX = "aerial drone nadir overhead view:"

# Cache for annotated full-frame images served by /frame_img/{det_id}
_frame_img_cache: dict[str, bytes] = {}
_tile_img_cache:  dict[str, bytes] = {}  # hex key → annotated tile JPEG

_state: dict = {
    "map_center":      [32.08, 34.78],
    "map_zoom":        14,
    "detections":      {},
    "ingested":        False,
    "frame_count":     0,
    "tile_count":      0,
    "query_color_idx": 0,
    "show_coverage":   False,
    "ingest_progress": "",    # e.g. "frame 3 of 50"
}

# ─────────────────────────────────────────────────────────────────────────────
# TAE components  (lazy for CLIP)
# ─────────────────────────────────────────────────────────────────────────────
spatial  = SpatialEngine(
    settings.SENSOR_WIDTH_MM,
    settings.SENSOR_HEIGHT_MM,
    settings.FOCAL_LENGTH_MM,
)
db       = TacticalDatabase()
analyst  = TacticalAnalyst(
    settings.AI_PROVIDER,
    settings.VLM_MODEL,
    settings.OPENROUTER_API_KEY,
)
intent_clf = IntentClassifier(
    api_key    = settings.OPENROUTER_API_KEY,
    model_name = getattr(settings, "INTENT_MODEL", None),  # optional override
)
_search_lib: SearchLibrarian | None = None


def get_search_lib() -> SearchLibrarian:
    global _search_lib
    if _search_lib is None:
        logger.info("Loading CLIP model – this may take a moment…")
        _search_lib = SearchLibrarian(settings.CLIP_MODEL)
    return _search_lib


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
    """
    Estimate a Leaflet zoom level that fits a lat/lon bounding box.
    Uses the larger of the two spans mapped against rough degree-per-tile widths.
    """
    import math
    span = max(lat_span, lon_span)
    if span <= 0:
        return 17
    # degrees visible at each zoom at ~800px wide viewport
    # zoom: degrees
    thresholds = [
        (0.002,  18), (0.005,  17), (0.01,   16), (0.02,   15),
        (0.05,   14), (0.1,    13), (0.2,    12), (0.5,    11),
        (1.0,    10), (2.0,     9), (5.0,     8),
    ]
    for deg, z in thresholds:
        if span <= deg:
            return z
    return 7


def _recenter_on_detections(det_ids: list[str]) -> None:
    """
    Recompute map center and zoom from a specific set of detection ids.
    Uses weighted centroid (equal weight per marker).
    """
    lats = [_state["detections"][d]["lat"] for d in det_ids if d in _state["detections"]]
    lons = [_state["detections"][d]["lon"] for d in det_ids if d in _state["detections"]]
    if not lats:
        return

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)

    # Add 30% padding around the extent
    zoom = _optimal_zoom(lat_span * 1.3, lon_span * 1.3)

    _state["map_center"] = [center_lat, center_lon]
    _state["map_zoom"]   = zoom
    logger.info(
        f"Map recentered | center ({center_lat:.6f}, {center_lon:.6f}) | "
        f"zoom {zoom} | span ({lat_span:.5f}° lat, {lon_span:.5f}° lon)"
    )


def _build_map() -> None:
    """Regenerate MAP_PATH/map.html from current state."""
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
        if confirmed:
            dot_html = (
                f'<div style="width:18px;height:18px;background:{color};'
                f'border-radius:50%;border:2.5px solid #fff;'
                f'box-shadow:0 0 8px {color};cursor:pointer"></div>'
            )
        else:
            dot_html = (
                f'<div style="width:18px;height:18px;background:transparent;'
                f'border-radius:50%;border:2.5px solid {color};'
                f'box-shadow:0 0 6px {color};cursor:pointer"></div>'
            )
        icon = folium.DivIcon(
            html=dot_html,
            icon_size=(18, 18),
            icon_anchor=(9, 9),
        )
        popup_html = (
            f'<div style="font-family:\'JetBrains Mono\',monospace;color:#1a1b26;'
            f'min-width:210px;padding:6px 2px">'
            f'<div style="color:{color};font-weight:700;font-size:13px;margin-bottom:5px">'
            f'● {det["label"][:40]}</div>'
            f'<div style="font-size:11px;line-height:1.7;color:#334155">'
            f'LAT &nbsp;{det["lat"]:.6f}<br>'
            f'LON &nbsp;{det["lon"]:.6f}<br>'
            f'GSD &nbsp;{det.get("gsd", "—")} cm/px</div>'
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

    # ── Coverage polygon (Ctrl+P toggle) ─────────────────────────────────────
    if _state.get("show_coverage") and db.table is not None:
        try:
            # Hull fix (v2): the previous per-frame top-left/bottom-right approach
            # only contributed the NW corner of tile[0] and SE corner of tile[-1],
            # leaving the frame's true NE and SW corners unrepresented — so markers
            # for edge tiles could still fall outside the hull.
            # Correct fix: include ALL tile footprint corners.  With 2400 tiles ×
            # 4 corners = 9600 points the convex hull is still computed in <50 ms.
            df_all = db.table.to_pandas()[
                ["parent_path",
                 "fp_nw_lat","fp_nw_lon","fp_ne_lat","fp_ne_lon",
                 "fp_se_lat","fp_se_lon","fp_sw_lat","fp_sw_lon"]
            ]
            n_frames = df_all["parent_path"].nunique()
            logger.info(f"Coverage: {n_frames} unique frames | {len(df_all)} tiles → building hull")

            # Collect every tile's 4 corners
            pts = []
            for _, r in df_all.iterrows():
                pts += [
                    (r.fp_nw_lat, r.fp_nw_lon), (r.fp_ne_lat, r.fp_ne_lon),
                    (r.fp_se_lat, r.fp_se_lon),  (r.fp_sw_lat, r.fp_sw_lon),
                ]

            # Pure-numpy convex hull (Andrew's monotone chain — no scipy needed)
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
            logger.info(f"Coverage hull: {len(hull_pts)} vertices from {len(pts)} corner points")

            folium.Polygon(
                locations=hull_pts,   # (lat, lon) tuples
                color="#7aa2f7",
                weight=2,
                fill=True,
                fill_color="#7aa2f7",
                fill_opacity=0.15,
            ).add_to(m)

        except Exception as e:
            import traceback as _tb
            logger.error(f"Coverage polygon error: {e}\n{_tb.format_exc()}")

    map_file = MAP_PATH / "map.html"
    m.save(str(map_file))


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile(candidate: dict):
    """
    Reconstructs a tile by cropping its parent frame.
    Tiles are never stored on disk — this mirrors main.py's _load_tile().
    """
    parent = candidate.get("parent_path", "")
    logger.info(f"_load_tile: reading parent frame: {parent}")
    if not parent or not Path(parent).exists():
        logger.error(f"_load_tile: parent frame not found on disk: {parent}")
        return None
    img = cv2.imread(parent)
    if img is None:
        logger.error(f"_load_tile: cv2.imread returned None for: {parent}")
        return None
    x, y = candidate["tile_x"], candidate["tile_y"]
    w, h = candidate["tile_w"], candidate["tile_h"]
    return img[y:y+h, x:x+w]


def _annotate_and_save(candidate: dict, bbox_px: list | None, label: str) -> str | None:
    """Crop tile, draw bbox, cache bytes in memory, return /tile_img/<key> URL."""
    img = _load_tile(candidate)
    if img is None:
        return None

    if bbox_px and len(bbox_px) == 4:
        xmin, ymin, xmax, ymax = [int(v) for v in bbox_px]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (74, 222, 128), 4)
        cv2.putText(
            img, label[:24], (xmin, max(ymin - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (74, 222, 128), 2,
        )

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        logger.error("cv2.imencode failed for tile")
        return None
    key = uuid.uuid4().hex[:16]
    _tile_img_cache[key] = buf.tobytes()
    return f"/tile_img/{key}"

def _tile_to_static_url(candidate: dict) -> str | None:
    """Crop tile (no bbox), cache, return URL."""
    return _annotate_and_save(candidate, None, "")


def _extract_video_frames(video_path: str, out_dir: Path, fps: float = 1.0) -> list[str]:
    """Extract frames from a video at `fps` frames-per-second."""
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
# Chat message components
# ─────────────────────────────────────────────────────────────────────────────

def _msg(content: str, role: str = "sys") -> FT:
    ts = datetime.now().strftime("%H:%M")
    return Div(
        Span(ts, cls="msg-time"),
        Div(content, cls="msg-bubble"),
        cls=f"msg {role}",
    )

def _msg_html(html_content: str, role: str = "sys") -> FT:
    """Like _msg but renders raw HTML inside the bubble (for colored icons etc)."""
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
  --blue-dim: rgba(122,162,247,.15);
  --text: #c0caf5;
  --muted: #6272a4;
  --danger: #f7768e;
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
  padding: 0 18px;
  gap: 14px;
}

.brand {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .12em;
  text-transform: uppercase;
}

.brand-sub {
  font-size: 9px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: .15em;
  text-transform: uppercase;
  align-self: flex-end;
  margin-bottom: 3px;
}

.sep { width:1px; height:28px; background: var(--border); margin: 0 6px; }

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

/* ── Image panel ─────────────────────────────────────────────────────────── */
.img-panel {
  width: 380px;
  min-width: 380px;
  background: var(--bg2);
  border-left: 1px solid var(--border);
  display: none;
  flex-direction: column;
  overflow-y: auto;
  transition: width .25s ease;
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
  cursor: pointer;
  font-size: 20px;
  line-height: 1;
  color: var(--muted);
  transition: color .15s;
}
.panel-close:hover { color: var(--text); }

.det-card {
  margin: 12px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
.det-img { width: 100%; display: block; }
.det-meta {
  padding: 8px 12px;
  font-size: 10px;
  color: var(--muted);
  background: var(--bg3);
  line-height: 1.7;
}
.det-meta .label {
  color: var(--accent);
  font-weight: 600;
  font-size: 11px;
  margin-bottom: 3px;
  text-transform: uppercase;
  letter-spacing: .05em;
}

/* ── Chat ────────────────────────────────────────────────────────────────── */
.chat-wrap {
  position: fixed;
  bottom: 20px;
  left: 20px;
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
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  user-select: none;
}
.chat-head-label {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--blue);
}
.chat-head-chevron { color: var(--muted); font-size: 11px; transition: transform .2s; }

.chat-msgs {
  height: 260px;
  overflow-y: auto;
  padding: 12px 14px;
}
.chat-msgs::-webkit-scrollbar { width: 3px; }
.chat-msgs::-webkit-scrollbar-thumb { background: var(--border); border-radius:2px; }

.msg { margin-bottom: 10px; }
.msg-time { font-size: 9px; color: var(--muted); margin-bottom: 3px; }
.msg-bubble {
  display: inline-block;
  padding: 7px 11px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.6;
  max-width: 92%;
}
.msg.user { text-align: right; }
.msg.user .msg-time { text-align: right; }
.msg.user .msg-bubble { background: var(--bg4); color: var(--text); text-align:left; }
.msg.sys  .msg-bubble { background: var(--bg3); color: #a9b1d6; }

.chat-divider { height:1px; background: var(--border); }

.chat-input-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
}
.chat-input-row input {
  flex: 1;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 7px 11px;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text);
  outline: none;
  transition: border-color .15s;
}
.chat-input-row input::placeholder { color: var(--muted); }
.chat-input-row input:focus { border-color: var(--blue); }

.send-btn {
  background: var(--blue);
  color: var(--bg0);
  border: none;
  border-radius: 7px;
  padding: 7px 14px;
  cursor: pointer;
  font-weight: 700;
  font-size: 13px;
  font-family: var(--font-mono);
  transition: background .15s;
}
.send-btn:hover { background: #89b4fa; }

/* ── Upload progress indicator ───────────────────────────────────────────── */
.htmx-indicator { display: none; }
.htmx-request ~ .htmx-indicator,
.htmx-request.htmx-indicator { display: inline-flex; align-items:center; gap:6px; }

@keyframes spin  { to { transform: rotate(360deg); } }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

.spinner {
  width: 12px; height: 12px;
  border: 2px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink:0;
}
.pulse-txt { font-size: 11px; color: var(--accent); animation: pulse 1.5s ease-in-out infinite; }

/* ── Empty state panel ───────────────────────────────────────────────────── */
.empty-state {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:10px; padding:30px; text-align:center;
}
.empty-icon { font-size:32px; opacity:.4; }
.empty-txt  { font-size:12px; color:var(--muted); line-height:1.7; }
""")

# ─────────────────────────────────────────────────────────────────────────────
# FastHTML app
# ─────────────────────────────────────────────────────────────────────────────
_JS = Script("""
    function scrollChat() {
        const el = document.getElementById('tae-msgs');
        if (el) el.scrollTop = el.scrollHeight;
    }

    // Marker click → open image panel
    // Ctrl+P → toggle coverage polygon
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'p') {
            e.preventDefault();
            htmx.ajax('GET', '/toggle_coverage', {
                target: '#tae-msgs', swap: 'beforeend'
            });
            // htmx.ajax is not a Promise in v1.x — use setTimeout to refresh after route completes
            setTimeout(() => { scrollChat(); refreshMap(); }, 400);
        }
    });

    window.addEventListener('message', function(e) {
        if (!e.data || e.data.type !== 'show_images') return;
        htmx.ajax('GET', '/images/' + e.data.id, {
            target: '#tae-imgpanel', swap: 'innerHTML'
        });
        document.getElementById('tae-imgpanel').classList.add('open');
    });

    function closeImages() {
        document.getElementById('tae-imgpanel').classList.remove('open');
    }

    function refreshMap() {
        const f = document.getElementById('tae-map-frame');
        if (f) f.src = '/map?' + Date.now();
    }
""")

_MAX_UPLOAD_MB = 500

app, rt = fast_app(
    max_upload_size=_MAX_UPLOAD_MB * 1024 * 1024,
    hdrs=(
        Theme.slate.headers(),
        Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        ),
        _CSS,
        _JS,
    ),
    pico=False,
    bodykw={"style": "margin:0"},
)

# Mount static directories
# Also serve detections and map directly from their persistent paths
# Images and map served via FileResponse routes — see /detections/<fname> and /map below


# ─────────────────────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────────────────────

def _navbar() -> FT:
    return Div(
        # Brand
        Span("TAE", cls="brand"),
        Span("Intelligence", cls="brand-sub"),
        Div(cls="sep"),

        # Upload button (images + video)
        Label(
            I(cls="fas fa-cloud-upload-alt", style="font-size:11px"),
            " Upload",
            Input(
                type="file",
                name="files",
                multiple=True,
                accept=".jpg,.jpeg,.png,.mp4,.mov,.avi,.mkv",
                hx_post="/upload",
                hx_target="#tae-msgs",
                hx_swap="beforeend",
                hx_encoding="multipart/form-data",
                hx_indicator="#upload-ind",
                **{"hx-on::after-request": "scrollChat(); refreshMap();"},
            ),
            cls="upload-btn",
        ),

        # Upload spinner
        Span(
            Span(cls="spinner"),
            Span("Processing…", cls="pulse-txt"),
            id="upload-ind",
            cls="htmx-indicator",
        ),

        # Status badge
        _status_badge(),
        cls="tae-nav",
    )


def _chat_panel(collapsed: bool = False) -> FT:
    body_style = "" if not collapsed else "display:none"
    chev_cls   = "fas fa-chevron-up" if not collapsed else "fas fa-chevron-down"

    return Div(
        Div(
            # Header (toggle)
            Div(
                Span("🎯  Query Theater", cls="chat-head-label"),
                I(cls=f"{chev_cls} chat-head-chevron"),
                cls="chat-head",
                hx_get="/toggle_chat",
                hx_target="#tae-chat-panel",
                hx_swap="outerHTML",
            ),

            # Messages
            Div(
                _msg("TAE ready. Upload imagery to build the theater index, "
                     "then query in natural language.", "sys"),
                id="tae-msgs",
                cls="chat-msgs",
                style=body_style,
            ),

            Div(cls="chat-divider", style=body_style),

            # Input
            Div(
                Form(
                    Input(
                        placeholder='e.g. "Find a helipad marked H"',
                        name="message",
                        id="tae-query-input",
                        autocomplete="off",
                    ),
                    Button("→", type="submit", cls="send-btn"),
                    hx_post="/query",
                    hx_target="#tae-msgs",
                    hx_swap="beforeend",
                    hx_indicator="#query-ind",
                    **{
                        "hx-on::after-request": (
                            "document.getElementById('tae-query-input').value='';"
                            "scrollChat();"
                            "refreshMap();"
                        )
                    },
                ),
                Span(
                    Span(cls="spinner"),
                    Span("Querying…", cls="pulse-txt"),
                    id="query-ind",
                    cls="htmx-indicator",
                    style="padding:0 12px 8px",
                ),
                cls="chat-input-row",
                style=body_style,
            ),
            cls="chat-box",
        ),
        id="tae-chat-panel",
        cls="chat-wrap",
    )


def _image_panel_empty() -> FT:
    """Empty state for the image panel (before any marker is clicked)."""
    return (
        Div(
            Span("Detection View"),
            Span("×", cls="panel-close", onclick="closeImages()"),
            cls="panel-header",
        ),
        Div(
            Div(cls="empty-icon", style="font-size:36px"),
            P("Click a map marker to inspect detected objects.",
              cls="empty-txt"),
            cls="empty-state",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@rt("/")
def index():
    _build_map()
    return (
        Title("TAE · Tactical Awareness Engine"),
        _navbar(),
        Div(
            # Map iframe
            Div(
                Iframe(src="/map", id="tae-map-frame"),
                cls="map-wrap",
            ),
            # Image panel (right side)
            Div(
                *_image_panel_empty(),
                id="tae-imgpanel",
                cls="img-panel",
            ),
            cls="tae-main",
        ),
        # Chat widget (floating, bottom-left)
        _chat_panel(),
    )


@rt("/toggle_chat")
def toggle_chat():
    # Read collapse state from query params; default toggle on each call
    # We store it simply by alternating; for robustness use a session cookie
    # For now just always return expanded panel
    return _chat_panel(collapsed=False)


def _ingest_background(saved_images: list[str], meta_file: Path, meta: dict):
    from core.sim_provider import SimD3Environment
    from core.ingestion import run_ingestion
    try:
        db.initialize_table(vector_dim=getattr(settings, "CLIP_DIM", 512))
        sim      = SimD3Environment(str(meta_file))
        lib      = get_search_lib()
        total    = len(sim.frame_names)
        logger.info("Starting CLIP ingestion…")

        def _on_frame(idx: int, name: str):
            _state["ingest_progress"] = f"frame {idx} of {total} — {name}"

        tiles_ok, frames_failed = run_ingestion(sim, spatial, lib, db,
                                                on_frame=_on_frame)
        logger.info(f"Ingestion done — {tiles_ok} tiles, {frames_failed} frame(s) failed.")
# Count unique frames from DB — correct even with incremental uploads
        try:
            df = db.table.to_pandas()
            total_frames = df["parent_path"].nunique()
            # Re-center map from all known frame GPS positions
            lats = df.groupby("parent_path")["lat"].first().tolist()
            lons = df.groupby("parent_path")["lon"].first().tolist()
            if lats and lons:
                _state["map_center"] = [sum(lats)/len(lats), sum(lons)/len(lons)]
                _state["map_zoom"]   = 16
        except Exception:
            total_frames = _state["frame_count"]
        _state["frame_count"] = total_frames
        _state["ingested"]    = True
        _state["tile_count"]  = tiles_ok
        _state["detections"]  = {}
        _state["ingesting"]   = False
        _state["ingest_msg"]  = (
            f"Indexed {tiles_ok} tiles across {total_frames} frame(s). "
            f"Theater is ready — query below."
        )
        _build_map()
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
        _state["ingesting"]  = False
        _state["ingest_msg"] = f"Ingestion failed: {e}"


@rt("/upload", methods=["POST"])
async def upload(request: Request):
    from starlette.requests import ClientDisconnect
    try:
        form  = await request.form()
        files = form.getlist("files")
    except ClientDisconnect:
        logger.warning("Client disconnected during upload — connection dropped or request too large.")
        return _msg(
            "⚠️  Upload failed: connection dropped mid-transfer. "
            "Try uploading fewer images at once (10 max recommended), "
            "or check your Railway HTTP timeout setting.",
            "sys"
        )

    if not files or all(not f.filename for f in files):
        return _msg("⚠️  No files received.", "sys")


    saved_images: list[str] = []
    video_count = 0

    for f in files:
        if not f.filename:
            continue
        fname  = Path(f.filename).name
        suffix = Path(fname).suffix.lower()
        dest   = UPLOAD_PATH / fname
        dest.write_bytes(await f.read())

        if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            frames = _extract_video_frames(str(dest), UPLOAD_PATH, fps=1.0)
            saved_images.extend(frames)
            video_count += 1
        elif suffix in {".jpg", ".jpeg", ".png"}:
            saved_images.append(str(dest))

    if not saved_images:
        return _msg("⚠️  No processable images found in upload.", "sys")

    # ── Metadata extraction ──────────────────────────────────────────────────
    logger.info(f"Extracting XMP/EXIF metadata from {len(saved_images)} image(s)…")
    gen  = TAESimGenerator(str(UPLOAD_PATH), str(UPLOAD_PATH))
    meta = {}
    for img_path in saved_images:
        try:
            meta[Path(img_path).name] = gen._extract_dji_data(Path(img_path))
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {img_path}: {e}")

    if not meta:
        return _msg("⚠️  No parseable metadata in uploaded images.", "sys")

    meta_file = UPLOAD_PATH / "pose_metadata.json"
    meta_file.write_text(json.dumps(meta))

    # ── Update map center ────────────────────────────────────────────────────
    lats = [v["lat"] for v in meta.values() if v.get("lat") not in (None, 0.0)]
    lons = [v["lon"] for v in meta.values() if v.get("lon") not in (None, 0.0)]
    if lats and lons:
        _state["map_center"] = [sum(lats) / len(lats), sum(lons) / len(lons)]
        _state["map_zoom"]   = 16

    # ── CLIP ingestion in background ─────────────────────────────────────────
    from starlette.background import BackgroundTasks
    _state["ingesting"]   = True
    _state["ingest_msg"]  = None
    _state["frame_count"] = len(meta)
    tasks = BackgroundTasks()
    tasks.add_task(_ingest_background, saved_images, meta_file, meta)
    return (
        _msg(
            f"{len(saved_images)} image(s) received. Indexing in background…",
            "sys"
        ),
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
            label = f"Indexing {progress}..." if progress else "Indexing..."
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

@rt("/query", methods=["POST"])
async def query(message: str):
    if not message.strip():
        return ""

    user_bubble = _msg(message, "user")

    if not _state["ingested"] and db.row_count() == 0:
        return (
            user_bubble,
            _msg("⚠️  No imagery indexed yet. Upload images first.", "sys"),
        )
    _state["ingested"] = True  # sync flag if restored from DB

    lib = get_search_lib()

    # ── Intent classification ─────────────────────────────────────────────────
    # Classify first — intent determines the entire downstream flow.
    # Non-object-search intents short-circuit here before any CLIP/VLM work.
    classified = intent_clf.classify(message)
    intent     = classified.params.intent
    logger.info(
        f"Intent: {intent} | conf={classified.confidence:.2f} | {classified.reasoning}"
    )

    if intent == "anomaly_detection":
        p = classified.params  # AnomalyDetectionParams
        return (
            user_bubble,
            _msg(
                f"⚠️ Anomaly detection is not yet implemented.<br>"
                f"Classified: scene <i>'{p.scene_context}'</i>, "
                f"looking for <i>'{p.anomaly_hint}'</i>. "
                f"Try rephrasing as a specific object search for now.",
                "sys",
            ),
        )

    if intent == "moving_object":
        p    = classified.params  # MovingObjectParams
        hint = f" for <i>'{p.motion_hint}'</i>" if p.motion_hint else ""
        return (
            user_bubble,
            _msg(
                f"⚠️ Moving object detection{hint} is not yet implemented — "
                f"it requires consecutive frame pairs for temporal differencing. "
                f"Upload a frame sequence and re-query when this feature ships.",
                "sys",
            ),
        )

    # intent == "object_search" — fall through to CLIP + VLM flow

    # ── Vector search ────────────────────────────────────────────────────────
    # Bug A fix: limit=3 let LanceDB return multiple tiles from the same
    # parent frame (one dominant frame monopolises all 3 slots).
    # Retrieve a larger ANN pool, then diversity-filter to 1 tile per frame
    # (handled inside db.semantic_search via frames_to_return).
    #
    # Aerial context: prepend _CLIP_AERIAL_CTX to the raw query before CLIP
    # encoding to compensate for the domain gap between CLIP's ground-level
    # training distribution and UAV nadir imagery.  The original user query
    # is kept intact for VLM prompting — the VLM already has UAV framing.
    logger.info(f"Encoding query: '{message}'")
    clip_query = f"{_CLIP_AERIAL_CTX} {message}"
    logger.info(f"CLIP query (with aerial context): '{clip_query}'")
    q_vec      = lib.encode_text(clip_query)
    candidates = db.semantic_search(q_vec, limit=20, frames_to_return=8)
    logger.info(f"Vector search returned {len(candidates)} candidate tile(s).")

    if not candidates:
        return (
            user_bubble,
            _msg("No matching frames found in the index.", "sys"),
        )

    # ── Pick query color (cycles through palette per query) ──────────────────
    color = _MARKER_COLORS[_state["query_color_idx"] % len(_MARKER_COLORS)]
    _state["query_color_idx"] += 1
    new_det_ids: list[str] = []  # track det_ids added by this query
    logger.info(f"Query color: {color} | Sending {len(candidates)} tiles to VLM…")

    # ── VLM grounding ────────────────────────────────────────────────────────
    intel   = analyst.analyze_multiple_views(candidates, message)
    targets: list[dict] = intel.get("targets", [])
    logger.info(f"VLM returned {len(targets)} target(s). Summary: {intel.get('summary', '')}")

    # ── Build detections (with geo-NMS via merge_detections) ─────────────────
    #
    # merge_detections() does two things in one pass:
    #   1. Geo-clusters raw VLM targets by haversine distance (3m default) so
    #      the same physical object seen from multiple overlapping frames
    #      becomes ONE ObjectInstance instead of N duplicate markers.
    #   2. Within each cluster, keeps the best detection per parent frame and
    #      caps at MAX_ANGLES views — so the image panel shows up to 3 angles
    #      of the same object rather than just the highest-confidence crop.
    #
    # The set of tile names that reached VLM (used below for hollow markers)
    # comes from all candidates — analyst.py logs a hit or miss for each one
    # it successfully loads. Tiles that failed to load are absent from targets
    # AND from the miss count, so we can't distinguish them without a deeper
    # analyst.py change; hollow markers fire conservatively on load failures.
    instances = merge_detections(targets, candidates)
    logger.info(
        f"Geo-NMS | {len(targets)} raw VLM target(s) → "
        f"{len(instances)} unique object instance(s)"
    )

    # ── Confirmed instances → solid markers ──────────────────────────────────
    for instance in instances:
        det_id = uuid.uuid4().hex[:10]
        best   = instance.best

        # Annotate each angle's tile and collect thumbnails (up to MAX_ANGLES)
        img_urls: list[str] = []
        for det in instance.detections:
            ann_url = _annotate_and_save(det.candidate, det.bbox, message[:20])
            img_urls.append(ann_url or _tile_to_static_url(det.candidate))

        angle_str = (
            f" · {len(instance.detections)} angles"
            if instance.is_multiangle else ""
        )
        _state["detections"][det_id] = {
            "lat":          instance.lat,           # geo-center of best detection
            "lon":          instance.lon,
            "label":        message,
            "color":        color,
            "confirmed":    True,
            "img_urls":     img_urls,               # one URL per angle
            "is_multiangle": instance.is_multiangle,
            "source_count": len(instance.detections),
            "gsd":          f"{best.candidate.get('gsd_cm_px', 0):.1f}",
            "bbox":         best.bbox,
            "source":       Path(best.parent_path).name,
            "parent_path":  best.parent_path,
            "tile_x":       best.candidate["tile_x"],
            "tile_y":       best.candidate["tile_y"],
            "tile_w":       best.candidate["tile_w"],
            "tile_h":       best.candidate["tile_h"],
        }
        new_det_ids.append(det_id)
        logger.info(
            f"Instance {instance.instance_id}{angle_str} | "
            f"{Path(best.parent_path).name} | "
            f"LAT {instance.lat:.6f} LON {instance.lon:.6f} | "
            f"conf {best.confidence:.2f}"
        )

    # ── Unvetted tiles → hollow markers (VLM load errors only) ───────────────
    # A tile is "confirmed processed" if VLM returned any target for it OR
    # explicitly confirmed it absent. Pragmatic proxy: all candidates are
    # assumed processed unless their tile name never appeared in targets at all
    # AND we have zero targets total (suggesting a systematic load failure).
    # Individual tile load errors are rare; this keeps the logic simple until
    # analyst.py returns a per-tile processed/failed status.
    confirmed_tile_names: set[str] = {
        Path(t["filename"]).name.lower() for t in targets
    }
    # All candidate names — tiles VLM was asked to process
    vlm_processed_tile_names: set[str] = {
        Path(c["image_path"]).name.lower() for c in candidates
    }

    for cand in candidates:
        tile_name   = Path(cand["image_path"]).name
        parent_name = Path(cand["parent_path"]).name
        tile_lat, tile_lon = tile_center_geo(cand)

        if tile_name.lower() in confirmed_tile_names:
            # Already represented in a merged instance above — skip
            continue

        if tile_name.lower() not in vlm_processed_tile_names:
            # Tile was never sent to VLM (load error before analyst call).
            # Hollow marker — operator can manually inspect the source tile.
            det_id  = uuid.uuid4().hex[:10]
            img_url = _tile_to_static_url(cand)
            _state["detections"][det_id] = {
                "lat":          tile_lat,
                "lon":          tile_lon,
                "label":        f"Unvetted (load error): {message[:36]}",
                "color":        color,
                "confirmed":    False,
                "img_urls":     [img_url] if img_url else [],
                "is_multiangle": False,
                "source_count": 1,
                "gsd":          f"{cand.get('gsd_cm_px', 0):.1f}",
                "bbox":         None,
                "source":       parent_name,
                "parent_path":  cand["parent_path"],
                "tile_x":       cand["tile_x"],
                "tile_y":       cand["tile_y"],
                "tile_w":       cand["tile_w"],
                "tile_h":       cand["tile_h"],
            }
            new_det_ids.append(det_id)
            logger.info(
                f"Unvetted tile (VLM load error) | {parent_name} | "
                f"LAT {tile_lat:.6f} LON {tile_lon:.6f}"
            )
        else:
            # VLM processed this tile and confirmed the object is absent.
            # No marker — confirmed-empty tiles are not useful to the operator.
            logger.info(
                f"Confirmed absent (suppressed marker) | {parent_name} | "
                f"tile {tile_name}"
            )

    # Recenter map on the new markers from this query
    _recenter_on_detections(list(new_det_ids))
    _build_map()

    # ── Format chat response ─────────────────────────────────────────────────
    n_confirmed  = sum(1 for d in _state["detections"].values()
                       if d.get("confirmed") and d.get("color") == color)
    n_multiangle = sum(1 for d in _state["detections"].values()
                       if d.get("confirmed") and d.get("color") == color
                       and d.get("is_multiangle"))
    n_unvetted   = sum(1 for d in _state["detections"].values()
                       if not d.get("confirmed") and d.get("color") == color)

    raw_report = intel.get("report", "")
    dot_solid  = f'<span style="color:{color};font-size:13px">&#9679;</span>'
    dot_hollow = f'<span style="color:{color};font-size:13px">&#9675;</span>'

    lines = []
    if raw_report and targets:
        parts = [p.strip() for p in raw_report.split(" | ") if p.strip()]
        for p in parts:
            if ": " in p:
                p = p.split(": ", 1)[1]
            if p:
                lines.append(f"{dot_solid} {p}")

    if n_multiangle:
        lines.append(
            f"{dot_solid} {n_multiangle} object(s) confirmed from multiple angles "
            f"— click markers to browse all views."
        )

    if n_unvetted:
        lines.append(
            f"{dot_hollow} {n_unvetted} tile(s) flagged by CLIP could not be "
            f"vetted by VLM (tile load error) — click hollow markers to inspect."
        )

    if not lines:
        lines = [raw_report or "No objects matching the query were found."]

    reply_html = "<br>".join(lines)

    return (
        user_bubble,
        _msg_html(reply_html),
    )


@rt("/images/{det_id}")
def images(det_id: str):
    det = _state["detections"].get(det_id)
    if not det:
        return (
            Div(
                Span("Detection View"),
                Span("×", cls="panel-close", onclick="closeImages()"),
                cls="panel-header",
            ),
            Div(P("Detection not found.", style="color:var(--danger);padding:20px"),
                cls="empty-state"),
        )

    # Use frame_view widget (tile/frame toggle) instead of raw img
    fv_widget = frame_view(det_id, mode="tile")
    meta_div  = Div(
        Div(det["label"][:60], cls="label"),
        f"LAT {det['lat']:.6f}  ·  LON {det['lon']:.6f}",
        Br(),
        f"GSD {det['gsd']} cm/px  ·  Source: {det['source']}",
        cls="det-meta",
        style="margin:0 12px 12px",
    )
    cards = [Div(fv_widget, meta_div, cls="det-card")]

    return (
        Div(
            Span(f"📍 {det['label'][:36]}"),
            Span("×", cls="panel-close", onclick="closeImages()"),
            cls="panel-header",
        ),
        *cards,
    )


@rt("/toggle_coverage")
def toggle_coverage():
    try:
        _state["show_coverage"] = not _state["show_coverage"]
        state_txt = "ON" if _state["show_coverage"] else "OFF"
        logger.info(
            f"Coverage polygon: {state_txt} | "
            f"db.table={'set' if db.table is not None else 'None'} | "
            f"rows={db.row_count()}"
        )
        # ── NEW: re-center on the indexed survey area ─────────────────────
        if db.table is not None and db.row_count() > 0:
            try:
                df = db.table.to_pandas()
                lats = df.groupby("parent_path")["lat"].first().tolist()
                lons = df.groupby("parent_path")["lon"].first().tolist()
                if lats and lons:
                    _state["map_center"] = [sum(lats)/len(lats), sum(lons)/len(lons)]
                    _state["map_zoom"]   = 16
                    _state["frame_count"] = df["parent_path"].nunique()
                    _state["ingested"]    = True
            except Exception as e:
                logger.warning(f"toggle_coverage: re-center failed: {e}")
        # ─────────────────────────────────────────────────────────────────
        _build_map()
        msg = f"Coverage polygon {'shown' if _state['show_coverage'] else 'hidden'} (Ctrl+P to toggle)."
        return _msg(msg, "sys"), _status_badge()
    except Exception as e:
        logger.warning(f"toggle_coverage error (non-fatal): {e}")
        return _msg(f"Coverage toggle failed: {e}", "sys"), _status_badge()
        #      ^^^ was returning "" — now always returns the badge so OOB fires
        

@rt("/frame_view/{det_id}")
def frame_view(det_id: str, mode: str = "tile"):
    """Return annotated tile or full-frame card for the image panel."""
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
            # Tile boundary — vivid magenta, very thick, with semi-transparent fill
            TILE_CLR = (180, 0, 255)   # BGR: vivid magenta
            # Semi-transparent fill overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), TILE_CLR, -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            # Black shadow border then colored border
            cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), (0, 0, 0), 20)
            cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), TILE_CLR, 12)
            # Label
            cv2.putText(img, "TILE", (tx + 8, ty + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
            cv2.putText(img, "TILE", (tx + 8, ty + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, TILE_CLR, 3)
            # Detection bbox inside tile
            if bbox and len(bbox) == 4:
                bx1 = tx + int(bbox[0])
                by1 = ty + int(bbox[1])
                bx2 = tx + int(bbox[2])
                by2 = ty + int(bbox[3])
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), 10)  # shadow
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (74, 222, 128), 6)
            # Frame is already in UPLOAD_PATH — annotate in memory, stream via /frame_img/
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return P("Failed to encode frame.", style="color:var(--danger);padding:12px")
            _frame_img_cache[det_id] = buf.tobytes()
            img_url = f"/frame_img/{det_id}"
            other_mode, other_label = "tile", "Tile view"
        else:
            return P("Could not load parent frame.", style="color:var(--danger);padding:12px")
    else:
        # Tile view — check static/detections/ file still exists, else regenerate
        stored = det["img_urls"][0] if det.get("img_urls") else None
        if stored and stored.startswith("/tile_img/"):
            key = stored.split("/")[-1]
            img_url = stored if key in _tile_img_cache else None
        else:
            img_url = None
        if not img_url:
            logger.info(f"Tile cache miss — regenerating for {det_id}")
            img_url = _tile_to_static_url(det) or ""
        other_mode, other_label = "frame", "Full frame"

    toggle_btn = Button(
        other_label,
        hx_get=f"/frame_view/{det_id}?mode={other_mode}",
        hx_target=f"#fv-{det_id}",
        hx_swap="outerHTML",
        style=(
            "margin:8px 12px;padding:5px 14px;"
            "background:var(--bg4);border:1px solid var(--border);"
            "border-radius:5px;cursor:pointer;font-size:11px;"
            "font-family:var(--font-mono);color:var(--text);"
        ),
    )
    return Div(
        toggle_btn,
        Img(src=img_url, style="width:100%;display:block", loading="lazy") if img_url else "",
        id=f"fv-{det_id}",
    )


@rt("/map")
def serve_map():
    """Serve map.html from MAP_PATH."""
    map_file = MAP_PATH / "map.html"
    if not map_file.exists():
        _build_map()
    return FileResponse(str(map_file), media_type="text/html")


@rt("/tile_img/{key}")
def serve_tile_img(key: str):
    """Stream annotated tile from in-memory cache."""
    from starlette.responses import Response
    data = _tile_img_cache.get(key)
    if data is None:
        return Response("Tile not in cache — re-query to regenerate.", status_code=404)
    return Response(content=data, media_type="image/jpeg")


@rt("/frame_img/{det_id}")
def serve_frame_img(det_id: str):
    """Stream annotated full-frame image directly — no disk write needed.
    The frame is read from UPLOAD_PATH (parent_path), annotated in frame_view,
    cached in _frame_img_cache, and streamed here.
    """
    from starlette.responses import Response
    data = _frame_img_cache.get(det_id)
    if data is None:
        return Response("Frame not rendered yet — click 'Full frame' first.", status_code=404)
    return Response(content=data, media_type="image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
# Restore state from persistent DB on startup
_restore_done = False

def _restore_state():
    global _restore_done
    if _restore_done:
        return
    _restore_done = True

    try:
        db.initialize_table(vector_dim=getattr(settings, 'CLIP_DIM', 512))
        count = db.row_count()
        if count > 0:
            _state["ingested"]   = True
            _state["tile_count"] = count
            # ── NEW: derive frame count + map center from DB ──────────────────
            try:
                df = db.table.to_pandas()
                frame_count = df["parent_path"].nunique()
                _state["frame_count"] = frame_count
                lats = df.groupby("parent_path")["lat"].first().tolist()
                lons = df.groupby("parent_path")["lon"].first().tolist()
                if lats and lons:
                    _state["map_center"] = [sum(lats)/len(lats), sum(lons)/len(lons)]
                    _state["map_zoom"]   = 16
            except Exception as e:
                logger.warning(f"_restore_state: could not derive frame stats: {e}")
            # ─────────────────────────────────────────────────────────────────
            logger.info(
                f"Restored index from DB — {count} tiles / "
                f"{_state['frame_count']} frames already indexed."
            )
            _build_map()
        else:
            logger.info("DB is empty — waiting for upload.")
    except Exception as e:
        logger.warning(f"Could not restore DB state: {e}")

_restore_state()
_PORT = int(os.environ.get('PORT', 8000))
logger.info(f'Starting TAE on port {_PORT}')
serve(host='0.0.0.0', port=_PORT)