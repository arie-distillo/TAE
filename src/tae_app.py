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
from tools.ingest_telemetry import TAESimGenerator

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

# Cache for annotated full-frame images served by /frame_img/{det_id}
_frame_img_cache: dict[str, bytes] = {}
_tile_img_cache:  dict[str, bytes] = {}  # hex key → annotated tile JPEG

_state: dict = {
    "map_center":      [32.08, 34.78],
    "map_zoom":        14,
    "detections":      {},    # det_id → DetectionRecord dict
    "ingested":        False,
    "frame_count":     0,
    "tile_count":      0,
    "query_color_idx": 0,     # increments each query
    "show_coverage": False,   # Ctrl+P toggle
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
            df = db.table.to_pandas()[
                ["parent_path",
                 "fp_nw_lat","fp_nw_lon","fp_ne_lat","fp_ne_lon",
                 "fp_se_lat","fp_se_lon","fp_sw_lat","fp_sw_lon"]
            ].drop_duplicates(subset=["parent_path"])  # one footprint per frame

            logger.info(f"Coverage: {len(df)} unique frames to draw")

            # Collect all corner points for the convex hull
            pts = []
            for _, r in df.iterrows():
                pts += [
                    (r.fp_nw_lat, r.fp_nw_lon), (r.fp_ne_lat, r.fp_ne_lon),
                    (r.fp_se_lat, r.fp_se_lon), (r.fp_sw_lat, r.fp_sw_lon),
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

app, rt = fast_app(
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


@rt("/upload", methods=["POST"])
async def upload(request: Request):
    form  = await request.form()
    files = form.getlist("files")

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
    gen = TAESimGenerator(str(UPLOAD_PATH), str(UPLOAD_PATH))
    gen.generate()
    meta_file = UPLOAD_PATH / "pose_metadata.json"

    if not meta_file.exists():
        return _msg("⚠️  Metadata extraction failed.", "sys")

    meta: dict = json.loads(meta_file.read_text())
    if not meta:
        return _msg("⚠️  No parseable metadata in uploaded images.", "sys")

    # ── Update map center ────────────────────────────────────────────────────
    lats = [v["lat"] for v in meta.values() if v.get("lat") not in (None, 0.0)]
    lons = [v["lon"] for v in meta.values() if v.get("lon") not in (None, 0.0)]
    if lats and lons:
        _state["map_center"] = [sum(lats) / len(lats), sum(lons) / len(lons)]
        _state["map_zoom"]   = 16

    # ── CLIP ingestion ───────────────────────────────────────────────────────
    from core.sim_provider import SimD3Environment
    from core.ingestion import run_ingestion

    logger.info(f"Metadata extracted for {len(meta)} frame(s).")
    db.initialize_table(vector_dim=getattr(settings, 'CLIP_DIM', 512))

    if db.row_count() > 0:
        existing = db.row_count()
        logger.info(f"Index already contains {existing} tiles — skipping ingestion.")
        ok, fail = existing, 0
    else:
        sim = SimD3Environment(str(meta_file))
        lib = get_search_lib()
        logger.info("Starting CLIP ingestion…")
        tiles_ok, frames_failed = run_ingestion(sim, spatial, lib, db)
        logger.info(f"Ingestion done — {tiles_ok} tiles, {frames_failed} frame(s) failed.")
        ok, fail = tiles_ok, frames_failed

    first_error = None  # run_ingestion logs errors internally

    _state["ingested"]    = True
    _state["frame_count"] = len(meta)
    _state["tile_count"]   = ok
    _state["detections"]  = {}   # Clear old detections on re-upload
    _build_map()

    video_note = (
        f" (from {video_count} video{'s' if video_count > 1 else ''})"
        if video_count else ""
    )
    center_txt = (
        f"{_state['map_center'][0]:.5f}, {_state['map_center'][1]:.5f}"
        if lats else "unknown (no GPS)"
    )

    return (
        _msg(
            f"✅  Indexed {ok} tiles across {_state['frame_count']} frame{'s' if _state['frame_count'] != 1 else ''}{video_note}. "
            f"{f'({fail} frames failed). ' if fail else ''}"
            f"Map centered at {center_txt}. "
            f"Theater is ready — query below."
            + (f" First error: {first_error}" if first_error else ""),
            "sys",
        ),
        # OOB-swap the status badge in the navbar
        _status_badge(),
    )


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

    # ── Vector search ────────────────────────────────────────────────────────
    logger.info(f"Encoding query: '{message}'")
    q_vec      = lib.encode_text(message)
    candidates = db.semantic_search(q_vec, limit=3)
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

    # ── Build detections ─────────────────────────────────────────────────────
    for cand in candidates:
        tile_name   = Path(cand["image_path"]).name
        parent_name = Path(cand["parent_path"]).name
        frame_targets = [
            t for t in targets
            if Path(t["filename"]).name.lower() == tile_name.lower()
        ]

        if frame_targets:
            for t in frame_targets:
                det_id  = uuid.uuid4().hex[:10]
                ann_url = _annotate_and_save(cand, t.get("bbox"), message[:20])
                img_urls = [ann_url] if ann_url else [_tile_to_static_url(cand)]
                _state["detections"][det_id] = {
                    "lat":         cand["lat"],
                    "lon":         cand["lon"],
                    "label":       message,
                    "color":       color,
                    "confirmed":   True,
                    "img_urls":    img_urls,
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
                logger.info(
                    f"Detection | {parent_name} | "
                    f"LAT {cand['lat']:.6f} LON {cand['lon']:.6f} | "
                    f"conf {t.get('confidence', '—')}"
                )
        else:
            # CLIP found this tile relevant but VLM couldn't confirm with a bbox.
            # Still show as a candidate marker (hollow) so the operator can inspect.
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
            logger.info(
                f"Candidate (no VLM bbox) | {parent_name} | "
                f"LAT {cand['lat']:.6f} LON {cand['lon']:.6f}"
            )

    # Recenter map on the new markers from this query
    _recenter_on_detections(list(new_det_ids))
    _build_map()

    # ── Format chat response ─────────────────────────────────────────────────
    n_confirmed  = sum(1 for d in _state["detections"].values()
                       if d.get("confirmed") and d.get("color") == color)
    n_candidates = sum(1 for d in _state["detections"].values()
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

    if n_candidates:
        lines.append(
            f"{dot_hollow} {n_candidates} additional frame(s) matched by semantic search "
            f"but not confirmed by VLM — click hollow markers to inspect."
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
    _state["show_coverage"] = not _state["show_coverage"]
    state_txt = "ON" if _state["show_coverage"] else "OFF"
    logger.info(
        f"Coverage polygon: {state_txt} | "
        f"db.table={'set' if db.table is not None else 'None'} | "
        f"rows={db.row_count()}"
    )
    _build_map()
    return _msg(f"Coverage polygon {'shown' if _state['show_coverage'] else 'hidden'} (Ctrl+P to toggle).", "sys")


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
def _restore_state():
    try:
        db.initialize_table(vector_dim=getattr(settings, 'CLIP_DIM', 512))
        count = db.row_count()
        if count > 0:
            _state["ingested"]    = True
            _state["tile_count"]  = count
            logger.info(f"Restored index from DB — {count} tiles already indexed.")
            _build_map()
        else:
            logger.info("DB is empty — waiting for upload.")
    except Exception as e:
        logger.warning(f"Could not restore DB state: {e}")

_restore_state()
_PORT = int(os.environ.get('PORT', 8000))
logger.info(f'Starting TAE on port {_PORT}')
serve(host='0.0.0.0', port=_PORT)