"""
core/services.py — Shared stateful service layer
=================================================
Contains every function that involves processing data at runtime:
building the Folium map, loading/annotating tile images, persisting
detections, and extracting video frames.

NOT responsible for:
  - Query execution (→ ai/detection_pipeline.py for object_detection)
  - CLIP search or VLM calls (→ ai/clip.py, ai/vlm.py)
  - Ingest-time tile encoding (→ core/ingestion.py)
  - HTTP routing or FastHTML types (→ main.py)

Dependencies on the database, VLM, and CLIP library are injected once
at startup via init() so this module is testable in isolation.

Replaces: pipeline.py  (same content, correct name, _execute_analysis removed)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import traceback
import uuid
from pathlib import Path
from typing import Callable
import time

import cv2
import folium

from core.app_state import (
    _state,
    _frame_img_cache,
    _tile_img_cache,
)

from config import Settings
settings = Settings()

logger = logging.getLogger("TAE.Services")

# ── Map tile configuration ────────────────────────────────────────────────────
_MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoiYXJpZWdlbmtpbiIsImEiOiJjaWg2bWR1djIwMDBodmdtM2YxeXR5bzYzIn0."
    "UvReMoVzH7szfW_1xszBhg",
)
_DARK_TILES = (
    "https://api.mapbox.com/styles/v1/mapbox/navigation-night-v1"
    f"/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={_MAPBOX_TOKEN}"
)

# ── Injected dependencies (populated by init()) ───────────────────────────────
_db:             object   = None
_analyst:        object   = None
_get_search_lib: Callable = None
_spatial:        object   = None


def init(db, analyst, get_search_lib: Callable, spatial=None) -> None:
    global _db, _analyst, _get_search_lib, _spatial
    _db             = db
    _analyst        = analyst
    _get_search_lib = get_search_lib   # stores main.py's factory, not this module's wrapper
    _spatial        = spatial
    logger.info("Services layer initialised.")


def get_search_lib():
    """Return the SearchLibrarian singleton via the injected factory."""
    if _get_search_lib is None:
        raise RuntimeError("services.init() has not been called yet")
    if _get_search_lib is get_search_lib:          # ← guard against self-reference
        raise RuntimeError(
            "services._get_search_lib points to itself — "
            "pass main.get_search_lib to services.init(), not services.get_search_lib"
        )
    return _get_search_lib()

# ─────────────────────────────────────────────────────────────────────────────
# Map helpers
# ─────────────────────────────────────────────────────────────────────────────

def _optimal_zoom(lat_span: float, lon_span: float) -> int:
    """Estimate a Leaflet zoom level that fits a lat/lon bounding box."""
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
    """Recentre the map on a set of newly added detections."""
    lats = [_state["detections"][d]["lat"]
            for d in det_ids if d in _state["detections"]]
    lons = [_state["detections"][d]["lon"]
            for d in det_ids if d in _state["detections"]]
    if not lats:
        return
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    zoom = _optimal_zoom(
        (max(lats) - min(lats)) * 1.3,
        (max(lons) - min(lons)) * 1.3,
    )
    _state["map_center"] = [center_lat, center_lon]
    _state["map_zoom"]   = zoom


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain — pure Python, no scipy dependency."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts

    def _cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _build_map() -> None:
    """Regenerate map.html for the active mission."""
    paths = _state.get("mission_paths")
    if paths is None:
        return

    lat, lon = _state["map_center"]
    m = folium.Map(
        location   = [lat, lon],
        zoom_start = _state["map_zoom"],
        tiles      = _DARK_TILES,
        attr       = "Mapbox",
        control_scale = False,
        zoomControl   = True,
    )

    # ── Detection markers ─────────────────────────────────────────────────────
    for det_id, det in _state["detections"].items():
        color     = det.get("color", "#4ade80")
        confirmed = det.get("confirmed", True)
        label     = det.get("label", "")
        tip       = f"{label[:50]} | {det['lat']:.5f}, {det['lon']:.5f}"

        # Clicking the dot sends a postMessage to the parent page (no popup needed)
        onclick = (
            f"window.parent.postMessage("
            f"{{type:'show_images',id:'{det_id}'}},"
            f"'*')"
        )
        filled = "background:" + color if confirmed else "background:transparent"
        glow   = "8" if confirmed else "6"
        dot_html = (
            f'<div onclick="{onclick}" title="{tip}" '
            f'style="width:18px;height:18px;{filled};'
            f'border-radius:50%;border:2.5px solid {color};'
            f'box-shadow:0 0 {glow}px {color};'
            f'cursor:pointer;transition:transform .1s" '
            f'onmouseenter="this.style.transform=\'scale(1.4)\'" '
            f'onmouseleave="this.style.transform=\'scale(1.0)\'">'
            f'</div>'
        )
        icon = folium.DivIcon(html=dot_html, icon_size=(18, 18), icon_anchor=(9, 9))
        folium.Marker(
            location=[det["lat"], det["lon"]],
            icon=icon,
        ).add_to(m)

    # ── Coverage polygon ──────────────────────────────────────────────────────
    # Computed from ALL tile footprint corners (not GPS waypoints) so the hull
    # correctly represents the surveyed ground area rather than the flight path.
    if _state.get("show_coverage") and _db is not None and _db.table is not None:
        try:
            df = _db.table.to_pandas()[[
                "parent_path",
                "fp_nw_lat", "fp_nw_lon", "fp_ne_lat", "fp_ne_lon",
                "fp_se_lat", "fp_se_lon", "fp_sw_lat", "fp_sw_lon",
            ]]
            n_frames = df["parent_path"].nunique()
            logger.info(
                "Coverage: %d frames | %d tiles → building hull", n_frames, len(df)
            )

            pts: list[tuple[float, float]] = []
            for _, r in df.iterrows():
                pts += [
                    (r.fp_nw_lat, r.fp_nw_lon), (r.fp_ne_lat, r.fp_ne_lon),
                    (r.fp_se_lat, r.fp_se_lon),  (r.fp_sw_lat, r.fp_sw_lon),
                ]

            # Outlier filter: drop corners more than 0.15° from the median
            # (guards against bad GPS fixes from a different session)
            if len(pts) >= 6:
                med_lat = statistics.median(p[0] for p in pts)
                med_lon = statistics.median(p[1] for p in pts)
                MAX_DEG = 0.15
                pts_clean = [
                    p for p in pts
                    if abs(p[0] - med_lat) < MAX_DEG
                    and abs(p[1] - med_lon) < MAX_DEG
                ]
                if len(pts_clean) >= 3:
                    removed = len(pts) - len(pts_clean)
                    if removed:
                        logger.info(
                            "Hull outlier filter: dropped %d of %d corners",
                            removed, len(pts),
                        )
                    pts = pts_clean

            hull_pts = _convex_hull(pts)
            logger.info("Coverage hull: %d vertices", len(hull_pts))

            if hull_pts and len(hull_pts) >= 3:
                folium.Polygon(
                    locations   = hull_pts,
                    color       = "#7aa2f7",
                    weight      = 2,
                    fill        = True,
                    fill_color  = "#7aa2f7",
                    fill_opacity= 0.15,
                ).add_to(m)

        except Exception as e:
            logger.error("Coverage polygon error: %s\n%s", e, traceback.format_exc())

    # ── Track polylines ───────────────────────────────────────────────────────
    try:
        if paths:
            tracks_file = paths.detections / "tracks.json"
            if tracks_file.exists():
                tracks = json.loads(tracks_file.read_text(encoding="utf-8"))
                for trk in tracks:
                    traj  = trk.get("trajectory", [])
                    color = trk.get("color", "#60a5fa")
                    label = trk.get("label", "object")
                    tid   = trk.get("id", "?")
                    if len(traj) >= 2:
                        coords = [(p["lat"], p["lon"]) for p in traj]
                        folium.PolyLine(
                            locations = coords,
                            color     = color,
                            weight    = 3,
                            opacity   = 0.85,
                            tooltip   = f"Track {tid}: {label} ({len(traj)} pts)",
                        ).add_to(m)
                        folium.CircleMarker(
                            location=coords[0], radius=5,
                            color=color, fill=True, fill_opacity=1.0, weight=1,
                            tooltip=f"Track {tid} start",
                        ).add_to(m)
                        folium.CircleMarker(
                            location=coords[-1], radius=7,
                            color=color, fill=True, fill_opacity=1.0, weight=2,
                            tooltip=f"Track {tid} end",
                        ).add_to(m)
    except Exception as _te:
        logger.warning("Track rendering error: %s", _te)

    # ── Motion track polylines / motion detections ─────────────────────────────
    try:
        if paths and hasattr(paths, "motion_tracks"):
            mt_file = paths.motion_tracks / "motion_tracks.json"
            if mt_file.exists():
                motion_tracks = json.loads(mt_file.read_text(encoding="utf-8"))
                min_pts      = getattr(settings, "MOTION_MIN_TRAJ_PTS", 8)
                show_circles = getattr(settings, "MOTION_SHOW_CIRCLES", False)

                for mtrk in motion_tracks:
                    traj = mtrk.get("trajectory", [])
                    if len(traj) < min_pts:
                        continue            # skip — too short to be informative

                    color    = mtrk.get("map_color") or mtrk.get("color", "#fb923c")
                    tid      = mtrk.get("track_id", "?")
                    label    = mtrk.get("label", "mover")
                    variance = mtrk.get("wf_variance_m", -1.0)
                    hits     = mtrk.get("hit_count", len(traj))
                    tip      = (f"Motion {tid}: {label}  hits={hits}  "
                                f"var={'n/a' if variance < 0 else f'{variance:.1f}m'}")

                    coords = [(p["lat"], p["lon"]) for p in traj]
                    folium.PolyLine(
                        locations  = coords,
                        color      = color,
                        weight     = 2,
                        opacity    = 0.80,
                        dash_array = "5 4",
                        tooltip    = tip,
                    ).add_to(m)

                    if show_circles:
                        folium.CircleMarker(
                            location=coords[0], radius=4,
                            color=color, fill=True, fill_opacity=0.8, weight=1,
                            tooltip=tip + " ▶ start",
                        ).add_to(m)
                        folium.CircleMarker(
                            location=coords[-1], radius=6,
                            color=color, fill=True, fill_opacity=1.0, weight=2,
                            tooltip=tip + " ■ end",
                        ).add_to(m)
    except Exception as _me:
        logger.warning("Motion track rendering error: %s", _me)

    # ── postMessage listener — allow parent page to re-centre the map ─────────
    map_var = f"map_{m._id}"
    m.get_root().html.add_child(folium.Element(
        f'<script>window.addEventListener("message",function(ev){{'
        f'if(ev.data&&ev.data.type==="center_on"){{'
        f'{map_var}.setView([ev.data.lat,ev.data.lon],'
        f'ev.data.zoom||17,{{animate:true}});}}}});</script>'
    ))

    map_file = paths.maps / "map.html"
    m.save(str(map_file))
    _state["map_version"] = int(time.time() * 1000)
    logger.info("Map saved to %s (version %s)", map_file, _state["map_version"])

# ─────────────────────────────────────────────────────────────────────────────
# Image / tile helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile(candidate: dict):
    """
    Reconstruct a tile by cropping its parent frame.
    Tiles are never stored on disk; this is the single reconstruction point.
    """
    parent = candidate.get("parent_path", "")
    if not parent or not Path(parent).exists():
        logger.error("_load_tile: parent frame not found: %s", parent)
        return None
    img = cv2.imread(parent)
    if img is None:
        logger.error("_load_tile: cv2.imread returned None for: %s", parent)
        return None
    x, y = candidate["tile_x"], candidate["tile_y"]
    w, h = candidate["tile_w"], candidate["tile_h"]
    return img[y: y + h, x: x + w]


def _annotate_and_save(
    candidate: dict,
    bbox_px:   list | None,
    label:     str,
    persist: bool = True,
    gdino_confidence: float | None = None,
    vlm_confidence: float | None = None,
) -> str | None:
    """Crop tile, optionally draw bbox, cache JPEG bytes, return URL."""
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
        return None
    
    key = uuid.uuid4().hex[:16]
    img_bytes = buf.tobytes()
    _tile_img_cache[key] = img_bytes

    # Save to disk for debugging/analysis
    disk_path = ""
    if persist:
        try:
            paths = _state.get("mission_paths")
            if paths:
                det_dir = Path(paths.uploads).parent / "detections"
                det_dir.mkdir(exist_ok=True)
                safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label[:20])
                g_conf = gdino_confidence if gdino_confidence is not None \
                         else float(candidate.get("gdino_confidence", candidate.get("confidence", 0.0)))
                v_conf = vlm_confidence if vlm_confidence is not None \
                         else float(candidate.get("vlm_confidence", 0.0))
                g_str  = f"{g_conf:.2f}"
                v_str  = f"{v_conf:.2f}"
                # Filename: {label}_{vlm_conf}_{gdino_conf}_{id}.jpg  e.g. car_0.91_0.73_a3b4c5d6.jpg
                disk_path = str(det_dir / f"{safe_label}_{v_str}_{g_str}_{key[:8]}.jpg")
                Path(disk_path).write_bytes(img_bytes)
        except Exception:
            pass

    return f"/tile_img/{key}", disk_path

def _tile_to_static_url(candidate: dict) -> str | None:
    """Crop tile (no annotation), cache, return URL."""
    url, _ = _annotate_and_save(candidate, None, "", persist=False)
    return url

def _extract_video_frames(
    video_path: str,
    out_dir:    Path,
    fps:        float = 1.0,
) -> list[str]:
    """Extract frames from a video at the given fps rate."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
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
    logger.info("Extracted %d frames from %s", saved, video_path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Detection persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_detections(detections_path) -> None:
    """
    Persist _state["detections"] to detections.json.
    img_urls excluded — they are in-memory cache keys that don't survive restart.
    """
    try:
        dets_file = Path(detections_path) / "detections.json"
        data = {
            det_id: {k: v for k, v in det.items() if k != "img_urls"}
            for det_id, det in _state["detections"].items()
        }
        dets_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved %d detection(s) → %s", len(data), dets_file)
    except Exception as e:
        logger.warning("Could not save detections: %s", e)

def _save_motion_tracks(motion_tracks_path: Path) -> None:
    """Persist _state['motion_tracks'] to motion_tracks/motion_tracks.json."""
    try:
        out = motion_tracks_path / "motion_tracks.json"
        tracks = list(_state.get("motion_tracks", {}).values())
        out.write_text(json.dumps(tracks, indent=2), encoding="utf-8")
        logger.info("Motion tracks saved (%d tracks)", len(tracks))
    except Exception as exc:
        logger.warning("Could not save motion tracks: %s", exc)

def _save_tracks(tracks: list, color: str) -> None:
    """
    Persist multi-frame track trajectories to paths.detections/tracks.json.
 
    _build_map() (services.py) reads this file and renders each trajectory
    as a Folium PolyLine with start/end circle markers.
 
    Only tracks with ≥ 2 detections are written; single-frame (parked,
    seen once) objects are already shown as dot markers.
 
    The file is append-safe across multiple queries in the same session:
    tracks from earlier queries are preserved and new ones are merged by
    track_id.
    """
    paths = _state.get("mission_paths")
    if not paths:
        return
 
    frame_ts: dict = _state.get("frame_timestamps", {})
 
    new_entries: list[dict] = []
    for track in tracks:
        if len(track.detections) < 2:
            continue
 
        # Sort chronologically using stored video timestamps
        dets_sorted = sorted(
            track.detections,
            key=lambda d: frame_ts.get(Path(d.parent_path).name, 0),
        )
 
        trajectory = [
            {
                "lat":    d.lat,
                "lon":    d.lon,
                "source": Path(d.parent_path).name,
                "ts_ms":  frame_ts.get(Path(d.parent_path).name, 0),
            }
            for d in dets_sorted
        ]
 
        new_entries.append({
            "id":         track.track_id,
            "label":      track.label,
            "color":      color,
            "trajectory": trajectory,
            "speed_ms":   round(getattr(track, "_speed_ms", 0.0), 2),
        })
 
    if not new_entries:
        return
 
    tracks_file = paths.detections / "tracks.json"
    existing: list[dict] = []
    if tracks_file.exists():
        try:
            existing = json.loads(tracks_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
 
    existing_by_id: dict[str, int] = {t["id"]: i for i, t in enumerate(existing)}
    added = updated = 0
    for entry in new_entries:
        if entry["id"] in existing_by_id:
            existing[existing_by_id[entry["id"]]] = entry   # replace in-place
            updated += 1
        else:
            existing_by_id[entry["id"]] = len(existing)
            existing.append(entry)
            added += 1

    tracks_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info(
        "_save_tracks: %d new + %d updated multi-frame track(s) → %s",
        added, updated, tracks_file,
    )
    
def _load_detections(detections_path) -> dict:
    """Load detections from detections.json. Returns {} if absent."""
    try:
        dets_file = Path(detections_path) / "detections.json"
        if not dets_file.exists():
            return {}
        data = json.loads(dets_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d detection(s) from disk", len(data))
        return data
    except Exception as e:
        logger.warning("Could not load detections: %s", e)
        return {}

def _load_motion_tracks(motion_tracks_path) -> dict:
    """Load motion tracks from motion_tracks.json. Returns {} if absent."""
    try:
        mt_file = Path(motion_tracks_path) / "motion_tracks.json"
        if not mt_file.exists():
            return {}
        data = json.loads(mt_file.read_text(encoding="utf-8"))
        keyed = {t["track_id"]: t for t in data if "track_id" in t}
        logger.info("Loaded %d motion track(s) from disk", len(keyed))
        return keyed
    except Exception as e:
        logger.warning("Could not load motion tracks: %s", e)
        return {}
   

