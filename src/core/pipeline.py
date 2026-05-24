"""
core/analysis.py — Analysis service layer.

Contains every function that involves processing data: building the Folium
map, loading/annotating tile images, and executing CLIP+VLM queries.

No FastHTML types, no HTTP routing, no UI components.

Dependencies on the database, VLM analyst, and CLIP library are injected
once at startup via init() so this module is testable in isolation.
"""

import logging
import os
import statistics
import traceback
import uuid
from pathlib import Path
from typing import Callable

import cv2
import folium

from core.app_state import (
    _state,
    _frame_img_cache,
    _tile_img_cache,
    _CLIP_AERIAL_CTX,
)

logger = logging.getLogger("TAE.Analysis")

# ── Map tile configuration ────────────────────────────────────────────────────
_MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoiYXJpZWdlbmtpbiIsImEiOiJjaWg2bWR1djIwMDBodmdtM2YxeXR5bzYzIn0.UvReMoVzH7szfW_1xszBhg",
)
_DARK_TILES = (
    "https://api.mapbox.com/styles/v1/mapbox/navigation-night-v1"
    f"/tiles/256/{{z}}/{{x}}/{{y}}@2x?access_token={_MAPBOX_TOKEN}"
)

# ── Injected dependencies (populated by init()) ───────────────────────────────
_db:              object = None
_analyst:         object = None
_get_search_lib:  Callable = None
_spatial:         object = None


def init(db, analyst, get_search_lib: Callable, spatial=None) -> None:
    """
    Wire up external dependencies.  Called once from main.py after the
    singletons (TacticalDatabase, TacticalAnalyst, SearchLibrarian) are ready.
    spatial is needed for the auto-tracking pipeline (YOLO-World + SORT).
    """
    global _db, _analyst, _get_search_lib, _spatial
    _db             = db
    _analyst        = analyst
    _get_search_lib = get_search_lib
    _spatial        = spatial


# ─────────────────────────────────────────────────────────────────────────────
# Map helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    lats = [_state["detections"][d]["lat"]
            for d in det_ids if d in _state["detections"]]
    lons = [_state["detections"][d]["lon"]
            for d in det_ids if d in _state["detections"]]
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
        label     = det.get("label", "")
        tip = f"{label[:50]} | {det['lat']:.5f}, {det['lon']:.5f}"
        # onclick directly on the dot — no popup intermediate step
        onclick = (
            f"window.parent.postMessage("
            f"{{type:\'show_images\',id:\'{det_id}\'}},"
            f"\'*\')"
        )
        dot_html = (
            f'<div onclick="{onclick}" title="{tip}" '
            f'style="width:18px;height:18px;'
            f'background:{"" if not confirmed else color};'
            f'border-radius:50%;border:2.5px solid {color};'
            f'box-shadow:0 0 {"8" if confirmed else "6"}px {color};'
            f'cursor:pointer;transition:transform .1s" '
            f'onmouseenter="this.style.transform=\'scale(1.4)\'" '
            f'onmouseleave="this.style.transform=\'scale(1.0)\'"'
            f'></div>'
        )
        icon = folium.DivIcon(html=dot_html, icon_size=(18, 18), icon_anchor=(9, 9))
        folium.Marker(
            location=[det["lat"], det["lon"]],
            icon=icon,
        ).add_to(m)

    if _state.get("show_coverage") and _db is not None and _db.table is not None:
        try:
            df = _db.table.to_pandas()[
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

            # Outlier filter: remove corners from geographically distant sessions.
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
                            "Hull outlier filter: dropped %d corners (%d of %d kept)",
                            removed, len(pts_clean), len(pts),
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
            if hull_pts and len(hull_pts) >= 3:
                folium.Polygon(
                    locations=hull_pts,
                    color="#7aa2f7", weight=2,
                    fill=True, fill_color="#7aa2f7", fill_opacity=0.15,
                ).add_to(m)

        except Exception as e:
            logger.error("Coverage polygon error: %s\n%s", e, traceback.format_exc())

    map_file = paths.maps / "map.html"
    # ── Track polylines (Phase D) ──────────────────────────────────────────
    try:
        paths = _state.get("mission_paths")
        if paths:
            from core.tracker import load_tracks
            tracks = load_tracks(paths.maps / "tracks.json")
            for trk in tracks:
                traj = trk.get("trajectory", [])
                color = trk.get("color", "#60a5fa")
                label = trk.get("label", "object")
                tid   = trk.get("id", "?")
                if len(traj) >= 2:
                    coords = [(p["lat"], p["lon"]) for p in traj]
                    folium.PolyLine(
                        locations  = coords,
                        color      = color,
                        weight     = 3,
                        opacity    = 0.85,
                        tooltip    = f"Track {tid}: {label} ({len(traj)} pts)",
                    ).add_to(m)
                    # Start dot
                    folium.CircleMarker(
                        location  = coords[0],
                        radius    = 5, color=color, fill=True,
                        fill_opacity=1.0, weight=1,
                        tooltip   = f"Track {tid} start",
                    ).add_to(m)
                    # End arrow (slightly larger)
                    folium.CircleMarker(
                        location  = coords[-1],
                        radius    = 7, color=color, fill=True,
                        fill_opacity=1.0, weight=2,
                        tooltip   = f"Track {tid} end",
                    ).add_to(m)
    except Exception as _te:
        logger.warning("Track rendering error: %s", _te)

    # Allow parent window to center the map via postMessage
    map_var = f"map_{m._id}"
    m.get_root().html.add_child(folium.Element(
        f'<script>window.addEventListener("message",function(ev){{'
        f'if(ev.data&&ev.data.type==="center_on"){{'
        f'{map_var}.setView([ev.data.lat,ev.data.lon],'
        f'ev.data.zoom||17,{{animate:true}});}}}});</script>'
    ))
    m.save(str(map_file))


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile(candidate: dict):
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


def _extract_video_frames(
    video_path: str, out_dir: Path, fps: float = 1.0
) -> list[str]:
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

def _save_detections(maps_path) -> None:
    """
    Persist _state["detections"] to detections.json.
    img_urls are excluded — they are in-memory cache keys that don't
    survive a restart.  Annotations are regenerated on demand by the
    /images/{det_id} route.
    """
    try:
        dets_file = Path(maps_path) / "detections.json"
        data = {
            det_id: {k: v for k, v in det.items() if k != "img_urls"}
            for det_id, det in _state["detections"].items()
        }
        dets_file.write_text(
            __import__("json").dumps(data, indent=2), encoding="utf-8"
        )
        logger.debug("Saved %d detection(s) → %s", len(data), dets_file)
    except Exception as e:
        logger.warning("Could not save detections: %s", e)


def _load_detections(maps_path) -> dict:
    """Load detections from detections.json.  Returns {} if file absent."""
    try:
        dets_file = Path(maps_path) / "detections.json"
        if not dets_file.exists():
            return {}
        data = __import__("json").loads(dets_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d detection(s) from disk", len(data))
        return data
    except Exception as e:
        logger.warning("Could not load detections: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Query execution — single shared pipeline for manual queries and auto-analysis
# ─────────────────────────────────────────────────────────────────────────────

def _execute_analysis(
    message:          str,
    color:            str,
    search_limit:     int  = 20,
    frames_to_return: int  = 8,
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
    lib        = _get_search_lib()
    clip_query = f"{_CLIP_AERIAL_CTX} {message}"
    q_vec      = lib.encode_text(clip_query)
    logger.info("CLIP query: '%s'", clip_query)

    candidates = _db.semantic_search(
        q_vec,
        limit            = search_limit,
        frames_to_return = frames_to_return,
    )
    if not candidates:
        return {"det_ids": [], "n_confirmed": 0, "n_unconfirmed": 0,
                "report": "", "color": color}

    logger.info("%d candidate tile(s) → VLM", len(candidates))
    intel   = _analyst.analyze_multiple_views(candidates, message)
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
                # Use VLM per-detection description (needs "description" in prompt schema)
                # Fallback chain: description → label → report snippet → query
                det_label = (
                    (t.get("description") or t.get("label") or "").strip()
                    or message
                )
                # Cap to 60 chars for display
                det_label = det_label[:60]
                ann_url = _annotate_and_save(cand, t.get("bbox"), det_label[:20])
                _state["detections"][det_id] = {
                    "lat":         cand["lat"],
                    "lon":         cand["lon"],
                    "label":       det_label,
                    "query":       message,
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
                "label":       "Candidate match",
                "query":       message,
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

    # Auto-trigger YOLO-World + SORT tracking in background.
    # Does not block the query response — map polylines appear once done.
    # Persist to disk so restarts load from file instead of replaying API calls
    paths = _state.get("mission_paths")
    if paths:
        _save_detections(paths.maps)

    # Note: _auto_track_background is NOT called here.
    # Tracking runs once per ingestion (called in main._ingest_background),
    # not on every query. User queries add detections only.

    return {
        "det_ids":       new_det_ids,
        "n_confirmed":   n_confirmed,
        "n_unconfirmed": n_unconfirmed,
        "report":        intel.get("report", ""),
        "color":         color,
    }


def _auto_track_background(message: str) -> None:
    """
    Spawn a background thread that runs YOLO-World dense detection
    + SORT tracking on all mission frames, then rebuilds the map
    with trajectory polylines.

    Called automatically at the end of every _execute_analysis().
    Non-blocking: the CLIP+VLM detections are already on the map
    before this function returns.
    """
    paths = _state.get("mission_paths")
    if paths is None or _db is None or _db.table is None:
        return

    import threading

    def _run() -> None:
        try:
            from core.tracker import run_tracking, save_tracks
            import re as _re

            # img_w_px/img_h_px not in LanceDB — reconstruct from tile extents
            df_all = _db.table.to_pandas()[
                ["parent_path", "lat", "lon", "alt_m", "gimbal_yaw",
                 "tile_x", "tile_y", "tile_w", "tile_h"]
            ]
            if df_all.empty:
                return

            frame_ts = _state.get("frame_timestamps", {})
            frames = []
            for parent_path, grp in df_all.groupby("parent_path"):
                ref   = grp.iloc[0]
                img_w = int((grp["tile_x"] + grp["tile_w"]).max())
                img_h = int((grp["tile_y"] + grp["tile_h"]).max())
                fname = Path(parent_path).name
                m_ts  = _re.search(r"_s(\d{5})", fname)
                fidx  = int(m_ts.group(1)) if m_ts else 0
                frames.append({
                    "parent_path": parent_path,
                    "lat":         float(ref["lat"]),
                    "lon":         float(ref["lon"]),
                    "alt_m":       float(ref.get("alt_m", 80.0)),
                    "gimbal_yaw":  float(ref.get("gimbal_yaw", 0.0)),
                    "img_w_px":    img_w if img_w > 0 else 1920,
                    "img_h_px":    img_h if img_h > 0 else 1080,
                    "frame_idx":   fidx,
                    "timestamp_ms": frame_ts.get(fname, fidx * 2000),
                })
            frames.sort(key=lambda r: r["frame_idx"])

            logger.info("Auto-tracking %d frames for: %s", len(frames), message)
            tracks = run_tracking(
                frames    = frames,
                spatial   = _spatial,
                query     = message,
                confidence= 0.15,
            )
            save_tracks(tracks, paths.maps / "tracks.json")
            _build_map()   # re-render with polylines
            logger.info("Auto-tracking done: %d track(s)", len(tracks))
        except Exception as e:
            logger.error("Auto-tracking failed: %s", e)

    threading.Thread(target=_run, daemon=True).start()