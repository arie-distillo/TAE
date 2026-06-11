"""
ai/detection_session.py  —  Shared detection-result commit logic.
==================================================================
Single source of truth for code that was duplicated across three paths in main.py:
  • /query route               (query path)
  • _ingest_background          (auto-analysis)
  • _stream_on_frame_telem /
    _bg_detect_callback         (streaming paths)

Public API
----------
collect_img_urls(track)                        annotate + cache tile crops
build_detection_record(track, color)           canonical _state["detections"] entry
commit_tracks(tracks, color, *, replace)       write tracks → state, return det_ids
flush_to_map(new_det_ids, tracks, color)       recenter + save tracks + rebuild map
pipeline_call_kwargs(color, actual_alt_m)      centralised settings access
stream_detect_and_commit(...)                  stages 1-5 + accumulate + track + commit

Import hierarchy — no upward imports:
    main.py
      └─ detection_session.py
           ├─ ai/detection_pipeline.py
           ├─ core/services.py
           └─ core/app_state.py
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from config import settings

logger = logging.getLogger("TAE.DetectionSession")


# ─────────────────────────────────────────────────────────────────────────────
# Image URL collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_img_urls(track) -> list[str]:
    """
    Annotate and cache tile crops for every detection in a track.

    Uses _state["det_crops"] (id(det) → disk_path) to skip detections that
    have already been annotated in this session.  _annotate_and_save returns
    (url, disk_path); we store disk_path so the trajectory's detection_image
    field can be populated by build_detection_record.

    MUST be called before build_detection_record so that det_crops is
    populated before the trajectory entries read it.

    Returns list of /tile_img/<key> URL strings for newly annotated crops.
    """
    from core.services import _annotate_and_save
    from core.app_state import _state

    det_crops = _state.setdefault("det_crops", {})
    urls: list[str] = []
    for det in track.detections:
        if id(det) in det_crops:
            continue   # already annotated this detection object in this session
        try:
            url, disk_path = _annotate_and_save(
                {
                    "parent_path": det.parent_path,
                    "tile_x":      det.tile_x,
                    "tile_y":      det.tile_y,
                    "tile_w":      det.tile_w,
                    "tile_h":      det.tile_h,
                },
                det.bbox_tile,
                track.best.label,
            )
            det_crops[id(det)] = disk_path
        except Exception:
            url = None
        if url:
            urls.append(url)
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Canonical detection record
# ─────────────────────────────────────────────────────────────────────────────

def build_detection_record(track, color: str) -> dict:
    """
    Build the canonical _state['detections'] entry for a Track.

    Call collect_img_urls(track) FIRST — it populates _state["det_crops"]
    which is read here to set detection_image on each trajectory entry.

    img_urls is intentionally [] here; commit_tracks sets it to the return
    value of collect_img_urls after calling this function.
    """
    from core.app_state import _state
    best      = track.best
    det_crops = _state.get("det_crops", {})
    return {
        "lat":              track.lat,
        "lon":              track.lon,
        "label":            best.label,
        "color":            color,
        "confirmed":        True,
        "img_urls":         [],
        "is_multiangle":    len(track.detections) > 1,
        "source_count":     len(track.detections),
        "gsd":              "—",
        "bbox":             best.bbox_tile,
        "source":           Path(best.parent_path).name,
        "parent_path":      best.parent_path,
        "tile_x":           best.tile_x,
        "tile_y":           best.tile_y,
        "tile_w":           best.tile_w,
        "tile_h":           best.tile_h,
        "gdino_confidence": round(best.confidence, 4),
        "vlm_confidence":   round(getattr(best, "vlm_confidence", 0.0), 4),
        "vlm_reason":       getattr(best, "vlm_reason", ""),
        "vlm_report":       best.vlm_report,
        "track_id":         track.track_id,
        "trajectory": [
            {
                "lat":              d.lat,
                "lon":              d.lon,
                "source":           Path(d.parent_path).name,
                "tile_x":           d.tile_x,
                "tile_y":           d.tile_y,
                "tile_w":           d.tile_w,
                "tile_h":           d.tile_h,
                "bbox":             d.bbox_tile,
                "gdino_confidence": round(d.confidence, 4),
                "vlm_confidence":   round(getattr(d, "vlm_confidence", 0.0), 4),
                "vlm_reason":       getattr(d, "vlm_reason", ""),
                "detection_image":  det_crops.get(id(d), ""),
            }
            for d in track.detections
        ],
        "is_moving":        getattr(track, "_speed_ms", 0.0) > 1.0,
        "speed_ms":         round(getattr(track, "_speed_ms", 0.0), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State commit
# ─────────────────────────────────────────────────────────────────────────────

def commit_tracks(
    tracks,
    color:   str,
    *,
    replace: bool = False,
) -> list[str]:
    """
    Write tracks into _state['detections'] and populate img_urls.

    replace=False (query + auto-analysis):
        Appends to existing detections.  Uses uuid4() det_ids so results
        from multiple queries accumulate without overwriting each other.

    replace=True (streaming paths):
        Wipes _state["detections"] first, then uses stable
        f"{label}_{track_id}" det_ids so map markers don't flicker on the
        per-frame state rebuild (track IDs are sha1-stable across frames).

    collect_img_urls is called BEFORE build_detection_record so det_crops
    is populated before the trajectory's detection_image fields are read.

    Returns the list of new det_ids (for _recenter_on_detections).
    """
    import uuid as _uuid
    from core.app_state import _state

    if replace:
        _state["detections"] = {}

    new_det_ids: list[str] = []
    for track in tracks:
        det_id = (
            f"{track.label}_{track.track_id}"
            if replace
            else _uuid.uuid4().hex[:10]
        )
        img_urls = collect_img_urls(track)       # FIRST — populates det_crops
        record   = build_detection_record(track, color)  # reads det_crops
        record["img_urls"] = img_urls
        _state["detections"][det_id] = record
        new_det_ids.append(det_id)

    return new_det_ids


# ─────────────────────────────────────────────────────────────────────────────
# Map flush
# ─────────────────────────────────────────────────────────────────────────────

def flush_to_map(
    new_det_ids: list[str],
    tracks:      list,
    color:       str,
) -> None:
    """
    Standard post-commit flush: recenter, write tracks.json, rebuild map,
    persist detections.json.

    Replaces the repeated sequence in every detection path:
        if new_det_ids: _recenter_on_detections(new_det_ids)
        if tracks:      _save_tracks(tracks, color)
        _build_map()
        if paths:       _save_detections(paths.detections)

    Parameters
    ----------
    new_det_ids : pass [] to skip recentering (background path, _stream_on_analyse)
    tracks      : pass [] when there are no multi-frame tracks (anomaly path)
    color       : hex colour string for this query
    """
    from core.services import (
        _recenter_on_detections, _save_tracks, _build_map, _save_detections,
    )
    from core.app_state import _state

    if new_det_ids:
        _recenter_on_detections(new_det_ids)
    if tracks:
        _save_tracks(tracks, color)
    _build_map()
    paths = _state.get("mission_paths")
    if paths:
        _save_detections(paths.detections)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline kwargs helper
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_call_kwargs(color: str, actual_alt_m: float | None = None) -> dict:
    """
    Standard kwargs for run_detection_pipeline() and individual stage calls.

    Centralises settings access — callers no longer repeat:
        api_key       = getattr(settings, "REPLICATE_API_KEY", "")
        model_version = getattr(settings, "DETECTOR_REPLICATE_MODEL", "")
        ...

    SAM remains disabled: Replicate's model only accepts center-point prompts,
    which fail on nadir aerial imagery.
    """
    from core.app_state import _state
    return dict(
        api_key       = getattr(settings, "REPLICATE_API_KEY", ""),
        model_version = getattr(settings, "DETECTOR_REPLICATE_MODEL", ""),
        timeout_s     = getattr(settings, "DETECTOR_TIMEOUT_S", 180),
        actual_alt_m  = actual_alt_m
                        if actual_alt_m is not None
                        else _state.get("mean_alt_m", 100.0),
        color         = color,
        sam_segmentor = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming core
# ─────────────────────────────────────────────────────────────────────────────

def stream_detect_and_commit(
    tiles:        list[dict],
    params,
    definition:   str,
    color:        str,
    actual_alt_m: float,
    analyst,
) -> tuple[list, list[str]]:
    """
    Run pipeline stages 1-5 on a tile batch, accumulate into stream_confirmed,
    re-run track_stage on the full session history, commit to state.

    Shared by both streaming paths:
      _stream_on_frame_telem  (foreground tiles — caller owns throttled flush)
      _bg_detect_callback     (background tiles — caller calls flush_to_map)

    Includes GDINO enriched class selection (Tasks 5/6) and per-tile
    confidence capping (Task 1) that were duplicated across both paths.

    Returns (all_tracks, new_det_ids).  Both are empty on no result —
    callers should early-return when all_tracks is empty.

    commit_tracks is called with replace=True so _state["detections"] is
    rebuilt from full accumulated history each call.  Stable sha1 track IDs
    mean no map marker flicker despite the full replacement.
    """
    from ai.detection_pipeline import (
        run_detector_stage, cross_tile_nms,
        sam_refine_stage, vlm_verify_stage,
        geolocate_stage, track_stage,
    )
    from core.app_state import _state

    if not tiles:
        return [], []

    kw = pipeline_call_kwargs(color, actual_alt_m)

    # GDINO enriched classes when Tasks 5/6 are active
    gdino_classes = params.yolo_classes
    if getattr(settings, "GDINO_ENRICHED_QUERY", True):
        enriched = getattr(params, "gdino_classes", [])
        if enriched:
            gdino_classes = enriched

    raw = run_detector_stage(
        tiles         = tiles,
        classes       = gdino_classes,
        confidence    = params.yolo_confidence,
        api_key       = kw["api_key"],
        model_version = kw["model_version"],
        timeout_s     = kw["timeout_s"],
    )
    if not raw:
        return [], []

    candidates = cross_tile_nms(raw)
    candidates = [
        d for d in candidates
        if (d.bbox_tile[2] - d.bbox_tile[0]) >= settings.DETECTION_MIN_BBOX_PX
        and (d.bbox_tile[3] - d.bbox_tile[1]) >= settings.DETECTION_MIN_BBOX_PX
    ]
    if not candidates:
        return [], []

    # Per-tile candidate capping by GDINO confidence (Task 1)
    max_per_tile = getattr(settings, "MAX_CANDIDATES_PER_TILE", 8)
    buckets: dict = defaultdict(list)
    for d in candidates:
        buckets[(d.parent_path, d.tile_x, d.tile_y)].append(d)
    capped = []
    for b in buckets.values():
        b.sort(key=lambda x: x.confidence, reverse=True)
        capped.extend(b[:max_per_tile])
    candidates = capped

    candidates = sam_refine_stage(
        candidates, params.shape_priors, actual_alt_m, None
    )
    if not candidates:
        return [], []

    confirmed = vlm_verify_stage(candidates, params, definition, analyst)
    if not confirmed:
        return [], []

    confirmed = geolocate_stage(confirmed)
    if not confirmed:
        return [], []

    _state.setdefault("stream_confirmed", []).extend(confirmed)
    logger.info(
        "stream_detect: +%d confirmed | session total %d",
        len(confirmed), len(_state["stream_confirmed"]),
    )

    all_tracks  = track_stage(_state["stream_confirmed"], color=color)
    new_det_ids = commit_tracks(all_tracks, color, replace=True)
    return all_tracks, new_det_ids
