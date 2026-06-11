"""
ai/query_handlers.py  —  Route-level detection logic extracted from main.py.
==============================================================================
Handler functions called by the /query route.  Returns plain Python data;
main.py is the only place that constructs _msg() / _msg_html() elements.

Callers (main.py /query route):

    # object_detection
    tracks, err = handle_object_detection(message, params, color, analyst, db)
    if err:
        return user_bubble, _msg(err, "sys")
    new_det_ids = commit_tracks(tracks, color)
    flush_to_map(new_det_ids, tracks, color)
    return user_bubble, _msg_html(_build_obj_reply(tracks, message, color))

    # anomaly_detection
    det_ids, reply_text, use_html = handle_anomaly_query(
        message, params, color, analyst, _main_get_search_lib(), db
    )
    response = _msg_html(reply_text) if use_html else _msg(reply_text, "sys")
    return user_bubble, response

Import hierarchy — no imports from main.py, no FastHTML imports:
    main.py
      └─ query_handlers.py
           ├─ ai/detection_pipeline.py   (run_detection_pipeline)
           ├─ ai/detection_session.py    (pipeline_call_kwargs, flush_to_map)
           ├─ ai/anomaly.py              (VocabularyBuilder, score_segments)
           ├─ core/segment_store.py      (SegmentStore)
           └─ core/app_state.py         (_state)
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

logger = logging.getLogger("TAE.QueryHandlers")


# ─────────────────────────────────────────────────────────────────────────────
# Object detection
# ─────────────────────────────────────────────────────────────────────────────

def handle_object_detection(
    message:  str,
    params,           # ObjectDetectionParams
    color:    str,
    analyst,          # TacticalAnalyst instance (module-level in main.py)
    db,               # TacticalDatabase instance (module-level in main.py)
) -> tuple[list, str | None]:
    """
    Run the full 6-stage pipeline for a manual /query or auto-analysis call.

    Returns (tracks, error).
    On success: tracks is the track list, error is None.
    On early exit: tracks is [], error is a ⚠️ string for _msg(..., "sys").
    Reply formatting (colored dot, frame count) stays in main.py.
    """
    from ai.detection_pipeline import run_detection_pipeline
    from ai.detection_session import pipeline_call_kwargs

    all_tiles = db.get_all_tiles()
    if not all_tiles:
        return [], "⚠️  No tiles in index. Upload imagery first."

    tracks = run_detection_pipeline(
        params         = params,
        all_tiles      = all_tiles,
        original_query = message,
        analyst        = analyst,
        **pipeline_call_kwargs(color),
    )
    return tracks, None


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def handle_anomaly_query(
    message:    str,
    params,           # AnomalyDetectionParams
    color:      str,
    analyst,          # TacticalAnalyst instance
    search_lib,       # SearchLibrarian — caller passes _main_get_search_lib()
    db,               # TacticalDatabase instance
) -> tuple[list[str], str, bool]:
    """
    Handle anomaly_detection intent:
    CLIP scene retrieval → SAM segments (SegmentStore) → CLIP scoring → VLM verify.

    Moved from main.py._handle_anomaly_query with these changes:
      • No FastHTML imports; returns data not HTMX elements.
      • user_bubble removed — main.py adds it at the call-site.
      • _main_get_search_lib() replaced by search_lib parameter.
      • _recenter_on_detections + _build_map replaced by flush_to_map()
        (also adds _save_detections, which the original omitted — fix).
      • Spurious track.detections / track._speed_ms references removed
        (anomaly path has no Track objects — was a silent copy-paste bug).

    Returns (new_det_ids, reply_text, use_html).
      use_html=True  → main.py wraps reply_text in _msg_html() (colored dot)
      use_html=False → main.py wraps reply_text in _msg(..., "sys") (plain)
    """
    from ai.anomaly import VocabularyBuilder, score_segments
    from ai.detection_session import flush_to_map
    from core.app_state import _state
    from core.segment_store import SegmentStore

    try:
        paths = _state.get("mission_paths")
        if not paths:
            return [], "⚠️  No active mission.", False

        seg_db_path = getattr(paths, "segments_db_path",
                              getattr(paths, "segments_db", None))
        if not seg_db_path or not Path(str(seg_db_path)).exists():
            return [], (
                "⚠️  No segment index for this mission. "
                "Enable 'anomaly_detection' intent before uploading imagery "
                "so SAM2 segments are computed at ingest time."
            ), False

        vocab = VocabularyBuilder(search_lib)
        store = SegmentStore(str(seg_db_path))

        clip_query = f"aerial drone nadir overhead view: {message}"
        q_vec      = search_lib.encode_text(clip_query)
        candidates = db.semantic_search(q_vec, limit=20, frames_to_return=8)

        if not candidates:
            return [], "No candidate frames found for anomaly search.", False

        scored = score_segments(
            store       = store,
            frame_paths = [c["parent_path"] for c in candidates],
            vocab       = vocab,
            query       = message,
            top_n       = 5,
        )

        if not scored:
            return [], f"No anomalies detected for: <i>{message}</i>", False

        new_det_ids: list[str] = []
        for seg in scored:
            if seg.crop is None:
                continue

            verify = analyst.verify_detection(
                image          = seg.crop,
                criteria       = params.vlm_verification_criteria,
                report_fields  = params.vlm_reporting_fields,
                original_query = message,
            )
            if not verify.get("confirmed"):
                continue

            det_id = uuid.uuid4().hex[:10]
            cand_match = next(
                (c for c in candidates if c["parent_path"] == seg.frame_path),
                None,
            )
            lat = cand_match["lat"] if cand_match else 0.0
            lon = cand_match["lon"] if cand_match else 0.0

            # Anomaly detections have no Track — no trajectory, track_id,
            # is_multiangle, source_count, is_moving, speed_ms fields.
            _state["detections"][det_id] = {
                "lat":              lat,
                "lon":              lon,
                "label":            message,
                "color":            color,
                "confirmed":        True,
                "img_urls":         [],
                "gsd":              "—",
                "bbox":             seg.bbox,
                "source":           Path(seg.frame_path).name,
                "parent_path":      seg.frame_path,
                "tile_x":           seg.bbox[0] if seg.bbox else 0,
                "tile_y":           seg.bbox[1] if seg.bbox else 0,
                "tile_w":           (seg.bbox[2] - seg.bbox[0]) if seg.bbox else 640,
                "tile_h":           (seg.bbox[3] - seg.bbox[1]) if seg.bbox else 640,
                "gdino_confidence": 0.0,
                "vlm_confidence":   round(float(verify.get("confidence", 0.0)), 4),
                "vlm_reason":       "",
                "vlm_report":       verify.get("report", {}),
            }
            new_det_ids.append(det_id)

        # Recenter + rebuild + persist (original only did recenter + rebuild;
        # _save_detections was missing — now added via flush_to_map).
        flush_to_map(new_det_ids, [], color)

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
        return new_det_ids, reply, True

    except Exception as exc:
        logger.error("Anomaly query failed: %s", exc, exc_info=True)
        return [], f"⚠️  Anomaly detection error: {exc}", False
