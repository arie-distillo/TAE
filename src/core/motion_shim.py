"""
core/motion_shim.py — TAE Adapter for MotionDetectionWorker
============================================================
Keeps all TAE-specific concerns (threading, HLS segment watching, _state
updates, JPEG encoding, map persistence) in one place, delegating every
byte of detection logic to MotionProcessor from core.motion_worker.

This file is the ONLY point of coupling between TAE infrastructure and the
motion detection stack.  core/motion.py and core/motion_worker.py remain
identical to the standalone tools and are never touched for TAE integration.

Integration steps
-----------------
1. Copy core/motion.py        — direct drop-in (no changes)
2. Copy core/motion_worker.py — direct drop-in (no changes)
3. Copy core/motion_shim.py   — this file (new)
4. Update src/config.py       — add MOTION_* settings (see motion_config.py)
5. Update main.py line 46:
       from core.motion_worker import MotionDetectionWorker   # old
       from core.motion_shim   import MotionDetectionWorker   # new
   That is the only change required in any existing TAE file.

Public interface of MotionDetectionWorker is identical to the old version:
    __init__()
    start(paths, segments_dir, get_srt_frames, analyst)
    stop()
    frames_processed  (property)
"""

from __future__ import annotations

import sys, importlib
_cm = importlib.import_module('core.motion')
sys.modules.setdefault('motion', _cm)

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from config import settings, MissionPaths
from core.app_state import _state
from core.motion_worker import MotionProcessor
from core.motion import interpolate_telem
from core.video import SRTFrame

logger = logging.getLogger("TAE.MotionWorker")

# ── TAE-side tunables (not in MotionConfig — specific to the streaming harness)
POLL_INTERVAL_S = 1.0   # seconds between segment-dir polls
PERSIST_EVERY_N = 30    # write motion_tracks.json + rebuild map every N frames
MONITOR_MAX_W   = 960   # max width for Monitor panel JPEG


import types

def _cfg_from_settings() -> types.SimpleNamespace:
    """
    Map TAE's MOTION_* settings to the attribute names MotionProcessor expects.
    Returns a SimpleNamespace that duck-types as a MotionConfig.
    """
    s = settings
    cfg = types.SimpleNamespace()
    cfg.SENSOR_W_MM        = getattr(s, "MOTION_SENSOR_W_MM",        6.3)
    cfg.FOCAL_MM           = getattr(s, "MOTION_FOCAL_MM",            4.5)
    cfg.MIN_OBJECT_M       = getattr(s, "MOTION_MIN_OBJECT_M",        0.2)
    cfg.MAX_OBJECT_M       = getattr(s, "MOTION_MAX_OBJECT_M",       50.0)
    cfg.ISOLATION_RADIUS_M = getattr(s, "MOTION_ISOLATION_RADIUS_M",  0.0)
    cfg.MAX_RANGE_M        = getattr(s, "MOTION_MAX_RANGE_M",          0.0)
    cfg.PERSIST_FRAMES     = getattr(s, "MOTION_PERSIST_FRAMES",       16)
    cfg.WF_THRESHOLD_M     = getattr(s, "MOTION_WF_THRESHOLD_M",       0.5)
    cfg.PROCESS_SCALE      = getattr(s, "MOTION_PROCESS_SCALE",        0.5)
    cfg.MIN_CONFIDENCE     = getattr(s, "MOTION_MIN_CONFIDENCE",       0.70)
    cfg.VLM_ENABLED        = getattr(s, "MOTION_VLM_ENABLED",         False)
    cfg.MAX_SCENE_SPEED    = getattr(s, "MOTION_MAX_SCENE_SPEED",      15.0)
    cfg.PERSIST_SPEED_STEP = getattr(s, "MOTION_PERSIST_SPEED_STEP",   5.0)
    return cfg

# ── MotionTrack serialization for _state storage ─────────────────────────────
def _track_to_dict(t) -> dict:
    # Full trajectory — all geo_history points, not just last 50.
    # The bird's flight path spans 200+ frames; truncating to 50 shows only
    # its final hover cluster and makes it invisible on the map.
    trajectory = [
        {"lat": lat, "lon": lon}
        for lat, lon in t.geo_history
    ]

    # Confidence-based map color (mirrors old variance_to_color behaviour).
    # conf=1.00 → #00cc44 green  (genuine mover — bird)
    # conf≥0.60 → #ffaa00 amber  (probable mover)
    # conf<0.60 → #ff6600 orange (borderline / FP — needs VLM to confirm)
    conf = t.confidence
    if conf >= 0.85:
        map_color = "#00cc44"
    elif conf >= 0.60:
        map_color = "#ffaa00"
    else:
        map_color = "#ff6600"

    return {
        "track_id":      t.track_id,
        "hit_count":     t.hit_count,
        "suppressed":    t.suppressed,
        "suppressed_by": t.suppressed_by,
        "confidence":    round(t.confidence, 3),
        "wf_variance_m": round(t.wf_variance_m, 4) if t.wf_variance_m >= 0 else None,
        "map_color":     map_color,
        "trajectory":    trajectory,
    }

class MotionDetectionWorker:
    """
    Processes HLS video segments at full frame rate for Pipeline 1.

    Lifecycle (identical to the old implementation — main.py is unchanged):
        1. Instantiate once at module level.
        2. Call start(paths, segments_dir, get_srt_frames, analyst).
        3. Worker polls segments_dir; processes each completed segment via
           MotionProcessor (core.motion_worker).
        4. Call stop() when the stream ends.
    """

    def __init__(self) -> None:
        self._thread:         Optional[threading.Thread] = None
        self._running:        bool = False
        self._paths:          Optional[MissionPaths] = None
        self._segments_dir:   Optional[Path] = None
        self._get_srt_frames: Optional[Callable] = None
        self._processor:      Optional[MotionProcessor] = None
        self._frames_processed: int = 0
        self._segment_base_ms:  int = 0
        # det_scale and mode stored so _process_segment can pass them through
        self._det_scale: float = 0.5
        self._mode:      str   = "mog2"

    # ── Public interface (unchanged from old MotionDetectionWorker) ───────────

    def start(
        self,
        paths:          MissionPaths,
        segments_dir:   Path,
        get_srt_frames: Optional[Callable] = None,
        analyst         = None,              # kept for API compatibility; unused
    ) -> None:
        if self._running:
            self.stop()

        self._paths          = paths
        self._segments_dir   = segments_dir
        self._get_srt_frames = get_srt_frames
        self._frames_processed = 0
        self._segment_base_ms  = 0

        cfg = _cfg_from_settings()
        self._det_scale = cfg.PROCESS_SCALE   # detect == annotate scale in streaming
        self._mode      = getattr(settings, "MOTION_MODE", "mog2")

        self._processor = MotionProcessor(cfg)

        # Reset _state motion keys
        _state["motion_tracks"]      = {}
        _state["motion_frame_count"] = 0
        _state["motion_last_frame"]  = None
        _state["motion_enabled"]     = True

        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="motion-detect",
        )
        self._thread.start()
        logger.info("MotionDetectionWorker started (segments_dir=%s)", segments_dir)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        _state["motion_enabled"] = False
        logger.info(
            "MotionDetectionWorker stopped (%d frames processed)",
            self._frames_processed,
        )

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    # ── Segment watcher loop (identical logic to old implementation) ──────────

    def _loop(self) -> None:
        processed_segments: set[str] = set()

        while self._running:
            time.sleep(POLL_INTERVAL_S)
            if not self._segments_dir or not self._segments_dir.exists():
                continue

            segs = sorted(self._segments_dir.glob("*.mp4"))
            if len(segs) < 2:
                continue  # need ≥ 2 to know the first is complete

            for seg_path in segs[:-1]:   # all but the one ffmpeg is still writing
                if seg_path.name in processed_segments:
                    continue
                try:
                    self._process_segment(seg_path)
                except Exception as exc:
                    logger.error("MotionWorker segment error %s: %s",
                                 seg_path.name, exc, exc_info=True)
                processed_segments.add(seg_path.name)

        logger.debug("MotionWorker: loop exited")

    # ── Segment processing ────────────────────────────────────────────────────

    def _process_segment(self, seg_path: Path) -> None:
        """Process every frame of one completed HLS segment."""
        srt_frames = self._get_srt_frames() if self._get_srt_frames else []

        # ── Telemetry diagnostic ──────────────────────────────────────────────────
        if not srt_frames:
            logger.warning("MotionWorker: %s — srt_frames EMPTY, WF filter disabled",
                        seg_path.name)
        else:
            logger.info("MotionWorker: %s — %d telem frames, first alt=%.0fm lat=%.4f",
                        seg_path.name, len(srt_frames),
                        srt_frames[0].alt_m, srt_frames[0].lat)
            
        cap = cv2.VideoCapture(str(seg_path))
        if not cap.isOpened():
            logger.warning("MotionWorker: cannot open %s", seg_path.name)
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        logger.info("MotionWorker: processing %s (%.0f fps)",
                    seg_path.name, src_fps)
        
        # For a continuous RTSP stream, HLS segments are temporally contiguous:
        # last frame of seg N and first frame of seg N+1 are a normal consecutive
        # frame pair. MOG2 must carry over — resetting it here caused a ~5s cold-start
        # relearning period per segment boundary where movers were missed and their
        # tracks fragmented (the bird appeared as T1042 instead of T999, with 70 hits
        # instead of 238). prev_gray also carries over so H estimation is correct.
        # True stream restarts are handled by creating a fresh MotionProcessor in start().

        frame_local_idx = 0

        while self._running:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            video_abs_ms = self._segment_base_ms + int(
                frame_local_idx * 1000 / src_fps
            )
            telem = interpolate_telem(srt_frames, video_abs_ms)

            try:
                annotated, tracks = self._processor.process_frame(
                    frame          = frame_bgr,
                    telem          = telem,
                    frame_idx      = self._frames_processed,
                    frame_ms       = video_abs_ms,
                    det_scale      = self._det_scale,
                    mode           = self._mode,
                )

                # ── Update _state ─────────────────────────────────────────────
                all_tracks = [t for t in self._processor.all_tracks_ever() if not t.suppressed]
                _state["motion_tracks"] = {
                    t.track_id: _track_to_dict(t) for t in all_tracks
                }
                _state["motion_last_frame"]  = self._encode_monitor(annotated)
                self._frames_processed      += 1
                _state["motion_frame_count"] = self._frames_processed

                logger.debug(
                    "MotionWorker #%d: %d confirmed  %d visible",
                    self._frames_processed,
                    self._processor.stats()["confirmed_tracks"],
                    self._processor.stats()["visible_movers"],
                )

                if self._frames_processed % PERSIST_EVERY_N == 0:
                    self._persist_and_rebuild()

            except Exception as exc:
                logger.error("MotionWorker frame error: %s", exc, exc_info=True)

            frame_local_idx += 1

        cap.release()
        self._segment_base_ms += int(frame_local_idx * 1000 / src_fps)
        logger.info(
            "MotionWorker: %s done  %d frames  %d confirmed track(s)",
            seg_path.name, frame_local_idx,
            self._processor.stats()["confirmed_tracks"],
        )

    # ── Monitor JPEG encoding ─────────────────────────────────────────────────

    def _encode_monitor(self, frame_bgr) -> Optional[bytes]:
        """
        Resize annotated frame to MONITOR_MAX_W and encode as JPEG.
        The annotated frame from MotionProcessor is already at proc_scale
        (typically 960×540 for 4K input at scale=0.5).
        """
        h, w = frame_bgr.shape[:2]
        if w > MONITOR_MAX_W:
            scale  = MONITOR_MAX_W / w
            out    = cv2.resize(
                frame_bgr,
                (MONITOR_MAX_W, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            out = frame_bgr

        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None


    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_and_rebuild(self) -> None:
        try:
            from core.services import _save_motion_tracks, _build_map
            if self._paths and hasattr(self._paths, "motion_tracks"):
                _save_motion_tracks(self._paths.motion_tracks)
            _build_map()
        except Exception as exc:
            logger.warning("MotionWorker: persist/rebuild failed: %s", exc)
