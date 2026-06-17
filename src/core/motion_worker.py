"""
core/motion_worker.py — MotionDetectionWorker: Pipeline 1 threading harness.
=============================================================================

Fix C: the worker now watches segments_dir for new .mp4 files and processes
each one at FULL frame rate using its own cv2.VideoCapture loop — independent
of VideoSampler's sparse Pipeline 2 sampling.

This decouples Pipeline 1 timing from Pipeline 2 and gives FrameDiff the
consecutive frames it needs for KLT homography estimation.

Thread model
------------
One daemon thread polls segments_dir every POLL_INTERVAL_S seconds.
Newly completed segments are processed in order.  A segment is considered
"complete" once a newer segment file exists (the current segment is still
being written by ffmpeg until then).

Telemetry
---------
srt_frames is supplied as a callable (get_srt_frames) so it can be resolved
lazily after stream_mgr.start() has parsed the djmd stream.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from config import settings, MissionPaths
from core.app_state import _state
from core.motion import (
    FrameDiff, MotionGeoProjector, MotionTracker, VlmMotionFilter,
    compute_gsd, sky_boundary_row, detect_blobs,
    DEFAULT_PERSIST, WF_THRESHOLD_M,
)
from core.video import SRTFrame

logger = logging.getLogger("TAE.MotionWorker")

# ── tunables ──────────────────────────────────────────────────────────────────
POLL_INTERVAL_S  = 1.0    # seconds between segment-dir polls
PERSIST_EVERY_N  = 30     # write motion_tracks.json + rebuild map every N frames
MONITOR_MAX_W    = 960    # max width for Monitor panel JPEG

# Physics parameters — match standalone tool defaults
MIN_OBJECT_M     = 0.2    # minimum object dimension in metres
MAX_OBJECT_M     = 50.0   # maximum object dimension in metres
PROCESS_SCALE    = 0.5    # detect-resolution scale (matches standalone tool)


# ─────────────────────────────────────────────────────────────────────────────
# MotionDetectionWorker
# ─────────────────────────────────────────────────────────────────────────────

class MotionDetectionWorker:
    """
    Processes video segments at full frame rate for Pipeline 1 motion detection.

    Lifecycle
    ---------
    1. Instantiate once at module level in main.py alongside _bg_worker.
    2. Call start(paths, segments_dir, get_srt_frames, analyst) after stream_mgr.start().
    3. Worker thread polls segments_dir; processes each completed segment.
    4. Call stop() when the stream ends.
    """

    def __init__(self) -> None:
        self._thread:        Optional[threading.Thread] = None
        self._running:       bool  = False
        self._paths:         Optional[MissionPaths] = None
        self._segments_dir:  Optional[Path] = None
        self._get_srt_frames: Optional[Callable] = None

        # Pipeline components
        self._frame_diff:  Optional[FrameDiff]          = None
        self._projector:   Optional[MotionGeoProjector] = None
        self._tracker:     Optional[MotionTracker]       = None
        self._vlm_filter:  Optional[VlmMotionFilter]    = None

        self._frames_processed: int = 0

    # ── public interface ──────────────────────────────────────────────────────

    def start(
        self,
        paths:          MissionPaths,
        segments_dir:   Path,
        get_srt_frames: Optional[Callable] = None,
        analyst         = None,
    ) -> None:
        """
        (Re-)start the worker for a new stream session.

        Parameters
        ----------
        paths         : MissionPaths for the active mission.
        segments_dir  : Directory where ffmpeg writes segment .mp4 files.
        get_srt_frames: Callable returning list[SRTFrame] — called lazily so
                        it resolves after stream_mgr.start() has parsed telemetry.
        analyst       : TacticalAnalyst singleton (only needed when MOTION_VLM_ENABLED).
        """
        if self._running:
            self.stop()

        self._paths          = paths
        self._segments_dir   = segments_dir
        self._get_srt_frames = get_srt_frames
        self._frames_processed = 0
        self._segment_base_ms: int = 0

        persist = getattr(settings, "MOTION_PERSIST_FRAMES", DEFAULT_PERSIST)
        wf_thr  = getattr(settings, "MOTION_WF_THRESHOLD_M", WF_THRESHOLD_M)

        self._frame_diff = FrameDiff(
            scale = getattr(settings, "MOTION_PROCESS_SCALE", PROCESS_SCALE),
            lr    = getattr(settings, "MOTION_LEARNING_RATE", 0.02),
        )
        self._projector = MotionGeoProjector()
        self._tracker   = MotionTracker(
            persist        = persist,
            wf_threshold_m = wf_thr,
        )

        vlm_enabled = getattr(settings, "MOTION_VLM_ENABLED", False)
        if vlm_enabled and analyst is not None:
            self._vlm_filter = VlmMotionFilter(analyst=analyst)
            logger.info("MotionWorker: VLM filter enabled")
        else:
            self._vlm_filter = None

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

    # ── segment watcher loop ──────────────────────────────────────────────────

    def _loop(self) -> None:
        processed_segments: set[str] = set()

        while self._running:
            time.sleep(POLL_INTERVAL_S)
            if not self._segments_dir or not self._segments_dir.exists():
                continue

            # Collect all .mp4 segments, sorted by name (chronological)
            segs = sorted(self._segments_dir.glob("*.mp4"))
            if len(segs) < 2:
                # Need at least 2 to know the first is complete
                continue

            # All segments except the last one are complete (ffmpeg is still
            # writing the newest file)
            complete = segs[:-1]

            for seg_path in complete:
                name = seg_path.name
                if name in processed_segments:
                    continue
                try:
                    self._process_segment(seg_path)
                except Exception as exc:
                    logger.error("MotionWorker: segment error %s: %s",
                                 name, exc, exc_info=True)
                processed_segments.add(name)

        logger.debug("MotionWorker: loop exited")

    # ── segment processing ────────────────────────────────────────────────────

    def _process_segment(self, seg_path: Path) -> None:
        """Read every frame from one completed segment and run Pipeline 1."""
        srt_frames = self._get_srt_frames() if self._get_srt_frames else []

        cap = cv2.VideoCapture(str(seg_path))
        if not cap.isOpened():
            logger.warning("MotionWorker: cannot open %s", seg_path.name)
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        logger.info(
            "MotionWorker: processing %s (%.0f fps)", seg_path.name, src_fps,
        )

        # Reset warp state between segments — frames in a new segment are not
        # temporally adjacent to the previous segment's frames in general.
        self._frame_diff.reset()

        # H-accumulation for refined camera position (world-fixed filter accuracy)
        cum_cam_delta_E = 0.0
        cum_cam_delta_N = 0.0
        cam_lat_ref:  Optional[float] = None
        cam_lon_ref:  Optional[float] = None

        frame_local_idx = 0

        while self._running:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            video_abs_ms = self._segment_base_ms + int(frame_local_idx * 1000 / src_fps)
            telem = _interpolate_telem(srt_frames, video_abs_ms)

            try:
                self._process_one_frame(
                    frame_bgr,
                    telem,
                    video_abs_ms,
                    src_fps,
                    cum_cam_delta_E,
                    cum_cam_delta_N,
                    cam_lat_ref,
                    cam_lon_ref,
                )

                # Update H-accumulated camera offset for next frame
                # (set after process_one_frame so it uses the H from this frame)
                H         = self._last_H
                gsd_m     = self._last_gsd_m
                if H is not None and gsd_m > 0:
                    cum_cam_delta_E -= float(H[0, 2]) * gsd_m
                    cum_cam_delta_N += float(H[1, 2]) * gsd_m

                if (cam_lat_ref is None and telem is not None
                        and telem.alt_m > 0 and telem.lat != 0.0):
                    cam_lat_ref = telem.lat
                    cam_lon_ref = telem.lon

            except Exception as exc:
                logger.error("MotionWorker frame error: %s", exc, exc_info=True)

            frame_local_idx += 1

        cap.release()
        self._segment_base_ms += int(frame_local_idx * 1000 / src_fps)
        
        logger.info(
            "MotionWorker: %s done  %d frames  %d confirmed track(s)",
            seg_path.name,
            frame_local_idx,
            len([t for t in self._tracker.all_tracks()]),
        )

    # ── per-frame processing ──────────────────────────────────────────────────

    # Store H and gsd_m as instance vars so _process_segment can read them
    # without changing the signature of _process_one_frame.
    _last_H:     Optional[object] = None
    _last_gsd_m: float = 0.0

    def _process_one_frame(
        self,
        frame_bgr:        object,
        telem:            Optional[SRTFrame],
        ts_ms:            int,
        src_fps:          float,
        cum_cam_delta_E:  float,
        cum_cam_delta_N:  float,
        cam_lat_ref:      Optional[float],
        cam_lon_ref:      Optional[float],
    ) -> None:
        import numpy as _np  # already imported at module level via motion; alias for clarity
        full_h, full_w = frame_bgr.shape[:2]

        alt_m        = telem.alt_m        if telem else 80.0
        gimbal_pitch = telem.gimbal_pitch if telem else -90.0
        gimbal_yaw   = telem.gimbal_yaw   if telem else 0.0

        # GSD at detect-resolution
        det_scale  = self._frame_diff.scale
        det_w      = max(1, int(full_w * det_scale))
        gsd_m      = compute_gsd(alt_m, settings.SENSOR_WIDTH_MM,
                                  settings.FOCAL_LENGTH_MM, det_w)
        self._last_gsd_m = gsd_m

        # Sky boundary for KLT grid exclusion
        sky_row = sky_boundary_row(gimbal_pitch, full_h)

        # ── Fix A: warp-compensated foreground mask ───────────────────────────
        fg_mask, H, n_inliers, scene_speed = self._frame_diff.apply(frame_bgr, sky_row)
        self._last_H = H

        if fg_mask is None:
            # First frame — no foreground; advance counter only
            self._frames_processed += 1
            _state["motion_frame_count"] = self._frames_processed
            return

        # Physics area gate (re-computed every frame from current GSD)
        min_dim  = getattr(settings, "MOTION_MIN_OBJECT_M", MIN_OBJECT_M) / gsd_m
        max_dim  = getattr(settings, "MOTION_MAX_OBJECT_M", MAX_OBJECT_M) / gsd_m
        min_area = max(4.0, min_dim ** 2)
        max_area = max_dim ** 2

        blobs = detect_blobs(fg_mask, min_area, max_area)

        # ── Fix B: refined camera position using H-accumulated offset ─────────
        footprint: Optional[tuple] = None
        cam_pos:   Optional[tuple] = None

        if cam_lat_ref is not None and telem is not None and alt_m > 0:
            m_lat = 111320.0
            m_lon = 111320.0 * math.cos(math.radians(cam_lat_ref))
            lat_c = cam_lat_ref + cum_cam_delta_N / m_lat
            lon_c = cam_lon_ref + cum_cam_delta_E / m_lon
            cam_pos   = (lat_c, lon_c, alt_m)
            footprint = self._projector.compute_footprint_tuple(
                lat_c, lon_c, alt_m, gimbal_yaw, full_w, full_h,
            )
        elif telem is not None and alt_m > 0:
            cam_pos   = (telem.lat, telem.lon, alt_m)
            footprint = self._projector.compute_footprint_tuple(
                telem.lat, telem.lon, alt_m, gimbal_yaw, full_w, full_h,
            )

        det_to_full = 1.0 / det_scale

        # ── Tracker update (pixel-space matching + world-fixed filter) ────────
        active_tracks = self._tracker.update(
            blobs        = blobs,
            frame_idx    = self._frames_processed,
            ts_ms        = ts_ms,
            scene_speed  = scene_speed,
            gsd_m        = gsd_m,
            footprint    = footprint,
            cam_pos      = cam_pos,
            det_to_full  = det_to_full,
            full_w       = full_w,
            full_h       = full_h,
        )

        # ── VLM filter (optional checkpoint) ─────────────────────────────────
        if (self._vlm_filter
                and self._vlm_filter.should_filter(self._frames_processed)):
            visible = self._tracker.confirmed_visible
            candidates = []
            for t in visible:
                if t.bbox and len(t.bbox) >= 4:
                    x, y, w, h = t.bbox
                    # Scale bbox to full-resolution for VLM context
                    candidates.append({
                        "track_id": t.track_id,
                        "bbox": [
                            int(x * det_to_full), int(y * det_to_full),
                            int((x+w) * det_to_full), int((y+h) * det_to_full),
                        ],
                        "label": t.label,
                    })
            if candidates:
                confirmed_ids, rejected_ids = self._vlm_filter.filter(
                    frame_bgr, candidates
                )
                for tid in rejected_ids:
                    self._tracker.remove_track(tid)
                for cand in candidates:
                    tid = cand["track_id"]
                    vlm_label = cand.get("vlm_label")
                    if tid in confirmed_ids and vlm_label and vlm_label != "mover":
                        self._tracker.relabel_track(tid, vlm_label)

        # ── Update _state ─────────────────────────────────────────────────────
        all_tracks = self._tracker.all_tracks()
        _state["motion_tracks"]      = {t.track_id: t.to_dict() for t in all_tracks}
        _state["motion_last_frame"]  = self._annotate_and_encode(
            frame_bgr, active_tracks, det_to_full
        )

        self._frames_processed      += 1
        _state["motion_frame_count"] = self._frames_processed

        logger.debug(
            "MotionWorker #%d: %d blobs  %d active  %d confirmed",
            self._frames_processed, len(blobs), len(active_tracks),
            len(self._tracker.confirmed_visible),
        )

        # ── Periodic persistence + map rebuild ────────────────────────────────
        if self._frames_processed % PERSIST_EVERY_N == 0:
            self._persist_and_rebuild()

    def _annotate_and_encode(
        self,
        frame_bgr:     object,
        active_tracks: list,
        det_to_full:   float,
    ) -> Optional[bytes]:
        """Draw confirmed non-suppressed tracks on the frame; return JPEG bytes."""
        annotated = frame_bgr.copy()

        for track in active_tracks:
            if not track.confirmed or track.suppressed:
                continue
            if not track.bbox or len(track.bbox) < 4:
                continue
            x, y, w, h = track.bbox
            x1 = int(x * det_to_full);  y1 = int(y * det_to_full)
            x2 = int((x+w) * det_to_full); y2 = int((y+h) * det_to_full)

            color_bgr = _hex_to_bgr(track.color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            label = f"{track.label[:12]}  {track.hit_count}f"
            for thick, col in ((3, (0, 0, 0)), (1, color_bgr)):
                cv2.putText(annotated, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, thick)

        full_h, full_w = frame_bgr.shape[:2]
        if full_w > MONITOR_MAX_W:
            scale     = MONITOR_MAX_W / full_w
            annotated = cv2.resize(
                annotated,
                (MONITOR_MAX_W, int(full_h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def _persist_and_rebuild(self) -> None:
        try:
            from core.services import _save_motion_tracks, _build_map
            if self._paths and hasattr(self._paths, "motion_tracks"):
                _save_motion_tracks(self._paths.motion_tracks)
            _build_map()
        except Exception as exc:
            logger.warning("MotionWorker: persist/rebuild failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to OpenCV (B, G, R)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except (ValueError, IndexError):
        return (60, 146, 251)   # fallback orange


def _interpolate_telem(frames: list, ms: int) -> Optional[SRTFrame]:
    """Linear interpolation of SRTFrame list at a given video timestamp."""
    if not frames:
        return None
    if ms <= frames[0].timestamp_ms:
        return frames[0]
    if ms >= frames[-1].timestamp_ms:
        return frames[-1]
    lo, hi = 0, len(frames) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if frames[mid].timestamp_ms <= ms:
            lo = mid
        else:
            hi = mid
    f0, f1 = frames[lo], frames[hi]
    dt = f1.timestamp_ms - f0.timestamp_ms
    if dt == 0:
        return f0
    t = (ms - f0.timestamp_ms) / dt
    from dataclasses import replace
    return replace(
        f0,
        timestamp_ms = ms,
        lat          = f0.lat          + t * (f1.lat          - f0.lat),
        lon          = f0.lon          + t * (f1.lon          - f0.lon),
        alt_m        = f0.alt_m        + t * (f1.alt_m        - f0.alt_m),
        gimbal_pitch = f0.gimbal_pitch + t * (f1.gimbal_pitch - f0.gimbal_pitch),
        gimbal_yaw   = f0.gimbal_yaw   + t * (f1.gimbal_yaw   - f0.gimbal_yaw),
    )