"""
motion_worker.py — Stateful Per-Frame Motion Processor

Key design: H-accumulated camera position persists across segment boundaries.
See stats() for diagnostic counters.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from motion import (
    FALLBACK_ALT_M, FALLBACK_GIMBAL_PITCH,
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MORPH_KSIZE,
    WF_MIN_FRAMES, WF_RETEST_EVERY,
    MAX_EXTRA_PERSIST, MATCH_DIST_M,
    VLM_SNAPSHOT_INTERVAL, VLM_SNAP_LONG_EDGE,MATCH_DIST_PX_FALLBACK,
    compute_gsd, sky_boundary_row,
    oblique_frame_corners, frame_corners, pixel_to_geo, pixel_ground_range,
    estimate_homography, warp_frame,
    foreground_mog2, foreground_diff, foreground_flow,
    detect_blobs, apply_isolation_gate,
    world_fixed_variance_test, _wf_update_confidence,
    MotionTrack, MotionTracker,
    annotate, DEFAULT_MIN_CONFIDENCE,
    vlm_classify_visible_movers, _HAS_OPENAI,
)

logger = logging.getLogger(__name__)


class MotionProcessor:
    """
    Stateful per-frame motion detection processor.

    Key design: H-accumulated camera position (cum_cam_delta_E/N) and GPS
    anchor (cam_lat_ref/lon_ref) are INSTANCE state, persisted across segment
    boundaries.  reset_segment() resets only the foreground model — geo state
    is never touched.  This is the fix for the TAE bird-detection bug where
    resetting these as local variables in _process_segment() caused coordinate
    frame jumps in camera_history, corrupting the WF variance test.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._tracker = MotionTracker(persist=cfg.PERSIST_FRAMES,
                               match_dist_px=MATCH_DIST_PX_FALLBACK)
        self._mog2:      Optional[cv2.BackgroundSubtractorMOG2] = None
        self._prev_gray: Optional[np.ndarray] = None

        # PERSISTENT ACROSS SEGMENTS — never reset between segments
        self._cum_cam_delta_E: float = 0.0
        self._cum_cam_delta_N: float = 0.0
        self._cam_lat_ref:  Optional[float] = None
        self._cam_lon_ref:  Optional[float] = None

        self._warp_failures = 0
        self._raw_blobs = 0
        self._gated_blobs = 0
        self._isolation_filtered = 0
        self._range_suppressed = 0
        self._seen_track_ids: set = set()
        self._vlm_snapshots: list[tuple] = []
        self._vlm_suppressed = 0
        self._proc_count = 0
        self._frame_times: list[float] = []
        self._t_last: float = time.time()

    def reset_segment(self) -> None:
        """
        Call at every HLS segment boundary.
        Resets foreground model only — H-accumulation and geo anchor persist.
        """
        self._mog2      = None
        self._prev_gray = None

    def process_frame(
        self,
        frame: np.ndarray,
        telem,
        frame_idx: int,
        frame_ms: int,
        det_scale: float = 0.5,
        mode: str = "mog2",
        isolation_radius_m: float = 0.0,
        max_range_m: float = 0.0,
        min_displacement_m: float = 0.0,
        flow_threshold_px: float = 2.0,
        vlm_classify: bool = False,
        vlm_api_key: str = "",
        crops_dir: Optional[Path] = None,
    ) -> tuple[np.ndarray, list]:
        cfg = self._cfg
        self._proc_count += 1
        full_h, full_w = frame.shape[:2]

        det_w = max(1, int(full_w * det_scale))
        det_h = max(1, int(full_h * det_scale))
        det_to_full = 1.0 / det_scale

        proc_scale = getattr(cfg, "PROCESS_SCALE", 0.5)
        proc_w = max(1, int(full_w * proc_scale))
        proc_h = max(1, int(full_h * proc_scale))
        ann_scale = det_scale / proc_scale
        morph_k = max(1, int(MORPH_KSIZE * det_scale / 0.5))

        alt_m        = telem.alt_m        if telem else FALLBACK_ALT_M
        gimbal_pitch = getattr(telem, "gimbal_pitch", FALLBACK_GIMBAL_PITCH) if telem else FALLBACK_GIMBAL_PITCH
        gimbal_yaw   = getattr(telem, "gimbal_yaw", 0.0) if telem else 0.0
        has_telem    = telem is not None and alt_m > 0

        gsd_m  = compute_gsd(alt_m, cfg.SENSOR_W_MM, cfg.FOCAL_MM, det_w)
        gsd_cm = gsd_m * 100.0
        sky_row = sky_boundary_row(gimbal_pitch, det_h)

        min_area = max(4.0, (cfg.MIN_OBJECT_M / max(gsd_m, 1e-9)) ** 2)
        max_area = (cfg.MAX_OBJECT_M / max(gsd_m, 1e-9)) ** 2
        self._tracker.match_dist_px = max(20.0, MATCH_DIST_M / max(gsd_m, 1e-9))
        isolation_radius_px = isolation_radius_m / gsd_m if isolation_radius_m > 0 and gsd_m > 0 else 0.0

        detect = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray   = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        small  = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA) if det_w != proc_w else detect

        fg_mask    = None
        warp_ok    = False
        n_inliers  = 0
        scene_speed = 0.0
        H = None

        if self._prev_gray is not None:
            H, n_inliers = estimate_homography(self._prev_gray, gray, sky_row=sky_row)
            if H is not None:
                warp_ok = True
                scene_speed = math.hypot(float(H[0, 2]), float(H[1, 2]))
                warped_prev = warp_frame(self._prev_gray, H, (det_h, det_w))
                if gsd_m > 0:
                    self._cum_cam_delta_E -= float(H[0, 2]) * gsd_m
                    self._cum_cam_delta_N += float(H[1, 2]) * gsd_m
            else:
                warped_prev = self._prev_gray
                self._warp_failures += 1

            if mode == "mog2":
                if self._mog2 is None:
                    self._mog2 = cv2.createBackgroundSubtractorMOG2(
                        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
                        detectShadows=False)
                fg_mask = foreground_mog2(self._mog2, warped_prev, gray)
            elif mode == "flow":
                fg_mask = foreground_flow(self._prev_gray, gray, H=H if warp_ok else None,
                                          threshold_px=flow_threshold_px)
            else:
                fg_mask = foreground_diff(warped_prev, gray)

        speed_excess      = max(0.0, scene_speed - cfg.MAX_SCENE_SPEED)
        extra_persist     = min(int(speed_excess / cfg.PERSIST_SPEED_STEP), MAX_EXTRA_PERSIST)
        effective_persist = cfg.PERSIST_FRAMES + extra_persist
        min_disp_px = min_displacement_m / gsd_m if min_displacement_m > 0 and gsd_m > 0 else 0.0
        fg_fraction = float(np.count_nonzero(fg_mask)) / fg_mask.size if fg_mask is not None else 0.0

        tracks: list = []
        if fg_mask is not None:
            blobs, n_raw = detect_blobs(fg_mask, min_area, max_area, morph_k=morph_k)
            self._raw_blobs += n_raw
            if isolation_radius_px > 0:
                n_before = len(blobs)
                blobs = apply_isolation_gate(blobs, isolation_radius_px)
                self._isolation_filtered += n_before - len(blobs)
            self._gated_blobs += len(blobs)
            tracks = self._tracker.update(blobs, frame_idx,
                                          effective_persist=effective_persist,
                                          min_displacement_px=min_disp_px)
            for t in self._tracker.confirmed:
                if t.track_id not in self._seen_track_ids:
                    self._seen_track_ids.add(t.track_id)
                    logger.debug("  ✓ T%d confirmed frame=%d alt=%.0fm gsd=%.1fcm hits=%d",
                                 t.track_id, frame_idx, alt_m, gsd_cm, t.hit_count)
                    t.confirm_frame_idx = frame_idx
                    if t.bbox:
                        x, y, w, h = t.bbox
                        t.confirm_bbox_full = (int(x*det_to_full), int(y*det_to_full),
                                               int(w*det_to_full), int(h*det_to_full))
                    if crops_dir and t.bbox:
                        x, y, w, h = t.bbox
                        fx=int(x*det_to_full); fy=int(y*det_to_full)
                        fw=int(w*det_to_full); fh=int(h*det_to_full)
                        crop = frame[max(0,fy):min(full_h,fy+fh), max(0,fx):min(full_w,fx+fw)]
                        if crop.size > 0:
                            cv2.imwrite(str(crops_dir/f"T{t.track_id:04d}_f{frame_idx:05d}.jpg"), crop)
            if vlm_classify and self._proc_count % VLM_SNAPSHOT_INTERVAL == 0:
                snap_scale = min(1.0, VLM_SNAP_LONG_EDGE / max(full_w, full_h))
                sh = int(full_h * snap_scale); sw = int(full_w * snap_scale)
                self._vlm_snapshots.append((frame_idx,
                    cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)))
        else:
            self._tracker.update([], frame_idx)

        # GPS anchor
        if self._cam_lat_ref is None and has_telem and telem.lat != 0.0:
            self._cam_lat_ref = telem.lat
            self._cam_lon_ref = telem.lon

        current_footprint = None
        lat_c_refined = lon_c_refined = None
        if self._cam_lat_ref is not None and has_telem:
            try:
                m_lat_r = 111320.0
                m_lon_r = 111320.0 * math.cos(math.radians(self._cam_lat_ref))
                lat_c_refined = self._cam_lat_ref + self._cum_cam_delta_N / m_lat_r
                lon_c_refined = self._cam_lon_ref + self._cum_cam_delta_E / m_lon_r
                current_footprint = oblique_frame_corners(
                    lat=lat_c_refined, lon=lon_c_refined, alt_m=telem.alt_m,
                    gimbal_yaw_deg=gimbal_yaw, gimbal_pitch_deg=gimbal_pitch,
                    img_w=full_w, img_h=full_h,
                    sensor_w_mm=cfg.SENSOR_W_MM, focal_mm=cfg.FOCAL_MM)
                if current_footprint is None:
                    current_footprint = frame_corners(
                        lat=lat_c_refined, lon=lon_c_refined, alt_m=telem.alt_m,
                        gimbal_yaw_deg=gimbal_yaw, img_w=full_w, img_h=full_h,
                        sensor_w_mm=cfg.SENSOR_W_MM, focal_mm=cfg.FOCAL_MM)
            except Exception:
                current_footprint = None

        if current_footprint is not None and tracks:
            nw_c, ne_c, se_c, sw_c = current_footprint
            for track in tracks:
                if len(track.geo_history) < track.hit_count:
                    cx_full = track.cx * det_to_full
                    cy_full = track.cy * det_to_full
                    lat_g, lon_g = pixel_to_geo(cx_full, cy_full, full_w, full_h,
                                                nw_c, ne_c, se_c, sw_c)
                    if max_range_m > 0 and scene_speed > 1.0:
                        dist_m = pixel_ground_range(cx_full, cy_full, full_w, full_h,
                            telem.alt_m, gimbal_yaw, gimbal_pitch,
                            cfg.SENSOR_W_MM, cfg.FOCAL_MM)
                        if dist_m > max_range_m:
                            if not track.suppressed:
                                track.suppressed = True
                                track.suppressed_by = "range"
                                self._range_suppressed += 1
                            continue
                    track.geo_history.append((lat_g, lon_g))
                if len(track.camera_history) < track.hit_count:
                    track.camera_history.append((lat_c_refined, lon_c_refined, alt_m))

        # In the WF retest loop — restore the sticky gate (identical to baseline):
        _wf_thr = cfg.WF_THRESHOLD_M
        for track in tracks:
            n_obs = len(track.geo_history)
            if n_obs < WF_MIN_FRAMES:
                continue
            if not (n_obs == WF_MIN_FRAMES or
                    (n_obs - WF_MIN_FRAMES) % WF_RETEST_EVERY == 0):
                continue
            # ── once classified world-fixed, never re-evaluate ───────────
            if track.suppressed and track.suppressed_by == "world_fixed":
                continue

            min_var, best_h = world_fixed_variance_test(track)
            track.wf_variance_m = min_var
            track.wf_best_h_m   = best_h

            if min_var < _wf_thr:
                was = track.suppressed
                track.suppressed    = True
                track.suppressed_by = "world_fixed"
                if not was:
                    logger.debug("  ○ T%d world-fixed var=%.3fm h=%.0fm",
                                track.track_id, min_var, best_h)
            else:
                _wf_update_confidence(track, min_var, _wf_thr)

        if isolation_radius_px > 0:
            active_conf = [t for t in tracks if t.confirmed and not t.suppressed]
            for t in active_conf:
                t.is_isolated = not any(
                    other.track_id != t.track_id and
                    math.hypot(t.cx - other.cx, t.cy - other.cy) <= isolation_radius_px
                    for other in active_conf)

        _now = time.time()
        self._frame_times.append(_now - self._t_last)
        self._t_last = _now
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        live_fps = 1.0 / (sum(self._frame_times) / len(self._frame_times)) if self._frame_times else 0.0

        min_conf = getattr(cfg, "MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        annotated = annotate(
            small, tracks, ann_scale,
            frame_idx, frame_ms, alt_m, gsd_cm,
            n_inliers, mode, warp_ok,
            scene_speed, fg_fraction, effective_persist,
            isolation_radius_px=isolation_radius_px,
            live_fps=live_fps,
            wf_threshold=cfg.WF_THRESHOLD_M,
            min_confidence=min_conf,
        )
        self._prev_gray = gray
        return annotated, tracks

    def run_vlm_postprocess(self, snap_scale: float, api_key: str) -> int:
        if not self._vlm_snapshots or not _HAS_OPENAI or not api_key:
            return 0
        n = vlm_classify_visible_movers(self._vlm_snapshots, snap_scale,
                                        self._tracker.all_confirmed_ever, api_key)
        self._vlm_suppressed = n
        return n

    def all_tracks_ever(self) -> list:
        return self._tracker.all_confirmed_ever

    def stats(self) -> dict:
        all_conf = self._tracker.all_confirmed_ever
        n_sup    = sum(1 for t in all_conf if t.suppressed)
        return {
            "confirmed_tracks":   len(self._seen_track_ids),
            "suppressed_tracks":  n_sup,
            "visible_movers":     len(all_conf) - n_sup,
            "vlm_suppressed":     self._vlm_suppressed,
            "warp_failures":      self._warp_failures,
            "blobs_raw":          self._raw_blobs,
            "blobs_passed_gate":  self._gated_blobs,
            "range_suppressed":   self._range_suppressed,
            "isolation_filtered": self._isolation_filtered,
        }
