"""
core/motion.py — Pipeline 1: ego-motion-compensated motion detection and tracking.
===================================================================================

Algorithm (mirrors tools/test_motion.py)
-----------------------------------------
1.  KLT optical flow on a ground grid → RANSAC homography (prev → curr).
2.  Warp prev frame into curr viewpoint; compute absolute difference.
3.  Feed warped diff to MOG2 — learns per-pixel residual distribution.
    Static scene residual ≈ 0 after compensation; movers leave non-zero diff.
4.  Physics-gated blob detection (area ∝ GSD², aspect ratio < MAX_ASPECT_RATIO).
5.  Greedy pixel-space tracker with dynamic persistence (higher during fast pans).
6.  World-fixed variance test: after WF_MIN_FRAMES hits, check whether the
    track's geo-projected position is fixed in world space.  If variance < threshold
    it is a camera-motion artefact (parallax residual) and is suppressed.

Design constraints
------------------
- No imports from main.py, app_state.py, or any FastHTML layer.
- SRTFrame is the sole telemetry contract (core.video).
- VlmMotionFilter is injected with a TacticalAnalyst; it does not construct one.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from core.video import SRTFrame

logger = logging.getLogger("TAE.Motion")

# ── Track colour palette ──────────────────────────────────────────────────────
_TRACK_COLORS: list[str] = [
    "#fb923c",  # orange-400
    "#f97316",  # orange-500
    "#fbbf24",  # amber-400
    "#f59e0b",  # amber-500
    "#fdba74",  # orange-300
    "#fcd34d",  # amber-300
    "#ef4444",  # red-400
    "#f87171",  # red-300
]

# ── KLT / RANSAC homography constants ────────────────────────────────────────
KLT_GRID_STEP_PX   = 50          # grid spacing at processing resolution
KLT_WIN_SIZE       = (21, 21)    # KLT search window
KLT_MAX_LEVEL      = 4           # pyramid levels
RANSAC_THRESH_PX   = 3.0         # RANSAC inlier pixel threshold
MIN_INLIERS        = 8           # minimum inliers to trust the homography

# ── Blob detection constants ──────────────────────────────────────────────────
MAX_ASPECT_RATIO   = 8.0         # reject very elongated blobs (wires, shadows)
MORPH_KSIZE        = 3           # morphological kernel size (at detect-res)

# ── Tracker constants ─────────────────────────────────────────────────────────
ORPHAN_FRAMES      = 6           # consecutive misses before retiring a track
MATCH_DIST_M       = 6.0         # real-world match gate radius in metres
MATCH_DIST_PX_FALLBACK = 50      # pixel fallback when GSD unavailable
DEFAULT_PERSIST    = 3           # base consecutive hits to confirm a track
MAX_SCENE_SPEED    = 15.0        # px/frame onset of dynamic-persist bonus
PERSIST_SPEED_STEP = 5.0         # px/frame excess per +1 extra persist frame
MAX_EXTRA_PERSIST  = 10          # cap on additional persist frames

# ── World-fixed variance filter constants ─────────────────────────────────────
WF_MIN_FRAMES   = 8              # minimum hits before the test is valid
WF_RETEST_EVERY = 15             # re-test every N additional hits after first
WF_N_SCAN       = 60             # altitude candidates in the h-scan
WF_THRESHOLD_M  = 0.4            # world-space RMS spread (m); below = world-fixed

# ─────────────────────────────────────────────────────────────────────────────
# Helper to interpolate telemetry for a given video timestamp (ms) from the SRT frames.
# ─────────────────────────────────────────────────────────────────────────────
 
def variance_to_color(wf_variance_m: float) -> str:
    """
    Map world-fixed variance to a colour encoding detection confidence.

    wf_variance_m < 0  : test not yet run (< WF_MIN_FRAMES observations)
                         → slate (unknown)
    wf_variance_m = inf: scan returned no valid result
                         → amber (inconclusive)
    wf_variance_m ≥ WF_THRESHOLD_M (surviving tracks only):
        0.4 – 2 m  → yellow  (barely above FP threshold, uncertain)
        2   – 5 m  → orange  (likely genuine mover)
        5   – 12 m → red     (clearly moving relative to ground)
        > 12 m     → bright red (fast / large displacement)
    """
    if wf_variance_m < 0:
        return "#94a3b8"    # slate-400   — not yet tested
    if not math.isfinite(wf_variance_m):
        return "#fbbf24"    # amber-400   — inconclusive scan
    if wf_variance_m < 2.0:
        return "#fcd34d"    # amber-300   — uncertain mover
    if wf_variance_m < 5.0:
        return "#fb923c"    # orange-400  — probable mover
    if wf_variance_m < 12.0:
        return "#ef4444"    # red-400     — confirmed mover
    return "#dc2626"        # red-600     — fast / large mover

# ─────────────────────────────────────────────────────────────────────────────
# MotionTrack — serialisable result unit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionTrack:
    """
    One tracked mover.

    Pixel-space fields (cx, cy, bbox) are at detect-resolution.
    Geo fields (geo_history, camera_history) are in WGS84.
    to_dict() serialises the lat/lon trajectory for _build_map and JSON persistence.
    """
    track_id:    str
    label:       str   = "mover"
    color:       str   = "#fb923c"

    # Tracker state
    cx:          float = 0.0
    cy:          float = 0.0
    bbox:        list  = field(default_factory=list)   # [x,y,w,h] detect-res
    hit_count:   int   = 0
    miss_count:  int   = 0
    confirmed:   bool  = False
    suppressed:  bool  = False   # True = world-fixed artefact

    # Per-hit histories
    px_history:      list = field(default_factory=list)  # [(cx, cy)]
    geo_history:     list = field(default_factory=list)  # [(lat, lon)]
    camera_history:  list = field(default_factory=list)  # [(cam_lat, cam_lon, alt)]
    ts_history:      list = field(default_factory=list)  # [ts_ms]

    # World-fixed diagnostics
    wf_variance_m: float = -1.0
    wf_best_h_m:   float = -1.0

    def to_dict(self) -> dict:
        trajectory = []
        for i, (lat, lon) in enumerate(self.geo_history):
            entry = {"lat": lat, "lon": lon}
            if i < len(self.ts_history):
                entry["ts_ms"] = self.ts_history[i]
            trajectory.append(entry)
        return {
            "track_id":      self.track_id,
            "label":         self.label,
            "color":         self.color,             # palette colour — Monitor panel
            "map_color":     variance_to_color(self.wf_variance_m),  # confidence colour — map
            "wf_variance_m": self.wf_variance_m,    # raw value for tooltip
            "hit_count":     self.hit_count,         # for dot sizing
            "trajectory":    trajectory,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Pure physics / geometry helpers  (ported from tools/test_motion.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gsd(
    alt_m:       float,
    sensor_w_mm: float,
    focal_mm:    float,
    frame_w_px:  int,
) -> float:
    """Ground sampling distance in metres/pixel at detect-resolution."""
    if focal_mm <= 0 or frame_w_px <= 0:
        return 0.05
    return alt_m * (sensor_w_mm / focal_mm) / frame_w_px


def sky_boundary_row(gimbal_pitch_deg: float, frame_h_px: int) -> int:
    """
    Pixel row below which ground content begins.
    Nadir (-90 deg) -> row 0.  Oblique -> excludes sky from KLT grid.
    """
    pitch_abs = abs(gimbal_pitch_deg)
    sky_frac  = max(0.0, (90.0 - pitch_abs) / 90.0) * 0.60
    return int(sky_frac * frame_h_px)


def frame_corners(
    lat:            float,
    lon:            float,
    alt_m:          float,
    gimbal_yaw_deg: float,
    img_w:          int,
    img_h:          int,
    sensor_w_mm:    float,
    focal_mm:       float,
) -> tuple:
    """
    WGS84 ground-plane corners (nw, ne, se, sw) of a nadir camera frame.
    Derives sensor height from aspect ratio — never uses SENSOR_HEIGHT_MM.
    Mirrors SpatialEngine.compute_footprint() exactly.
    """
    ground_w = alt_m * sensor_w_mm / focal_mm
    ground_h = ground_w * (img_h / img_w)
    hw, hh   = ground_w / 2, ground_h / 2

    corners_ned = np.array([
        [ hh, -hw],   # NW
        [ hh,  hw],   # NE
        [-hh,  hw],   # SE
        [-hh, -hw],   # SW
    ])
    yr  = math.radians(gimbal_yaw_deg)
    R   = np.array([[math.cos(yr), -math.sin(yr)],
                    [math.sin(yr),  math.cos(yr)]])
    rot = (R @ corners_ned.T).T

    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat))
    corners = [(lat + n / m_lat, lon + e / m_lon) for n, e in rot]
    return tuple(corners)   # (nw, ne, se, sw)


def pixel_to_geo(
    cx: float, cy: float,
    img_w: int, img_h: int,
    nw: tuple, ne: tuple, se: tuple, sw: tuple,
) -> tuple[float, float]:
    """Bilinear interpolation of a pixel position to WGS84 (lat, lon)."""
    u   = cx / img_w
    v   = cy / img_h
    lat = ((1 - v) * ((1 - u) * nw[0] + u * ne[0]) +
                v   * ((1 - u) * sw[0] + u * se[0]))
    lon = ((1 - v) * ((1 - u) * nw[1] + u * ne[1]) +
                v   * ((1 - u) * sw[1] + u * se[1]))
    return lat, lon


def estimate_homography(
    prev_gray:  np.ndarray,
    curr_gray:  np.ndarray,
    sky_row:    int = 0,
) -> tuple[Optional[np.ndarray], int]:
    """
    KLT optical flow on a uniform ground grid + RANSAC homography.
    Returns (H, n_inliers).  H is None when estimation fails.

    Grid-based keypoints (not feature detectors) handle the large textureless
    regions typical of nadir aerial footage (rooftops, grass, water).
    """
    h, w = prev_gray.shape
    gy, gx = np.mgrid[sky_row:h:KLT_GRID_STEP_PX, 0:w:KLT_GRID_STEP_PX]
    pts0   = (
        np.column_stack([gx.ravel(), gy.ravel()])
        .astype(np.float32)
        .reshape(-1, 1, 2)
    )
    if len(pts0) < MIN_INLIERS:
        return None, 0

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts0, None,
        winSize  = KLT_WIN_SIZE,
        maxLevel = KLT_MAX_LEVEL,
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )
    good = status.ravel() == 1
    if good.sum() < MIN_INLIERS:
        return None, 0

    src = pts0[good].reshape(-1, 2)
    dst = pts1[good].reshape(-1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH_PX)
    n_inliers = int(mask.sum()) if (mask is not None and H is not None) else 0

    if H is None or n_inliers < MIN_INLIERS:
        return None, 0
    return H, n_inliers


def warp_frame(src_gray: np.ndarray, H: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Warp src_gray into the coordinate system defined by H."""
    h, w = out_shape
    return cv2.warpPerspective(
        src_gray, H, (w, h),
        flags      = cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_REPLICATE,
    )


def clean_mask(mask: np.ndarray, ksize: int = MORPH_KSIZE) -> np.ndarray:
    """Morphological open (remove speckle) then close (fill gaps)."""
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_blobs(
    mask:       np.ndarray,
    min_area:   float,
    max_area:   float,
    max_aspect: float = MAX_ASPECT_RATIO,
    morph_k:    int   = MORPH_KSIZE,
) -> list[dict]:
    """
    Extract connected components from the foreground mask and apply
    physics-derived area and aspect-ratio gates.

    Returns list of {cx, cy, bbox:(x,y,w,h), area} dicts in detect-res coords.
    """
    mask = clean_mask(mask, ksize=morph_k)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[dict] = []
    for i in range(1, n):   # label 0 = background
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(min(bw, bh), 1) > max_aspect:
            continue
        blobs.append({
            "cx":   float(centroids[i, 0]),
            "cy":   float(centroids[i, 1]),
            "bbox": (int(stats[i, cv2.CC_STAT_LEFT]),
                     int(stats[i, cv2.CC_STAT_TOP]),
                     bw, bh),
            "area": area,
        })
    return blobs


def world_fixed_variance_test(
    track:  MotionTrack,
    n_scan: int = WF_N_SCAN,
) -> tuple[float, float]:
    """
    NumPy-vectorised world-fixed variance test.

    Scans candidate object altitudes h in [0, alt_camera].  For each h,
    computes the world-plane position of each observation by projecting the
    camera ray down to h and measures spatial variance.  The minimum variance
    across all h is returned.

    A static world-fixed point has near-zero variance at h=0 (ground level).
    A genuine mover has non-zero variance at every candidate altitude.

    Returns (min_variance_m, best_h_m).  Returns (inf, 0) when insufficient data.
    """
    n_geo = len(track.geo_history)
    n_cam = len(track.camera_history)
    if n_geo < WF_MIN_FRAMES or n_cam < WF_MIN_FRAMES:
        return float("inf"), 0.0

    n    = min(n_geo, n_cam)
    geos = track.geo_history[-n:]
    cams = track.camera_history[-n:]

    lat_ref, lon_ref, _ = cams[0]
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat_ref))

    cam_E = np.array([(lo - lon_ref) * m_lon for _, lo, _    in cams])
    cam_N = np.array([(la - lat_ref) * m_lat for la, _, _    in cams])
    gnd_E = np.array([(lo - lon_ref) * m_lon for _, lo       in geos])
    gnd_N = np.array([(la - lat_ref) * m_lat for la, _       in geos])
    alts  = np.array([alt_c                  for _, _, alt_c  in cams])

    alt_max = float(alts.max())
    if alt_max <= 0:
        return float("inf"), 0.0

    def _scan(h_arr: np.ndarray) -> tuple[float, float]:
        valid = (alts[None, :] > h_arr[:, None]) & (alts[None, :] > 0)
        frac  = np.where(
            valid,
            (alts[None, :] - h_arr[:, None]) / alts[None, :],
            0.0,
        )
        W_E = np.where(
            valid,
            cam_E[None, :] + (gnd_E[None, :] - cam_E[None, :]) * frac,
            np.nan,
        )
        W_N = np.where(
            valid,
            cam_N[None, :] + (gnd_N[None, :] - cam_N[None, :]) * frac,
            np.nan,
        )
        n_ok = valid.sum(axis=1)
        with np.errstate(invalid="ignore"):
            var = np.nanvar(W_E, axis=1) + np.nanvar(W_N, axis=1)
        var[n_ok < 3] = np.inf
        i = int(np.argmin(var))
        return float(np.sqrt(var[i])), float(h_arr[i])

    # Coarse scan then one refinement pass
    h_max  = alt_max * 0.95
    h_step = h_max / max(n_scan - 1, 1)
    best_v, best_h = _scan(np.linspace(0.0, h_max, n_scan))
    lo = max(0.0, best_h - h_step)
    hi = min(h_max, best_h + h_step)
    fv, fh = _scan(np.linspace(lo, hi, 21))
    if fv < best_v:
        best_v, best_h = fv, fh
    return best_v, best_h


# ─────────────────────────────────────────────────────────────────────────────
# FrameDiff — warp-compensated foreground extraction
# ─────────────────────────────────────────────────────────────────────────────

class FrameDiff:
    """
    Per-frame warp-compensated foreground detector.

    Each apply() call:
      1. Downscales to detect-resolution (scale factor).
      2. Estimates KLT+RANSAC homography prev->curr.
      3. Warps prev into curr viewpoint.
      4. Feeds warped absolute diff to MOG2.
      5. Returns the raw foreground mask (before physics area gate).

    Callers must run detect_blobs() on the returned mask with per-frame
    GSD-derived area limits.

    Parameters
    ----------
    scale : float
        Resize factor before all CV processing (0.5 = half resolution).
        Annotation and geo-projection use full-resolution coordinates.
    lr : float
        MOG2 learning rate.  0.02 is recommended for moving cameras.
    """

    def __init__(self, scale: float = 0.5, lr: float = 0.02) -> None:
        self._scale = scale
        self._lr    = lr
        self._mog2  = cv2.createBackgroundSubtractorMOG2(
            history       = 500,
            varThreshold  = 16,
            detectShadows = False,
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0

    @property
    def scale(self) -> float:
        return self._scale

    def apply(
        self,
        frame_bgr: np.ndarray,
        sky_row:   int = 0,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int, float]:
        """
        Process one BGR frame.

        Parameters
        ----------
        frame_bgr : Full-resolution BGR frame.
        sky_row   : Pixel row (full-resolution) below which ground begins;
                    used to exclude sky from the KLT grid.

        Returns
        -------
        (fg_mask, H, n_inliers, scene_speed_px)
            fg_mask       : foreground mask at detect-resolution, or None on first frame.
            H             : 3x3 homography (prev->curr), or None if estimation failed.
            n_inliers     : RANSAC inlier count (diagnostic).
            scene_speed_px: translation magnitude in px/frame at detect-resolution.
        """
        full_h, full_w = frame_bgr.shape[:2]
        det_w = max(1, int(full_w * self._scale))
        det_h = max(1, int(full_h * self._scale))

        detect = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray   = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)

        fg_mask:     Optional[np.ndarray] = None
        H:           Optional[np.ndarray] = None
        n_inliers:   int   = 0
        scene_speed: float = 0.0

        if self._prev_gray is not None:
            sky_det = int(sky_row * self._scale)
            H, n_inliers = estimate_homography(self._prev_gray, gray, sky_row=sky_det)

            warped_prev = (
                warp_frame(self._prev_gray, H, (det_h, det_w))
                if H is not None
                else self._prev_gray
            )
            if H is not None:
                scene_speed = math.hypot(float(H[0, 2]), float(H[1, 2]))

            diff = cv2.absdiff(warped_prev, gray)
            raw  = self._mog2.apply(diff, learningRate=self._lr)
            # Keep only definite foreground; drop shadow class (127)
            _, fg_mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)

        self._prev_gray    = gray
        self._frame_count += 1

        logger.debug(
            "FrameDiff #%d: warp=%s inliers=%d speed=%.1fpx",
            self._frame_count, "ok" if H is not None else "fail",
            n_inliers, scene_speed,
        )
        return fg_mask, H, n_inliers, scene_speed

    def reset(self) -> None:
        """Re-initialise — call between segments or on stream restart."""
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False,
        )
        self._prev_gray   = None
        self._frame_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# MotionGeoProjector — frame footprint for geo-projection
# ─────────────────────────────────────────────────────────────────────────────

class MotionGeoProjector:
    """
    Computes the WGS84 footprint of a camera frame and projects pixel
    coordinates to geo coordinates.

    Sensor height is always derived from the actual frame aspect ratio
    (16:9 recording crop of the 4:3 sensor) — never from SENSOR_HEIGHT_MM.
    """

    def __init__(
        self,
        sensor_w_mm: float | None = None,
        focal_mm:    float | None = None,
    ) -> None:
        self._sensor_w = float(sensor_w_mm or settings.SENSOR_WIDTH_MM)
        self._focal    = float(focal_mm    or settings.FOCAL_LENGTH_MM)

    def compute_footprint_tuple(
        self,
        lat:            float,
        lon:            float,
        alt_m:          float,
        gimbal_yaw_deg: float,
        img_w:          int,
        img_h:          int,
    ) -> Optional[tuple]:
        """Returns (nw, ne, se, sw) as (lat, lon) tuples, or None when alt <= 0."""
        if alt_m <= 0:
            return None
        return frame_corners(
            lat            = lat,
            lon            = lon,
            alt_m          = alt_m,
            gimbal_yaw_deg = gimbal_yaw_deg,
            img_w          = img_w,
            img_h          = img_h,
            sensor_w_mm    = self._sensor_w,
            focal_mm       = self._focal,
        )

    def project(
        self,
        cx: float, cy: float,
        img_w: int, img_h: int,
        telem: SRTFrame,
    ) -> tuple[float, float]:
        """Project a full-resolution pixel point to WGS84."""
        footprint = self.compute_footprint_tuple(
            telem.lat, telem.lon, telem.alt_m, telem.gimbal_yaw, img_w, img_h
        )
        if footprint is None:
            return telem.lat, telem.lon
        nw, ne, se, sw = footprint
        return pixel_to_geo(cx, cy, img_w, img_h, nw, ne, se, sw)


# ─────────────────────────────────────────────────────────────────────────────
# MotionTracker — pixel-space greedy tracker with world-fixed filter
# ─────────────────────────────────────────────────────────────────────────────

class MotionTracker:
    """
    Greedy nearest-neighbour tracker with temporal persistence and world-fixed filter.

    Matching is done in pixel space (detect-resolution) using a GSD-derived
    match radius.  Tracks become 'confirmed' after `persist` consecutive hits,
    where persist is raised dynamically during fast camera pans so parallax
    residuals (2-5 frames) never reach the threshold while real movers do.

    After WF_MIN_FRAMES geo observations, the world-fixed variance test runs.
    Tracks whose minimum world-plane variance < wf_threshold_m are suppressed
    — they are parallax artefacts of static structures, not genuine movers.
    """

    def __init__(
        self,
        persist:        int   = DEFAULT_PERSIST,
        wf_threshold_m: float = WF_THRESHOLD_M,
        colors:         Optional[list[str]] = None,
    ) -> None:
        self._persist        = persist
        self._wf_threshold_m = wf_threshold_m
        self._colors         = colors or _TRACK_COLORS
        self._color_idx      = 0
        self._next_id        = 1
        self._active:  list[MotionTrack] = []
        self._retired: list[MotionTrack] = []

    def update(
        self,
        blobs:       list[dict],
        frame_idx:   int,
        ts_ms:       int,
        scene_speed: float,
        gsd_m:       float,
        footprint:   Optional[tuple],
        cam_pos:     Optional[tuple],
        det_to_full: float,
        full_w:      int,
        full_h:      int,
    ) -> list[MotionTrack]:
        """
        Associate blobs to tracks; update geo/camera histories; apply world-fixed test.

        Parameters
        ----------
        blobs       : list of {cx, cy, bbox, area} at detect-resolution.
        frame_idx   : monotonic frame counter.
        ts_ms       : video timestamp in milliseconds.
        scene_speed : translation magnitude px/frame at detect-res (from H).
        gsd_m       : metres/pixel at detect-resolution.
        footprint   : (nw, ne, se, sw) of the FULL-resolution frame, or None.
        cam_pos     : (cam_lat, cam_lon, alt_m) refined camera position, or None.
        det_to_full : full_w / det_w (inverse of PROCESS_SCALE).
        full_w, full_h : full-resolution frame dimensions.

        Returns
        -------
        All currently alive active tracks (confirmed + unconfirmed).
        """
        # Dynamic persistence
        speed_excess      = max(0.0, scene_speed - MAX_SCENE_SPEED)
        extra_persist     = min(int(speed_excess / PERSIST_SPEED_STEP), MAX_EXTRA_PERSIST)
        effective_persist = self._persist + extra_persist

        # GSD-based match radius
        match_px = (MATCH_DIST_M / gsd_m) if gsd_m > 0 else MATCH_DIST_PX_FALLBACK

        unmatched = list(range(len(blobs)))

        # Match blobs to existing tracks
        for track in self._active:
            best_i, best_d = None, float("inf")
            for bi in unmatched:
                d = math.hypot(blobs[bi]["cx"] - track.cx,
                               blobs[bi]["cy"] - track.cy)
                if d < best_d and d < match_px:
                    best_d, best_i = d, bi

            if best_i is not None:
                b = blobs[best_i]
                track.cx         = b["cx"]
                track.cy         = b["cy"]
                track.bbox       = list(b["bbox"])
                track.hit_count += 1
                track.miss_count = 0
                track.px_history.append((b["cx"], b["cy"]))
                track.ts_history.append(ts_ms)
                if not track.confirmed and track.hit_count >= effective_persist:
                    track.confirmed = True
                    logger.debug(
                        "MotionTracker: %s confirmed  frame=%d  hits=%d",
                        track.track_id, frame_idx, track.hit_count,
                    )
                unmatched.remove(best_i)
            else:
                track.miss_count += 1

        # Spawn new tracks for unmatched blobs
        for bi in unmatched:
            b     = blobs[bi]
            color = self._colors[self._color_idx % len(self._colors)]
            self._color_idx += 1
            tid   = f"mt_{self._next_id}"
            self._next_id += 1
            t = MotionTrack(
                track_id   = tid,
                color      = color,
                cx         = b["cx"],
                cy         = b["cy"],
                bbox       = list(b["bbox"]),
                hit_count  = 1,
                px_history = [(b["cx"], b["cy"])],
                ts_history = [ts_ms],
            )
            self._active.append(t)

        # Geo-project new observations
        if footprint is not None and cam_pos is not None:
            nw, ne, se, sw = footprint
            cam_lat, cam_lon, cam_alt = cam_pos

            for track in self._active:
                if len(track.geo_history) < track.hit_count:
                    cx_full = track.cx * det_to_full
                    cy_full = track.cy * det_to_full
                    lat_g, lon_g = pixel_to_geo(
                        cx_full, cy_full, full_w, full_h, nw, ne, se, sw
                    )
                    track.geo_history.append((lat_g, lon_g))
                if len(track.camera_history) < track.hit_count:
                    track.camera_history.append((cam_lat, cam_lon, cam_alt))

        # World-fixed variance test
        for track in self._active:
            n_obs = len(track.geo_history)
            if n_obs < WF_MIN_FRAMES:
                continue
            if not (n_obs == WF_MIN_FRAMES or
                    (n_obs - WF_MIN_FRAMES) % WF_RETEST_EVERY == 0):
                continue

            min_var, best_h = world_fixed_variance_test(track)
            track.wf_variance_m = min_var
            track.wf_best_h_m   = best_h
            was_suppressed      = track.suppressed
            track.suppressed    = (min_var < self._wf_threshold_m)

            if track.suppressed and not was_suppressed:
                logger.debug(
                    "MotionTracker: %s suppressed (world-fixed) var=%.3fm h=%.0fm",
                    track.track_id, min_var, best_h,
                )
            elif was_suppressed and not track.suppressed:
                logger.info(
                    "MotionTracker: %s unsuppressed (now moving) var=%.3fm",
                    track.track_id, min_var,
                )

        # Retire dead tracks
        alive, dead = [], []
        for t in self._active:
            (alive if t.miss_count < ORPHAN_FRAMES else dead).append(t)
        self._retired.extend(dead)
        self._active = alive

        return list(self._active)

    @property
    def confirmed_visible(self) -> list[MotionTrack]:
        """Active, confirmed, non-suppressed tracks."""
        return [t for t in self._active if t.confirmed and not t.suppressed]

    def all_tracks(self) -> list[MotionTrack]:
        """
        All confirmed non-suppressed tracks for map/JSON persistence.
        Includes retired tracks so trajectories persist after a mover leaves frame.
        """
        all_t = self._active + self._retired
        return [t for t in all_t if t.confirmed and not t.suppressed]

    def remove_track(self, track_id: str) -> None:
        """Remove a track — used by VlmMotionFilter to purge false positives."""
        self._active  = [t for t in self._active  if t.track_id != track_id]
        self._retired = [t for t in self._retired if t.track_id != track_id]

    def relabel_track(self, track_id: str, label: str) -> None:
        """Update label after VLM classification."""
        for t in self._active + self._retired:
            if t.track_id == track_id:
                t.label = label

    def reset(self) -> None:
        """Clear all state — call on stream restart."""
        self._active.clear()
        self._retired.clear()
        self._color_idx = 0
        self._next_id   = 1


# ─────────────────────────────────────────────────────────────────────────────
# VlmMotionFilter — optional semantic false-positive gate (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _build_motion_filter_prompt(bboxes: list, img_w: int, img_h: int) -> str:
    entries = "\n".join(
        f"  {i}: bbox=[{b[0]},{b[1]},{b[2]},{b[3]}]"
        for i, b in enumerate(bboxes)
    )
    return (
        f"You are analysing an aerial drone frame ({img_w}x{img_h} px). "
        f"Numbered bounding boxes mark regions where motion was detected.\n\n"
        f"Numbered regions:\n{entries}\n\n"
        f"For EACH region decide:\n"
        f"  confirmed=true  -> genuine moving object: person, vehicle, or animal.\n"
        f"  confirmed=false -> false positive: camera artefact, parallax residual,\n"
        f"                     shadow, vegetation, stationary object, or noise.\n\n"
        f"Return ONLY a JSON array -- no markdown:\n"
        f'[{{"index":0,"confirmed":true,"confidence":0.9,"label":"vehicle","reason":"brief"}},...]\n'
    )


def _is_retryable_motion(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg or "403" in msg


class VlmMotionFilter:
    """
    Semantic filter using a VLM to reject false-positive tracks.
    One OpenRouter call per snapshot interval; all candidates in one call.
    Disabled by default (MOTION_VLM_ENABLED=False in settings).
    """

    def __init__(self, analyst, snapshot_interval: int = 0) -> None:
        self._analyst  = analyst
        self._interval = (
            snapshot_interval
            or getattr(settings, "MOTION_VLM_SNAPSHOT_INTERVAL", 90)
        )

    def should_filter(self, frame_idx: int) -> bool:
        return frame_idx > 0 and (frame_idx % self._interval == 0)

    def filter(
        self,
        frame_bgr:  np.ndarray,
        candidates: list[dict],
    ) -> tuple[set[str], set[str]]:
        if not candidates or frame_bgr is None:
            return set(), set()

        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        for i, cand in enumerate(candidates):
            x1, y1, x2, y2 = (int(v) for v in cand["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (251, 146, 60), 2)
            cv2.putText(annotated, str(i), (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (251, 146, 60), 2)

        bboxes = [cand["bbox"] for cand in candidates]
        prompt = _build_motion_filter_prompt(bboxes, w, h)
        raw    = self._call_vlm(annotated, prompt)

        if raw is None:
            logger.warning("VlmMotionFilter: call failed -- treating all as confirmed")
            return {c["track_id"] for c in candidates}, set()

        confirmed_ids: set[str] = set()
        rejected_ids:  set[str] = set()
        try:
            clean   = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            clean   = re.sub(r"^\s*>+\s*", "", clean)
            results = json.loads(clean)
            indexed = {
                r["index"]: r for r in results
                if isinstance(r.get("index"), int) and 0 <= r["index"] < len(candidates)
            }
            for i, cand in enumerate(candidates):
                tid = cand["track_id"]
                if i not in indexed:
                    confirmed_ids.add(tid)
                    continue
                r = indexed[i]
                if r.get("confirmed", False):
                    confirmed_ids.add(tid)
                    cand["vlm_label"] = r.get("label", "mover")
                else:
                    rejected_ids.add(tid)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("VlmMotionFilter: parse error %s -- treating all confirmed", exc)
            return {c["track_id"] for c in candidates}, set()

        logger.info("VlmMotionFilter: %d confirmed, %d rejected / %d candidates",
                    len(confirmed_ids), len(rejected_ids), len(candidates))
        return confirmed_ids, rejected_ids

    def _call_vlm(self, frame_bgr: np.ndarray, prompt: str) -> Optional[str]:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        @retry(
            retry   = retry_if_exception(_is_retryable_motion),
            wait    = wait_exponential(multiplier=1, min=2, max=30),
            stop    = stop_after_attempt(4),
            reraise = True,
        )
        def _call() -> str:
            response = self._analyst.client.chat.completions.create(
                model    = self._analyst.model_name,
                messages = [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens  = 512,
                temperature = 0.1,
                extra_body  = {"provider": {
                    "order": ["Together", "NovitaAI", "Nebius Token Factory"],
                    "allow_fallbacks": True,
                }},
            )
            return response.choices[0].message.content or ""

        try:
            return _call()
        except Exception as exc:
            logger.warning("VlmMotionFilter: retries exhausted: %s", exc)
            return None