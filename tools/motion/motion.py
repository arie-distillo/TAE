"""
motion.py — Core Motion Detection Library
==========================================
Pure algorithms, data structures, and annotation for the TAE motion pipeline.
No I/O, no threading, no CLI concerns.

Imported by:
  motion_worker.py  — stateful per-frame processor (standalone + TAE)
  motion_driver.py  — CLI driver (standalone only)

TAE integration: drop this file into src/core/ and import from there.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Optional: TAE telemetry parsers ──────────────────────────────────────────
# When running as part of TAE (src/ on path) use the canonical parsers.
# motion_driver.py adds the TAE src/ directory before importing this module.
try:
    from core.video import SRTParser, DJIProtobufParser, SRTFrame
    _HAS_TAE = True
except ImportError:
    _HAS_TAE = False
    SRTParser = DJIProtobufParser = SRTFrame = None  # type: ignore

# ── Optional: OpenAI / OpenRouter (VLM) ──────────────────────────────────────
try:
    import base64 as _b64
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants — override via MotionConfig or by passing values to constructors.
# These serve as fallback defaults when no config is provided.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SENSOR_W_MM   = 6.3
DEFAULT_FOCAL_MM      = 4.5
FALLBACK_ALT_M        = 80.0
FALLBACK_GIMBAL_PITCH = -90.0

DEFAULT_MIN_OBJECT_M  = 0.2
DEFAULT_MAX_OBJECT_M  = 50.0
MAX_ASPECT_RATIO      = 8.0

MOG2_HISTORY          = 150
MOG2_VAR_THRESHOLD    = 20.0

FLOW_PYR_SCALE     = 0.5
FLOW_LEVELS        = 3
FLOW_WINSIZE       = 9
FLOW_ITERATIONS    = 3
FLOW_POLY_N        = 5
FLOW_POLY_SIGMA    = 1.2
DEFAULT_FLOW_THRESHOLD_PX = 2.0

DEFAULT_ISOLATION_RADIUS_M = 0.0
DEFAULT_MAX_RANGE_M        = 0.0

MORPH_KSIZE           = 3

KLT_GRID_STEP_PX      = 50
KLT_WIN_SIZE          = (21, 21)
KLT_MAX_LEVEL         = 4
RANSAC_THRESH_PX      = 3.0
MIN_INLIERS           = 8

DEFAULT_PERSIST       = 16        # ≥ WF_MIN_FRAMES so pre-filter fires at confirmation
ORPHAN_FRAMES         = 6
MATCH_DIST_M          = 6.0
MATCH_DIST_PX_FALLBACK = 50

DEFAULT_MAX_SCENE_SPEED  = 15.0
PERSIST_SPEED_STEP       = 5.0
MAX_EXTRA_PERSIST        = 10
DEFAULT_MIN_DISPLACEMENT = 0.0

WF_MIN_FRAMES    = 8
WF_RETEST_EVERY  = 15
WF_N_SCAN        = 60
WF_THRESHOLD_M   = 0.5            # calibrated: static crane ~0.2–0.4 m, bird ~0.7–1.7 m
WF_RECOVERY_FACTOR = 2.0          # hysteresis: only un-suppress at factor × threshold

VLM_MODEL              = "qwen/qwen-2.5-vl-72b-instruct"
VLM_SNAPSHOT_INTERVAL  = 90
VLM_MAX_DET_PER_CALL   = 25
VLM_MAX_TOKENS         = 4096
VLM_SNAP_LONG_EDGE     = 1280

COL_PENDING   = (0, 200, 255)
COL_ISOLATED  = (255, 220, 0)
COL_HUD       = (240, 240, 240)
COL_HUD_SHD   = (20, 20, 20)
DEFAULT_MIN_CONFIDENCE = 0.70

def confidence_color(conf: float) -> tuple:
    """
    Map confidence [0, 1] to a BGR colour via HSV.
      0.0 → red    — not yet WF-tested, or variance just above threshold
      0.5 → yellow — moderate evidence of independent motion
      1.0 → green  — strong mover (high WF variance, many consistent hits)
    """
    conf  = max(0.0, min(1.0, conf))
    h_cv  = int(60 * conf)        # OpenCV H in [0, 180]: 0=red, 30=yellow, 60=green
    hsv   = np.uint8([[[h_cv, 230, 215]]])
    bgr   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ─────────────────────────────────────────────────────────────────────────────
# GSD and sky-row helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_gsd(
    alt_m: float,
    sensor_w_mm: float,
    focal_mm: float,
    frame_w_px: int,
) -> float:
    """
    Ground sampling distance in metres per pixel at processing resolution.

    Derivation: footprint_w = alt × sensor_w / focal  (pinhole model)
                GSD          = footprint_w / frame_w
    """
    if focal_mm <= 0 or frame_w_px <= 0:
        return 0.05  # 5 cm/px safe fallback
    return alt_m * (sensor_w_mm / focal_mm) / frame_w_px


def sky_boundary_row(gimbal_pitch_deg: float, frame_h_px: int) -> int:
    """
    Pixel row below which ground content begins.

    Nadir (pitch = -90°) → entire frame is ground → row 0.
    Oblique (pitch approaches 0°) → up to 60% of frame may be sky.

    We exclude sky rows from the KLT grid because sky pixels violate the
    flat-ground homography assumption and corrupt the warp estimate.
    """
    pitch_abs = abs(gimbal_pitch_deg)          # 90 = nadir, 0 = horizontal
    sky_frac  = max(0.0, (90.0 - pitch_abs) / 90.0) * 0.60
    return int(sky_frac * frame_h_px)


# ─────────────────────────────────────────────────────────────────────────────
# Geo-projection helpers  (inline mirror of SpatialEngine from core/spatial.py)
# ─────────────────────────────────────────────────────────────────────────────

def frame_corners(
    lat: float, lon: float,
    alt_m: float, gimbal_yaw_deg: float,
    img_w: int, img_h: int,
    sensor_w_mm: float, focal_mm: float,
) -> tuple:
    """
    Compute the WGS84 ground-plane corners (nw, ne, se, sw) of a camera frame.
    Mirrors SpatialEngine.compute_footprint() — same math, no class dependency.

    Assumes flat-earth, nadir projection (pitch = -90°); yaw rotates footprint.
    Consistent with how TAE uses compute_footprint() throughout.

    Returns
    -------
    (nw, ne, se, sw) where each is a (lat, lon) tuple.
    """
    ground_w = alt_m * sensor_w_mm / focal_mm
    # Derive sensor height from aspect ratio (assumes square pixels)
    ground_h = ground_w * (img_h / img_w)
    hw, hh   = ground_w / 2, ground_h / 2

    # Corners in local NED frame — matches spatial.py sign convention exactly
    corners_ned = np.array([
        [ hh, -hw],   # NW: +north, -east
        [ hh,  hw],   # NE: +north, +east
        [-hh,  hw],   # SE: -north, +east
        [-hh, -hw],   # SW: -north, -east
    ])

    yr = math.radians(gimbal_yaw_deg)
    R  = np.array([[math.cos(yr), -math.sin(yr)],
                   [math.sin(yr),  math.cos(yr)]])
    rot = (R @ corners_ned.T).T

    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat))
    corners = [(lat + n / m_lat, lon + e / m_lon) for n, e in rot]
    return tuple(corners)   # (nw, ne, se, sw)


def oblique_frame_corners(
    lat: float, lon: float,
    alt_m: float, gimbal_yaw_deg: float, gimbal_pitch_deg: float,
    img_w: int, img_h: int,
    sensor_w_mm: float, focal_mm: float,
) -> Optional[tuple]:
    """
    Compute the ground-plane intersections of the four image corner rays
    for an oblique camera at ANY gimbal pitch (including nadir).

    The nadir frame_corners() assumes pitch = −90° and produces a rectangle
    ±(alt × sensor/focal)/2 around the drone.  For an oblique shot at −30°
    to −45° pitch, this is completely wrong: the true footprint is a deep
    trapezoid that extends hundreds of metres forward and has very different
    aspect ratios near vs. far.  Using the wrong footprint means pixel_to_geo
    gives nonsense coordinates for any object more than ~56 m from nadir.

    This function traces the actual 3-D ray from the camera through each
    image corner and intersects it with the ground plane (z = 0).

    Camera convention
    -----------------
    gimbal_pitch_deg :
      −90 = nadir (pointing straight down)
      −45 = 45° below horizontal (looking diagonally forward-down)
        0 = horizontal
    gimbal_yaw_deg   : clockwise from north (0 = camera looking north)

    Returns
    -------
    (nw, ne, se, sw) as (lat, lon) tuples in image-corner order
    (TL, TR, BR, BL), which matches the bilinear convention expected by
    pixel_to_geo.  Returns None if any corner ray points above horizontal
    (i.e., the camera is so shallow that part of the image shows sky).
    """
    f_px = focal_mm / sensor_w_mm * img_w   # focal length in pixels

    P = math.radians(gimbal_pitch_deg)       # −π/2 = nadir
    Y = math.radians(gimbal_yaw_deg)

    # Camera axes expressed in world ENU (x = east, y = north, z = up)
    # Optical axis (camera z, pointing forward/down):
    cz_e =  math.cos(P) * math.sin(Y)   # east  (0 at nadir)
    cz_n =  math.cos(P) * math.cos(Y)   # north (0 at nadir)
    cz_u =  math.sin(P)                 # up    (−1 at nadir ← pointing down)

    # Image-right axis (camera x):
    cx_e =  math.cos(Y)
    cx_n = -math.sin(Y)
    cx_u =  0.0

    # Image-down axis (camera y = optical × right, in right-hand sense):
    cy_e = cz_n * cx_u - cz_u * cx_n   # = sin(P) · sin(Y)
    cy_n = cz_u * cx_e - cz_e * cx_u   # = sin(P) · cos(Y)
    cy_u = cz_e * cx_n - cz_n * cx_e   # = −cos(P)

    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat))

    # Image corners in order (TL, TR, BR, BL) = bilinear (nw, ne, se, sw)
    ground_pts = []
    for pu, pv in ((0, 0), (img_w, 0), (img_w, img_h), (0, img_h)):
        # Normalised image-plane offsets
        rx = (pu - img_w / 2.0) / f_px
        ry = (pv - img_h / 2.0) / f_px

        # Ray direction in world ENU
        r_e = rx * cx_e + ry * cy_e + cz_e
        r_n = rx * cx_n + ry * cy_n + cz_n
        r_u = rx * cx_u + ry * cy_u + cz_u

        if r_u >= 0.0:         # ray pointing up or horizontal → no ground hit
            return None
        t   = -alt_m / r_u    # t > 0 since r_u < 0
        ge  = t * r_e          # metres east  of drone nadir
        gn  = t * r_n          # metres north of drone nadir
        ground_pts.append((lat + gn / m_lat, lon + ge / m_lon))

    return tuple(ground_pts)   # (nw, ne, se, sw)


def pixel_ground_range(
    cx: float, cy: float,
    img_w: int, img_h: int,
    alt_m: float,
    gimbal_yaw_deg: float, gimbal_pitch_deg: float,
    sensor_w_mm: float, focal_mm: float,
) -> float:
    """
    Compute the horizontal ground distance (metres) from the camera nadir
    to the ground intersection of the ray through pixel (cx, cy).

    Unlike pixel_to_geo (bilinear within the footprint box), this uses the
    true perspective ray, giving correct distances for oblique cameras at
    any pixel — including near-horizon pixels that project thousands of
    metres from nadir.

    Returns float('inf') if the ray points above horizontal (sky pixels).
    Used by the --max-range gate to suppress far-field blobs.
    """
    f_px = focal_mm / sensor_w_mm * img_w
    P    = math.radians(gimbal_pitch_deg)
    Y    = math.radians(gimbal_yaw_deg)

    cz_e =  math.cos(P) * math.sin(Y)
    cz_n =  math.cos(P) * math.cos(Y)
    cz_u =  math.sin(P)

    cx_e =  math.cos(Y);  cx_n = -math.sin(Y);  cx_u = 0.0

    cy_e = cz_n * cx_u - cz_u * cx_n
    cy_n = cz_u * cx_e - cz_e * cx_u
    cy_u = cz_e * cx_n - cz_n * cx_e

    rx = (cx - img_w / 2.0) / f_px
    ry = (cy - img_h / 2.0) / f_px

    r_e = rx * cx_e + ry * cy_e + cz_e
    r_n = rx * cx_n + ry * cy_n + cz_n
    r_u = rx * cx_u + ry * cy_u + cz_u

    if r_u >= 0.0:
        return float("inf")          # ray above horizontal → sky pixel
    t = -alt_m / r_u
    return math.hypot(t * r_e, t * r_n)   # metres from nadir


def pixel_to_geo(
    cx: float, cy: float,
    img_w: int, img_h: int,
    nw: tuple, ne: tuple, se: tuple, sw: tuple,
) -> tuple:
    """
    Bilinear interpolation of a pixel position to WGS84 (lat, lon).
    Mirrors SpatialEngine._pixel_to_wgs84().
    """
    u   = cx / img_w
    v   = cy / img_h
    lat = ((1 - v) * ((1 - u) * nw[0] + u * ne[0]) +
                v   * ((1 - u) * sw[0] + u * se[0]))
    lon = ((1 - v) * ((1 - u) * nw[1] + u * ne[1]) +
                v   * ((1 - u) * sw[1] + u * se[1]))
    return lat, lon


def geo_to_pixel(
    lat_t: float, lon_t: float,
    img_w: int, img_h: int,
    nw: tuple, ne: tuple, se: tuple, sw: tuple,
    max_iter: int = 12,
    tol:      float = 1e-9,
) -> tuple:
    """
    Inverse bilinear interpolation: WGS84 (lat, lon) → pixel (cx, cy).
    Uses Newton's method; typically converges to machine precision in 3–4 steps.

    Works for any convex quadrilateral, including the yaw-rotated footprints
    produced by frame_corners().
    """
    u, v = 0.5, 0.5   # initial guess: centre of frame

    for _ in range(max_iter):
        # Forward mapping at current (u, v)
        lat_f = ((1 - v) * ((1 - u) * nw[0] + u * ne[0]) +
                      v   * ((1 - u) * sw[0] + u * se[0]))
        lon_f = ((1 - v) * ((1 - u) * nw[1] + u * ne[1]) +
                      v   * ((1 - u) * sw[1] + u * se[1]))

        # Jacobian  ∂(lat, lon) / ∂(u, v)
        dlat_du = (1 - v) * (ne[0] - nw[0]) + v * (se[0] - sw[0])
        dlat_dv = (1 - u) * (sw[0] - nw[0]) + u * (se[0] - ne[0])
        dlon_du = (1 - v) * (ne[1] - nw[1]) + v * (se[1] - sw[1])
        dlon_dv = (1 - u) * (sw[1] - nw[1]) + u * (se[1] - ne[1])

        dlat = lat_t - lat_f
        dlon = lon_t - lon_f

        det = dlat_du * dlon_dv - dlat_dv * dlon_du
        if abs(det) < 1e-20:
            break

        # Newton step  (Cramer's rule)
        du = ( dlon_dv * dlat - dlat_dv * dlon) / det
        dv = (-dlon_du * dlat + dlat_du * dlon) / det
        u += du
        v += dv

        if abs(du) < tol and abs(dv) < tol:
            break

    return u * img_w, v * img_h


def world_fixed_variance_test(track, n_scan: int = WF_N_SCAN) -> tuple[float, float]:
    """NumPy-vectorised world-fixed variance test. ~30× faster than pure Python."""
    n_geo = len(track.geo_history);  n_cam = len(track.camera_history)
    if n_geo < WF_MIN_FRAMES or n_cam < WF_MIN_FRAMES:
        return float("inf"), 0.0
    n = min(n_geo, n_cam)
    geos = track.geo_history[-n:];  cams = track.camera_history[-n:]
    lat_ref, lon_ref, _ = cams[0]
    m_lat = 111320.0;  m_lon = 111320.0 * math.cos(math.radians(lat_ref))
    cam_E = np.array([(lo-lon_ref)*m_lon for _,lo,_  in cams])
    cam_N = np.array([(la-lat_ref)*m_lat for la,_,_  in cams])
    gnd_E = np.array([(lo-lon_ref)*m_lon for _,lo    in geos])
    gnd_N = np.array([(la-lat_ref)*m_lat for la,_    in geos])
    alts  = np.array([alt_c              for _,_,alt_c in cams])
    alt_max = float(alts.max())

    def _scan(h_arr):
        valid = (alts[None,:] > h_arr[:,None]) & (alts[None,:] > 0)
        frac  = np.where(valid, (alts[None,:]-h_arr[:,None])/alts[None,:], 0.0)
        W_E   = np.where(valid, cam_E[None,:]+(gnd_E[None,:]-cam_E[None,:])*frac, np.nan)
        W_N   = np.where(valid, cam_N[None,:]+(gnd_N[None,:]-cam_N[None,:])*frac, np.nan)
        n_ok  = valid.sum(axis=1)
        with np.errstate(invalid="ignore"):
            var = np.nanvar(W_E, axis=1) + np.nanvar(W_N, axis=1)
        var[n_ok < 3] = np.inf
        i = int(np.argmin(var))
        return float(np.sqrt(var[i])), float(h_arr[i])

    h_max = alt_max * 0.95;  h_step = h_max / max(n_scan-1, 1)
    best_v, best_h = _scan(np.linspace(0.0, h_max, n_scan))
    lo = max(0.0, best_h-h_step);  hi = min(h_max, best_h+h_step)
    fv, fh = _scan(np.linspace(lo, hi, 21))
    if fv < best_v: best_v, best_h = fv, fh
    return best_v, best_h


def _wf_update_confidence(track, min_var: float, wf_threshold: float) -> None:
    """
    Update track.confidence after a world-fixed variance test.
    Called only for non-suppressed tracks (suppressed tracks keep confidence 0).

    Score components:
      WF variance (65%): ramps from 0 at threshold to 1 at 3× threshold
      Track lifetime (25%): caps at 60 frames confirmed
      Miss penalty (10%): proportion of total life spent as misses
    """
    var_conf  = min(1.0, max(0.0,
                    (min_var - wf_threshold) / max(2.0 * wf_threshold, 1e-9)))
    hit_conf  = min(1.0, track.hit_count / 60.0)
    miss_rate = track.miss_count / max(track.hit_count + track.miss_count, 1)
    track.confidence = 0.65 * var_conf + 0.25 * hit_conf + 0.10 * (1.0 - miss_rate)


# ─────────────────────────────────────────────────────────────────────────────
def estimate_homography(
    prev_gray:  np.ndarray,
    curr_gray:  np.ndarray,
    sky_row:    int = 0,
    grid_step:  int = KLT_GRID_STEP_PX,
    ransac_thr: float = RANSAC_THRESH_PX,
) -> tuple[Optional[np.ndarray], int]:
    """
    Estimate the homography that maps prev_gray into curr_gray's coordinate
    system using KLT optical flow on a uniform ground grid + RANSAC.

    Grid-based keypoints (not feature detectors) are used because aerial
    nadir footage has large textureless regions (rooftops, grass, water)
    where corner detectors produce too few stable points.

    Returns
    -------
    (H, n_inliers)
        H          : 3×3 homography matrix, or None if estimation failed.
        n_inliers  : RANSAC inlier count (diagnostic).
    """
    h, w = prev_gray.shape

    # Build uniform grid below the sky boundary
    gy, gx = np.mgrid[sky_row:h:grid_step, 0:w:grid_step]
    pts0 = (
        np.column_stack([gx.ravel(), gy.ravel()])
        .astype(np.float32)
        .reshape(-1, 1, 2)
    )

    if len(pts0) < MIN_INLIERS:
        return None, 0

    # KLT track from prev → curr
    pts1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts0, None,
        winSize=KLT_WIN_SIZE,
        maxLevel=KLT_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
    )

    good = status.ravel() == 1
    if good.sum() < MIN_INLIERS:
        return None, 0

    src = pts0[good].reshape(-1, 2)
    dst = pts1[good].reshape(-1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
    n_inliers = int(mask.sum()) if (mask is not None and H is not None) else 0

    if H is None or n_inliers < MIN_INLIERS:
        return None, 0

    return H, n_inliers


def warp_frame(
    src_gray: np.ndarray,
    H: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Warp src_gray into the target coordinate system defined by H."""
    h, w = out_shape
    return cv2.warpPerspective(
        src_gray, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Foreground mask — two modes
# ─────────────────────────────────────────────────────────────────────────────

def foreground_diff(
    warped_prev: np.ndarray,
    curr_gray:   np.ndarray,
    threshold:   int = 25,
) -> np.ndarray:
    """
    Plain absolute difference between warped-prev and current.
    Fast and interpretable; sensitive to scene illumination changes.
    """
    diff = cv2.absdiff(warped_prev, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def foreground_mog2(
    mog2:        cv2.BackgroundSubtractor,
    warped_prev: np.ndarray,
    curr_gray:   np.ndarray,
    lr:          float = 0.02,
) -> np.ndarray:
    """
    Feed the warped absolute difference to MOG2 with a slow learning rate.

    Why the diff instead of raw frames: MOG2's Gaussian mixture model
    was designed for static cameras.  For moving cameras, the background
    changes every frame and MOG2 never converges.  Feeding the
    motion-compensated diff (which is near-zero for static scene content)
    lets MOG2 learn a per-pixel residual distribution and flag only pixels
    whose residual exceeds the learned variance — naturally handling
    illumination drift from cloud shadows and camera auto-exposure.

    lr=0.02 → background model updates slowly; better for rapidly
    moving drone.  Increase to 0.05–0.10 for very slow hover.
    """
    diff = cv2.absdiff(warped_prev, curr_gray)
    raw  = mog2.apply(diff, learningRate=lr)
    # MOG2 returns 255 foreground / 127 shadow / 0 background;
    # treat only definite foreground (drop shadow class)
    _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
    return mask


def _expected_flow_from_H(
    H: np.ndarray,
    shape: tuple,
) -> np.ndarray:
    """
    Compute the dense optical flow field that a purely camera-induced motion
    would produce, given homography H (maps prev-frame pixels → curr-frame).

    For every pixel (x, y) in prev, H tells us where it lands in curr.
    The expected flow vector is simply that landing point minus the origin:
        flow_expected(x, y) = H(x, y) − (x, y)

    Uses cv2.perspectiveTransform for vectorised computation.

    Returns
    -------
    np.ndarray, shape (h, w, 2), dtype float32
        flow_expected[:,:,0] = dx  (horizontal expected displacement)
        flow_expected[:,:,1] = dy  (vertical   expected displacement)
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts    = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(h, w, 2)
    expected = np.empty((h, w, 2), dtype=np.float32)
    expected[:, :, 0] = mapped[:, :, 0] - xx
    expected[:, :, 1] = mapped[:, :, 1] - yy
    return expected


def foreground_flow(
    prev_gray:    np.ndarray,
    curr_gray:    np.ndarray,
    H:            Optional[np.ndarray],
    threshold_px: float,
) -> np.ndarray:
    """
    Dense optical flow residual foreground mask.

    Algorithm
    ---------
    1. Compute Gunnar-Farneback dense optical flow (prev → curr): measures the
       true per-pixel motion in the image, unaffected by illumination or
       auto-exposure differences that fool frame differencing.
    2. Compute the flow that camera motion alone would produce (from H).
    3. Subtract: residual = actual_flow − camera_flow.
    4. Threshold the residual magnitude.

    Why this beats frame-differencing for oblique shots with 3-D structure
    -----------------------------------------------------------------------
    Frame differencing lights up the *silhouette* of every elevated structure
    (building, crane) — the entire edge — creating large blobs because the
    warp cannot perfectly align texture across depth discontinuities.

    Optical flow residual instead measures *how fast each pixel moves* relative
    to what the camera motion predicts.  A 50 m crane at 80 m altitude with a
    5 m/s pan has a parallax residual of ≈ (5 × 50/80) / GSD / fps ≈ 1.8 px/frame.
    A bird at 5 m/s has ≈ 5 / GSD / fps ≈ 2.9 px/frame.  Threshold at 2.0 px
    separates them cleanly.  A static ground-plane point has 0 residual.

    Parameters
    ----------
    H            : 3×3 homography (prev → curr), or None (no compensation).
    threshold_px : residual magnitude below which a pixel is background.
                   Set via --flow-threshold; see FLOW_THRESHOLD_PX for guidance.
    """
    flow_actual = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow       = None,
        pyr_scale  = FLOW_PYR_SCALE,
        levels     = FLOW_LEVELS,
        winsize    = FLOW_WINSIZE,
        iterations = FLOW_ITERATIONS,
        poly_n     = FLOW_POLY_N,
        poly_sigma = FLOW_POLY_SIGMA,
        flags      = 0,
    )   # (h, w, 2) — (dx, dy) per pixel

    residual = (flow_actual - _expected_flow_from_H(H, prev_gray.shape)
                if H is not None else flow_actual)

    magnitude = np.sqrt(residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2)
    return (magnitude > threshold_px).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray, ksize: int = MORPH_KSIZE) -> np.ndarray:
    """Morphological open+close with caller-supplied kernel size.
    ksize is derived from detect resolution so small targets are not erased.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Blob detection (physics-gated)
# ─────────────────────────────────────────────────────────────────────────────

def detect_blobs(
    mask:       np.ndarray,
    min_area:   float,
    max_area:   float,
    max_aspect: float = MAX_ASPECT_RATIO,
    morph_k:    int   = MORPH_KSIZE,
) -> tuple[list[dict], int]:
    """
    Extract connected components from the foreground mask and apply
    physics-derived area + aspect-ratio gates.

    Returns
    -------
    (blobs, n_raw)
        blobs  : list of {"cx", "cy", "bbox", "area"} dicts that passed all gates
        n_raw  : total connected components before any gating (excludes background)

    n_raw - len(blobs) = blobs silently discarded by area / aspect gate.
    A large gap here (e.g. 200 raw, 0 passed) means the area gate is too tight
    for the objects of interest — lower --min-object or increase --max-object.
    """
    mask = clean_mask(mask, ksize=morph_k)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_raw = n - 1   # exclude background label

    blobs: list[dict] = []
    for i in range(1, n):   # label 0 is background
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > max_aspect:
            continue  # wire, road marking, tree shadow
        blobs.append({
            "cx":   float(centroids[i, 0]),
            "cy":   float(centroids[i, 1]),
            "bbox": (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                bw, bh,
            ),
            "area": area,
        })

    return blobs, n_raw


def apply_isolation_gate(
    blobs:     list[dict],
    radius_px: float,
) -> list[dict]:
    """
    Proximity exclusion filter: discard any blob that has at least one
    other blob within radius_px pixels (Euclidean centroid distance).

    Only blobs with NO neighbour within the radius pass — i.e. blobs that
    appear in isolation in the current frame.

    Rationale
    ---------
    Construction machinery, clustered workers, and crane assemblies generate
    *multiple* simultaneous blobs within 1–10 m of each other.  A lone flying
    object (bird, UAV) generates a *single* blob with no nearby companions.
    Discarding clustered blobs removes the bulk of construction-site false
    positives without touching isolated targets.

    Parameters
    ----------
    blobs     : list of blob dicts with 'cx' and 'cy' keys (proc-resolution).
    radius_px : exclusion radius in pixels.  0 or negative → no filtering.

    Returns
    -------
    Subset of blobs where every surviving blob has no neighbour within radius_px.
    Order is preserved.
    """
    if radius_px <= 0 or len(blobs) < 2:
        return blobs

    isolated: list[dict] = []
    for i, b in enumerate(blobs):
        has_neighbour = any(
            i != j and
            math.hypot(b["cx"] - other["cx"], b["cy"] - other["cy"]) <= radius_px
            for j, other in enumerate(blobs)
        )
        if not has_neighbour:
            isolated.append(b)
    return isolated
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionTrack:
    """Mutable state for one tracked motion blob."""
    track_id:    int
    cx:          float         # centroid x at processing resolution
    cy:          float         # centroid y at processing resolution
    bbox:        tuple         # (x, y, w, h) at processing resolution
    hit_count:   int  = 1
    miss_count:  int  = 0
    confirmed:   bool = False
    suppressed:  bool = False  # True = classified as world-fixed → hidden from output
    is_isolated: bool = True   # True = no other confirmed track within isolation radius
    history:     list = field(default_factory=list)    # (cx, cy) per hit
    geo_history: list = field(default_factory=list)    # (lat_g, lon_g) per hit — ground projection
    camera_history: list = field(default_factory=list) # (lat_c, lon_c, alt_c) per hit — camera pos
    # Diagnostics populated by world_fixed_variance_test
    wf_variance_m: float = -1.0   # minimum world-frame variance (m); -1 = not yet tested
    wf_best_h_m:   float = -1.0   # altitude (m) that minimised variance
    suppressed_by: str   = ""     # "world_fixed" | "range" | "vlm" | ""
    confidence:    float = 0.0    # [0, 1] — certainty this is a genuine mover (updated by WF test)
    # VLM classification support
    confirm_frame_idx:  int   = -1   # frame index when this track was first confirmed
    confirm_bbox_full:  tuple = ()   # full-res bbox (x,y,w,h) at confirmation

    def match_and_update(self, blob: dict) -> None:
        self.cx        = blob["cx"]
        self.cy        = blob["cy"]
        self.bbox      = blob["bbox"]
        self.hit_count += 1
        self.miss_count = 0
        self.history.append((blob["cx"], blob["cy"]))

    def mark_miss(self) -> None:
        self.miss_count += 1

    @property
    def alive(self) -> bool:
        return self.miss_count < ORPHAN_FRAMES

    @property
    def total_life(self) -> int:
        return self.hit_count + self.miss_count


class MotionTracker:
    """
    Greedy nearest-neighbour tracker with temporal consistency gate.

    Blobs are matched to open tracks by centroid Euclidean distance.
    A track becomes 'confirmed' after `persist` consecutive hits.
    A track is retired after ORPHAN_FRAMES consecutive misses.
    """

    def __init__(self, match_dist_px: float, persist: int) -> None:
        self._tracks:   list[MotionTrack] = []
        self._retired:  list[MotionTrack] = []
        self._next_id:  int = 1
        self.match_dist_px = match_dist_px
        self.persist       = persist

    def update(
        self,
        blobs:              list[dict],
        frame_idx:          int   = 0,
        effective_persist:  int   = 0,    # 0 = use self.persist
        min_displacement_px: float = 0.0, # 0 = disabled
    ) -> list[MotionTrack]:
        """
        Associate blobs to tracks; return all currently alive tracks.

        effective_persist:
            Override the base persist threshold for this frame.  Used to raise
            the confirmation bar during fast camera motion so that short-lived
            parallax residuals (3–5 frames) never reach the threshold while a
            real moving object (20+ frames) still does.

        min_displacement_px:
            Minimum Euclidean distance a track's centroid must have moved from
            its origin before it can be confirmed.  Filters fixed-world-point
            residuals that appear at a constant pixel location during a
            steady-speed camera pan (zero velocity in the warped frame).
            0.0 = disabled (default; always confirm at persist threshold).
        """
        persist = effective_persist if effective_persist > 0 else self.persist
        unmatched = list(range(len(blobs)))

        for track in self._tracks:
            if not track.alive:
                continue
            best_i, best_d = None, float("inf")
            for bi in unmatched:
                d = math.hypot(
                    blobs[bi]["cx"] - track.cx,
                    blobs[bi]["cy"] - track.cy,
                )
                if d < best_d and d < self.match_dist_px:
                    best_d, best_i = d, bi

            if best_i is not None:
                track.match_and_update(blobs[best_i])
                if track.hit_count >= persist:
                    if min_displacement_px > 0 and len(track.history) >= 2:
                        # Displacement from first recorded position to current
                        dx = track.history[-1][0] - track.history[0][0]
                        dy = track.history[-1][1] - track.history[0][1]
                        if math.hypot(dx, dy) >= min_displacement_px:
                            track.confirmed = True
                        # else: keep accumulating — track will confirm once it moves enough
                    else:
                        track.confirmed = True
                unmatched.remove(best_i)
            else:
                track.mark_miss()

        # Always spawn new tracks for unmatched blobs.
        # Discrimination of real objects vs. residuals is handled by effective_persist
        # and min_displacement_px, not by suppressing spawning.
        for bi in unmatched:
            b = blobs[bi]
            t = MotionTrack(
                track_id=self._next_id,
                cx=b["cx"], cy=b["cy"],
                bbox=b["bbox"],
                history=[(b["cx"], b["cy"])],
            )
            self._next_id += 1
            self._tracks.append(t)

        # Retire dead tracks
        alive, dead = [], []
        for t in self._tracks:
            (alive if t.alive else dead).append(t)
        self._retired.extend(dead)
        self._tracks = alive

        return alive

    @property
    def confirmed(self) -> list[MotionTrack]:
        return [t for t in self._tracks if t.confirmed]

    @property
    def all_confirmed_ever(self) -> list[MotionTrack]:
        return [t for t in (self._tracks + self._retired) if t.confirmed]


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry loading + interpolation
# ─────────────────────────────────────────────────────────────────────────────
def load_telemetry(video_path: Path, srt_path: Optional[Path]) -> list:
    """
    Load SRT / protobuf telemetry; return list of SRTFrame (may be empty).
    """
    if not _HAS_TAE:
        log.warning("core.video not importable — running without telemetry. "
                    "Add src/ to PYTHONPATH or run from project root.")
        return []

    # 1. Explicit sidecar
    if srt_path and srt_path.exists():
        log.info("SRT sidecar: %s", srt_path)
        return SRTParser().parse(srt_path)

    # 2. Auto-discover next to video
    for cand in [
        video_path.with_suffix(".SRT"),
        video_path.with_suffix(".srt"),
        video_path.parent / (video_path.stem + ".SRT"),
    ]:
        if cand.exists():
            log.info("Auto-discovered SRT: %s", cand)
            return SRTParser().parse(cand)

    # 3. Embedded protobuf
    log.info("No SRT found — probing embedded djmd telemetry …")
    frames = DJIProtobufParser().parse(video_path)
    if frames:
        log.info("Embedded telemetry: %d frames", len(frames))
        return frames

    log.warning("No telemetry found — using fallback alt=%.0fm, nadir assumed",
                FALLBACK_ALT_M)
    return []


def interpolate_telem(frames: list, ms: int):
    """Return linearly-interpolated SRTFrame at video position `ms`."""
    if not frames:
        return None
    if ms <= frames[0].timestamp_ms:
        return frames[0]
    if ms >= frames[-1].timestamp_ms:
        return frames[-1]

    # Binary search for bracketing pair
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
        lat          = f0.lat          + t * (f1.lat          - f0.lat),
        lon          = f0.lon          + t * (f1.lon          - f0.lon),
        alt_m        = f0.alt_m        + t * (f1.alt_m        - f0.alt_m),
        gimbal_pitch = f0.gimbal_pitch + t * (f1.gimbal_pitch - f0.gimbal_pitch),
        gimbal_yaw   = f0.gimbal_yaw   + t * (f1.gimbal_yaw   - f0.gimbal_yaw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
def annotate(
    frame:             np.ndarray,
    tracks:            list[MotionTrack],
    scale:             float,
    frame_idx:         int,
    frame_ms:          int,
    alt_m:             float,
    gsd_cm:            float,
    n_inliers:         int,
    mode:              str,
    warp_ok:           bool,
    scene_speed:       float = 0.0,
    fg_fraction:       float = 0.0,
    effective_persist: int   = 3,
    isolation_radius_px: float = 0.0,
    live_fps:            float = 0.0,
    wf_threshold:        float = 0.5,    # used for per-track confidence colour
    min_confidence:      float = 0.70,   # tracks below this are not drawn
) -> np.ndarray:
    """
    Draw bounding boxes, track trails, and HUD onto a full-resolution copy.
    All bbox / centroid coordinates are stored at processing resolution and
    scaled back to full resolution here via (1 / scale).

    Only confirmed tracks with confidence ≥ min_confidence are drawn.
    Tracks below the threshold are still tracked and written to JSON.
    """
    out      = frame.copy()
    inv_s    = 1.0 / scale
    trail_n  = 12   # history points to draw per track

    for t in tracks:
        if t.suppressed:
            continue                  # world-fixed residual — don't draw
        if t.confirmed and t.confidence < min_confidence:
            continue                  # below operator confidence threshold

        x, y, w, h = t.bbox

        fx  = int(x * inv_s);  fy  = int(y * inv_s)
        fw  = int(w * inv_s);  fh  = int(h * inv_s)

        if t.confirmed:
            col = confidence_color(t.confidence)
            # High-confidence isolated tracks get the special highlight
            if t.is_isolated and isolation_radius_px > 0 and t.confidence >= 0.4:
                col = COL_ISOLATED
        else:
            col = COL_PENDING
        thk = 2 if t.confirmed else 1

        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), col, thk)

        # Label: track id, hit count, confidence %  (confirmed only)
        if t.confirmed:
            conf_pct = int(t.confidence * 100)
            label = f"T{t.track_id}  {t.hit_count}f  {conf_pct}%"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            lx = max(0, fx)
            ly = max(th + 6, fy - 3)
            cv2.rectangle(
                out, (lx, ly - th - 4), (lx + tw + 6, ly + 1), col, -1
            )
            cv2.putText(
                out, label, (lx + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_HUD_SHD, 1, cv2.LINE_AA
            )

        # Motion trail
        if len(t.history) > 1:
            pts = [
                (int(p[0] * inv_s), int(p[1] * inv_s))
                for p in t.history[-trail_n:]
            ]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                fade_col = tuple(int(c * alpha) for c in col)
                cv2.line(out, pts[i - 1], pts[i], fade_col, 1)

    # ── HUD overlay ──────────────────────────────────────────────────────────
    warp_str = f"inliers={n_inliers}  spd={scene_speed:.1f}px" if warp_ok else "WARP FAIL"
    n_visible    = sum(1 for t in tracks if not t.suppressed and t.confirmed
                       and t.confidence >= min_confidence)
    n_suppressed = sum(1 for t in tracks if t.suppressed)
    n_isolated   = sum(1 for t in tracks if not t.suppressed and t.confirmed
                       and t.is_isolated and t.confidence >= min_confidence)
    isolation_str = (f"  isolated={n_isolated}" if isolation_radius_px > 0 else "")
    fps_str = f"   proc {live_fps:.1f} fps" if live_fps > 0 else ""
    hud_lines = [
        f"frame {frame_idx:05d}   t={frame_ms/1000:.2f}s   mode={mode}{fps_str}",
        f"alt {alt_m:.0f}m   gsd {gsd_cm:.1f} cm/px",
        f"warp {warp_str}   fg={fg_fraction*100:.1f}%   persist={effective_persist}f",
        f"tracks: {len(tracks)} alive   {n_visible} movers   {n_suppressed} suppressed{isolation_str}",
    ]
    for i, line in enumerate(hud_lines):
        y_pos = 22 + i * 22
        cv2.putText(out, line, (9, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_HUD_SHD, 3, cv2.LINE_AA)
        cv2.putText(out, line, (9, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_HUD,     1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# VLM classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vlm_motion_prompt(detections: list[dict], frame_w: int, frame_h: int) -> str:
    """
    Build a prompt asking the VLM to classify each numbered detection as a
    genuine independently-moving object or a false alarm on a static structure.

    The model sees a drone video frame with numbered green bounding boxes already
    drawn on it, plus this text description of each box.
    """
    cands = "\n".join(
        f"  {d['index']}: bbox=[{d['bbox'][0]},{d['bbox'][1]},"
        f"{d['bbox'][2]},{d['bbox'][3]}]"
        for d in detections
    )
    return (
        f"You are analyzing an aerial drone video frame (oblique camera, not nadir).\n"
        f"Frame size: {frame_w}×{frame_h} px.  Top-left is (0,0).\n\n"
        f"A motion detector found the following candidate regions, drawn as numbered\n"
        f"green boxes on the frame.  MANY ARE FALSE ALARMS caused by camera movement\n"
        f"over static structures (buildings, cranes, scaffolding, rooftops, trees).\n\n"
        f"Candidates:\n{cands}\n\n"
        f"For EACH candidate decide:\n"
        f"  confirmed: true  → genuinely independently-moving object\n"
        f"                     (flying bird, aircraft, UAV, moving vehicle, person)\n"
        f"  confirmed: false → false alarm on a static structure\n"
        f"                     (building edge, crane, scaffold, roof, tree, road)\n\n"
        f"Visual discriminators:\n"
        f"• Box overlaps or touches a visible structural element → false alarm\n"
        f"• Box in open sky / clear airspace, no nearby structure → genuine mover\n"
        f"• Box at the silhouette edge of a building or crane → false alarm\n"
        f"• Isolated blob in an open area between structures → genuine mover\n\n"
        f"Return ONLY a JSON list, one entry per candidate in index order:\n"
        f"[\n"
        f"  {{\"index\": 0, \"confirmed\": true,  \"confidence\": 0.9, "
        f"\"reason\": \"isolated blob in sky\"}},\n"
        f"  {{\"index\": 1, \"confirmed\": false, \"confidence\": 0.95}},\n"
        f"  ...\n"
        f"]\n"
        f"Rules:\n"
        f"- confirmed entries MUST include a brief reason.\n"
        f"- rejected entries: omit reason to save tokens.\n"
        f"- Be strict: when uncertain, reject.\n"
        f"- No markdown, no text outside the JSON array."
    )


def vlm_classify_visible_movers(
    snapshots:   list[tuple],       # (frame_idx, np.ndarray BGR snapshot at snap scale)
    snap_scale:  float,             # snapshot_dim / full_frame_dim
    all_tracks:  list,              # all confirmed MotionTrack objects
    api_key:     str,
    model:       str = VLM_MODEL,
) -> int:
    """
    Classify all visible (non-suppressed) confirmed tracks using the VLM.

    Each track is sent to the VLM once, in the snapshot frame closest to its
    first confirmation.  Detections are sent in batches of VLM_MAX_DET_PER_CALL
    per API call.  Tracks the VLM classifies as 'not confirmed' (i.e. false alarms
    on static structures) are suppressed in-place.

    Returns the number of tracks newly suppressed by VLM.
    """
    if not _HAS_OPENAI:
        log.warning("VLM classify skipped — 'openai' package not installed.")
        return 0
    if not snapshots:
        log.warning("VLM classify skipped — no snapshots collected.")
        return 0

    client = _OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key  = api_key,
        default_headers = {
            "HTTP-Referer": "https://github.com/arie-distillo/TAE",
            "X-Title": "TAE motion test",
        },
    )

    # Index snapshots by frame_idx for fast lookup
    snap_by_idx = {s[0]: s[1] for s in snapshots}
    snap_indices = sorted(snap_by_idx.keys())

    def nearest_snap(frame_idx: int) -> Optional[np.ndarray]:
        if not snap_indices:
            return None
        best = min(snap_indices, key=lambda s: abs(s - frame_idx))
        return snap_by_idx[best], best

    # Gather pending tracks (visible, not already VLM-processed)
    pending = [
        t for t in all_tracks
        if not t.suppressed
        and t.confirm_frame_idx >= 0
        and t.confirm_bbox_full
    ]
    if not pending:
        log.info("VLM: no visible movers with confirmation data to classify.")
        return 0

    log.info("VLM: classifying %d visible movers in batches of %d …",
             len(pending), VLM_MAX_DET_PER_CALL)

    # Group by nearest snapshot index so each batch shares a frame context
    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for t in pending:
        _, sidx = nearest_snap(t.confirm_frame_idx)
        groups[sidx].append(t)

    total_suppressed = 0
    total_calls      = 0

    for sidx, group_tracks in sorted(groups.items()):
        snap_img = snap_by_idx[sidx]
        sh, sw   = snap_img.shape[:2]

        # Sub-batch the group
        for batch_start in range(0, len(group_tracks), VLM_MAX_DET_PER_CALL):
            batch = group_tracks[batch_start : batch_start + VLM_MAX_DET_PER_CALL]

            # Draw numbered boxes on a copy of the snapshot
            disp = snap_img.copy()
            detections = []
            for det_idx, t in enumerate(batch):
                x, y, w, h = t.confirm_bbox_full
                # Scale full-res bbox to snapshot coords
                x1 = int(x * snap_scale);  y1 = int(y * snap_scale)
                x2 = int((x+w) * snap_scale); y2 = int((y+h) * snap_scale)
                cv2.rectangle(disp, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(disp, str(det_idx), (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                            cv2.LINE_AA)
                detections.append({
                    "index": det_idx,
                    "track_id": t.track_id,
                    "bbox": [x1, y1, x2, y2],
                })

            prompt  = _vlm_motion_prompt(detections, sw, sh)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                log.warning("VLM: failed to encode snapshot frame — skipping batch.")
                continue

            b64 = _b64.standard_b64encode(buf.tobytes()).decode()

            try:
                resp = client.chat.completions.create(
                    model    = model,
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    max_tokens  = VLM_MAX_TOKENS,
                    temperature = 0.1,
                    extra_body  = {
                        "provider": {
                            "order": ["Together", "NovitaAI", "Nebius Token Factory"],
                            "allow_fallbacks": True,
                        }
                    },
                )
                raw = resp.choices[0].message.content or ""
                total_calls += 1
            except Exception as exc:
                log.warning("VLM API call failed (snap=%d batch=%d): %s",
                            sidx, batch_start // VLM_MAX_DET_PER_CALL, exc)
                continue

            # Parse response
            try:
                import re as _re
                clean   = _re.sub(r"^```json\s*|\s*```$", "", raw.strip(),
                                  flags=_re.MULTILINE)
                results = json.loads(clean)
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("VLM parse error: %s\nRaw: %s", exc, raw[:300])
                continue

            # Apply decisions
            result_by_idx = {int(r.get("index", -1)): r for r in results
                             if isinstance(r, dict)}
            for det_idx, t in enumerate(batch):
                r = result_by_idx.get(det_idx)
                if r is None:
                    log.debug("VLM: no result for T%d (index %d)", t.track_id, det_idx)
                    continue
                confirmed = bool(r.get("confirmed", True))
                conf      = float(r.get("confidence", 0.5))
                reason    = r.get("reason", "")
                if not confirmed:
                    t.suppressed    = True
                    t.suppressed_by = "vlm"
                    total_suppressed += 1
                    log.debug("  ✗ VLM T%d suppressed (conf=%.2f)", t.track_id, conf)
                else:
                    log.info("  ✓ VLM T%d confirmed  conf=%.2f  %s",
                             t.track_id, conf, reason)

    log.info("VLM: %d API calls, %d tracks suppressed as artifacts.",
             total_calls, total_suppressed)
    return total_suppressed


