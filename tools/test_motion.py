#!/usr/bin/env python3
"""
tools/test_motion.py — Standalone motion detection test harness for TAE.

Runs an ego-motion-compensated motion detection pipeline on any DJI drone
video and writes an annotated output video showing detected motion tracks.

Algorithm
---------
1. Decode video at configured FPS (default: all frames, max 30 fps).
2. For each frame: estimate homography (prev → curr) via KLT on a ground
   grid + RANSAC, then warp prev frame into current viewpoint.
3. Compute foreground mask from warped abs-diff (mode=diff) or an adaptive
   MOG2 applied to that diff (mode=mog2 — handles shadow/illumination change).
4. Extract blobs; gate on physics-derived area bounds from GSD + altitude.
5. Associate blobs to tracks (greedy centroid match); promote to "confirmed"
   after PERSIST consecutive hits.
6. Annotate full-resolution frames; write output video + summary JSON.

Usage
-----
    # From the project root:
    python tools/test_motion.py path/to/DJI_0001.MP4

    # With explicit SRT sidecar:
    python tools/test_motion.py DJI_0001.MP4 DJI_0001.SRT

    # Process at 10 fps, half resolution, 5-frame confirmation:
    python tools/test_motion.py DJI_0001.MP4 --fps 10 --scale 0.5 --persist 5

    # Use 3-frame difference instead of MOG2:
    python tools/test_motion.py DJI_0001.MP4 --mode diff

    # Quick smoke test (first 300 source frames, no preview window):
    python tools/test_motion.py DJI_0001.MP4 --max-frames 300 --no-preview

    # Save crops of each newly-confirmed track:
    python tools/test_motion.py DJI_0001.MP4 --save-crops

Options
-------
    --fps N          Target processing frame rate (default: native, capped at 30)
    --scale F        Resize factor for CV processing (default 0.5 = half-res).
                     Annotation video is always written at full resolution.
    --mode {mog2,diff,flow}
                     mog2 (default): MOG2 adaptive threshold on warped diff.
                     diff: plain absolute diff on warped frames.
                     flow: Gunnar-Farneback dense optical flow residual.
                           Best for oblique shots with 3-D structures (buildings,
                           cranes): measures actual pixel velocity and subtracts
                           the camera-motion component, leaving only independently
                           moving objects. Parallax residuals from a 50 m crane at
                           80 m AGL, 5 m/s pan ≈ 1.8 px/frame vs a bird at 5 m/s
                           ≈ 2.9 px/frame — cleanly separable at 2 px threshold.
                           ~3-5× slower than diff; use --fps 10 to compensate.
    --persist N      Consecutive matched frames before a track is confirmed
                     (default 3).
    --min-object M   Minimum object dimension in metres (default 0.5 — person).
    --max-object M   Maximum object dimension in metres (default 50.0 — truck).
    --sensor-w MM    Camera sensor width in mm (default 6.3 — Mavic 3 Enterprise).
    --focal MM       Camera focal length in mm (default 4.5 — Mavic 3 Enterprise).
    --output PATH    Output video path (default: <input>_motion.mp4).
    --no-preview     Skip cv2.imshow window.
    --save-crops     Write JPEG crops of each confirmed track to _crops/.
    --max-frames N   Halt after N source frames (for quick tests).

Output
------
  <input>_motion.mp4   — annotated video (full resolution)
  <input>_motion.json  — summary: track count, warp stats, parameters

Telemetry discovery order
--------------------------
  1. Explicit SRT path (second positional argument)
  2. Auto-discovered <video>.SRT / <video>.srt in same directory
  3. Embedded DJI protobuf djmd stream (via DJIProtobufParser)
  4. Fallback: altitude=80m, gimbal_pitch=-90° (nadir assumed)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Detect GUI availability without triggering Qt platform plugin initialisation.
# Calling cv2.namedWindow() on a system with Qt but no active display causes
# Qt to call abort() — uncatchable.  We instead inspect the build info string
# (no window creation) and additionally check the DISPLAY env var on Linux/Mac.
def _cv2_has_gui() -> bool:
    """
    Return True only if this OpenCV build has a working GUI backend AND
    a display is reachable.  False for opencv-python-headless or for Qt
    builds running without a display (CI, SSH without X forwarding, WSL).
    """
    try:
        info = cv2.getBuildInformation()
        # Walk the GUI: section looking for any enabled backend
        in_gui = False
        gui_enabled = False
        for line in info.splitlines():
            stripped = line.strip()
            if stripped.startswith("GUI:"):
                in_gui = True
                continue
            if in_gui:
                # Section ends at the next top-level (non-indented) heading
                if stripped and not line.startswith(" "):
                    break
                if "YES" in line and any(
                    kw in line for kw in ("QT", "GTK", "Win32", "Cocoa")
                ):
                    gui_enabled = True
                    break

        if not gui_enabled:
            return False  # headless build — no GUI at all

        # Build has GUI support; on Linux/Mac also verify a display exists
        import os, platform
        if platform.system() == "Windows":
            return True   # Windows always has a display when interactive
        return bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    except Exception:
        return False   # safest default on any unexpected error

_CV2_GUI = _cv2_has_gui()

# ── TAE module import (src/ is the sibling of tools/) ─────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

try:
    from core.video import SRTParser, DJIProtobufParser, SRTFrame
    _HAS_TAE = True
except ImportError:
    _HAS_TAE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motion_test")


# ─────────────────────────────────────────────────────────────────────────────
# Tunable defaults  (override via CLI — never hardcode for a specific test case)
# ─────────────────────────────────────────────────────────────────────────────

# Camera (Mavic 3 Enterprise defaults; override with --sensor-w / --focal)
DEFAULT_SENSOR_W_MM   = 6.3
DEFAULT_FOCAL_MM      = 4.5
FALLBACK_ALT_M        = 80.0      # used when telemetry unavailable
FALLBACK_GIMBAL_PITCH = -90.0     # assume nadir when pitch unknown

# Object size bounds in real-world metres (physics-derived blob gates)
DEFAULT_MIN_OBJECT_M  = 0.5       # person (~50 cm wide)
DEFAULT_MAX_OBJECT_M  = 50.0      # large truck / building outline
MAX_ASPECT_RATIO      = 8.0       # reject very elongated blobs (wires, shadows)

# Background subtractor
MOG2_HISTORY          = 150       # frames to build background model
MOG2_VAR_THRESHOLD    = 20.0      # per-pixel variance threshold

# Dense optical flow (Gunnar-Farneback) — used by mode=flow
#
# Flow measures actual pixel VELOCITY; subtracting the homography-predicted
# (camera-motion) component leaves only object-independent motion.
# Elevated static structures (cranes, buildings) produce parallax residuals
# of only ≈ v_camera × height/altitude px/frame — much smaller than a moving
# object's residual — allowing the threshold to separate them cleanly.
#
# FLOW_THRESHOLD_PX : minimum residual flow magnitude (pixels/frame at proc
#   resolution) to classify a pixel as "moving".  Physics check:
#     50 m crane, 80 m alt, 5 m/s pan, 5.8 cm/px GSD, 30 fps
#       → parallax residual ≈ (5 × 50/80) / 0.058 / 30 ≈ 1.8 px/frame
#     Bird at 5 m/s
#       → flow residual ≈ 5 / 0.058 / 30 ≈ 2.9 px/frame
#   Default 2.0 px suppresses crane residuals while flagging fast birds.
#   Lower (1.5) catches slower movers; higher (3.0) reduces false positives
#   in very windy conditions.  Tune with --flow-threshold.
FLOW_PYR_SCALE     = 0.5    # pyramid downscale per level
FLOW_LEVELS        = 3      # pyramid depth — handles up to 2^3 = 8× displacement
FLOW_WINSIZE       = 9      # averaging window; smaller = less spatial blur on small blobs
FLOW_ITERATIONS    = 3      # iterations per pyramid level
FLOW_POLY_N        = 5      # polynomial neighbourhood size (5 or 7)
FLOW_POLY_SIGMA    = 1.2    # Gaussian s.d. for polynomial smoothing
DEFAULT_FLOW_THRESHOLD_PX = 2.0   # residual flow threshold (px/frame, proc-res)

# Morphological cleanup kernel (applied to foreground mask)
MORPH_KSIZE           = 3

# Ego-motion estimation (KLT + RANSAC homography)
KLT_GRID_STEP_PX      = 25        # grid spacing at processing resolution
KLT_WIN_SIZE          = (21, 21)  # KLT window
KLT_MAX_LEVEL         = 4         # pyramid levels
RANSAC_THRESH_PX      = 3.0       # RANSAC inlier pixel threshold
MIN_INLIERS           = 8         # minimum inliers to trust the homography

# Tracker
DEFAULT_PERSIST       = 3         # consecutive hits to confirm a track (base)
ORPHAN_FRAMES         = 6         # consecutive misses before retiring a track
MATCH_DIST_M          = 6.0       # real-world match gate radius (metres)
MATCH_DIST_PX_FALLBACK = 50       # pixel fallback when GSD unavailable

# Dynamic persist — discriminates real movers from parallax residuals.
#
# Core idea: real moving objects (birds, vehicles) produce coherent tracks
# that persist across many frames.  Parallax residuals from fast camera
# motion are short-lived: they appear at world-fixed depth edges and vanish
# when the camera sweeps past or slows.  Rather than suppressing entire frames
# (which would blind the system during fast pans), we raise the confirmation
# threshold when the scene is moving fast.  A residual lasting 3-5 frames
# never reaches the higher bar; a bird present for 20+ frames always does.
#
# Separately, a minimum-displacement check rejects the remaining residual class:
# fixed-world-point artefacts that can appear at the SAME pixel location for
# many frames during a constant-speed pan.  These have near-zero track
# displacement in the warped frame; a real moving object has nonzero velocity.
#
# DEFAULT_MAX_SCENE_SPEED : scene translation (px/frame at proc resolution)
#   above which extra persist frames are added.  15 px/frame ≈ 0.9 m/s ground
#   speed at 5.8 cm/px GSD.  Tune with --max-scene-speed.
#
# PERSIST_SPEED_STEP : px/frame of excess scene speed that adds 1 extra frame.
#   Default 5 → +1 frame per extra 5 px/frame.  At 30 px/frame excess (drone
#   moving ~2.6 m/s above threshold): +6 frames (total persist = 3+6 = 9).
#
# MIN_DISPLACEMENT_M : minimum real-world displacement (metres) a track must
#   show before it can be confirmed.  0 = no check (default).  Set to e.g. 0.5
#   to require half a metre of movement — filters stationary residuals while
#   allowing slow vehicles.  Tune with --min-displacement.
DEFAULT_MAX_SCENE_SPEED  = 15.0   # px/frame → onset of dynamic persist
PERSIST_SPEED_STEP       = 5.0    # px/frame excess per +1 extra persist frame
MAX_EXTRA_PERSIST        = 10     # cap on additional frames
DEFAULT_MIN_DISPLACEMENT = 0.0    # metres; 0 = disabled

# World-fixed point filter — uses DJI telemetry to test whether a track
# is consistent with a STATIC world location.
#
# Algorithm: triangulate the track's best-estimate geo position from its
# N-frame observation history (mean lat/lon), then re-project that position
# into the CURRENT camera frame and compare to the actual blob position.
#
#   reprojection error ≈ 0    → camera motion explains all pixel motion
#                            → world-fixed point (parallax residual) → suppress
#
#   reprojection error large  → blob moved independently of camera
#                            → genuine mover → show
#
# WF_MIN_FRAMES: minimum hits before the test is reliable.  With N=20
#   frames at 30 fps (0.67 s), a 5 m/s bird accumulates ~3.3 m of travel,
#   placing its mean position 1.65 m from its current position → reprojection
#   error ≈ 28 px (at 5.8 cm/px GSD), well above WF_THRESHOLD_PX.
#   A static world point has reprojection error ≈ centroid noise ≈ 2–4 px.
#
# WF_RETEST_EVERY: re-run the test every N hits after the first classification
#   so that a previously-static object (parked vehicle) that begins to move
#   is eventually un-suppressed.
#
# WF_THRESHOLD_PX: at processing resolution.  Below → world-fixed → suppress.
WF_MIN_FRAMES    = 20    # minimum track hits before world-fixed test is valid
WF_RETEST_EVERY  = 15    # re-test every N additional hits after first result
WF_THRESHOLD_PX  = 10.0  # reprojection error (px, proc-res) separating static vs. mover

# Annotation colours  (BGR)
COL_PENDING   = (0, 200, 255)     # yellow  — seen but not yet confirmed
COL_CONFIRMED = (50, 220, 50)     # green   — confirmed track
COL_HUD       = (240, 240, 240)   # white   — HUD text
COL_HUD_SHD   = (20, 20, 20)     # dark    — HUD text shadow


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


def world_fixed_reprojection_error(
    track,
    footprint: tuple,   # (nw, ne, se, sw) at FULL resolution for current frame
    full_w: int,
    full_h: int,
    scale:  float,
) -> float:
    """
    Estimate the reprojection error for the hypothesis
    "this track is a static world-fixed point."

    Algorithm
    ---------
    1. Mean of all geo observations in track.geo_history
       → best-estimate world position (lat_est, lon_est) assuming static.
    2. Project (lat_est, lon_est) into the CURRENT frame using current footprint
       → predicted pixel (cx_pred, cy_pred) at processing resolution.
    3. Euclidean distance between predicted and actual current centroid.

    Interpretation
    --------------
    Small error (< WF_THRESHOLD_PX):
        Camera motion fully explains the blob's pixel trajectory.
        The blob is a world-fixed point (building edge, crane top, terrain).
        → suppress.

    Large error (≥ WF_THRESHOLD_PX):
        The blob has moved independently of the camera between its first
        observed position and its current position.
        → genuine mover — confirm and show.

    Requires at least WF_MIN_FRAMES geo observations to be meaningful.
    Returns float('inf') when geo_history is empty.
    """
    if not track.geo_history:
        return float("inf")

    # Triangulate: simple mean of all geolocated observations
    n       = len(track.geo_history)
    lat_est = sum(g[0] for g in track.geo_history) / n
    lon_est = sum(g[1] for g in track.geo_history) / n

    nw, ne, se, sw = footprint

    # Re-project estimated world position into current frame (full resolution)
    cx_full, cy_full = geo_to_pixel(lat_est, lon_est, full_w, full_h, nw, ne, se, sw)

    # Scale to processing resolution and compare to actual centroid
    cx_pred = cx_full * scale
    cy_pred = cy_full * scale

    return math.hypot(track.cx - cx_pred, track.cy - cy_pred)
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


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological open (remove noise) then close (fill holes)."""
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE)
    )
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
    mask = clean_mask(mask)

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


# ─────────────────────────────────────────────────────────────────────────────
# Motion tracker (temporal consistency gate)
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
    history:     list = field(default_factory=list)    # (cx, cy) per hit
    geo_history: list = field(default_factory=list)    # (lat, lon) per hit

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
# ─────────────────────────────────────────────────────────────────────────────

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
) -> np.ndarray:
    """
    Draw bounding boxes, track trails, and HUD onto a full-resolution copy.
    All bbox / centroid coordinates are stored at processing resolution and
    scaled back to full resolution here via (1 / scale).
    """
    out      = frame.copy()
    inv_s    = 1.0 / scale
    trail_n  = 12   # history points to draw per track

    for t in tracks:
        if t.suppressed:
            continue                  # world-fixed residual — don't draw

        x, y, w, h = t.bbox

        fx  = int(x * inv_s);  fy  = int(y * inv_s)
        fw  = int(w * inv_s);  fh  = int(h * inv_s)
        col = COL_CONFIRMED if t.confirmed else COL_PENDING
        thk = 2 if t.confirmed else 1

        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), col, thk)

        # Label (confirmed only)
        if t.confirmed:
            label = f"T{t.track_id}  {t.hit_count}f"
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
    n_visible    = sum(1 for t in tracks if not t.suppressed and t.confirmed)
    n_suppressed = sum(1 for t in tracks if t.suppressed)
    hud_lines = [
        f"frame {frame_idx:05d}   t={frame_ms/1000:.2f}s   mode={mode}",
        f"alt {alt_m:.0f}m   gsd {gsd_cm:.1f} cm/px",
        f"warp {warp_str}   fg={fg_fraction*100:.1f}%   persist={effective_persist}f",
        f"tracks: {len(tracks)} alive   {n_visible} movers   {n_suppressed} suppressed",
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

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_motion.py",
        description="TAE motion detection test harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", type=Path,
                   help="Input drone video (.mp4)")
    p.add_argument("srt", nargs="?", type=Path, default=None,
                   help="DJI SRT sidecar (auto-discovered if omitted)")

    p.add_argument("--fps", type=float, default=None,
                   help="Target processing frame rate (default: native, ≤30)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Processing resolution scale factor")
    p.add_argument("--mode", choices=["mog2", "diff", "flow"], default="mog2",
                   help="Foreground extraction mode. "
                        "mog2: MOG2 on motion-compensated diff (robust to illumination). "
                        "diff: plain warped-frame absolute diff (fastest). "
                        "flow: Gunnar-Farneback dense optical flow residual (best for "
                        "oblique shots with 3-D structures — eliminates parallax residuals "
                        "from buildings/cranes that flood mog2 and diff with false blobs). "
                        "Default: %(default)s.")
    p.add_argument("--persist", type=int, default=DEFAULT_PERSIST,
                   help="Frames required to confirm a track")
    p.add_argument("--min-object", type=float, default=DEFAULT_MIN_OBJECT_M,
                   dest="min_object",
                   help="Minimum object size in metres")
    p.add_argument("--max-object", type=float, default=DEFAULT_MAX_OBJECT_M,
                   dest="max_object",
                   help="Maximum object size in metres")
    p.add_argument("--sensor-w", type=float, default=DEFAULT_SENSOR_W_MM,
                   dest="sensor_w",
                   help="Camera sensor width in mm")
    p.add_argument("--focal", type=float, default=DEFAULT_FOCAL_MM,
                   help="Camera focal length in mm")
    p.add_argument("--output", type=Path, default=None,
                   help="Output video path")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip real-time preview window")
    p.add_argument("--save-crops", action="store_true",
                   dest="save_crops",
                   help="Save confirmed-track crops to _crops/ directory")
    p.add_argument("--max-frames", type=int, default=None,
                   dest="max_frames",
                   help="Stop after N source frames (for quick tests)")
    p.add_argument("--max-scene-speed", type=float, default=DEFAULT_MAX_SCENE_SPEED,
                   dest="max_scene_speed",
                   help="Scene translation (px/frame at proc resolution) above which "
                        "extra persist frames are added. Default %(default).0f px. "
                        "Lower = stricter during slower pans.")
    p.add_argument("--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD_PX,
                   dest="flow_threshold",
                   help="Residual optical flow magnitude threshold in px/frame (proc-res) "
                        "for --mode flow.  Pixels below this are background; above = mover. "
                        "Physics guide: parallax from a 50 m crane at 80 m AGL, 5 m/s pan "
                        "≈ 1.8 px/frame; a 5 m/s bird ≈ 2.9 px/frame.  "
                        "Default %(default).1f px.  Lower (1.5) catches slower movers; "
                        "higher (3.0) reduces false positives in turbulent conditions.")
    p.add_argument("--min-displacement", type=float, default=DEFAULT_MIN_DISPLACEMENT,
                   dest="min_displacement",
                   help="Minimum real-world displacement (metres) a track must show "
                        "before it is confirmed. 0 = disabled (default). "
                        "E.g. 0.5 rejects fixed-world residuals during steady pans.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_args()

    if not args.video.exists():
        sys.exit(f"[error] Video not found: {args.video}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open: {args.video}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame skip so effective rate ≤ target_fps (and ≤ 30 always)
    target_fps  = min(args.fps or src_fps, 30.0)
    frame_skip  = max(1, round(src_fps / target_fps))
    eff_fps     = src_fps / frame_skip

    # Processing dimensions
    proc_w = max(1, int(full_w * args.scale))
    proc_h = max(1, int(full_h * args.scale))

    log.info("━" * 60)
    log.info("Video      : %s", args.video.name)
    log.info("Resolution : %d×%d  →  processing %d×%d (scale %.2f)",
             full_w, full_h, proc_w, proc_h, args.scale)
    log.info("FPS        : %.1f src  →  %.1f effective (skip=%d)",
             src_fps, eff_fps, frame_skip)
    log.info("Mode       : %s  |  persist=%d  |  orphan=%d",
             args.mode, args.persist, ORPHAN_FRAMES)
    log.info("Object     : %.1fm – %.1fm  |  sensor=%.1fmm  focal=%.1fmm",
             args.min_object, args.max_object, args.sensor_w, args.focal)
    if not _CV2_GUI:
        log.info("Preview    : disabled (opencv-python-headless build)")
    log.info("━" * 60)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    srt_frames = load_telemetry(args.video, args.srt)
    has_telem  = len(srt_frames) > 0

    # ── Output paths ──────────────────────────────────────────────────────────
    out_video = (
        args.output or
        args.video.parent / (args.video.stem + "_motion.mp4")
    )
    out_json  = out_video.with_suffix(".json")
    writer    = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        eff_fps,
        (full_w, full_h),
    )

    crops_dir: Optional[Path] = None
    if args.save_crops:
        crops_dir = out_video.parent / (out_video.stem + "_crops")
        crops_dir.mkdir(exist_ok=True)

    log.info("Output     : %s", out_video)
    if crops_dir:
        log.info("Crops      : %s", crops_dir)

    # ── Algorithm state ───────────────────────────────────────────────────────
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )

    tracker = MotionTracker(
        match_dist_px=MATCH_DIST_PX_FALLBACK,
        persist=args.persist,
    )

    prev_gray:  Optional[np.ndarray] = None   # processing-resolution grayscale
    frame_idx   = 0     # source frame counter (all frames, including skipped)
    proc_count  = 0     # frames actually processed
    t_start     = time.time()

    # Diagnostics
    warp_fail_count  = 0
    total_blob_count = 0          # blobs that passed the area/aspect gate
    total_raw_count  = 0          # all connected components before gating
    seen_track_ids:  set[int] = set()     # IDs that have been confirmed at least once
    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            if (frame_idx - 1) % frame_skip != 0:
                continue

            proc_count += 1
            frame_ms = int((frame_idx - 1) / src_fps * 1000)

            # ── Telemetry for this moment ─────────────────────────────────────
            telem        = interpolate_telem(srt_frames, frame_ms)
            alt_m        = telem.alt_m        if telem else FALLBACK_ALT_M
            gimbal_pitch = telem.gimbal_pitch if telem else FALLBACK_GIMBAL_PITCH

            # Physics-derived parameters (recomputed per frame as altitude varies)
            gsd_m   = compute_gsd(alt_m, args.sensor_w, args.focal, proc_w)
            gsd_cm  = gsd_m * 100.0
            sky_row = sky_boundary_row(gimbal_pitch, proc_h)

            # Blob area gates (squared: area is px²)
            min_dim   = args.min_object / gsd_m     # px
            max_dim   = args.max_object / gsd_m     # px
            min_area  = max(4.0, min_dim ** 2)      # px²  — never below 4 px²
            max_area  = max_dim ** 2

            # Match gate from real-world distance → pixels
            tracker.match_dist_px = max(20.0, MATCH_DIST_M / gsd_m)

            # ── Resize to processing resolution ───────────────────────────────
            small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # ── Ego-motion compensation ────────────────────────────────────────
            fg_mask:    Optional[np.ndarray] = None
            warp_ok     = False
            n_inliers   = 0
            scene_speed = 0.0   # translation magnitude from H (px/frame)

            if prev_gray is not None:
                H, n_inliers = estimate_homography(prev_gray, gray, sky_row=sky_row)

                if H is not None:
                    warp_ok     = True
                    # Translation component of H gives scene speed in px/frame
                    scene_speed = math.hypot(float(H[0, 2]), float(H[1, 2]))
                    warped_prev = warp_frame(prev_gray, H, (proc_h, proc_w))
                else:
                    # Homography failed — fall back to identity (no compensation).
                    # This will produce false positives on a flying drone, but
                    # temporal consistency gate keeps them from becoming tracks.
                    warped_prev = prev_gray
                    warp_fail_count += 1

                # ── Foreground mask ────────────────────────────────────────────
                if args.mode == "mog2":
                    fg_mask = foreground_mog2(mog2, warped_prev, gray)
                elif args.mode == "flow":
                    # Dense optical flow residual — measures actual pixel velocity,
                    # not texture difference.  Subtracts the camera-motion component
                    # (from H) leaving only object-independent motion.  Significantly
                    # cleaner than diff/mog2 for oblique shots with tall structures.
                    fg_mask = foreground_flow(
                        prev_gray, gray,
                        H            = H if warp_ok else None,
                        threshold_px = args.flow_threshold,
                    )
                else:  # diff
                    fg_mask = foreground_diff(warped_prev, gray)

            # ── Stability gate ─────────────────────────────────────────────────
            # Measure foreground density and scene speed (kept for HUD display).
            # Neither suppresses spawning — instead they inform the dynamic-persist
            # threshold below, which is the actual discrimination mechanism.
            fg_fraction = 0.0
            if fg_mask is not None:
                fg_fraction = float(np.count_nonzero(fg_mask)) / fg_mask.size

            # Dynamic persist: how many consecutive hits to require before confirming.
            # Base: args.persist.  Bonus: +1 for every PERSIST_SPEED_STEP px/frame
            # that scene_speed exceeds args.max_scene_speed, capped at MAX_EXTRA_PERSIST.
            # A parallax residual typically lasts only 2–5 frames; a bird or vehicle
            # lasts far longer, so raising the bar during fast motion discriminates them.
            speed_excess    = max(0.0, scene_speed - args.max_scene_speed)
            extra_persist   = min(int(speed_excess / PERSIST_SPEED_STEP), MAX_EXTRA_PERSIST)
            effective_persist = args.persist + extra_persist

            # Minimum displacement in pixels (physics-derived from GSD).
            # Rejects fixed-world-point residuals that appear at the same pixel
            # location for many frames during a constant-speed pan (velocity ≈ 0
            # in the warped frame).  0 = disabled when args.min_displacement == 0.
            min_disp_px = (
                args.min_displacement / gsd_m
                if args.min_displacement > 0 and gsd_m > 0
                else 0.0
            )

            # ── Blob detection + tracking ─────────────────────────────────────
            tracks: list[MotionTrack] = []

            if fg_mask is not None:
                blobs, n_raw = detect_blobs(fg_mask, min_area, max_area)
                total_blob_count += len(blobs)
                total_raw_count  += n_raw
                tracks = tracker.update(
                    blobs, frame_idx,
                    effective_persist=effective_persist,
                    min_displacement_px=min_disp_px,
                )

                # Log first confirmation of each track
                for t in tracker.confirmed:
                    if t.track_id not in seen_track_ids:
                        seen_track_ids.add(t.track_id)
                        log.info(
                            "  ✓ Track T%d confirmed  frame=%d  "
                            "alt=%.0fm  gsd=%.1fcm  hits=%d",
                            t.track_id, frame_idx,
                            alt_m, gsd_cm, t.hit_count,
                        )
                        # Save first-confirmation crop
                        if crops_dir:
                            x, y, w, h = t.bbox
                            inv_s = 1.0 / args.scale
                            fx = int(x * inv_s); fy = int(y * inv_s)
                            fw = int(w * inv_s); fh = int(h * inv_s)
                            crop = frame[
                                max(0, fy):min(full_h, fy + fh),
                                max(0, fx):min(full_w, fx + fw),
                            ]
                            if crop.size > 0:
                                cname = crops_dir / f"T{t.track_id:04d}_f{frame_idx:05d}.jpg"
                                cv2.imwrite(str(cname), crop)
            else:
                # First frame — no foreground, still advance tracker state
                tracker.update([], frame_idx)

            # ── Geo-history population + world-fixed filter ───────────────────
            # Requires DJI telemetry.  Skipped silently when not available.
            #
            # For each track matched this frame: geolocate its current centroid
            # and append (lat, lon) to track.geo_history.  Once WF_MIN_FRAMES
            # observations have accumulated, project the mean geo position back
            # into the current camera frame and measure the reprojection error.
            # Low error → blob is a world-fixed residual → suppress it.
            # High error → blob is moving independently → keep it.

            # 1. Compute current frame footprint (full-res, for geo-projection)
            current_footprint: Optional[tuple] = None
            if has_telem and telem is not None and telem.alt_m > 0:
                try:
                    gimbal_yaw = getattr(telem, "gimbal_yaw", 0.0)
                    current_footprint = frame_corners(
                        lat=telem.lat, lon=telem.lon,
                        alt_m=telem.alt_m,
                        gimbal_yaw_deg=gimbal_yaw,
                        img_w=full_w, img_h=full_h,
                        sensor_w_mm=args.sensor_w, focal_mm=args.focal,
                    )
                except Exception:
                    current_footprint = None

            if current_footprint is not None:
                nw_c, ne_c, se_c, sw_c = current_footprint

                # 2. Append geo observation for every track hit this frame
                for track in tracks:
                    if len(track.geo_history) < track.hit_count:
                        # Track was matched this frame — geolocate centroid
                        cx_full = track.cx / args.scale
                        cy_full = track.cy / args.scale
                        lat_b, lon_b = pixel_to_geo(
                            cx_full, cy_full,
                            full_w, full_h,
                            nw_c, ne_c, se_c, sw_c,
                        )
                        track.geo_history.append((lat_b, lon_b))

                # 3. World-fixed test: run at WF_MIN_FRAMES, then every WF_RETEST_EVERY
                for track in tracks:
                    n_geo = len(track.geo_history)
                    if n_geo < WF_MIN_FRAMES:
                        continue
                    if not (n_geo == WF_MIN_FRAMES or
                            (n_geo - WF_MIN_FRAMES) % WF_RETEST_EVERY == 0):
                        continue

                    reproj_err = world_fixed_reprojection_error(
                        track, current_footprint, full_w, full_h, args.scale
                    )
                    was_suppressed = track.suppressed
                    track.suppressed = (reproj_err < WF_THRESHOLD_PX)

                    if track.suppressed and not was_suppressed:
                        log.debug(
                            "  ○ Track T%d suppressed (world-fixed) "
                            "reproj=%.1fpx  hits=%d",
                            track.track_id, reproj_err, track.hit_count,
                        )
                    elif was_suppressed and not track.suppressed:
                        log.info(
                            "  ↑ Track T%d un-suppressed (now moving) "
                            "reproj=%.1fpx  hits=%d",
                            track.track_id, reproj_err, track.hit_count,
                        )

            # Count suppressed tracks for summary
            suppressed_this_frame = sum(1 for t in tracks if t.suppressed)

            # ── Annotate + write ──────────────────────────────────────────────
            annotated = annotate(
                frame, tracks, args.scale,
                frame_idx, frame_ms, alt_m, gsd_cm,
                n_inliers, args.mode, warp_ok,
                scene_speed, fg_fraction, effective_persist,
            )
            writer.write(annotated)

            # ── Preview ───────────────────────────────────────────────────────
            if not args.no_preview and _CV2_GUI:
                pw = min(full_w, 1280)
                ph = int(full_h * pw / full_w)
                preview = cv2.resize(annotated, (pw, ph), interpolation=cv2.INTER_AREA)
                cv2.imshow("TAE — Motion Detection  (q to quit)", preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    log.info("User quit at frame %d", frame_idx)
                    break

            # ── State advance ─────────────────────────────────────────────────
            prev_gray = gray

            # Periodic progress
            if proc_count % 100 == 0:
                elapsed = time.time() - t_start
                pct     = 100.0 * frame_idx / max(total_frames, 1)
                gate_pct = (
                    100.0 * (total_raw_count - total_blob_count) / max(total_raw_count, 1)
                )
                log.info(
                    "  [%5.1f%%]  frame %d  elapsed %.0fs  "
                    "blobs raw/gated: %d/%d (%.0f%% filtered)  confirmed tracks: %d",
                    pct, frame_idx, elapsed,
                    total_raw_count, total_blob_count, gate_pct,
                    len(seen_track_ids),
                )

    finally:
        cap.release()
        writer.release()
        if not args.no_preview and _CV2_GUI:
            cv2.destroyAllWindows()

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    elapsed  = time.time() - t_start
    warp_pct = 100.0 * warp_fail_count / max(proc_count - 1, 1)
    gate_filtered     = total_raw_count - total_blob_count
    gate_filtered_pct = 100.0 * gate_filtered / max(total_raw_count, 1)

    all_conf = tracker.all_confirmed_ever
    n_suppressed_ever = sum(1 for t in all_conf if t.suppressed)
    n_visible_movers  = sum(1 for t in all_conf if not t.suppressed)

    log.info("━" * 60)
    log.info("Finished in %.1f s", elapsed)
    log.info("Source frames read : %d  (processed: %d)", frame_idx, proc_count)
    log.info("Warp failures      : %d / %d frames  (%.1f%%)",
             warp_fail_count, max(proc_count - 1, 1), warp_pct)
    log.info("Blobs raw          : %d  (before area/aspect gate)", total_raw_count)
    log.info("Blobs passed gate  : %d  (%.0f%% filtered out)",
             total_blob_count, gate_filtered_pct)
    log.info("Confirmed tracks   : %d", len(seen_track_ids))
    if has_telem:
        log.info("  World-fixed (suppressed) : %d", n_suppressed_ever)
        log.info("  Genuine movers (visible) : %d", n_visible_movers)
    else:
        log.info("  (world-fixed filter disabled — no telemetry)")

    # Diagnose common "nothing detected" situations explicitly
    if n_visible_movers == 0 and has_telem:
        if len(seen_track_ids) > 0:
            log.warning(
                "  ⚠ %d confirmed tracks, but all classified as world-fixed. "
                "If genuine movers are expected: lower --wf-threshold (currently %.0f px) "
                "or ensure telemetry alt/yaw are correct.",
                len(seen_track_ids), WF_THRESHOLD_PX,
            )
        elif total_raw_count == 0:
            log.warning("  ⚠ No blobs reached detect_blobs() at all — "
                        "check warp failures or try --mode diff")
        elif gate_filtered_pct > 90:
            log.warning(
                "  ⚠ %.0f%% of blobs were filtered by the area gate "
                "(%d raw → %d passed). Try: --min-object %.2f",
                gate_filtered_pct, total_raw_count, total_blob_count,
                args.min_object / 2,
            )
        else:
            log.warning("  ⚠ Blobs detected but no track reached persist=%d. "
                        "Try --persist 2 or check for noisy warp.",
                        args.persist)

    if args.save_crops and n_visible_movers == 0:
        log.warning("  ⚠ --save-crops was set but no visible-mover crops were saved.")

    # Per-track table (confirmed movers only)
    visible_movers = [t for t in all_conf if not t.suppressed]
    if visible_movers:
        log.info("")
        log.info("  Genuine movers (reprojection error ≥ %.0f px):", WF_THRESHOLD_PX)
        log.info("  %-6s  %-8s  %-8s", "Track", "Hits", "Misses")
        for t in sorted(visible_movers, key=lambda x: x.track_id):
            log.info("  T%-5d  %-8d  %-8d", t.track_id, t.hit_count, t.miss_count)

    log.info("")
    log.info("Output video  : %s", out_video)

    # Summary JSON
    summary = {
        "video":                str(args.video),
        "mode":                 args.mode,
        "scale":                args.scale,
        "persist_frames":       args.persist,
        "wf_min_frames":        WF_MIN_FRAMES,
        "wf_threshold_px":      WF_THRESHOLD_PX,
        "sensor_w_mm":          args.sensor_w,
        "focal_mm":             args.focal,
        "min_object_m":         args.min_object,
        "max_object_m":         args.max_object,
        "src_fps":              src_fps,
        "effective_fps":        eff_fps,
        "has_telemetry":        has_telem,
        "frames_total":         frame_idx,
        "frames_processed":     proc_count,
        "warp_failures":        warp_fail_count,
        "warp_failure_pct":     round(warp_pct, 1),
        "blobs_raw":            total_raw_count,
        "blobs_passed_gate":    total_blob_count,
        "blobs_gate_filtered":  gate_filtered,
        "blobs_gate_filtered_pct": round(gate_filtered_pct, 1),
        "confirmed_tracks":     len(seen_track_ids),
        "suppressed_tracks":    n_suppressed_ever,
        "visible_movers":       n_visible_movers,
        "track_details": [
            {
                "id":         t.track_id,
                "hits":       t.hit_count,
                "misses":     t.miss_count,
                "suppressed": t.suppressed,
            }
            for t in sorted(all_conf, key=lambda x: x.track_id)
        ],
        "output_video":  str(out_video),
        "elapsed_s":     round(elapsed, 1),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Summary JSON  : %s", out_json)
    log.info("━" * 60)


if __name__ == "__main__":
    main()