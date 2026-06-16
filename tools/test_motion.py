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

# ── OpenAI / OpenRouter (for optional VLM classification stage) ────────────────
try:
    import base64 as _b64
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

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

# Isolation gate — proximity exclusion filter
#
# Observation: construction activity (crane jibs, excavators, concrete pumps,
# clustered workers) produces MULTIPLE simultaneous blobs within 1–5 m of each
# other.  A lone flying object (bird, UAV) produces a SINGLE isolated blob with
# no neighbours within 10–15 m.
#
# The gate discards any blob that has at least one other blob within
# ISOLATION_RADIUS_M metres.  Only truly isolated blobs reach the tracker.
#
# Tradeoffs accepted (evaluated before implementation):
#   ✗ Bird flocks dismissed (user-acknowledged)
#   ✗ Single isolated workers / parked vehicles also pass (acceptable for bird use-case)
#   ✗ Bird flying near a building edge may be suppressed (mitigated by flow mode reducing
#       building-edge blobs to near-zero)
#   ✓ Works well for the primary use-case: single bird over a busy construction site
#
# Physics guide (at proc-scale 0.5 from 4K, GSD≈5.8 cm/px, alt≈80 m):
#   5 m  → 86 px  — kills adjacent machine-part pairs, two workers walking side-by-side
#  10 m  → 172 px — kills workers 5–10 m apart, most crane-tip clusters
#  15 m  → 258 px — conservative; may clip bird flying near building edge
#
# Default: 0.0 (disabled).  Enable with --isolation-radius.
DEFAULT_ISOLATION_RADIUS_M = 0.0

# Maximum ground-range filter
#
# The flat-earth nadir footprint model (frame_corners / pixel_to_geo) breaks
# completely for objects far from the drone nadir — especially in oblique shots
# where the camera looks toward the horizon.  Pixels in the upper frame project
# to locations thousands of metres from the nadir; the variance formula then
# runs on nonsensical coordinates and classifies city-skyline buildings as
# "movers" despite being obviously static.
#
# This filter computes the horizontal distance from the drone's GPS position
# to each blob's ground projection.  Blobs beyond MAX_RANGE_M are suppressed
# immediately — the flat-earth model is invalid at those distances, so they
# can never be reliably geolocated or classified.
#
# Physics guide (80 m AGL, oblique camera):
#   300 m → cuts pixels beyond ~75° from vertical  (safe for most construction pits)
#   500 m → cuts pixels beyond ~81° from vertical  (keeps wide construction zones)
#   800 m → cuts pixels beyond ~84° from vertical  (very conservative)
#   City skyline in this footage: 1000–5000 m → suppressed at any of the above
#
# Default: 0 (disabled).  Enable with --max-range.
DEFAULT_MAX_RANGE_M = 0.0

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

# World-fixed point filter — uses DJI telemetry to identify blobs that are
# static in the world regardless of their altitude above ground.
#
# Algorithm (world-frame variance test)
# ──────────────────────────────────────
# For a blob tracked across N frames, we have for each frame i:
#   G_i = ground-plane geo projection of the blob pixel (from footprint)
#   C_i = camera position (lat, lon, alt)  ← from DJI telemetry
#
# At candidate altitude h, the blob's inferred WORLD position is:
#   W(h, i) = C_i + (G_i - C_i) × (alt_i - h) / alt_i
#
# This is a linear interpolation between camera and ground along the ray.
# For a WORLD-FIXED object at true height h*:
#   W(h*, i) ≈ same world point for all i  →  variance ≈ 0
#
# For a MOVING object (bird) at altitude h_bird:
#   W(h_bird, i) changes each frame as the bird moves  →  variance > 0
#
# We scan h from 0 to 0.95 × alt and take the MINIMUM variance.
# A small minimum means no matter what altitude is assumed, the object
# is world-fixed.  A large minimum means the object moves in the world.
#
# Why this is correct for ALL flight directions (including parallel to camera):
#   The camera position C_i changes each frame via telemetry.  Even if the
#   bird flies in the same direction as the camera, the same pixel maps to
#   a different world coordinate in each frame because C_i has moved.
#   The only true degeneracy is v_bird = v_camera exactly — zero relative
#   motion — which is physically undetectable by any single-camera method.
#
# WF_THRESHOLD_M : minimum variance (metres) that separates static from moving.
#   Simulation with realistic noise (GPS 0.5m σ, pixel 3px):
#     Static crane at 50m  →  min variance ≈ 0.2–0.4 m
#     Bird at 5 m/s        →  min variance ≈ 0.7–1.0 m
#     Bird at 3 m/s        →  min variance ≈ 0.5–0.7 m
#   Default 0.4 m gives clean separation for birds at ≥ 3 m/s.
#   Known edge case — undetectable by this method:
#     A bird flying *directly toward* the camera creates a degenerate altitude
#     h_deg = alt × v_bird/(v_cam + v_bird) + v_cam × h_bird/(v_cam + v_bird)
#     at which its apparent world position is constant, indistinguishable from a
#     static structure at that height.  This is a fundamental single-camera
#     observability limit (relative motion ≈ 0 in the degenerate projection).
#     Any lateral velocity component breaks the degeneracy.
WF_MIN_FRAMES    = 8     # minimum track hits before test is valid; 64% of tracks
                          # never reached 20 — 8 hits (0.27s) exposes ~78% of tracks
WF_RETEST_EVERY  = 15    # re-test every N additional hits after first result
WF_N_SCAN        = 60    # number of altitude candidates in the h-scan
WF_THRESHOLD_M   = 0.4   # world-space RMS spread (metres); below = world-fixed

# VLM classification stage (optional post-processing, --vlm-classify)
#
# After the video is fully processed, confirmed visible_movers are passed to
# Qwen2.5-VL (via OpenRouter) for semantic classification.  The VLM is asked to
# distinguish genuine independently-moving objects (birds, vehicles) from false
# alarms caused by camera parallax over static structures (buildings, cranes).
#
# Triggered once per track at first confirmation, grouped into batches of
# VLM_MAX_DET_PER_CALL detections per API call to keep response within token limits.
# A snapshot frame is saved every VLM_SNAPSHOT_INTERVAL processed frames; each
# batch is sent with the snapshot closest to the tracks' confirmation times.
#
# Requires:  pip install openai
#            OPENROUTER_API_KEY env var  (or --vlm-api-key)
VLM_MODEL              = "qwen/qwen-2.5-vl-72b-instruct"  # same as TAE
VLM_SNAPSHOT_INTERVAL  = 90    # save one frame every N processed frames (~3 s at 30 fps)
VLM_MAX_DET_PER_CALL   = 25    # max detections per API call (token budget)
VLM_MAX_TOKENS         = 2048  # response token budget (≈80 per confirmed + 20 per rejected)
VLM_SNAP_LONG_EDGE     = 1280  # resize snapshot to this long edge before sending

# Annotation colours  (BGR)
COL_PENDING   = (0, 200, 255)     # yellow  — seen but not yet confirmed
COL_CONFIRMED = (50, 220, 50)     # green   — confirmed track
COL_ISOLATED  = (255, 220, 0)     # bright cyan-yellow — confirmed AND isolated (primary target)
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


def world_fixed_variance_test(
    track,
    n_scan: int = WF_N_SCAN,
) -> tuple[float, float]:
    """
    Test whether this track is consistent with a static world point at ANY altitude.

    For each candidate altitude h (scanned 0 → 0.95 × camera alt):

        W(h, i) = C_i + (G_i − C_i) × (alt_i − h) / alt_i

    where C_i is the camera position in frame i (from telemetry) and G_i is
    the ground-plane geo projection of the blob pixel.  W(h, i) is the world
    position the blob WOULD have if it were at height h in frame i.

    A world-FIXED object at true height h*:
        W(h*, i) ≈ same point for all i  →  spatial variance ≈ 0

    A moving object (bird, vehicle):
        W(h_obj, i) changes each frame regardless of h  →  variance > 0

    We take the MINIMUM variance across all scanned h values.  This
    automatically finds the correct altitude without any prior knowledge,
    and correctly handles elevated structures (cranes, buildings) that the
    previous flat-ground reprojection test incorrectly classified as movers.

    Direction independence: because camera position C_i changes each frame
    via telemetry, even a bird flying parallel to the camera maps to a
    different world coordinate each frame → variance > 0.  There is no
    parallel-motion degeneracy in this formulation.

    Parameters
    ----------
    track   : MotionTrack with geo_history and camera_history populated.
    n_scan  : number of altitude candidates (coarse phase); 60 gives ~1m
              resolution at 80m altitude.

    Returns
    -------
    (min_variance_m, best_h_m)
        min_variance_m  RMS world-space positional spread (metres) at best h.
        best_h_m        altitude (m) that minimises the variance.
    """
    n_geo = len(track.geo_history)
    n_cam = len(track.camera_history)
    if n_geo < WF_MIN_FRAMES or n_cam < WF_MIN_FRAMES:
        return float("inf"), 0.0

    n = min(n_geo, n_cam)
    geos = track.geo_history[-n:]      # list of (lat_g, lon_g)
    cams = track.camera_history[-n:]   # list of (lat_c, lon_c, alt_c)

    # ENU reference: first camera position
    lat_ref, lon_ref, _ = cams[0]
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat_ref))

    # Pre-convert to local ENU metres — avoids repeated trig in the inner loop
    cam_E  = [(lo - lon_ref) * m_lon for _, lo, _ in cams]
    cam_N  = [(la - lat_ref) * m_lat for la, _, _ in cams]
    gnd_E  = [(lo - lon_ref) * m_lon for _, lo    in geos]
    gnd_N  = [(la - lat_ref) * m_lat for la, _    in geos]
    alts   = [alt_c                  for _, _, alt_c in cams]
    alt_max = max(alts)

    def variance_at_h(h: float) -> float:
        """RMS 2-D positional spread of W(h, i) across all frames."""
        Es, Ns = [], []
        for Ce, Cn, Ge, Gn, alt_c in zip(cam_E, cam_N, gnd_E, gnd_N, alts):
            if alt_c <= h or alt_c <= 0:
                continue
            frac = (alt_c - h) / alt_c
            Es.append(Ce + (Ge - Ce) * frac)
            Ns.append(Cn + (Gn - Cn) * frac)
        if len(Es) < 3:
            return float("inf")
        em = sum(Es) / len(Es);  nm = sum(Ns) / len(Ns)
        vE = sum((e - em) ** 2 for e in Es) / len(Es)
        vN = sum((n_i - nm) ** 2 for n_i in Ns) / len(Ns)
        return math.sqrt(vE + vN)

    # ── Coarse scan ────────────────────────────────────────────────────────────
    h_max   = alt_max * 0.95
    h_step  = h_max / max(n_scan - 1, 1)
    coarse  = sorted((variance_at_h(h_step * i), h_step * i) for i in range(n_scan))
    best_v, best_h = coarse[0]

    # ── Fine scan (±1 step around best coarse point, 20 sub-steps) ───────────
    lo = max(0.0, best_h - h_step)
    hi = min(h_max, best_h + h_step)
    fine_step = (hi - lo) / 20
    for k in range(21):
        h_f = lo + fine_step * k
        v_f = variance_at_h(h_f)
        if v_f < best_v:
            best_v, best_h = v_f, h_f

    return best_v, best_h


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
    isolation_radius_px: float = 0.0,
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
        # Isolated confirmed tracks get a brighter cyan highlight so the
        # operator can immediately spot lone flyers among clustered detections.
        if t.confirmed and t.is_isolated and isolation_radius_px > 0:
            col = COL_ISOLATED
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
    n_isolated   = sum(1 for t in tracks if not t.suppressed and t.confirmed and t.is_isolated)
    isolation_str = (f"  isolated={n_isolated}" if isolation_radius_px > 0 else "")
    hud_lines = [
        f"frame {frame_idx:05d}   t={frame_ms/1000:.2f}s   mode={mode}",
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
    p.add_argument("--vlm-classify", action="store_true", dest="vlm_classify",
                   help="Run a VLM pass after video processing.  Sends confirmed visible "
                        "movers to Qwen2.5-VL (via OpenRouter) to distinguish genuine movers "
                        "(birds, vehicles) from false alarms on static structures (buildings, "
                        "cranes, scaffolding).  Makes ~5–10 API calls for a 16 s clip.  "
                        "Requires: pip install openai  and  OPENROUTER_API_KEY env var "
                        "(or --vlm-api-key).")
    p.add_argument("--vlm-api-key", type=str, default=None, dest="vlm_api_key",
                   help="OpenRouter API key for --vlm-classify.  "
                        "Defaults to OPENROUTER_API_KEY env var.")
    p.add_argument("--max-frames", type=int, default=None,
                   dest="max_frames",
                   help="Stop after N source frames (for quick tests)")
    p.add_argument("--max-scene-speed", type=float, default=DEFAULT_MAX_SCENE_SPEED,
                   dest="max_scene_speed",
                   help="Scene translation (px/frame at proc resolution) above which "
                        "extra persist frames are added. Default %(default).0f px. "
                        "Lower = stricter during slower pans.")
    p.add_argument("--wf-threshold", type=float, default=WF_THRESHOLD_M,
                   dest="wf_threshold",
                   help="World-frame variance threshold (metres) for the world-fixed "
                        "filter.  Tracks whose minimum positional variance across all "
                        "scanned altitudes is below this are classified as world-fixed "
                        "(parallax residuals from static structures) and suppressed. "
                        "Lower → suppress more aggressively; higher → miss fewer movers. "
                        "Physics guide: GPS noise 0.5m σ gives static-structure variance "
                        "≈ 0.2–0.35 m; birds at ≥ 3 m/s give ≥ 0.5 m. "
                        "Default %(default).2f m.")
    p.add_argument("--flow-threshold", type=float, default=DEFAULT_FLOW_THRESHOLD_PX,
                   dest="flow_threshold",
                   help="Residual optical flow magnitude threshold in px/frame (proc-res) "
                        "for --mode flow.  Pixels below this are background; above = mover. "
                        "Physics guide: parallax from a 50 m crane at 80 m AGL, 5 m/s pan "
                        "≈ 1.8 px/frame; a 5 m/s bird ≈ 2.9 px/frame.  "
                        "Default %(default).1f px.  Lower (1.5) catches slower movers; "
                        "higher (3.0) reduces false positives in turbulent conditions.")
    p.add_argument("--isolation-radius", type=float, default=DEFAULT_ISOLATION_RADIUS_M,
                   dest="isolation_radius",
                   help="Proximity exclusion radius in metres.  Any blob that has at least "
                        "one other blob within this distance is discarded before tracking. "
                        "Rationale: construction machinery produces CLUSTERS of simultaneous "
                        "blobs; a lone bird produces ONE isolated blob.  0 = disabled (default). "
                        "Recommended for bird detection over a busy construction site: 5–10 m. "
                        "Physics at GSD≈5.8 cm/px (80 m alt): "
                        "5 m → 86 px, 10 m → 172 px.  Tradeoff: also dismisses single "
                        "isolated workers/vehicles and birds flying near structure edges.")
    p.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE_M,
                   dest="max_range",
                   help="Maximum ground-range (metres from camera nadir) for blob acceptance. "
                        "Blobs whose flat-earth ground projection falls beyond this distance "
                        "are suppressed immediately — the nadir footprint model is invalid "
                        "at those distances, especially for oblique shots where the city "
                        "skyline or horizon appears in the upper frame. "
                        "0 = disabled (default). "
                        "Recommended for oblique urban shots: 300–500 m. "
                        "Physics: at 80 m AGL, 500 m range corresponds to pixels beyond "
                        "~81° from vertical; city skyline at 1–5 km is cleanly suppressed.")
    p.add_argument("--min-displacement", type=float, default=DEFAULT_MIN_DISPLACEMENT,
                   dest="min_displacement",
                   help="Minimum real-world displacement (metres) a track must show "
                        "before it is confirmed. 0 = disabled (default). "
                        "E.g. 0.5 rejects fixed-world residuals during steady pans.")
    return p.parse_args()


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
    total_blob_count = 0          # blobs that passed the area/aspect gate AND isolation gate
    total_raw_count  = 0          # all connected components before gating
    total_isolation_filtered = 0  # blobs removed by isolation gate
    total_range_suppressed   = 0  # tracks suppressed by max-range gate

    # VLM snapshot buffer: (frame_idx, snapshot_BGR_img)
    vlm_snapshots:  list[tuple] = []
    snap_scale      = min(1.0, VLM_SNAP_LONG_EDGE / max(full_w, full_h))
    seen_track_ids:  set[int] = set()     # IDs that have been confirmed at least once

    # Cumulative camera displacement from homography chain (30 Hz accuracy).
    # The SRT telemetry is typically 1 Hz — the same lat/lon repeated for 30 frames.
    # Raw SRT camera positions in camera_history would be a step function, making
    # W(h, i) = C_i + (G_i − C_i)·frac compute large variance even for static
    # world-fixed blobs (because G_i drifts while C_i is frozen).
    # Fix: accumulate per-frame H translations to get a 30 Hz camera trajectory,
    # then compute a refined footprint each frame from the refined camera position.
    # Both C_i (camera_history) and G_i (geo_history) are then accurate per-frame.
    cum_cam_delta_E = 0.0   # metres east  accumulated from H[0,2]
    cum_cam_delta_N = 0.0   # metres north accumulated from H[1,2]
    cam_lat_ref:  Optional[float] = None   # GPS anchor (set at first valid telem)
    cam_lon_ref:  Optional[float] = None
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

                    # Accumulate per-frame camera displacement at 30 Hz.
                    # H maps prev→curr; scene shifts (H[0,2], H[1,2]) px.
                    # Camera moves OPPOSITE to scene in world space:
                    #   scene right (H[0,2]>0) → camera moved west → ΔE<0
                    #   scene down  (H[1,2]>0) → camera moved north → ΔN>0 (nadir)
                    if gsd_m > 0:
                        cum_cam_delta_E -= float(H[0, 2]) * gsd_m
                        cum_cam_delta_N += float(H[1, 2]) * gsd_m
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

            # Compute isolation radius in pixels for this frame (GSD may vary).
            # Initialised here so annotate() always has a valid value even when
            # fg_mask is None (first frame).
            isolation_radius_px = (
                args.isolation_radius / gsd_m
                if args.isolation_radius > 0 and gsd_m > 0
                else 0.0
            )

            if fg_mask is not None:
                blobs, n_raw = detect_blobs(fg_mask, min_area, max_area)
                total_raw_count  += n_raw

                # Isolation gate: discard blobs that have a neighbour within
                # isolation_radius_px.  Applied before the tracker so clustered
                # blobs never form tracks, reducing both false positives and
                # tracker bookkeeping overhead.
                if isolation_radius_px > 0:
                    n_before = len(blobs)
                    blobs = apply_isolation_gate(blobs, isolation_radius_px)
                    total_isolation_filtered += (n_before - len(blobs))

                total_blob_count += len(blobs)
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
                        # Store for VLM classification
                        t.confirm_frame_idx = frame_idx
                        if t.bbox:
                            x, y, w, h = t.bbox
                            inv_s = 1.0 / args.scale
                            t.confirm_bbox_full = (int(x*inv_s), int(y*inv_s),
                                                   int(w*inv_s), int(h*inv_s))
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

                # Periodic VLM snapshot (full frame, downsampled)
                if args.vlm_classify and proc_count % VLM_SNAPSHOT_INTERVAL == 0:
                    sh = int(full_h * snap_scale); sw = int(full_w * snap_scale)
                    snap = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
                    vlm_snapshots.append((frame_idx, snap))
            else:
                # First frame — no foreground, still advance tracker state
                tracker.update([], frame_idx)

            # ── World-frame variance filter ───────────────────────────────────
            # Requires DJI telemetry.  Skipped silently when unavailable.
            #
            # Root cause of previous failure: SRT telemetry is ~1 Hz but video
            # is 30 fps.  Raw telem.lat/lon in camera_history was a step function
            # (same value repeated 30× per SRT block).  With C_i frozen while G_i
            # drifted (blob pixel moves as camera pans), the variance formula
            # produced multi-metre values even for static world-fixed structures.
            #
            # Fix: build a REFINED footprint and camera position for each frame by
            # adding the H-accumulated displacement to the GPS anchor.  This gives
            # both C_i and G_i at 30 Hz accuracy (homography error ≈ 6 cm over
            # a 20-frame window, vs. 1–5 m GPS step error per SRT block).

            # 1. GPS anchor — set once at first valid telemetry
            if (cam_lat_ref is None and has_telem
                    and telem is not None and telem.alt_m > 0
                    and telem.lat != 0.0):
                cam_lat_ref = telem.lat
                cam_lon_ref = telem.lon

            # 2. Refined footprint: SRT GPS anchor + H-accumulated offset
            current_footprint: Optional[tuple] = None
            lat_c_refined: Optional[float] = None
            lon_c_refined: Optional[float] = None

            if (cam_lat_ref is not None and has_telem
                    and telem is not None and telem.alt_m > 0):
                try:
                    gimbal_yaw   = getattr(telem, "gimbal_yaw",   0.0)
                    gimbal_pitch = getattr(telem, "gimbal_pitch", -90.0)
                    m_lat_r    = 111320.0
                    m_lon_r    = 111320.0 * math.cos(math.radians(cam_lat_ref))
                    lat_c_refined = cam_lat_ref + cum_cam_delta_N / m_lat_r
                    lon_c_refined = cam_lon_ref + cum_cam_delta_E / m_lon_r

                    # Use oblique ray-tracing footprint when pitch is available.
                    # This correctly maps pixels to ground for oblique cameras:
                    #   - City skyline at 1–5 km → projects to 1–5 km from nadir ✓
                    #   - Nearby crane at 200 m → projects to ~200 m from nadir ✓
                    #   - Construction pit at 50 m → correct ground position ✓
                    # The nadir frame_corners() would map everything within ±56 m,
                    # making the range gate and world-fixed test useless for oblique shots.
                    current_footprint = oblique_frame_corners(
                        lat=lat_c_refined, lon=lon_c_refined,
                        alt_m=telem.alt_m,
                        gimbal_yaw_deg=gimbal_yaw,
                        gimbal_pitch_deg=gimbal_pitch,
                        img_w=full_w, img_h=full_h,
                        sensor_w_mm=args.sensor_w, focal_mm=args.focal,
                    )
                    # Fallback to nadir model if oblique rays don't all hit ground
                    # (e.g., camera pitched so shallow that top of frame is sky)
                    if current_footprint is None:
                        current_footprint = frame_corners(
                            lat=lat_c_refined, lon=lon_c_refined,
                            alt_m=telem.alt_m,
                            gimbal_yaw_deg=gimbal_yaw,
                            img_w=full_w, img_h=full_h,
                            sensor_w_mm=args.sensor_w, focal_mm=args.focal,
                        )
                except Exception:
                    current_footprint = None

            if current_footprint is not None:
                nw_c, ne_c, se_c, sw_c = current_footprint

                # 3. Append geo + camera observation for every track hit this frame
                for track in tracks:
                    if len(track.geo_history) < track.hit_count:
                        cx_full = track.cx / args.scale
                        cy_full = track.cy / args.scale
                        lat_g, lon_g = pixel_to_geo(
                            cx_full, cy_full,
                            full_w, full_h,
                            nw_c, ne_c, se_c, sw_c,
                        )

                        # ── Maximum ground-range gate ─────────────────────────────
                        # Applied only during CAMERA MOTION (scene_speed > threshold).
                        # When the camera is static, the world-fixed variance test
                        # handles suppression correctly on its own — static objects
                        # get zero variance and flying objects get non-zero variance.
                        # If we apply the range gate during static periods it
                        # incorrectly eliminates the bird (whose ray projects far past
                        # it to the ground below).
                        if (args.max_range > 0
                                and scene_speed > 1.0):   # 1 px/frame ≈ 3.5 m/s
                            dist_m = pixel_ground_range(
                                cx_full, cy_full,
                                full_w, full_h,
                                telem.alt_m,
                                getattr(telem, "gimbal_yaw",   0.0),
                                getattr(telem, "gimbal_pitch", -90.0),
                                args.sensor_w, args.focal,
                            )
                            if dist_m > args.max_range:
                                if not track.suppressed:
                                    track.suppressed   = True
                                    track.suppressed_by = "range"
                                    total_range_suppressed += 1
                                continue   # don't append geo to history

                        track.geo_history.append((lat_g, lon_g))
                    if len(track.camera_history) < track.hit_count:
                        track.camera_history.append(
                            (lat_c_refined, lon_c_refined, telem.alt_m)
                        )

                # 4. World-fixed variance test: at WF_MIN_FRAMES, then every WF_RETEST_EVERY
                for track in tracks:
                    n_obs = len(track.geo_history)
                    if n_obs < WF_MIN_FRAMES:
                        continue
                    if not (n_obs == WF_MIN_FRAMES or
                            (n_obs - WF_MIN_FRAMES) % WF_RETEST_EVERY == 0):
                        continue

                    min_var, best_h = world_fixed_variance_test(track)
                    track.wf_variance_m = min_var
                    track.wf_best_h_m   = best_h
                    was_suppressed  = track.suppressed
                    track.suppressed = (min_var < args.wf_threshold)
                    if track.suppressed:
                        track.suppressed_by = "world_fixed"

                    if track.suppressed and not was_suppressed:
                        log.debug(
                            "  ○ T%d world-fixed  var=%.3fm  h=%.0fm  hits=%d",
                            track.track_id, min_var, best_h, track.hit_count,
                        )
                    elif was_suppressed and not track.suppressed:
                        log.info(
                            "  ↑ T%d now moving   var=%.3fm  h=%.0fm  hits=%d",
                            track.track_id, min_var, best_h, track.hit_count,
                        )

            # Count suppressed tracks for summary
            suppressed_this_frame = sum(1 for t in tracks if t.suppressed)

            # ── Isolation status update ───────────────────────────────────────
            # Re-evaluate is_isolated for every active confirmed track at the
            # TRACK level each frame.  Even though isolated blobs are pre-filtered
            # before the tracker, two tracks whose blobs were separately isolated
            # could drift within radius of each other — this dynamic check catches it.
            # Tracks that lose isolation (enter a cluster) are de-highlighted but
            # not killed; they can regain isolated status if they move apart again.
            if isolation_radius_px > 0:
                active_conf = [t for t in tracks if t.confirmed and not t.suppressed]
                for t in active_conf:
                    t.is_isolated = not any(
                        other.track_id != t.track_id and
                        math.hypot(t.cx - other.cx, t.cy - other.cy) <= isolation_radius_px
                        for other in active_conf
                    )

            # ── Annotate + write ──────────────────────────────────────────────
            annotated = annotate(
                frame, tracks, args.scale,
                frame_idx, frame_ms, alt_m, gsd_cm,
                n_inliers, args.mode, warp_ok,
                scene_speed, fg_fraction, effective_persist,
                isolation_radius_px=isolation_radius_px,
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
    n_isolated_ever   = sum(1 for t in all_conf if not t.suppressed and t.is_isolated)

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
        if args.max_range > 0:
            log.info("  Range-suppressed (>%.0fm): %d", args.max_range, total_range_suppressed)
    else:
        log.info("  (world-fixed filter disabled — no telemetry)")
    if args.isolation_radius > 0:
        log.info("  Isolation gate (radius=%.1f m): %d blobs pre-filtered across run",
                 args.isolation_radius, total_isolation_filtered)
        log.info("  Isolated confirmed tracks (primary targets): %d", n_isolated_ever)

    # Diagnose common "nothing detected" situations explicitly
    if n_visible_movers == 0 and has_telem:
        if len(seen_track_ids) > 0:
            log.warning(
                "  ⚠ %d confirmed tracks but all world-fixed. "
                "Try lowering --wf-threshold (currently %.2f m).",
                len(seen_track_ids), args.wf_threshold,
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
        log.info("  Genuine movers (world-variance ≥ %.2f m):", args.wf_threshold)
        log.info("  %-6s  %-8s  %-8s", "Track", "Hits", "Misses")
        for t in sorted(visible_movers, key=lambda x: x.track_id):
            log.info("  T%-5d  %-8d  %-8d", t.track_id, t.hit_count, t.miss_count)

    log.info("")
    log.info("Output video  : %s", out_video)

    # ── VLM classification (optional post-processing) ─────────────────────────
    vlm_suppressed = 0
    if args.vlm_classify:
        api_key = args.vlm_api_key or __import__("os").environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            log.warning("VLM classify requested but no API key found.  "
                        "Set OPENROUTER_API_KEY or use --vlm-api-key.")
        elif not _HAS_OPENAI:
            log.warning("VLM classify requires 'openai' package: pip install openai")
        else:
            log.info("━" * 60)
            log.info("VLM classification pass …")
            all_confirmed = tracker.all_confirmed_ever
            vlm_suppressed = vlm_classify_visible_movers(
                snapshots  = vlm_snapshots,
                snap_scale = snap_scale,
                all_tracks = all_confirmed,
                api_key    = api_key,
                model      = VLM_MODEL,
            )
            # Recount after VLM
            n_visible_movers = sum(1 for t in all_confirmed if not t.suppressed)
            log.info("After VLM: visible_movers=%d  vlm_suppressed=%d",
                     n_visible_movers, vlm_suppressed)

    # Summary JSON
    summary = {
        "video":                str(args.video),
        "mode":                 args.mode,
        "scale":                args.scale,
        "persist_frames":       args.persist,
        "wf_min_frames":        WF_MIN_FRAMES,
        "wf_threshold_m":       args.wf_threshold,
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
        "isolation_radius_m":   args.isolation_radius,
        "blobs_isolation_filtered": total_isolation_filtered,
        "max_range_m":          args.max_range,
        "range_suppressed":     total_range_suppressed,
        "confirmed_tracks":     len(seen_track_ids),
        "suppressed_tracks":    n_suppressed_ever,
        "visible_movers":       n_visible_movers,
        "vlm_suppressed":       vlm_suppressed,
        "isolated_movers":      n_isolated_ever,
        "track_details": [
            {
                "id":           t.track_id,
                "hits":         t.hit_count,
                "misses":       t.miss_count,
                "suppressed":   t.suppressed,
                "suppressed_by": t.suppressed_by,
                "isolated":     t.is_isolated,
                "wf_variance_m": round(t.wf_variance_m, 4) if t.wf_variance_m >= 0 else None,
                "wf_best_h_m":   round(t.wf_best_h_m,  1) if t.wf_best_h_m  >= 0 else None,
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