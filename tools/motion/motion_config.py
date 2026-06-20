"""
motion_config.py — Motion Detection Configuration
==================================================
All tunable parameters for the TAE motion-detection pipeline.

Standalone usage:
    from motion_config import MotionConfig
    cfg = MotionConfig()

TAE integration — copy the field definitions below into TAE's Settings class
(src/core/config.py).  They are written as plain class attributes so they
can be pasted in directly without modification.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MotionConfig:
    # ── Camera (Mavic 3 Enterprise defaults) ─────────────────────────────────
    SENSOR_W_MM:           float = 6.3
    FOCAL_MM:              float = 4.5
    FALLBACK_ALT_M:        float = 80.0    # used when telemetry unavailable
    FALLBACK_GIMBAL_PITCH: float = -90.0   # assume nadir when pitch unknown

    # ── Object size gates (physics-derived) ──────────────────────────────────
    MIN_OBJECT_M:    float = 0.2    # minimum object dimension (metres)
    MAX_OBJECT_M:    float = 50.0   # maximum object dimension (metres)
    MAX_ASPECT_RATIO: float = 8.0   # reject very elongated blobs

    # ── Background subtractor (MOG2) ─────────────────────────────────────────
    MOG2_HISTORY:       int   = 150    # frames to build background model
    MOG2_VAR_THRESHOLD: float = 20.0   # per-pixel variance threshold

    # ── Dense optical flow (mode=flow) ───────────────────────────────────────
    FLOW_PYR_SCALE:    float = 0.5
    FLOW_LEVELS:       int   = 3
    FLOW_WINSIZE:      int   = 9
    FLOW_ITERATIONS:   int   = 3
    FLOW_POLY_N:       int   = 5
    FLOW_POLY_SIGMA:   float = 1.2
    FLOW_THRESHOLD_PX: float = 2.0    # residual flow threshold (px/frame, detect-res)

    # ── Isolation gate ────────────────────────────────────────────────────────
    ISOLATION_RADIUS_M: float = 0.0   # 0 = disabled; 10m kills most construction clusters

    # ── Maximum ground-range gate ─────────────────────────────────────────────
    MAX_RANGE_M: float = 0.0          # 0 = disabled; 500m good for oblique shots

    # ── Morphological cleanup ─────────────────────────────────────────────────
    MORPH_KSIZE: int = 3

    # ── Ego-motion estimation (KLT + RANSAC homography) ──────────────────────
    KLT_GRID_STEP_PX:  int   = 50
    KLT_WIN_SIZE:      tuple = (21, 21)
    KLT_MAX_LEVEL:     int   = 4
    RANSAC_THRESH_PX:  float = 3.0
    MIN_INLIERS:       int   = 8

    # ── Tracker ───────────────────────────────────────────────────────────────
    PERSIST_FRAMES:   int   = 16   # consecutive hits to confirm; ≥ WF_MIN_FRAMES
    ORPHAN_FRAMES:    int   = 6    # consecutive misses before retiring
    MATCH_DIST_M:     float = 6.0  # real-world match gate radius (metres)

    # ── Dynamic persist ───────────────────────────────────────────────────────
    MAX_SCENE_SPEED:  float = 15.0  # px/frame above which extra persist is added
    PERSIST_SPEED_STEP: float = 5.0 # px/frame excess per +1 extra persist frame
    MAX_EXTRA_PERSIST:  int   = 10  # cap on additional frames
    MIN_DISPLACEMENT_M: float = 0.0 # 0 = disabled

    # ── World-fixed point filter ──────────────────────────────────────────────
    WF_MIN_FRAMES:   int   = 8     # minimum hits before WF test fires
    WF_RETEST_EVERY: int   = 15    # retest every N additional hits
    WF_N_SCAN:       int   = 60    # altitude candidates in h-scan
    WF_THRESHOLD_M:  float = 0.5   # variance below this → world-fixed
    # Hysteresis: only un-suppress if variance > WF_RECOVERY_FACTOR × threshold
    # Blocks oscillating FP tracks (median escape variance 0.7 m) while allowing
    # genuine movers that were briefly mis-classified (bird: 1.71 m) to recover.

    # ── Display ───────────────────────────────────────────────────────────────
    PROCESS_SCALE:   float = 0.5   # annotation resolution scale
    MIN_CONFIDENCE:  float = 0.70  # minimum confidence to show track in UI

    # ── VLM (optional semantic filter) ───────────────────────────────────────
    VLM_ENABLED:          bool  = False
    VLM_MODEL:            str   = "qwen/qwen-2.5-vl-72b-instruct"
    VLM_SNAPSHOT_INTERVAL: int  = 90    # frames between snapshots
    VLM_MAX_DET_PER_CALL:  int  = 25    # max detections per API call
    VLM_MAX_TOKENS:        int  = 4096
    VLM_SNAP_LONG_EDGE:    int  = 1280


# ─────────────────────────────────────────────────────────────────────────────
# TAE INTEGRATION REFERENCE
# Copy the following field definitions into TAE's Settings class
# (src/core/config.py) inside the Settings(BaseSettings) body:
# ─────────────────────────────────────────────────────────────────────────────
#
#     # ── Motion pipeline (calibrated against tools/test_motion.py) ────────
#     MOTION_SENSOR_W_MM:       float = 6.3
#     MOTION_FOCAL_MM:          float = 4.5
#     MOTION_MIN_OBJECT_M:      float = 0.2
#     MOTION_MAX_OBJECT_M:      float = 50.0
#     MOTION_ISOLATION_RADIUS_M: float = 0.0
#     MOTION_MAX_RANGE_M:        float = 0.0
#     MOTION_PERSIST_FRAMES:    int   = 16
#     MOTION_WF_THRESHOLD_M:    float = 0.5
#     MOTION_WF_RECOVERY_FACTOR: float = 2.0
#     MOTION_PROCESS_SCALE:     float = 0.5
#     MOTION_MIN_CONFIDENCE:    float = 0.70
#     MOTION_VLM_ENABLED:       bool  = False
#     MOTION_MIN_TRAJ_PTS:      int   = 8
