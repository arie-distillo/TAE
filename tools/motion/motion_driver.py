#!/usr/bin/env python3
"""
motion_driver.py — CLI orchestrator for the motion detection pipeline.
Produces identical JSON and annotated-video output to tools/test_motion.py.

Usage:
    python motion_driver.py path/to/DJI_0001.MP4
    python motion_driver.py DJI_0001.MP4 --mode mog2 --persist 16 --wf-threshold 0.5

Compare outputs with the standalone baseline:
    python tools/test_motion.py  video.MP4 --no-preview  -> video_motion.mp4 + .json
    python motion_driver.py      video.MP4 --no-preview  -> video_motion.mp4 + .json
    # Outputs should be identical (same tracks, same suppression decisions).
"""
from __future__ import annotations

import json
import logging
import queue as _queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import argparse

import cv2

# Allow running from either tools/ or src/core/
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


from motion_config import MotionConfig

from motion import (
    load_telemetry, interpolate_telem,
    FALLBACK_ALT_M, VLM_SNAP_LONG_EDGE,
    # defaults referenced by build_args()
    DEFAULT_PERSIST, DEFAULT_MIN_OBJECT_M, DEFAULT_MAX_OBJECT_M,
    DEFAULT_SENSOR_W_MM, DEFAULT_FOCAL_MM,
    DEFAULT_MAX_SCENE_SPEED, WF_THRESHOLD_M,
    DEFAULT_FLOW_THRESHOLD_PX, DEFAULT_ISOLATION_RADIUS_M,
    DEFAULT_MAX_RANGE_M, DEFAULT_MIN_DISPLACEMENT,
)
from motion_worker import MotionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motion_driver")


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
                   help="Output/annotation resolution scale (default 0.5).")
    p.add_argument("--detect-scale", type=float, default=None, dest="detect_scale",
                   help="CV-processing scale (default: auto from physics). "
                        "All expensive ops (MOG2, KLT, warp, blob detection) run here. "
                        "Auto-clamped so min-object is at least 4 px at detect-res. "
                        "Only useful to set manually for objects ≥ 0.5 m.")
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
                   help="Skip real-time preview window (legacy flag, kept for compatibility)")
    p.add_argument("--live", action="store_true",
                   help="Show annotated frames in a window as they are processed. "
                        "Frames appear at processing rate (~18 fps on CPU). "
                        "Press q to quit early.")
    p.add_argument("--no-output", action="store_true", dest="no_output",
                   help="Skip writing the output mp4. "
                        "Use with --live for display-only mode.")
    p.add_argument("--save-crops", action="store_true",
                   dest="save_crops",
                   help="Save confirmed-track crops to _crops/ directory")
    p.add_argument("--min-confidence", type=float, default=0.70,
                   dest="min_confidence",
                   metavar="F",
                   help="Minimum confidence score [0–1] for a confirmed track to be "
                        "drawn on the output video and live preview.  Tracks below this "
                        "threshold are suppressed visually (still written to JSON).  "
                        "Default: 0.70.  Set to 0.0 to show all confirmed movers.")
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




def _cv2_has_gui() -> bool:
    try:
        info = cv2.getBuildInformation()
        in_gui = False
        for line in info.splitlines():
            if line.strip().startswith("GUI:"):
                in_gui = True
                continue
            if in_gui:
                if line.strip() and not line.startswith(" "):
                    break
                if "YES" in line and any(kw in line for kw in ("QT", "GTK", "Win32", "Cocoa")):
                    return True
        return False
    except Exception:
        return False

_CV2_GUI = _cv2_has_gui()


def main() -> None:
    args = build_args()

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        sys.exit(1)

    # ── Video probe ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        log.error("Cannot open video: %s", args.video)
        sys.exit(1)
    src_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    eff_fps    = min(src_fps, args.fps or src_fps)
    frame_skip = max(1, round(src_fps / eff_fps))

    # ── Output paths ──────────────────────────────────────────────────────────
    out_video = (Path(args.output) if args.output
                 else args.video.parent / (args.video.stem + "_motion.mp4"))
    out_json  = out_video.with_suffix(".json")

    crops_dir: Optional[Path] = None
    if args.save_crops:
        crops_dir = out_video.parent / (out_video.stem + "_crops")
        crops_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = MotionConfig(
        SENSOR_W_MM      = args.sensor_w,
        FOCAL_MM         = args.focal,
        MIN_OBJECT_M     = args.min_object,
        MAX_OBJECT_M     = args.max_object,
        PERSIST_FRAMES   = args.persist,
        MAX_SCENE_SPEED  = args.max_scene_speed,
        MIN_DISPLACEMENT_M = args.min_displacement,
        ISOLATION_RADIUS_M = args.isolation_radius,
        MAX_RANGE_M      = args.max_range,
        WF_THRESHOLD_M   = args.wf_threshold,
        PROCESS_SCALE    = args.scale,
        MIN_CONFIDENCE   = args.min_confidence,
        VLM_ENABLED      = args.vlm_classify,
    )

    # ── Telemetry ─────────────────────────────────────────────────────────────
    srt_path = args.srt if hasattr(args, "srt") else None
    srt_frames = load_telemetry(args.video, srt_path)
    has_telem  = bool(srt_frames)

    # ── Scale / detect resolution ─────────────────────────────────────────────
    if has_telem and srt_frames:
        first_telem = srt_frames[0]
        alt0 = getattr(first_telem, "alt_m", FALLBACK_ALT_M) or FALLBACK_ALT_M
    else:
        alt0 = FALLBACK_ALT_M
    from motion import compute_gsd
    _MIN_BLOB_PX = 4
    _gsd0 = compute_gsd(alt0, args.sensor_w, args.focal, int(full_w * args.scale))
    _min_safe_scale = args.min_object / (_gsd0 * _MIN_BLOB_PX) if _gsd0 > 0 else args.scale
    det_scale = _min_safe_scale if args.detect_scale is None else max(args.detect_scale, _min_safe_scale)
    det_scale = min(det_scale, args.scale)

    # ── Processor ─────────────────────────────────────────────────────────────
    proc = MotionProcessor(cfg)

    # ── Video writer (background thread) ─────────────────────────────────────
    writer = None
    _write_q: _queue.Queue = _queue.Queue(maxsize=4)

    def _encode_worker():
        nonlocal writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, eff_fps, (full_w, full_h))
        while True:
            item = _write_q.get()
            if item is None:
                break
            if writer is not None:
                fr = cv2.resize(item, (full_w, full_h), interpolation=cv2.INTER_LINEAR) \
                     if item.shape[:2] != (full_h, full_w) else item
                writer.write(fr)

    _encode_thread = threading.Thread(target=_encode_worker, daemon=True)
    _encode_thread.start()

    # ── Frame reader (background thread) ─────────────────────────────────────
    _frame_q: _queue.Queue = _queue.Queue(maxsize=8)

    def _reader_worker():
        cap2 = cv2.VideoCapture(str(args.video))
        while True:
            ret, fr = cap2.read()
            _frame_q.put((ret, fr if ret else None))
            if not ret:
                break
        cap2.release()

    _reader_thread = threading.Thread(target=_reader_worker, daemon=True)
    _reader_thread.start()

    log.info("━" * 60)
    log.info("Video      : %s", args.video.name)
    log.info("Resolution : %d×%d  →  processing %.0f×%.0f (scale %.2f)",
             full_w, full_h, full_w * det_scale, full_h * det_scale, det_scale)
    log.info("FPS        : %.1f src  →  %.1f effective (skip=%d)",
             src_fps, eff_fps, frame_skip)
    log.info("Mode       : %s  |  persist=%d", args.mode, args.persist)
    log.info("━" * 60)

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    proc_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = _frame_q.get()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            if (frame_idx - 1) % frame_skip != 0:
                continue
            proc_count += 1
            frame_ms = int((frame_idx - 1) / src_fps * 1000)
            telem = interpolate_telem(srt_frames, frame_ms)

            annotated, tracks = proc.process_frame(
                frame, telem, frame_idx, frame_ms,
                det_scale=det_scale,
                mode=args.mode,
                isolation_radius_m=args.isolation_radius,
                max_range_m=args.max_range,
                min_displacement_m=args.min_displacement,
                flow_threshold_px=args.flow_threshold,
                vlm_classify=args.vlm_classify,
                vlm_api_key=getattr(args, "vlm_api_key", "") or "",
                crops_dir=crops_dir,
            )
            _write_q.put(annotated)

            if not args.no_preview and _CV2_GUI:
                pw = min(int(full_w * args.scale), 1280)
                ph = int(full_h * pw / full_w)
                disp = cv2.resize(annotated, (pw, ph), interpolation=cv2.INTER_AREA)
                cv2.imshow("TAE Motion Driver  (q=quit)", disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if proc_count % 100 == 0:
                elapsed = time.time() - t_start
                pct = 100.0 * frame_idx / max(total_frames, 1)
                s = proc.stats()
                log.info("  [%5.1f%%]  frame %d  elapsed %.0fs  blobs raw/gated: %d/%d",
                         pct, frame_idx, elapsed, s["blobs_raw"], s["blobs_passed_gate"])
    finally:
        _write_q.put(None)
        _encode_thread.join(timeout=10.0)
        _reader_thread.join(timeout=2.0)
        if writer is not None:
            writer.release()
        if not args.no_preview and _CV2_GUI:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # ── VLM post-processing ───────────────────────────────────────────────────
    if args.vlm_classify:
        snap_scale = min(1.0, VLM_SNAP_LONG_EDGE / max(full_w, full_h))
        api_key = getattr(args, "vlm_api_key", "") or ""
        if api_key:
            log.info("VLM post-processing …")
            proc.run_vlm_postprocess(snap_scale, api_key)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_conf = proc.all_tracks_ever()
    s = proc.stats()
    n_vis = s["visible_movers"]
    n_sup = s["suppressed_tracks"]
    n_conf = s["confirmed_tracks"]

    log.info("━" * 60)
    log.info("Finished in %.1f s", elapsed)
    log.info("Source frames read : %d  (processed: %d)", frame_idx, proc_count)
    log.info("Confirmed tracks   : %d", n_conf)
    log.info("  World-fixed (suppressed) : %d", n_sup)
    log.info("  Genuine movers (visible) : %d", n_vis)

    visible_movers = [t for t in all_conf if not t.suppressed]
    if visible_movers:
        log.debug("  Genuine movers (world-variance ≥ %.2f m):", args.wf_threshold)
        log.debug("  %-6s  %-8s  %-8s  %-10s  %-6s", "Track", "Hits", "Misses",
                  "WF var", "Conf")
        for t in sorted(visible_movers, key=lambda x: x.track_id):
            log.debug("  T%-5d  %-8d  %-8d  %-10.3f  %.0f%%",
                      t.track_id, t.hit_count, t.miss_count,
                      t.wf_variance_m if t.wf_variance_m >= 0 else -1,
                      t.confidence * 100)

    log.info("Output video  : %s", out_video)

    gate_filtered = s["blobs_raw"] - s["blobs_passed_gate"]
    gate_pct = 100.0 * gate_filtered / max(s["blobs_raw"], 1)

    summary = {
        "video":                str(args.video),
        "mode":                 args.mode,
        "scale":                args.scale,
        "detect_scale":         det_scale,
        "persist_frames":       args.persist,
        "wf_min_frames":        cfg.WF_MIN_FRAMES,
        "wf_threshold_m":       cfg.WF_THRESHOLD_M,
        "sensor_w_mm":          cfg.SENSOR_W_MM,
        "focal_mm":             cfg.FOCAL_MM,
        "min_object_m":         cfg.MIN_OBJECT_M,
        "max_object_m":         cfg.MAX_OBJECT_M,
        "src_fps":              src_fps,
        "effective_fps":        eff_fps,
        "has_telemetry":        has_telem,
        "frames_total":         frame_idx,
        "frames_processed":     proc_count,
        "warp_failures":        s["warp_failures"],
        "blobs_raw":            s["blobs_raw"],
        "blobs_passed_gate":    s["blobs_passed_gate"],
        "blobs_gate_filtered":  gate_filtered,
        "blobs_gate_filtered_pct": round(gate_pct, 1),
        "isolation_radius_m":   cfg.ISOLATION_RADIUS_M,
        "blobs_isolation_filtered": s["isolation_filtered"],
        "max_range_m":          cfg.MAX_RANGE_M,
        "range_suppressed":     s["range_suppressed"],
        "confirmed_tracks":     n_conf,
        "suppressed_tracks":    n_sup,
        "visible_movers":       n_vis,
        "vlm_suppressed":       s["vlm_suppressed"],
        "min_confidence":       cfg.MIN_CONFIDENCE,
        "elapsed_s":            round(elapsed, 1),
        "track_details": [
            {
                "id":            t.track_id,
                "hits":          t.hit_count,
                "misses":        t.miss_count,
                "suppressed":    t.suppressed,
                "suppressed_by": t.suppressed_by,
                "isolated":      t.is_isolated,
                "wf_variance_m": round(t.wf_variance_m, 4) if t.wf_variance_m >= 0 else None,
                "wf_best_h_m":   round(t.wf_best_h_m, 1)  if t.wf_best_h_m  >= 0 else None,
                "confidence":    round(t.confidence, 3),
            }
            for t in sorted(all_conf, key=lambda x: x.track_id)
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("Summary JSON  : %s", out_json)
    log.info("━" * 60)


if __name__ == "__main__":
    main()
