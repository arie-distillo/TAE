"""
test_motion_worker.py — Standalone validation for MotionDetectionWorker.
=========================================================================

Run from the TAE src/ directory:

    python test_motion_worker.py                      # synthetic frames
    python test_motion_worker.py --video city1-25-40.mov   # real footage, fixed GPS
    python test_motion_worker.py --video city1-25-40.mov --srt city1-25-40.SRT

Synthetic mode
--------------
Creates 50 BGR frames (640×480) with a white rectangle moving horizontally
across a gray background — guarantees detectable motion without needing the
actual drone footage.

Real-footage mode
-----------------
Decodes frames from the supplied video with cv2.VideoCapture.  Telemetry
priority mirrors main.py: embedded djmd protobuf stream first (via
DJIProtobufParser, ffmpeg-based), then .SRT sidecar if --srt is supplied,
then EmbeddedTelemetryParser (exiftool) as fallback, then a fixed GPS anchor
as last resort.

What is validated
-----------------
✔ _state["motion_frame_count"] equals number of frames fed
✔ _state["motion_last_frame"] is non-empty JPEG bytes
✔ _state["motion_tracks"] is a dict (may be empty if MOG2 still warming up)
✔ Any track that IS present has a trajectory list with valid lat/lon entries
✔ All track lat/lon values fall within 0.02° of the GPS anchor (geo sanity)
✔ motion_tracks.json is written to the temp mission directory

_build_map() is patched to a no-op so the test runs without a live DB or
Folium state — map rendering is covered by the Step 4 validation.
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# ── ensure src/ is on sys.path ────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np

# ── TAE imports ───────────────────────────────────────────────────────────────
from core.app_state import _state
from core.video import SRTFrame


# ─────────────────────────────────────────────────────────────────────────────
# Frame generators
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_frames(
    tmp_dir: Path,
    n_frames: int = 50,
    width:    int = 640,
    height:   int = 480,
    anchor_lat: float = 32.0890,
    anchor_lon: float = 34.8820,
    alt_m:      float = 80.0,
) -> list[tuple[Path, SRTFrame]]:
    """
    Generate JPEG frames with a moving white rectangle on a gray background.

    Frames 0-14  : static background (MOG2 model warm-up period).
    Frames 15-49 : white 80×60 rectangle moves 8 px/frame to the right,
                   giving FrameDiff a clear, consistent foreground signal.

    Returns
    -------
    List of (jpeg_path, SRTFrame) pairs ready to feed to the worker.
    """
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    bg_color  = np.full((height, width, 3), 120, dtype=np.uint8)
    rect_h, rect_w = 60, 80

    pairs: list[tuple[Path, SRTFrame]] = []

    for i in range(n_frames):
        frame = bg_color.copy()

        if i >= 15:                          # motion phase
            rx = ((i - 15) * 8) % (width - rect_w)
            ry = height // 2 - rect_h // 2
            cv2.rectangle(
                frame,
                (rx, ry), (rx + rect_w, ry + rect_h),
                (240, 240, 240), -1,         # near-white rectangle
            )

        jpeg_path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        telem = SRTFrame(
            frame_idx    = i,
            timestamp_ms = i * 100,          # 10 fps synthetic cadence
            lat          = anchor_lat,
            lon          = anchor_lon,
            alt_m        = alt_m,
            gimbal_pitch = -90.0,
            gimbal_yaw   = 0.0,
            gimbal_roll  = 0.0,
        )
        pairs.append((jpeg_path, telem))

    print(f"  Generated {n_frames} synthetic frames → {frames_dir}")
    return pairs


def load_video_frames(
    video_path: Path,
    srt_path:   Path | None,
    tmp_dir:    Path,
    max_frames: int = 200,
    anchor_lat: float = 32.0890,
    anchor_lon: float = 34.8820,
    alt_m:      float = 80.0,
) -> tuple[list[tuple[Path, SRTFrame]], float, float]:
    """
    Decode frames from a real video file.

    Returns (frame_pairs, actual_lat, actual_lon) where actual_lat/lon is
    the median GPS position from the loaded telemetry — used as the geo
    validation reference instead of the command-line anchor, which is only
    meaningful as a fixed-GPS fallback when no embedded telemetry exists.
    """
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {video_path.name} | {total} frames | {fps:.1f} fps")

    # Load telemetry: embedded djmd stream first, SRT sidecar second,
    # fixed GPS anchor as last resort.  Mirrors main.py lines 2045-2056.
    from core.video import DJIProtobufParser, EmbeddedTelemetryParser, SRTParser
    _srt_parser = SRTParser()
    telem_frames: list = []

    if srt_path and srt_path.exists():
        telem_frames = _srt_parser.parse(srt_path)
        print(f"  SRT: {len(telem_frames)} telemetry records from {srt_path.name}")
    else:
        # Try embedded djmd protobuf stream (ffmpeg-based, no exiftool needed)
        telem_frames = DJIProtobufParser().parse(video_path)
        if telem_frames:
            print(f"  Embedded djmd: {len(telem_frames)} telemetry records")
        else:
            # Exiftool fallback — single GPS fix, less dense
            telem_frames = EmbeddedTelemetryParser().parse(video_path)
            if telem_frames:
                print(f"  Embedded exiftool: {len(telem_frames)} records")
            else:
                print(f"  No embedded telemetry — using fixed anchor "
                      f"({anchor_lat}, {anchor_lon}), {alt_m} m AGL")

    pairs: list[tuple[Path, SRTFrame]] = []
    idx = 0

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        jpeg_path = frames_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        ts_ms = int(idx * 1000 / fps)

        if telem_frames:
            telem = _srt_parser.interpolate(telem_frames, ts_ms)
            if telem is None:
                telem = telem_frames[-1]
        else:
            telem = SRTFrame(
                frame_idx    = idx,
                timestamp_ms = ts_ms,
                lat          = anchor_lat,
                lon          = anchor_lon,
                alt_m        = alt_m,
                gimbal_pitch = -90.0,
                gimbal_yaw   = 0.0,
                gimbal_roll  = 0.0,
            )

        pairs.append((jpeg_path, telem))
        idx += 1

    cap.release()
    print(f"  Loaded {len(pairs)} frames → {frames_dir}")

    # Derive actual geo reference from telemetry (median avoids outlier fixes)
    if telem_frames:
        import statistics as _st
        actual_lat = _st.median(f.lat for f in telem_frames if f.lat != 0)
        actual_lon = _st.median(f.lon for f in telem_frames if f.lon != 0)
        print(f"  Telemetry median GPS: ({actual_lat:.5f}, {actual_lon:.5f})")
    else:
        actual_lat, actual_lon = anchor_lat, anchor_lon

    return pairs, actual_lat, actual_lon


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    frame_pairs: list[tuple[Path, SRTFrame]],
    tmp_dir:     Path,
    anchor_lat:  float,
    anchor_lon:  float,
    timeout_s:   int = 120,
) -> bool:
    """
    Feed frame_pairs through MotionDetectionWorker and validate _state.

    Returns True if all checks pass, False otherwise.
    """
    import types

    # ── mission paths in temp dir ──────────────────────────────────────────────
    # SimpleNamespace instead of MissionPaths so the test runs both before
    # and after Step 2 (motion_tracks field not yet in MissionPaths dataclass).
    # The worker only reads paths.motion_tracks and paths.root.
    paths = types.SimpleNamespace(
        root          = tmp_dir,
        motion_tracks = tmp_dir / "motion_tracks",
    )
    paths.motion_tracks.mkdir(parents=True, exist_ok=True)

    # ── prime _state with motion keys (Step 2) ────────────────────────────────
    _state.setdefault("motion_tracks",      {})
    _state.setdefault("motion_enabled",     False)
    _state.setdefault("motion_frame_count", 0)
    _state.setdefault("motion_last_frame",  None)
    _state.setdefault("map_center",         [anchor_lat, anchor_lon])
    _state.setdefault("map_zoom",           16)
    _state.setdefault("mission_paths",      None)

    # ── import worker AFTER _state is primed ──────────────────────────────────
    from core.motion_worker import MotionDetectionWorker

    worker = MotionDetectionWorker()
    n_frames = len(frame_pairs)

    # ── patch _build_map to a no-op (no DB available in standalone test) ──────
    print(f"\n  Starting worker | {n_frames} frames to process …")

    import core.services as _svc  # ensure loaded before patch
    with patch.object(_svc, "_build_map", return_value=None):
        worker.start(paths=paths, analyst=None)

        # Feed frames with blocking put() — not worker.enqueue() which uses
        # put_nowait() and silently drops when the queue is full.
        # In production, non-blocking drops are correct (the streaming
        # callback must not block).  In this test we want guaranteed delivery,
        # so we block: the bounded queue (QUEUE_MAXSIZE=60) provides natural
        # backpressure that paces feeding to match the worker's throughput.
        #
        # NOTE: with real 4K footage, each frame takes ~200-400ms on CPU.
        # The main thread will block here for the full processing duration.
        # The progress line shows it is alive.
        t0 = time.time()
        for i, (jpeg_path, telem) in enumerate(frame_pairs):
            worker._queue.put((jpeg_path, telem))   # blocks when queue is full
            fc  = _state.get("motion_frame_count", 0)
            ela = time.time() - t0
            fps = fc / ela if ela > 0 else 0.0
            print(
                f"\r  Feeding {i+1}/{n_frames} | processed {fc} | "
                f"{fps:.1f} fps  ",
                end="", flush=True,
            )
        print()   # newline after progress line

        # Wait for the worker to drain whatever is still in the queue
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            fc = _state.get("motion_frame_count", 0)
            print(
                f"\r  Draining … processed {fc}/{n_frames}  ",
                end="", flush=True,
            )
            if fc >= n_frames:
                break
            time.sleep(0.25)
        print()   # newline after progress line

        worker.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # Assertions
    # ─────────────────────────────────────────────────────────────────────────
    passed  = True
    results = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        symbol = "✔" if ok else "✘"
        msg    = f"  {symbol}  {label}"
        if detail:
            msg += f"  [{detail}]"
        results.append(msg)
        if not ok:
            passed = False

    fc = _state.get("motion_frame_count", 0)
    check(
        f"motion_frame_count == {n_frames}",
        fc == n_frames,
        f"got {fc}",
    )

    last_frame = _state.get("motion_last_frame")
    check(
        "motion_last_frame is JPEG bytes",
        isinstance(last_frame, (bytes, bytearray)) and len(last_frame) > 100,
        f"{len(last_frame) if last_frame else 0} bytes",
    )

    tracks = _state.get("motion_tracks", {})
    check(
        "motion_tracks is a dict",
        isinstance(tracks, dict),
        type(tracks).__name__,
    )

    # Track count — MOG2 needs ~15 frames to warm up; with 50 synthetic frames
    # we expect at least one track in the motion phase.  Real footage may vary.
    check(
        "at least one motion track detected",
        len(tracks) >= 1,
        f"{len(tracks)} track(s)",
    )

    # Schema + geo sanity for each track
    GEO_TOLERANCE = 0.02   # degrees — generous for flat-earth projection test
    geo_ok = True
    schema_ok = True
    for tid, td in tracks.items():
        # Schema check
        for key in ("track_id", "label", "color", "trajectory"):
            if key not in td:
                schema_ok = False
        traj = td.get("trajectory", [])
        for pt in traj:
            if "bbox" in pt:
                schema_ok = False        # bbox must be stripped by to_dict()
            lat = pt.get("lat", 0.0)
            lon = pt.get("lon", 0.0)
            if (abs(lat - anchor_lat) > GEO_TOLERANCE or
                    abs(lon - anchor_lon) > GEO_TOLERANCE):
                geo_ok = False

    check("track schema valid (no bbox in trajectory)", schema_ok)
    check(
        f"all track lat/lon within {GEO_TOLERANCE}° of anchor",
        geo_ok,
        f"anchor=({anchor_lat}, {anchor_lon})",
    )

    # Persistence check
    mt_json = paths.motion_tracks / "motion_tracks.json"
    json_ok = mt_json.exists()
    if json_ok:
        data = json.loads(mt_json.read_text())
        json_ok = isinstance(data, list)
    check(
        "motion_tracks.json written as JSON list",
        json_ok,
        str(mt_json),
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n  Results:")
    for line in results:
        print(f"  {line}")

    n_tracks = len(tracks)
    if n_tracks:
        print(f"\n  Track details ({n_tracks} track(s)):")
        for tid, td in list(tracks.items())[:5]:    # cap at 5 for readability
            traj = td.get("trajectory", [])
            first = traj[0] if traj else {}
            last  = traj[-1] if traj else {}
            print(
                f"    {tid}  label={td['label']}  "
                f"pts={len(traj)}  "
                f"lat={first.get('lat', 0):.5f}→{last.get('lat', 0):.5f}  "
                f"lon={first.get('lon', 0):.5f}→{last.get('lon', 0):.5f}"
            )
        if n_tracks > 5:
            print(f"    … and {n_tracks - 5} more")

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone MotionDetectionWorker test")
    p.add_argument("--video",   type=Path, default=None,
                   help="Path to drone video (mp4/mov). Omit for synthetic frames.")
    p.add_argument("--srt",     type=Path, default=None,
                   help="Path to matching .SRT sidecar. Optional with --video.")
    p.add_argument("--lat",     type=float, default=32.0890,
                   help="GPS anchor latitude (used when no SRT). Default: Petah Tikva.")
    p.add_argument("--lon",     type=float, default=34.8820,
                   help="GPS anchor longitude.")
    p.add_argument("--alt",     type=float, default=80.0,
                   help="AGL altitude in metres.")
    p.add_argument("--frames",  type=int,   default=200,
                   help="Max frames to load from real video (default 200).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("MotionDetectionWorker — standalone test")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="tae_motion_test_") as tmp:
        tmp_dir = Path(tmp)

        if args.video:
            print(f"\n[Mode] Real footage: {args.video}")
            frame_pairs, actual_lat, actual_lon = load_video_frames(
                video_path = args.video,
                srt_path   = args.srt,
                tmp_dir    = tmp_dir,
                max_frames = args.frames,
                anchor_lat = args.lat,
                anchor_lon = args.lon,
                alt_m      = args.alt,
            )
        else:
            print("\n[Mode] Synthetic frames (moving rectangle)")
            frame_pairs = make_synthetic_frames(
                tmp_dir    = tmp_dir,
                anchor_lat = args.lat,
                anchor_lon = args.lon,
                alt_m      = args.alt,
            )
            actual_lat, actual_lon = args.lat, args.lon

        if not frame_pairs:
            print("ERROR: no frames to process")
            sys.exit(1)

        passed = run_test(
            frame_pairs = frame_pairs,
            tmp_dir     = tmp_dir,
            anchor_lat  = actual_lat,
            anchor_lon  = actual_lon,
        )

    print()
    if passed:
        print("=" * 60)
        print("PASS — all checks passed")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("FAIL — one or more checks failed (see ✘ lines above)")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()