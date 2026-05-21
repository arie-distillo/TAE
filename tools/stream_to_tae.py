#!/usr/bin/env python3
"""
stream_to_tae.py  —  TAE Test Harness · Tool 2
===============================================
Simulate a live drone video feed by posting frames to a running TAE
instance at a controlled rate.  Exercises the exact same ingestion
code path as a real operator uploading images.

Two streaming modes
───────────────────
IMAGE mode (--source <dir>)
    Posts original DJI JPEGs in sorted order.
    XMP metadata is preserved in the file → TAE extracts it normally.
    Recommended for functional testing.

VIDEO mode (--source <file.mp4> --srt <file.srt>)
    Extracts frames from an MP4 produced by make_drone_video.py.
    Re-injects telemetry from the SRT sidecar as a raw XMP packet
    into each extracted JPEG before posting.
    Use this to test the full video → frame → TAE pipeline.

Usage
─────
# Stream from source images (cleanest, preserves all XMP):
python stream_to_tae.py \\
    --source  ./data/raw \\
    --tae-url http://localhost:8000 \\
    --fps 2

# Stream from synthetic video + SRT:
python stream_to_tae.py \\
    --source  ./data/test_video/drone_mission.mp4 \\
    --srt     ./data/test_video/drone_mission.srt \\
    --tae-url http://localhost:8000 \\
    --fps 2

# Speed multiplier (2× real-time):
python stream_to_tae.py --source ./data/raw --tae-url http://localhost:8000 \\
    --fps 5 --speed 2.0

# Dry-run (no HTTP posts, just show what would be sent):
python stream_to_tae.py --source ./data/raw --dry-run

Dependencies:
    pip install requests opencv-python pillow
    # For VIDEO mode only: pip install moviepy
"""

import argparse
import io
import json
import re
import struct
import sys
import time
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# SRT parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_srt(srt_path: Path) -> list[dict]:
    """
    Parse an SRT file produced by make_drone_video.py.
    Returns list of telemetry dicts, one per subtitle entry, in order.
    """
    text = srt_path.read_text(encoding="utf-8")
    # Split on blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    entries = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        # lines[0] = index, lines[1] = timestamps, lines[2+] = text
        telem_text = " ".join(lines[2:]).strip()
        try:
            entries.append(json.loads(telem_text))
        except json.JSONDecodeError:
            # Gracefully handle DJI-native SRT format (non-JSON subtitles)
            entries.append({"raw_subtitle": telem_text})
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# XMP re-injection
# ─────────────────────────────────────────────────────────────────────────────

_XMP_PACKET_TEMPLATE = """\
<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description rdf:about=''
        xmlns:drone-dji='http://www.dji.com/drone-dji/1.0/'
        drone-dji:RelativeAltitude="{alt_m}"
        drone-dji:GimbalPitchDegree="{gimbal_pitch}"
        drone-dji:GimbalRollDegree="{gimbal_roll}"
        drone-dji:GimbalYawDegree="{gimbal_yaw}"
        drone-dji:FlightYawDegree="{flight_yaw}"
        drone-dji:FlightPitchDegree="{flight_pitch}"
        drone-dji:FlightRollDegree="{flight_roll}"
        drone-dji:GPSLatitude="{lat}"
        drone-dji:GPSLongitude="{lon}"
    />
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
# NOTE: attribute style (field="value") is required — TAE's _extract_dji_data
# uses re.search(rf'{field}="([^"]+)"', content) for all field extraction.

# DJI XMP is embedded as an APP1 marker with this namespace URI prefix
_XMP_MARKER_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"


def _build_xmp_packet(telem: dict) -> bytes:
    """Render XMP XML for a telemetry dict, encoded as bytes."""
    def _v(key: str) -> str:
        v = telem.get(key)
        return str(v) if v is not None else "0.0"

    xml = _XMP_PACKET_TEMPLATE.format(
        alt_m       = _v("alt_m"),
        gimbal_pitch= _v("gimbal_pitch"),
        gimbal_roll = _v("gimbal_roll"),
        gimbal_yaw  = _v("gimbal_yaw"),
        flight_yaw  = _v("flight_yaw"),
        flight_pitch= _v("flight_pitch"),
        flight_roll = _v("flight_roll"),
        lat         = _v("lat"),
        lon         = _v("lon"),
    )
    return xml.encode("utf-8")


def _inject_xmp_into_jpeg(jpeg_bytes: bytes, telem: dict) -> bytes:
    """
    Insert a DJI-style XMP APP1 marker into a JPEG byte string.

    The JPEG structure is:
        SOI (2 bytes: FF D8)
        [optional existing markers]
        ...image data...
        EOI

    We insert our XMP APP1 immediately after SOI, before any existing APP1
    markers, so TAE's regex scan finds it in the first read window.
    """
    if not jpeg_bytes.startswith(b"\xff\xd8"):
        raise ValueError("Not a valid JPEG")

    xmp_data    = _build_xmp_packet(telem)
    marker_body = _XMP_MARKER_PREFIX + xmp_data
    # APP1 marker: FF E1, 2-byte big-endian length (includes the 2 length bytes)
    marker_len  = len(marker_body) + 2
    app1_marker = (
        b"\xff\xe1"
        + struct.pack(">H", marker_len)
        + marker_body
    )
    # Insert after SOI
    return jpeg_bytes[:2] + app1_marker + jpeg_bytes[2:]


# ─────────────────────────────────────────────────────────────────────────────
# Frame sources
# ─────────────────────────────────────────────────────────────────────────────

def _image_frames(source_dir: Path):
    """
    Yield (filename, jpeg_bytes) for each JPEG in source_dir, sorted by name.
    XMP is already embedded — no modification needed.
    """
    images = sorted(
        [p for p in source_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if not images:
        print(f"ERROR: no JPEG images found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    for img_path in images:
        yield img_path.name, img_path.read_bytes()


def _video_frames(mp4_path: Path, srt_entries: list[dict]):
    """
    Extract frames from an MP4 and pair each with the corresponding SRT
    telemetry entry.  XMP is re-injected into each extracted JPEG.

    Yields (filename, jpeg_bytes_with_xmp).
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required for video mode. "
              "pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open video {mp4_path}", file=sys.stderr)
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame as JPEG
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            print(f"  ⚠  Frame {frame_idx}: JPEG encode failed — skipped")
            frame_idx += 1
            continue

        jpeg_bytes = buf.tobytes()

        # Get matching telemetry from SRT (best-effort: clamp to last entry)
        if srt_entries:
            telem = srt_entries[min(frame_idx, len(srt_entries) - 1)]
        else:
            telem = {}

        # Re-inject XMP so TAE's _extract_dji_data can parse it
        try:
            jpeg_bytes = _inject_xmp_into_jpeg(jpeg_bytes, telem)
        except Exception as e:
            print(f"  ⚠  Frame {frame_idx}: XMP injection failed ({e}) — sending raw")

        fname = f"stream_frame_{frame_idx:05d}.jpg"
        yield fname, jpeg_bytes
        frame_idx += 1

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP posting
# ─────────────────────────────────────────────────────────────────────────────

def _post_frame(
    session: requests.Session,
    tae_url: str,
    fname: str,
    jpeg_bytes: bytes,
) -> bool:
    """
    POST a single JPEG frame to TAE's /upload endpoint.
    Returns True on HTTP 200, False otherwise.
    """
    url = tae_url.rstrip("/") + "/upload"
    files = {"files": (fname, jpeg_bytes, "image/jpeg")}
    try:
        resp = session.post(url, files=files, timeout=30)
        return resp.status_code == 200
    except requests.RequestException as e:
        print(f"    HTTP error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Progress display
# ─────────────────────────────────────────────────────────────────────────────

def _progress_bar(current: int, total: int | None, width: int = 30) -> str:
    if total:
        filled = int(width * current / total)
        bar    = "█" * filled + "░" * (width - filled)
        pct    = f"{100 * current / total:5.1f}%"
        return f"[{bar}] {pct} ({current}/{total})"
    return f"[{'·' * width}] frame {current}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate live drone video streaming to a TAE instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source",  "-s", required=True,
                        help="Source: JPEG directory (image mode) or MP4 file (video mode)")
    parser.add_argument("--srt",           default=None,
                        help="SRT sidecar (required for video mode)")
    parser.add_argument("--tae-url", "-u", default="http://localhost:8000",
                        help="TAE base URL (default: http://localhost:8000)")
    parser.add_argument("--fps",     "-f", type=float, default=1.0,
                        help="Streaming rate in frames/sec (default: 1.0)")
    parser.add_argument("--speed",         type=float, default=1.0,
                        help="Speed multiplier (e.g. 2.0 = twice real-time, default: 1.0)")
    parser.add_argument("--batch",         type=int,   default=1,
                        help="Frames per POST batch (default: 1 — truly frame-by-frame)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Show what would be sent without making HTTP requests")
    parser.add_argument("--loop",          action="store_true",
                        help="Loop the frame sequence indefinitely (Ctrl-C to stop)")
    parser.add_argument("--start-frame",   type=int, default=0,
                        help="Skip the first N frames (default: 0)")
    args = parser.parse_args()

    source_path = Path(args.source)
    inter_frame_s = (1.0 / args.fps) / args.speed

    # ── Banner ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  TAE Live Stream Simulator")
    print(f"{'─'*65}")
    mode = "VIDEO" if source_path.is_file() else "IMAGE"
    print(f"  Mode      : {mode}")
    print(f"  Source    : {source_path}")
    print(f"  TAE URL   : {args.tae_url}")
    print(f"  Rate      : {args.fps} fps  ×{args.speed}  "
          f"(Δt = {inter_frame_s*1000:.0f} ms / frame)")
    print(f"  Batch     : {args.batch} frame(s) / POST")
    if args.dry_run:
        print(f"  *** DRY RUN — no HTTP requests will be made ***")
    print(f"{'─'*65}\n")

    # ── Verify TAE reachability ─────────────────────────────────────────────
    if not args.dry_run:
        try:
            r = requests.get(args.tae_url.rstrip("/") + "/", timeout=5)
            print(f"  ✓  TAE reachable  (HTTP {r.status_code})\n")
        except requests.RequestException as e:
            print(f"  ✗  Cannot reach TAE at {args.tae_url}: {e}")
            print("     Is the TAE server running?  Use --dry-run to skip this check.")
            sys.exit(1)

    # ── Build frame source ──────────────────────────────────────────────────
    srt_entries: list[dict] = []
    if source_path.is_file():
        # VIDEO mode
        if args.srt:
            srt_entries = _parse_srt(Path(args.srt))
            print(f"  Loaded {len(srt_entries)} SRT telemetry entries")
        else:
            print("  ⚠  No --srt provided — frames will be sent without XMP metadata")

        def frame_source():
            yield from _video_frames(source_path, srt_entries)
    elif source_path.is_dir():
        # IMAGE mode
        def frame_source():
            yield from _image_frames(source_path)
    else:
        print(f"ERROR: {source_path} is neither a file nor a directory", file=sys.stderr)
        sys.exit(1)

    # ── Stream loop ─────────────────────────────────────────────────────────
    session = requests.Session()
    stats = {"sent": 0, "ok": 0, "fail": 0, "bytes": 0}

    def _run_once():
        batch: list[tuple[str, bytes]] = []

        for frame_idx, (fname, jpeg_bytes) in enumerate(frame_source()):
            if frame_idx < args.start_frame:
                continue

            batch.append((fname, jpeg_bytes))

            if len(batch) < args.batch:
                continue  # accumulate batch

            # ── Post batch ──────────────────────────────────────────────
            t0 = time.monotonic()

            if args.dry_run:
                for fn, jb in batch:
                    print(f"  [DRY-RUN] would POST {fn}  ({len(jb):,} bytes)")
                    stats["sent"] += 1
                    stats["bytes"] += len(jb)
            else:
                for fn, jb in batch:
                    ok = _post_frame(session, args.tae_url, fn, jb)
                    stats["sent"] += 1
                    stats["bytes"] += len(jb)
                    if ok:
                        stats["ok"] += 1
                        status = "✓"
                    else:
                        stats["fail"] += 1
                        status = "✗"

                    print(
                        f"  {status}  {fn:<35} "
                        f"{len(jb)/1024:6.1f} KB  "
                        f"[sent={stats['sent']} ok={stats['ok']} fail={stats['fail']}]"
                    )

            batch.clear()

            # ── Rate limiting ───────────────────────────────────────────
            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, inter_frame_s * args.batch - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Flush any remaining partial batch
        if batch and not args.dry_run:
            for fn, jb in batch:
                ok = _post_frame(session, args.tae_url, fn, jb)
                stats["sent"] += 1
                stats["bytes"] += len(jb)
                (stats["ok"] if ok else stats["fail"])
                print(f"  {'✓' if ok else '✗'}  {fn} (flush)")

    try:
        if args.loop:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n  ── Loop iteration {iteration} ──────────────────────────────")
                _run_once()
                print(f"  Loop complete. Restarting in 2 s … (Ctrl-C to stop)")
                time.sleep(2)
        else:
            _run_once()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Stream summary")
    print(f"{'─'*65}")
    print(f"  Frames sent : {stats['sent']}")
    print(f"  Successful  : {stats['ok']}")
    print(f"  Failed      : {stats['fail']}")
    print(f"  Data sent   : {stats['bytes'] / 1024 / 1024:.2f} MB")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
