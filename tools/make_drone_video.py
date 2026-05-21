#!/usr/bin/env python3
"""
make_drone_video.py  —  TAE Test Harness · Tool 1
==================================================
Compose a directory of DJI JPEG drone images into a synthetic MP4 video
with a parallel SRT sidecar that carries per-frame telemetry, plus a
pose_metadata.json in the format TAE's ingest pipeline already expects.

Dependencies:
    pip install moviepy pillow

Usage:
    python make_drone_video.py \\
        --input  ./data/raw \\
        --output ./data/test_video \\
        --fps 5 \\
        --resolution 1920x1080 \\
        --sort-by name          # name | mtime | gps_seq

Outputs (all in <output>/):
    drone_mission.mp4        synthetic video (H.264)
    drone_mission.srt        SRT sidecar — each subtitle = one source frame,
                             text block = telemetry JSON
    pose_metadata.json       TAE-compatible frame-keyed metadata dict

SRT telemetry block format (one subtitle per source frame):
    {"lat": 32.08, "lon": 34.78, "alt_m": 50.0,
     "gimbal_pitch": -90.0, "gimbal_yaw": 12.5, "gimbal_roll": 0.0,
     "flight_yaw": 12.5, "flight_pitch": 0.0, "flight_roll": 0.0,
     "source_image": "DJI_0031.JPG", "frame_index": 3}
"""

import argparse
import json
import re
import struct
import sys
from datetime import timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# XMP extraction  (same logic as TAE's _extract_dji_data)
# ─────────────────────────────────────────────────────────────────────────────

_XMP_FIELDS = [
    "RelativeAltitude",
    "AbsoluteAltitude",
    "GimbalPitchDegree",
    "GimbalRollDegree",
    "GimbalYawDegree",
    "FlightYawDegree",
    "FlightPitchDegree",
    "FlightRollDegree",
    "GPSLatitude",
    "GPSLongitude",
]


def _read_xmp(image_path: Path) -> dict:
    """
    Read DJI XMP fields from a JPEG.
    Returns a dict with all _XMP_FIELDS as keys; missing fields are None.
    """
    with open(image_path, "rb") as f:
        raw = f.read(131072)  # first 128 KB is more than enough for XMP
    content = raw.decode("latin-1", errors="ignore")

    result = {}
    for field in _XMP_FIELDS:
        m = re.search(rf'{field}="([^"]+)"', content)
        result[field] = m.group(1) if m else None
    return result


def _parse_gps(xmp: dict) -> tuple[float | None, float | None]:
    """Parse GPSLatitude / GPSLongitude from XMP (handles decimal and DMS)."""
    def _to_float(val: str | None) -> float | None:
        if val is None:
            return None
        val = val.strip()
        # Decimal degrees: "+32.080000" or "32.080000"
        try:
            return float(val.lstrip("+"))
        except ValueError:
            pass
        # DMS: "32,4.800000N" or "32/1,4800000/100000,0/1"
        # best-effort: just return None and let caller handle
        return None

    return _to_float(xmp.get("GPSLatitude")), _to_float(xmp.get("GPSLongitude"))


def _xmp_to_telemetry(image_name: str, xmp: dict, frame_index: int) -> dict:
    """Convert raw XMP dict to the TAE-compatible telemetry record."""
    lat, lon = _parse_gps(xmp)

    def _f(key: str) -> float | None:
        v = xmp.get(key)
        if v is None:
            return None
        try:
            return float(v.lstrip("+"))
        except (ValueError, AttributeError):
            return None

    return {
        "lat":          lat,
        "lon":          lon,
        "alt_m":        _f("RelativeAltitude"),
        "abs_alt_m":    _f("AbsoluteAltitude"),
        "gimbal_pitch": _f("GimbalPitchDegree"),
        "gimbal_roll":  _f("GimbalRollDegree"),
        "gimbal_yaw":   _f("GimbalYawDegree"),
        "flight_yaw":   _f("FlightYawDegree"),
        "flight_pitch": _f("FlightPitchDegree"),
        "flight_roll":  _f("FlightRollDegree"),
        "source_image": image_name,
        "frame_index":  frame_index,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sorting strategies
# ─────────────────────────────────────────────────────────────────────────────

def _sort_images(images: list[Path], strategy: str) -> list[Path]:
    if strategy == "name":
        return sorted(images, key=lambda p: p.name)
    if strategy == "mtime":
        return sorted(images, key=lambda p: p.stat().st_mtime)
    if strategy == "gps_seq":
        # Sort by (lat, lon) to approximate boustrophedon flight order —
        # imperfect but better than random for survey patterns.
        def _gps_key(p: Path):
            xmp = _read_xmp(p)
            lat, lon = _parse_gps(xmp)
            return (lat or 0.0, lon or 0.0)
        return sorted(images, key=_gps_key)
    raise ValueError(f"Unknown sort strategy: {strategy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# SRT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    td = timedelta(milliseconds=ms)
    total_s = int(td.total_seconds())
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    frac = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{frac:03d}"


def _write_srt(telemetry_list: list[dict], fps: float, out_path: Path) -> None:
    """
    Write an SRT file where each subtitle corresponds to one source frame.
    The text block is a JSON telemetry object — parseable by stream_to_tae.py.
    """
    frame_ms = int(1000 / fps)
    lines = []
    for i, telem in enumerate(telemetry_list):
        start_ms = i * frame_ms
        end_ms   = start_ms + frame_ms
        lines.append(str(i + 1))
        lines.append(f"{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}")
        lines.append(json.dumps(telem, separators=(",", ":")))
        lines.append("")  # blank line between entries

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  SRT  → {out_path}  ({len(telemetry_list)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# Video builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_video(
    images: list[Path],
    fps: float,
    resolution: tuple[int, int] | None,
    out_path: Path,
) -> None:
    """
    Assemble images into an MP4 using moviepy's ImageSequenceClip.
    Each image is shown for 1/fps seconds.
    """
    try:
        from moviepy import ImageSequenceClip  # moviepy 2.x
    except ImportError:
        from moviepy.editor import ImageSequenceClip  # moviepy 1.x

    str_paths = [str(p) for p in images]
    clip = ImageSequenceClip(str_paths, fps=fps)

    if resolution:
        w, h = resolution
        clip = clip.resized((w, h))

    print(f"  Encoding {len(images)} frames @ {fps} fps → {out_path} …")
    clip.write_videofile(
        str(out_path),
        codec="libx264",
        audio=False,
        preset="fast",
        ffmpeg_params=["-crf", "23"],
        logger=None,  # suppress moviepy progress bar (use our own)
    )
    print(f"  ✓  MP4  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a synthetic drone video + SRT + pose_metadata.json from DJI JPEGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  "-i", required=True,  help="Directory of DJI JPEG images")
    parser.add_argument("--output", "-o", required=True,  help="Output directory")
    parser.add_argument("--fps",    "-f", type=float, default=5.0,
                        help="Video frame rate (default: 5)")
    parser.add_argument("--resolution", "-r", default=None,
                        help="Output resolution WxH e.g. 1920x1080 (default: original)")
    parser.add_argument("--sort-by", choices=["name", "mtime", "gps_seq"], default="name",
                        help="Image ordering strategy (default: name)")
    parser.add_argument("--stem", default="drone_mission",
                        help="Output file stem (default: drone_mission)")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Gather images ──────────────────────────────────────────────────────
    images = sorted(
        [p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not images:
        print(f"ERROR: no JPEG/PNG images found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  TAE Drone Video Builder")
    print(f"{'─'*60}")
    print(f"  Input  : {in_dir}  ({len(images)} images)")
    print(f"  Output : {out_dir}")
    print(f"  FPS    : {args.fps}")
    print(f"  Sort   : {args.sort_by}")
    print(f"{'─'*60}\n")

    # ── Sort ───────────────────────────────────────────────────────────────
    print("  Sorting images …")
    images = _sort_images(images, args.sort_by)

    # ── Extract telemetry ──────────────────────────────────────────────────
    print("  Extracting XMP telemetry …")
    telemetry_list: list[dict] = []
    pose_meta: dict = {}
    missing_xmp = 0

    for idx, img_path in enumerate(images):
        xmp   = _read_xmp(img_path)
        telem = _xmp_to_telemetry(img_path.name, xmp, idx)
        telemetry_list.append(telem)
        pose_meta[img_path.name] = telem

        if telem["lat"] is None or telem["lon"] is None:
            missing_xmp += 1
            print(f"    ⚠  No GPS in {img_path.name}")

    print(f"  Extracted {len(telemetry_list)} telemetry records "
          f"({missing_xmp} missing GPS)")

    # ── Parse resolution ───────────────────────────────────────────────────
    resolution = None
    if args.resolution:
        try:
            w, h = map(int, args.resolution.lower().split("x"))
            resolution = (w, h)
        except ValueError:
            print(f"ERROR: invalid resolution {args.resolution!r} — use WxH e.g. 1920x1080",
                  file=sys.stderr)
            sys.exit(1)

    # ── Build MP4 ──────────────────────────────────────────────────────────
    mp4_path = out_dir / f"{args.stem}.mp4"
    _build_video(images, args.fps, resolution, mp4_path)

    # ── Write SRT ──────────────────────────────────────────────────────────
    srt_path = out_dir / f"{args.stem}.srt"
    _write_srt(telemetry_list, args.fps, srt_path)

    # ── Write pose_metadata.json ───────────────────────────────────────────
    meta_path = out_dir / "pose_metadata.json"
    meta_path.write_text(json.dumps(pose_meta, indent=2), encoding="utf-8")
    print(f"  ✓  JSON → {meta_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    duration_s = len(images) / args.fps
    lats = [t["lat"] for t in telemetry_list if t["lat"] is not None]
    lons = [t["lon"] for t in telemetry_list if t["lon"] is not None]
    alts = [t["alt_m"] for t in telemetry_list if t["alt_m"] is not None]

    print(f"\n{'─'*60}")
    print(f"  Done!")
    print(f"  Frames   : {len(images)}")
    print(f"  Duration : {duration_s:.1f} s  ({duration_s/60:.1f} min)")
    if lats:
        print(f"  Center   : {sum(lats)/len(lats):.6f}, {sum(lons)/len(lons):.6f}")
    if alts:
        print(f"  Altitude : {min(alts):.1f} – {max(alts):.1f} m AGL")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
