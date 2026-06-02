"""
core/video.py – Video telemetry extraction and adaptive frame sampling.

The critical difference between still images and drone video
-----------------------------------------------------------
Still images  : XMP metadata embedded in every JPEG → _extract_dji_data() works.
Video frames  : ffmpeg-extracted JPEGs carry NO XMP metadata.
                Telemetry must come from either:
                  (a) the DJI .SRT sidecar file the drone generates alongside
                      every .MP4  (preferred — always try this first), or
                  (b) the binary 'djmd' data stream embedded directly in the
                      .MP4 container (parsed via exiftool -ee).

This module provides:
  SRTFrame               – telemetry dataclass shared by all telemetry sources
  SRTParser              – parse 3 DJI SRT format variants into SRTFrame records
  EmbeddedTelemetryParser– extract telemetry from the djmd stream via exiftool
  AdaptiveSampler        – compute the optimal frame sampling interval
  VideoSampler           – extract JPEG frames paired with interpolated telemetry
"""

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger("TAE.Video")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SRTFrame:
    """Telemetry for one subtitle block (≈ one video frame at recording FPS)."""
    frame_idx:    int
    timestamp_ms: int    # milliseconds from start of video
    lat:          float
    lon:          float
    alt_m:        float  # AGL altitude (RelativeAltitude equivalent)
    gimbal_pitch: float  # degrees; -90 = nadir
    gimbal_yaw:   float  # degrees clockwise from North
    gimbal_roll:  float  # degrees


# ─────────────────────────────────────────────────────────────────────────────
# SRT parser — handles 3 DJI format families
# ─────────────────────────────────────────────────────────────────────────────
#
# Format A  (Mini 2, Air 2, Mavic Air 2 — no gimbal fields):
#   [latitude: 31.235678] [longitude: 34.567890] [altitude: 98.34]
#
# Format B  (Air 2S, Mavic 3, newer firmware — includes gimbal):
#   [latitude : 31.235678] [longitude : 34.567890] [altitude : 98.34]
#   [gb_yaw : 12.4] [gb_pitch : -89.8] [gb_roll : 0.0]
#
# Format C  (FPV / Avata):
#   GPS (31.235678, 34.567890, 0), D 0.00m, H 98.34m, H.S 0.0m/s, V.S 0.00m/s

_TC_RE      = re.compile(r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})')
_GPS_FPV_RE = re.compile(r'GPS\s*\(([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)', re.I)
_LAT_RE     = re.compile(r'\[latitude\s*:\s*([+-]?\d+\.?\d*)', re.I)
_LON_RE     = re.compile(r'\[longitude\s*:\s*([+-]?\d+\.?\d*)', re.I)
_ALT_RE     = re.compile(r'\[altitude\s*:\s*([+-]?\d+\.?\d*)', re.I)
_H_AGL_RE   = re.compile(r'(?:^|[,\s])H\s+([+-]?\d+\.?\d*)m')   # FPV: "H 98.34m"
_PITCH_RE   = re.compile(r'\[gb_pitch\s*:\s*([+-]?\d+\.?\d*)', re.I)
_YAW_RE     = re.compile(r'\[gb_yaw\s*:\s*([+-]?\d+\.?\d*)', re.I)
_ROLL_RE    = re.compile(r'\[gb_roll\s*:\s*([+-]?\d+\.?\d*)', re.I)


def _tc_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def _parse_block(content: str, frame_idx: int, start_ms: int) -> "SRTFrame | None":
    """Parse the text body of one SRT block. Returns None if no GPS found."""
    # Strip font tags (present in most DJI formats)
    text = re.sub(r'<[^>]+>', '', content)

    lat = lon = alt = None

    # ── Format C: FPV GPS(lat,lon,alt_abs) ───────────────────────────────────
    m = _GPS_FPV_RE.search(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        h_m = _H_AGL_RE.search(text)
        alt = float(h_m.group(1)) if h_m else 0.0
    else:
        # ── Formats A / B ─────────────────────────────────────────────────────
        lat_m = _LAT_RE.search(text)
        lon_m = _LON_RE.search(text)
        alt_m = _ALT_RE.search(text)
        if lat_m and lon_m:
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            alt = float(alt_m.group(1)) if alt_m else 0.0

    if lat is None or lon is None:
        return None

    pitch_m      = _PITCH_RE.search(text)
    yaw_m        = _YAW_RE.search(text)
    roll_m       = _ROLL_RE.search(text)
    absolute_yaw = float(yaw_m.group(1)) if yaw_m else 0.0

    return SRTFrame(
        frame_idx    = frame_idx,
        timestamp_ms = start_ms,
        lat          = lat,
        lon          = lon,
        alt_m        = alt or 0.0,
        gimbal_pitch = float(pitch_m.group(1)) if pitch_m else -90.0,
        gimbal_yaw   = absolute_yaw,
        gimbal_roll  = float(roll_m.group(1))  if roll_m  else 0.0,
    )


class SRTParser:
    """Parse DJI .SRT sidecar files and convert frames to pose_metadata format."""

    # ── Parsing ───────────────────────────────────────────────────────────────

    def parse(self, srt_path: Path) -> list[SRTFrame]:
        """
        Parse a DJI SRT file into a list of SRTFrame records (one per block).
        Blocks with no GPS data are silently skipped.
        """
        text   = srt_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n\s*\n", text.strip())

        frames: list[SRTFrame] = []
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            # Line 0: subtitle index
            try:
                idx = int(lines[0].strip()) - 1
            except ValueError:
                continue
            # Line 1: timecode  "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            tc_m = _TC_RE.search(lines[1])
            if not tc_m:
                continue
            start_ms = _tc_to_ms(*tc_m.groups()[:4])
            content  = "\n".join(lines[2:])
            frame    = _parse_block(content, idx, start_ms)
            if frame:
                frames.append(frame)

        logger.info(
            f"SRT '{srt_path.name}': {len(frames)} telemetry frames "
            f"({len(blocks)} blocks)"
        )
        return frames

    # ── Interpolation ─────────────────────────────────────────────────────────

    def interpolate(
        self, frames: list[SRTFrame], timestamp_ms: int
    ) -> "SRTFrame | None":
        """
        Return the SRTFrame whose timestamp is closest to timestamp_ms,
        with linear interpolation of position/angles between neighbouring frames.
        """
        if not frames:
            return None
        if len(frames) == 1 or timestamp_ms <= frames[0].timestamp_ms:
            return frames[0]
        if timestamp_ms >= frames[-1].timestamp_ms:
            return frames[-1]

        # Binary search
        lo, hi = 0, len(frames) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if frames[mid].timestamp_ms <= timestamp_ms:
                lo = mid
            else:
                hi = mid

        f0, f1 = frames[lo], frames[hi]
        dt = f1.timestamp_ms - f0.timestamp_ms
        t  = (timestamp_ms - f0.timestamp_ms) / dt if dt else 0.0
        t  = max(0.0, min(1.0, t))

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * t

        return SRTFrame(
            frame_idx    = f0.frame_idx,
            timestamp_ms = timestamp_ms,
            lat          = lerp(f0.lat,          f1.lat),
            lon          = lerp(f0.lon,           f1.lon),
            alt_m        = lerp(f0.alt_m,         f1.alt_m),
            gimbal_pitch = lerp(f0.gimbal_pitch,  f1.gimbal_pitch),
            gimbal_yaw   = lerp(f0.gimbal_yaw,    f1.gimbal_yaw),
            gimbal_roll  = lerp(f0.gimbal_roll,   f1.gimbal_roll),
        )

    # ── Conversion to pose_metadata format ───────────────────────────────────

    def to_meta_entry(
        self, srt_frame: SRTFrame, jpeg_path: Path, img_w: int, img_h: int
    ) -> dict:
        """
        Convert an SRTFrame to the pose_metadata.json entry format consumed by
        the ingestion pipeline (same structure as _extract_dji_data() output).
        """
        return {
            "full_path":    str(jpeg_path),
            "lat":          srt_frame.lat,
            "lon":          srt_frame.lon,
            "z":            srt_frame.alt_m,
            "gimbal_pitch": srt_frame.gimbal_pitch,
            "gimbal_yaw":   srt_frame.gimbal_yaw,
            "gimbal_roll":  srt_frame.gimbal_roll,
            "img_w_px":     img_w,
            "img_h_px":     img_h,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Embedded telemetry parser  (via exiftool -ee)
# ─────────────────────────────────────────────────────────────────────────────

def _exif_float(rec: dict, *keys: str) -> "float | None":
    """
    Try each tag name in *keys* against the exiftool JSON record and return
    the first value that can be coerced to float, or None if none succeed.
    """
    for key in keys:
        v = rec.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


class EmbeddedTelemetryParser:
    """
    Extract DJI flight telemetry from the proprietary 'djmd' binary data stream
    embedded in .MP4 files produced by DJI drones (Mavic 3, Air 3, Mini 4 Pro, …).

    Why exiftool?
    -------------
    The djmd stream is protobuf-encoded with an unpublished schema.  ExifTool's
    DJI module has reverse-engineered this format and is the only reliable public
    decoder.  Home-rolled protobuf parsers are brittle because field IDs, scaling
    factors, and coordinate encoding differ across firmware versions.

    Usage priority
    --------------
    Always prefer an .SRT sidecar when one is available — it is a plain-text
    format that is trivially parseable and guaranteed to be complete.  Use this
    class as a fallback when the video has been transferred without its sidecar.

    Falls back gracefully (returns []) when:
      • exiftool is not installed
      • the file has no embedded telemetry track
      • exiftool cannot decode the stream (rare firmware variant)
    """

    # Tag names exiftool uses for DJI embedded-stream samples.
    # Multiple candidates handle the variation across drone models / exiftool versions.
    _LAT_KEYS   = ("GPSLatitude",)
    _LON_KEYS   = ("GPSLongitude",)
    _ALT_KEYS   = ("RelativeAltitude", "AbsoluteAltitude", "GPSAltitude")
    _PITCH_KEYS = ("GimbalPitch",)
    _YAW_KEYS   = ("GimbalYaw",)
    _ROLL_KEYS  = ("GimbalRoll",)
    _TIME_KEYS  = ("SampleTime",)

    # ── Availability check ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if exiftool is installed and callable."""
        try:
            subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Main parse entry point ────────────────────────────────────────────────

    def parse(self, video_path: Path) -> list[SRTFrame]:
        """
        Shell out to ``exiftool -ee -j -n`` and convert per-sample records to
        SRTFrame objects.

        exiftool -ee (ExtractEmbedded) reads the binary djmd track and emits
        one JSON object per telemetry sample (≈ one per source video frame).
        The -n flag returns numeric values without unit strings so floats parse
        cleanly.  Records without a GPSLatitude/GPSLongitude pair are skipped
        (the first record in the array is always the top-level file metadata).

        Parameters
        ----------
        video_path : Path to the .MP4 (or other DJI video container).

        Returns
        -------
        list[SRTFrame] — one entry per telemetry sample, sorted by timestamp.
        Empty list on any failure.
        """
        cmd = ["exiftool", "-ee", "-j", "-n", str(video_path)]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,   # large 4K files at high bitrate can be slow
            )
        except FileNotFoundError:
            logger.warning(
                "exiftool not found — install it to parse embedded DJI telemetry. "
                "Debian/Ubuntu: sudo apt-get install libimage-exiftool-perl"
            )
            return []
        except subprocess.TimeoutExpired:
            logger.warning(
                f"exiftool timed out after 120 s on '{video_path.name}'"
            )
            return []

        if res.returncode != 0:
            logger.warning(
                f"exiftool exited {res.returncode} for '{video_path.name}': "
                f"{res.stderr.strip()[:200]}"
            )
            return []

        try:
            records: list[dict] = json.loads(res.stdout)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"exiftool produced malformed JSON for '{video_path.name}': {exc}"
            )
            return []

        frames: list[SRTFrame] = []
        for rec in records:
            lat = _exif_float(rec, *self._LAT_KEYS)
            lon = _exif_float(rec, *self._LON_KEYS)
            if lat is None or lon is None:
                continue    # top-level file record or sample without GPS fix

            timestamp_ms = int(
                (_exif_float(rec, *self._TIME_KEYS) or 0.0) * 1000
            )
            alt_m        = _exif_float(rec, *self._ALT_KEYS)   or 0.0
            gimbal_pitch = _exif_float(rec, *self._PITCH_KEYS) or -90.0
            gimbal_yaw   = _exif_float(rec, *self._YAW_KEYS)   or 0.0
            gimbal_roll  = _exif_float(rec, *self._ROLL_KEYS)  or 0.0

            frames.append(SRTFrame(
                frame_idx    = len(frames),
                timestamp_ms = timestamp_ms,
                lat          = lat,
                lon          = lon,
                alt_m        = alt_m,
                gimbal_pitch = gimbal_pitch,
                gimbal_yaw   = gimbal_yaw,
                gimbal_roll  = gimbal_roll,
            ))

        if frames:
            logger.info(
                f"Embedded telemetry: {len(frames)} frames from "
                f"'{video_path.name}' (exiftool parsed {len(records)} records)"
            )
        else:
            logger.info(
                f"No embedded telemetry in '{video_path.name}' "
                f"(exiftool returned {len(records)} record(s), none with GPS)"
            )
        return frames


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive sampler
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveSampler:
    """
    Computes the recommended frame-sampling interval for video ingestion.

    The key insight: the sampling interval should be set so that consecutive
    sampled frames have the desired ground overlap.  This depends on:
      - AGL altitude (determines footprint size)
      - Flight speed (determines how fast the footprint moves)
      - Target overlap %
    """

    # Fallback heuristic when speed is unknown
    _ALT_TABLE = [
        (30,         1.0),   # < 30 m  → 1 s
        (60,         2.0),   # 30–60 m → 2 s
        (120,        3.0),   # 60–120 m→ 3 s
        (float("inf"), 5.0), # > 120 m → 5 s
    ]

    def compute_interval(
        self,
        alt_m:              float,
        speed_ms:           float,
        sensor_w_mm:        float,
        focal_mm:           float,
        target_overlap_pct: float = 60.0,
    ) -> float:
        """
        Return recommended sampling interval in seconds.

        Derivation
        ----------
        footprint_w_m = alt × (sensor_width_mm / focal_length_mm)
        non_overlap_m = (1 − overlap/100) × footprint_w_m
        T             = non_overlap_m / speed_m_s
        """
        if alt_m <= 0 or speed_ms <= 0 or focal_mm <= 0:
            return self.altitude_heuristic(alt_m)
        footprint_w = alt_m * (sensor_w_mm / focal_mm)
        non_overlap = (1.0 - target_overlap_pct / 100.0) * footprint_w
        interval    = non_overlap / speed_ms
        # Clamp to operationally sensible range
        return max(0.5, min(interval, 10.0))

    def altitude_heuristic(self, alt_m: float) -> float:
        """Conservative fallback when speed is unavailable."""
        for threshold, interval in self._ALT_TABLE:
            if alt_m <= threshold:
                return interval
        return 5.0

    def from_srt_frames(
        self,
        srt_frames:         list[SRTFrame],
        sensor_w_mm:        float,
        focal_mm:           float,
        target_overlap_pct: float = 60.0,
    ) -> float:
        """
        Estimate interval from the first few SRT frames.
        Speed is derived from consecutive GPS positions; altitude from frame 0.
        """
        if not srt_frames:
            return 2.0

        alt_m    = srt_frames[0].alt_m if srt_frames[0].alt_m > 1 else 80.0
        speed_ms = 0.0

        # Estimate ground speed from the first two GPS-distinct frames
        for i in range(1, min(15, len(srt_frames))):
            f0, f1 = srt_frames[i - 1], srt_frames[i]
            dt_s   = (f1.timestamp_ms - f0.timestamp_ms) / 1000.0
            if dt_s <= 0:
                continue
            dlat_m = (f1.lat - f0.lat) * 111_320.0
            dlon_m = (
                (f1.lon - f0.lon)
                * 111_320.0
                * math.cos(math.radians((f0.lat + f1.lat) / 2))
            )
            dist_m = math.hypot(dlat_m, dlon_m)
            if dist_m > 0.2:  # at least 20 cm movement between consecutive SRT entries
                speed_ms = dist_m / dt_s
                break

        interval = self.compute_interval(
            alt_m, speed_ms, sensor_w_mm, focal_mm, target_overlap_pct
        )
        logger.info(
            f"Adaptive sampling: alt={alt_m:.0f}m  speed={speed_ms:.1f}m/s  "
            f"→ interval={interval:.1f}s"
        )
        return interval


# ─────────────────────────────────────────────────────────────────────────────
# Video sampler
# ─────────────────────────────────────────────────────────────────────────────

class VideoSampler:
    """
    Extracts JPEG frames from a video at adaptive or fixed intervals,
    pairing each frame with its interpolated SRT telemetry.

    The output is a list of (jpeg_path, SRTFrame | None) pairs that drop
    directly into the existing ingestion pipeline with no further changes —
    SRTFrame is converted to the pose_metadata.json format by SRTParser.to_meta_entry().
    """

    def __init__(
        self,
        sampler: AdaptiveSampler | None = None,
        parser:  SRTParser | None = None,
    ):
        self._sampler = sampler or AdaptiveSampler()
        self._parser  = parser  or SRTParser()

    def sample_file(
        self,
        video_path:         Path,
        out_dir:            Path,
        srt_frames:         list[SRTFrame] | None = None,
        interval_sec:       float = 2.0,
        adaptive:           bool  = True,
        target_overlap_pct: float = 60.0,
        sensor_w_mm:        float = 6.3,
        focal_mm:           float = 4.5,
        jpeg_quality:       int   = 92,
        on_progress:        Callable[[int, int], None] | None = None,
    ) -> list[tuple[Path, "SRTFrame | None"]]:
        """
        Extract frames at computed intervals and pair with telemetry.

        Parameters
        ----------
        video_path          : source video file
        out_dir             : directory for extracted JPEG frames
        srt_frames          : parsed SRT telemetry (None if no SRT available)
        interval_sec        : fixed interval (used when adaptive=False or no SRT)
        adaptive            : if True, compute interval from SRT altitude + speed
        target_overlap_pct  : target ground overlap between consecutive frames
        sensor_w_mm/focal_mm: camera params for footprint calculation
        on_progress         : callback(frames_done, frames_estimated)

        Returns
        -------
        list of (jpeg_path, SRTFrame | None)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms  = int(total_frames / src_fps * 1000)

        # Peek at first frame for image dimensions
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Cannot read first frame: {video_path}")
        img_h, img_w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Determine sampling interval
        if adaptive and srt_frames:
            interval_sec = self._sampler.from_srt_frames(
                srt_frames, sensor_w_mm, focal_mm, target_overlap_pct
            )
        else:
            logger.info(f"Fixed sampling interval: {interval_sec:.1f}s")

        interval_ms      = int(interval_sec * 1000)
        estimated_frames = max(1, duration_ms // interval_ms)

        results:   list[tuple[Path, SRTFrame | None]] = []
        prev_gray: np.ndarray | None = None
        sample_idx = 0
        fi         = 0
        next_ms    = 0

        logger.info(
            f"Sampling '{video_path.name}' | "
            f"duration={duration_ms/1000:.1f}s | "
            f"src_fps={src_fps:.0f} | "
            f"interval={interval_sec:.1f}s | "
            f"~{estimated_frames} frames expected"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_ms = int(fi / src_fps * 1000)

            if current_ms >= next_ms:
                # Motion filter: skip frames that are nearly identical to the
                # previous accepted frame (drone hovering / very slow movement).
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and gray.shape == prev_gray.shape:
                    mean_diff = cv2.absdiff(gray, prev_gray).mean()
                    if mean_diff < 1.5:   # empirically: < 1.5 ≈ < 1% scene change
                        fi      += 1
                        continue
                prev_gray = gray

                # Write JPEG
                fname = out_dir / f"{stem}_s{sample_idx:05d}.jpg"
                cv2.imwrite(
                    str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )

                # Pair with interpolated telemetry
                telem: SRTFrame | None = None
                if srt_frames:
                    telem = self._parser.interpolate(srt_frames, current_ms)

                results.append((fname, telem))
                sample_idx += 1
                next_ms    += interval_ms

                if on_progress:
                    on_progress(sample_idx, estimated_frames)

            fi += 1

        cap.release()
        logger.info(
            f"Extracted {len(results)} frames from '{video_path.name}'"
        )
        return results