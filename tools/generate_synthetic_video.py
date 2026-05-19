#!/usr/bin/env python3
"""
tools/generate_synthetic_video.py
==================================
Generate a synthetic DJI-style drone survey video and .SRT telemetry sidecar
from a satellite/aerial image exported from QGIS (or any GIS tool).

The tool simulates a boustrophedon (lawnmower) or straight-line survey flight
over the image area, producing:
  • <output_dir>/drone_video.mp4    — simulated camera footage
  • <output_dir>/drone_video.SRT    — DJI-format telemetry sidecar
  • <output_dir>/frames/            — extracted sample frames (optional)
  • <output_dir>/pose_metadata.json — ready for TAE ingestion (optional)

Geo-bounds input
----------------
Option A: pass --bounds lat_min,lon_min,lat_max,lon_max on the command line.
Option B: place a QGIS world file (.pgw / .jgw / .wld) alongside the image;
          the tool auto-detects it.
Option C: for GeoTIFF input, bounds are read from the file metadata via rasterio
          (rasterio is optional — install with: pip install rasterio).

Usage examples
--------------
# Basic — 80m AGL, 8 m/s, boustrophedon, 60% overlap
python tools/generate_synthetic_video.py \\
    --image qgis_export.jpg \\
    --bounds 31.230,34.560,31.250,34.590

# Custom altitude and speed, with synthetic detection targets
python tools/generate_synthetic_video.py \\
    --image qgis_export.jpg \\
    --bounds 31.230,34.560,31.250,34.590 \\
    --altitude 60 \\
    --speed 6 \\
    --overlap 70 \\
    --objects "31.240,34.570,red;31.235,34.580,blue" \\
    --extract-frames \\
    --output output/mission_sim/
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Geo helpers
# ─────────────────────────────────────────────────────────────────────────────

M_PER_DEG_LAT = 111_320.0  # metres per degree latitude (constant)


def m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def latlon_to_metres(lat: float, lon: float, origin_lat: float, origin_lon: float):
    """Convert (lat, lon) to (east_m, north_m) relative to origin."""
    north_m = (lat - origin_lat) * M_PER_DEG_LAT
    east_m  = (lon - origin_lon) * m_per_deg_lon(origin_lat)
    return east_m, north_m


def metres_to_latlon(east_m: float, north_m: float, origin_lat: float, origin_lon: float):
    lat = origin_lat + north_m / M_PER_DEG_LAT
    lon = origin_lon + east_m  / m_per_deg_lon(origin_lat)
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# World-file / geo-bounds reader
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeoBounds:
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def center_lat(self): return (self.lat_min + self.lat_max) / 2
    @property
    def center_lon(self): return (self.lon_min + self.lon_max) / 2
    @property
    def width_m(self):
        return (self.lon_max - self.lon_min) * m_per_deg_lon(self.center_lat)
    @property
    def height_m(self):
        return (self.lat_max - self.lat_min) * M_PER_DEG_LAT


def _web_mercator_to_wgs84(x_m: float, y_m: float) -> tuple[float, float]:
    """
    Inverse Web Mercator (EPSG:3857) → WGS84.
    No external dependencies — uses the closed-form inverse Mercator formula.
    Returns (lat_deg, lon_deg).
    """
    R   = 6_378_137.0          # WGS84 equatorial radius in metres
    lon = math.degrees(x_m / R)
    lat = math.degrees(2.0 * math.atan(math.exp(y_m / R)) - math.pi / 2.0)
    return lat, lon


def _try_world_file(image_path: Path, img_w: int, img_h: int) -> GeoBounds | None:
    """
    Try to read a QGIS world file (.pgw / .jgw / .wld) alongside the image.

    World file format (6 lines):
      pixel_size_x   rotation_x   rotation_y   pixel_size_y(negative)   X_UL   Y_UL

    The CRS of the world file is whatever CRS the QGIS project used:
      • WGS84 (EPSG:4326) → values are degrees, works directly.
      • Web Mercator (EPSG:3857) → values are metres; converted automatically.
      • Other projected CRS → user must re-export in EPSG:4326.
    """
    for ext in (".pgw", ".jgw", ".wld", ".PGW", ".JGW", ".WLD"):
        wf = image_path.with_suffix(ext)
        if wf.exists():
            vals = [float(l.strip()) for l in wf.read_text().splitlines() if l.strip()]
            if len(vals) < 6:
                continue

            px_w = vals[0]        # pixel size in X direction (CRS units)
            px_h = abs(vals[3])   # pixel size in Y direction (positive, CRS units)
            x_ul = vals[4]        # X of upper-left pixel centre (CRS units)
            y_ul = vals[5]        # Y of upper-left pixel centre (CRS units)

            # Outer edges of the image in the world CRS
            x_min = x_ul - px_w * 0.5
            y_max = y_ul + px_h * 0.5
            x_max = x_min + px_w * img_w
            y_min = y_max - px_h * img_h

            # ── Detect projected vs geographic coordinates ────────────────────
            if abs(x_ul) > 180 or abs(y_ul) > 90:
                # Projected CRS — try Web Mercator inverse first
                print(f"  World file: {wf.name}  (projected coordinates detected)")
                wm_range = 2.0037e7   # Web Mercator valid extent in metres
                if abs(x_ul) < wm_range * 1.05 and abs(y_ul) < wm_range * 1.05:
                    try:
                        lat_max, lon_min = _web_mercator_to_wgs84(x_min, y_max)
                        lat_min, lon_max = _web_mercator_to_wgs84(x_max, y_min)
                        bounds = GeoBounds(lat_min, lon_min, lat_max, lon_max)
                        # Sanity check
                        if not (-90 < bounds.lat_min < bounds.lat_max < 90
                                and -180 < bounds.lon_min < bounds.lon_max < 180):
                            raise ValueError(f"Out-of-range after conversion: {bounds}")
                        print(f"  Converted from Web Mercator (EPSG:3857):")
                        print(f"    lat [{bounds.lat_min:.6f}, {bounds.lat_max:.6f}]  "
                              f"lon [{bounds.lon_min:.6f}, {bounds.lon_max:.6f}]")
                        return bounds
                    except Exception as e:
                        print(f"  Web Mercator conversion failed: {e}")

                # Fallback: try pyproj if installed
                try:
                    from pyproj import Transformer
                    tf = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
                    lon_min, lat_min = tf.transform(x_min, y_min)
                    lon_max, lat_max = tf.transform(x_max, y_max)
                    print("  Converted via pyproj.")
                    return GeoBounds(
                        min(lat_min, lat_max), min(lon_min, lon_max),
                        max(lat_min, lat_max), max(lon_min, lon_max),
                    )
                except ImportError:
                    pass

                print("Cannot convert projected coordinates automatically.")

                print("  Fix: in QGIS, re-export the image with CRS = EPSG:4326 (WGS84):")
                print("    Project > Import/Export > Export Map to Image")
                print("    Set CRS to EPSG:4326, enable 'Append georeference information'.")
                print("  Or pass the bounds manually:  --bounds lat_min,lon_min,lat_max,lon_max")
                return None

            else:
                # Already geographic degrees
                print(f"  World file: {wf.name}  (WGS84 degrees)")
                return GeoBounds(y_min, x_min, y_max, x_max)

    return None


def _try_geotiff(image_path: Path) -> GeoBounds | None:
    try:
        import rasterio
        with rasterio.open(str(image_path)) as ds:
            b = ds.bounds
            return GeoBounds(b.bottom, b.left, b.top, b.right)
    except ImportError:
        return None
    except Exception as e:
        print(f"  rasterio read failed: {e}")
        return None


def read_geo_bounds(image_path: Path, img_w: int, img_h: int,
                    cli_bounds: str | None) -> GeoBounds:
    if cli_bounds:
        parts = [float(x) for x in cli_bounds.split(",")]
        if len(parts) != 4:
            sys.exit("--bounds must be lat_min,lon_min,lat_max,lon_max")
        return GeoBounds(*parts)

    # GeoTIFF
    if image_path.suffix.lower() in (".tif", ".tiff"):
        b = _try_geotiff(image_path)
        if b:
            return b

    # World file
    b = _try_world_file(image_path, img_w, img_h)
    if b:
        return b

    sys.exit(
        "Cannot determine geo-bounds. Provide --bounds lat_min,lon_min,lat_max,lon_max\n"
        "or place a .pgw / .jgw world file alongside the image."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic object painter
# ─────────────────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "red":    (0,   0,   220),
    "blue":   (220, 80,  10),
    "green":  (30,  180, 30),
    "yellow": (20,  220, 220),
    "white":  (240, 240, 240),
    "orange": (0,   140, 255),
}


def parse_objects(spec: str) -> list[dict]:
    """
    Parse --objects "lat,lon,color[,radius_m];lat,lon,color[,radius_m];..."
    Returns list of dicts: {lat, lon, color_bgr, radius_m}
    """
    objects = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = [t.strip() for t in part.split(",")]
        if len(tokens) < 3:
            print(f"  Skipping malformed object spec: {part}")
            continue
        lat   = float(tokens[0])
        lon   = float(tokens[1])
        color_name = tokens[2].lower()
        radius_m   = float(tokens[3]) if len(tokens) > 3 else 3.0
        color_bgr  = COLOR_MAP.get(color_name, (0, 0, 220))
        objects.append({"lat": lat, "lon": lon, "color": color_bgr, "radius_m": radius_m})
    return objects


def paint_objects(image: np.ndarray, objects: list[dict], bounds: GeoBounds) -> np.ndarray:
    """Draw synthetic detection targets onto the satellite image."""
    img_h, img_w = image.shape[:2]
    img = image.copy()
    for obj in objects:
        # Convert lat/lon to pixel coordinates
        px_x = int((obj["lon"] - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * img_w)
        px_y = int((bounds.lat_max - obj["lat"]) / (bounds.lat_max - bounds.lat_min) * img_h)
        # Convert radius from metres to pixels
        m_per_px = bounds.width_m / img_w
        r_px = max(4, int(obj["radius_m"] / m_per_px))
        cv2.circle(img, (px_x, px_y), r_px, obj["color"], -1)
        cv2.circle(img, (px_x, px_y), r_px, (0, 0, 0), max(1, r_px // 6))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Flight path generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    lat:        float
    lon:        float
    bearing:    float   # degrees clockwise from North (flight direction)
    timestamp_s: float


def generate_boustrophedon(
    bounds:       GeoBounds,
    alt_m:        float,
    speed_ms:     float,
    sensor_w_mm:  float,
    focal_mm:     float,
    overlap_pct:  float,
    direction_deg: float = 90.0,  # 90 = East-West passes, 0 = North-South passes
) -> list[Waypoint]:
    """
    Generate waypoints for a boustrophedon (lawnmower) survey.

    direction_deg = 90: drone flies East↔West, progresses North.
    direction_deg = 0:  drone flies North↔South, progresses East.
    """
    # Footprint in the progression axis (perpendicular to flight direction)
    sensor_h_mm  = sensor_w_mm * 9 / 16   # assume 16:9 aspect ratio
    footprint_m  = alt_m * (sensor_h_mm / focal_mm)  if direction_deg == 90 \
                   else alt_m * (sensor_w_mm / focal_mm)
    lane_spacing = footprint_m * (1.0 - overlap_pct / 100.0)

    waypoints: list[Waypoint] = []
    t = 0.0

    if direction_deg == 90:   # East-West passes
        area_h_m = bounds.height_m
        area_w_m = bounds.width_m
        n_lanes  = max(1, math.ceil(area_h_m / lane_spacing))

        for i in range(n_lanes):
            north_m = i * lane_spacing
            lat, _  = metres_to_latlon(0, north_m, bounds.lat_min, bounds.lon_min)
            lat     = min(lat, bounds.lat_max)

            if i % 2 == 0:   # fly East
                lon_start = bounds.lon_min
                lon_end   = bounds.lon_max
                bearing   = 90.0
            else:             # fly West
                lon_start = bounds.lon_max
                lon_end   = bounds.lon_min
                bearing   = 270.0

            waypoints.append(Waypoint(lat, lon_start, bearing, t))
            t += area_w_m / speed_ms
            waypoints.append(Waypoint(lat, lon_end,   bearing, t))

            # Short transition pause between lanes
            if i < n_lanes - 1:
                t += max(1.0, lane_spacing / speed_ms)

    else:   # North-South passes
        area_w_m = bounds.width_m
        area_h_m = bounds.height_m
        n_lanes  = max(1, math.ceil(area_w_m / lane_spacing))

        for i in range(n_lanes):
            east_m  = i * lane_spacing
            _, lon  = metres_to_latlon(east_m, 0, bounds.lat_min, bounds.lon_min)
            lon     = min(lon, bounds.lon_max)

            if i % 2 == 0:   # fly North
                lat_start = bounds.lat_min
                lat_end   = bounds.lat_max
                bearing   = 0.0
            else:
                lat_start = bounds.lat_max
                lat_end   = bounds.lat_min
                bearing   = 180.0

            waypoints.append(Waypoint(lat_start, lon, bearing, t))
            t += area_h_m / speed_ms
            waypoints.append(Waypoint(lat_end,   lon, bearing, t))

            if i < n_lanes - 1:
                t += max(1.0, lane_spacing / speed_ms)

    return waypoints


def generate_straight(
    bounds: GeoBounds, speed_ms: float
) -> list[Waypoint]:
    """Single straight-line pass from SW to NE corner."""
    diag_m = math.hypot(bounds.width_m, bounds.height_m)
    return [
        Waypoint(bounds.lat_min, bounds.lon_min, 45.0, 0.0),
        Waypoint(bounds.lat_max, bounds.lon_max, 45.0, diag_m / speed_ms),
    ]


def interpolate_position(
    waypoints: list[Waypoint], t_s: float
) -> tuple[float, float, float]:
    """Return (lat, lon, bearing) at time t_s by linear interpolation."""
    if not waypoints:
        return 0.0, 0.0, 0.0
    if t_s <= waypoints[0].timestamp_s:
        w = waypoints[0]
        return w.lat, w.lon, w.bearing
    if t_s >= waypoints[-1].timestamp_s:
        w = waypoints[-1]
        return w.lat, w.lon, w.bearing

    for i in range(len(waypoints) - 1):
        w0, w1 = waypoints[i], waypoints[i + 1]
        if w0.timestamp_s <= t_s <= w1.timestamp_s:
            dt = w1.timestamp_s - w0.timestamp_s
            f  = (t_s - w0.timestamp_s) / dt if dt > 0 else 0.0
            lat  = w0.lat + (w1.lat - w0.lat) * f
            lon  = w0.lon + (w1.lon - w0.lon) * f
            return lat, lon, w0.bearing

    w = waypoints[-1]
    return w.lat, w.lon, w.bearing


# ─────────────────────────────────────────────────────────────────────────────
# Image crop helpers
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_image_px(
    lat: float, lon: float, bounds: GeoBounds, img_w: int, img_h: int
) -> tuple[int, int]:
    """Convert (lat, lon) → pixel coordinates in the satellite image."""
    px_x = (lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min) * img_w
    px_y = (bounds.lat_max - lat) / (bounds.lat_max - bounds.lat_min) * img_h
    return int(round(px_x)), int(round(px_y))


def crop_footprint(
    image:       np.ndarray,
    cx: int, cy: int,
    fp_w_px: int, fp_h_px: int,
    out_w:   int, out_h:   int,
) -> np.ndarray:
    """
    Crop a (fp_w_px × fp_h_px) window centred at (cx, cy) from the satellite
    image, padding with black if the crop extends beyond the image boundary,
    then resize to (out_w × out_h).
    """
    img_h, img_w = image.shape[:2]
    x0 = cx - fp_w_px // 2
    y0 = cy - fp_h_px // 2
    x1 = x0 + fp_w_px
    y1 = y0 + fp_h_px

    # Compute valid region
    src_x0 = max(x0, 0);  src_x1 = min(x1, img_w)
    src_y0 = max(y0, 0);  src_y1 = min(y1, img_h)
    dst_x0 = src_x0 - x0; dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = src_y0 - y0; dst_y1 = dst_y0 + (src_y1 - src_y0)

    canvas = np.zeros((fp_h_px, fp_w_px, 3), dtype=np.uint8)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = \
            image[src_y0:src_y1, src_x0:src_x1]

    return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def add_jitter(frame: np.ndarray, max_px: int = 2) -> np.ndarray:
    """Apply sub-pixel gimbal jitter for realism."""
    dx = np.random.randint(-max_px, max_px + 1)
    dy = np.random.randint(-max_px, max_px + 1)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))


# ─────────────────────────────────────────────────────────────────────────────
# SRT writer
# ─────────────────────────────────────────────────────────────────────────────

def ms_to_tc(ms: int) -> str:
    h   = ms // 3_600_000; ms %= 3_600_000
    m   = ms //    60_000; ms %=    60_000
    s   = ms //     1_000; ms %=     1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    path:      Path,
    waypoints: list[Waypoint],
    alt_m:     float,
    fps:       float,
    total_frames: int,
) -> None:
    """Write a DJI Format-B SRT file (with gb_pitch/gb_yaw/gb_roll fields)."""
    lines = []
    frame_ms = int(1000 / fps)
    rng = np.random.default_rng(42)   # deterministic noise

    for fi in range(total_frames):
        t_s    = fi / fps
        t_ms   = int(t_s * 1000)
        lat, lon, bearing = interpolate_position(waypoints, t_s)

        # Slight altitude variation (±1 m) for realism
        alt_var = alt_m + rng.uniform(-1.0, 1.0)

        start_tc = ms_to_tc(t_ms)
        end_tc   = ms_to_tc(t_ms + frame_ms)

        block = (
            f"{fi + 1}\n"
            f"{start_tc} --> {end_tc}\n"
            f'<font size="28">FrameCnt : {fi}, DiffTime : {frame_ms}ms\n'
            f"[iso : 100] [shutter : 1/1000] [fnum : 280] [ev : 0] "
            f"[ct : 5500] [color_md : default] [focal_len : 240]\n"
            f"[latitude : {lat:.6f}] [longitude : {lon:.6f}] "
            f"[altitude : {alt_var:.2f}] "
            f"[gb_yaw : {bearing:.1f}] [gb_pitch : -90.0] [gb_roll : 0.0]"
            f"</font>"
        )
        lines.append(block)

    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"  SRT  → {path}  ({total_frames} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    image_path:    Path,
    bounds:        GeoBounds,
    out_dir:       Path,
    alt_m:         float = 80.0,
    speed_ms:      float = 8.0,
    pattern:       str   = "boustrophedon",
    overlap_pct:   float = 60.0,
    fps:           int   = 30,
    out_w:         int   = 1920,
    out_h:         int   = 1080,
    sensor_w_mm:   float = 6.3,
    sensor_h_mm:   float = 4.7,
    focal_mm:      float = 4.5,
    objects:       list[dict] | None = None,
    extract_frames: bool = False,
) -> tuple[Path, Path]:
    """
    Generate the synthetic video and SRT file.

    Returns (video_path, srt_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and optionally paint satellite image ─────────────────────────────
    print(f"Loading satellite image: {image_path}")
    sat_img = cv2.imread(str(image_path))
    if sat_img is None:
        sys.exit(f"Cannot load image: {image_path}")
    img_h, img_w = sat_img.shape[:2]
    print(f"  Image size : {img_w} × {img_h} px")
    print(f"  Geo bounds : lat [{bounds.lat_min:.5f}, {bounds.lat_max:.5f}]  "
          f"lon [{bounds.lon_min:.5f}, {bounds.lon_max:.5f}]")
    print(f"  Area       : {bounds.width_m:.0f} m × {bounds.height_m:.0f} m")

    if objects:
        print(f"  Painting {len(objects)} synthetic object(s) onto terrain…")
        sat_img = paint_objects(sat_img, objects, bounds)

    # ── Camera footprint in satellite-image pixels ────────────────────────────
    m_per_px_lat = bounds.height_m / img_h
    m_per_px_lon = bounds.width_m  / img_w
    fp_w_m       = alt_m * (sensor_w_mm / focal_mm)
    fp_h_m       = alt_m * (sensor_h_mm / focal_mm)
    fp_w_px      = max(16, int(fp_w_m / m_per_px_lon))
    fp_h_px      = max(9,  int(fp_h_m / m_per_px_lat))
    gsd_cm_px    = (m_per_px_lon * fp_w_px / out_w) * 100

    print(f"\nFlight parameters:")
    print(f"  Altitude   : {alt_m:.0f} m AGL")
    print(f"  Speed      : {speed_ms:.1f} m/s")
    print(f"  Footprint  : {fp_w_m:.1f} m × {fp_h_m:.1f} m  "
          f"({fp_w_px} × {fp_h_px} px in satellite image)")
    print(f"  GSD        : {gsd_cm_px:.1f} cm/px at output resolution")

    # ── Flight path ───────────────────────────────────────────────────────────
    print(f"\nGenerating {pattern} flight path…")
    if pattern == "boustrophedon":
        waypoints = generate_boustrophedon(
            bounds, alt_m, speed_ms, sensor_w_mm, focal_mm, overlap_pct
        )
    else:
        waypoints = generate_straight(bounds, speed_ms)

    if not waypoints:
        sys.exit("Flight path has no waypoints.")

    duration_s    = waypoints[-1].timestamp_s
    total_frames  = int(duration_s * fps)
    n_lanes       = sum(
        1 for i in range(len(waypoints) - 1)
        if waypoints[i].bearing in (90.0, 270.0, 0.0, 180.0)
    ) // 2 + 1

    print(f"  Duration   : {duration_s:.1f} s  ({total_frames} frames at {fps} fps)")

    # ── Video writer ──────────────────────────────────────────────────────────
    video_path = out_dir / "drone_video.mp4"
    srt_path   = out_dir / "drone_video.SRT"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        sys.exit(f"Cannot open video writer for: {video_path}")

    # ── Frame extraction output ───────────────────────────────────────────────
    frames_dir   = out_dir / "frames" if extract_frames else None
    meta_entries = {}
    sample_every = max(1, int(fps * 2.0))   # extract ~1 frame per 2 seconds

    if frames_dir:
        frames_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    print(f"\nRendering {total_frames} video frames…")

    sample_idx = 0
    for fi in range(total_frames):
        t_s = fi / fps
        lat, lon, bearing = interpolate_position(waypoints, t_s)

        # Convert to satellite image pixel
        cx, cy = latlon_to_image_px(lat, lon, bounds, img_w, img_h)

        # Crop + resize to output resolution
        frame = crop_footprint(sat_img, cx, cy, fp_w_px, fp_h_px, out_w, out_h)
        frame = add_jitter(frame, max_px=1)

        writer.write(frame)

        # Optional frame extraction
        if extract_frames and fi % sample_every == 0:
            fname = frames_dir / f"frame_{sample_idx:05d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta_entries[fname.name] = {
                "full_path":    str(fname),
                "lat":          lat,
                "lon":          lon,
                "z":            alt_m,
                "gimbal_pitch": -90.0,
                "gimbal_yaw":   bearing,
                "gimbal_roll":  0.0,
                "img_w_px":     out_w,
                "img_h_px":     out_h,
            }
            sample_idx += 1

        if fi % (fps * 10) == 0:
            pct = fi / total_frames * 100
            print(f"  {pct:5.1f}%  frame {fi}/{total_frames}  "
                  f"lat={lat:.5f} lon={lon:.5f}")

    writer.release()
    print(f"\n  Video → {video_path}  ({video_path.stat().st_size // 1024} KB)")

    # ── SRT ───────────────────────────────────────────────────────────────────
    write_srt(srt_path, waypoints, alt_m, fps, total_frames)

    # ── pose_metadata.json ────────────────────────────────────────────────────
    if extract_frames and meta_entries:
        meta_path = out_dir / "pose_metadata.json"
        meta_path.write_text(json.dumps(meta_entries, indent=2))
        print(f"  Frames → {frames_dir}  ({sample_idx} frames)")
        print(f"  Meta   → {meta_path}")

    return video_path, srt_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a synthetic drone video and DJI SRT from a satellite image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--image",    required=True, help="Satellite image (JPEG/PNG/GeoTIFF)")
    ap.add_argument("--bounds",   default=None,
                    help="lat_min,lon_min,lat_max,lon_max  (not needed for GeoTIFF or if .pgw/.jgw exists)")
    ap.add_argument("--altitude", type=float, default=80.0,  help="Flight altitude AGL in metres [80]")
    ap.add_argument("--speed",    type=float, default=8.0,   help="Flight speed in m/s [8]")
    ap.add_argument("--pattern",  choices=["boustrophedon","straight"], default="boustrophedon")
    ap.add_argument("--overlap",  type=float, default=60.0,  help="Lane overlap %% [60]")
    ap.add_argument("--fps",      type=int,   default=30,    help="Output video FPS [30]")
    ap.add_argument("--width",    type=int,   default=1920,  help="Output video width px [1920]")
    ap.add_argument("--height",   type=int,   default=1080,  help="Output video height px [1080]")
    ap.add_argument("--sensor-w", type=float, default=6.3,   help="Camera sensor width mm [6.3 = DJI Mini 2]")
    ap.add_argument("--sensor-h", type=float, default=4.7,   help="Camera sensor height mm [4.7]")
    ap.add_argument("--focal",    type=float, default=4.5,   help="Focal length mm [4.5]")
    ap.add_argument("--objects",  default=None,
                    help='Synthetic targets: "lat,lon,color[,radius_m];…"  '
                         'Colors: red,blue,green,yellow,white,orange')
    ap.add_argument("--extract-frames", action="store_true",
                    help="Also extract sample JPEG frames + pose_metadata.json")
    ap.add_argument("--output",   default="output/sim_video",
                    help="Output directory [output/sim_video]")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    # Peek at image size for world-file parsing
    probe = cv2.imread(str(image_path))
    if probe is None:
        sys.exit(f"Cannot load image: {image_path}")
    img_h, img_w = probe.shape[:2]

    bounds  = read_geo_bounds(image_path, img_w, img_h, args.bounds)
    objects = parse_objects(args.objects) if args.objects else []
    out_dir = Path(args.output)

    print("\n═══ TAE Synthetic Video Generator ═══\n")
    video_path, srt_path = generate(
        image_path     = image_path,
        bounds         = bounds,
        out_dir        = out_dir,
        alt_m          = args.altitude,
        speed_ms       = args.speed,
        pattern        = args.pattern,
        overlap_pct    = args.overlap,
        fps            = args.fps,
        out_w          = args.width,
        out_h          = args.height,
        sensor_w_mm    = args.sensor_w,
        sensor_h_mm    = args.sensor_h,
        focal_mm       = args.focal,
        objects        = objects,
        extract_frames = args.extract_frames,
    )

    print("\n═══ Done ═══")
    print(f"Upload to TAE:")
    print(f"  {video_path}")
    print(f"  {srt_path}")


if __name__ == "__main__":
    main()