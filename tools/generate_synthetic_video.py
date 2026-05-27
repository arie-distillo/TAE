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
from typing import Optional

import cv2
import numpy as np

# PyYAML — only required when --config is used
try:
    import yaml as _yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# tqdm — optional progress bar
try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# XMP injection  (for TAE-compatible JPEG frame export)
# ─────────────────────────────────────────────────────────────────────────────
# TAE's _extract_dji_data uses re.search(rf'{field}="([^"]+)"', content)
# so the XMP must use attribute style, not element style.

import struct as _struct

_XMP_NS_PREFIX = b"http://ns.adobe.com/xap/1.0/\x00"

_XMP_TEMPLATE = (
    "<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>\n"
    "<x:xmpmeta xmlns:x='adobe:ns:meta/'>\n"
    "  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
    "    <rdf:Description rdf:about=''\n"
    "        xmlns:drone-dji='http://www.dji.com/drone-dji/1.0/'\n"
    '        drone-dji:RelativeAltitude="{alt_m}"\n'
    '        drone-dji:AbsoluteAltitude="{alt_m}"\n'
    '        drone-dji:GimbalPitchDegree="{gimbal_pitch}"\n'
    '        drone-dji:GimbalRollDegree="{gimbal_roll}"\n'
    '        drone-dji:GimbalYawDegree="{gimbal_yaw}"\n'
    '        drone-dji:FlightYawDegree="{flight_yaw}"\n'
    '        drone-dji:FlightPitchDegree="0.0"\n'
    '        drone-dji:FlightRollDegree="0.0"\n'
    '        drone-dji:GPSLatitude="{lat}"\n'
    '        drone-dji:GPSLongitude="{lon}"\n'
    "    />\n"
    "  </rdf:RDF>\n"
    "</x:xmpmeta>\n"
    "<?xpacket end='w'?>"
)


def _inject_xmp(jpeg_bytes: bytes, lat: float, lon: float, alt_m: float,
                gimbal_yaw: float, gimbal_pitch: float = -90.0,
                gimbal_roll: float = 0.0) -> bytes:
    """
    Inject a DJI-style XMP APP1 marker into a JPEG byte string.
    Produces attribute-style XMP that TAE's _extract_dji_data regex can parse.
    """
    xmp_xml = _XMP_TEMPLATE.format(
        lat=f"{lat:.7f}", lon=f"{lon:.7f}",
        alt_m=f"{alt_m:.3f}",
        gimbal_pitch=f"{gimbal_pitch:.1f}",
        gimbal_roll=f"{gimbal_roll:.1f}",
        gimbal_yaw=f"{gimbal_yaw:.1f}",
        flight_yaw=f"{gimbal_yaw:.1f}",
    ).encode("utf-8")

    marker_body = _XMP_NS_PREFIX + xmp_xml
    marker_len  = len(marker_body) + 2          # +2 for the length field itself
    app1 = b"\xff\xe1" + _struct.pack(">H", marker_len) + marker_body

    # Insert immediately after SOI (first 2 bytes)
    return jpeg_bytes[:2] + app1 + jpeg_bytes[2:]


def _save_frame_with_xmp(path: Path, frame_bgr: np.ndarray,
                          lat: float, lon: float, alt_m: float,
                          gimbal_yaw: float, quality: int = 92) -> None:
    """Encode frame as JPEG, inject DJI XMP, write to disk."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path}")
    jpeg = _inject_xmp(buf.tobytes(), lat, lon, alt_m, gimbal_yaw)
    path.write_bytes(jpeg)

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
                    cli_bounds: str | None) -> "GeoBounds | None":
    """
    Try to resolve geo-bounds from CLI arg, GeoTIFF metadata, or world file.
    Returns None (instead of exiting) when none is available — caller can then
    fall back to synthetic_geo_bounds().
    """
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

    return None   # caller decides what to do (synthetic bounds or error)


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
# Tracking target system  (new)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Linear trajectory starting from (start_lat, start_lon)."""
    direction_deg: Optional[float]   # bearing CW from North; None → seeded random
    speed_ms:      float             # object speed in m/s


@dataclass
class TrackingTarget:
    """
    A PNG object planted into a contiguous window of frames.

    The object moves in geo-space along a linear trajectory — the camera
    sees it at different pixel positions in consecutive frames, making it
    a natural tracking subject.
    """
    id:          str
    png_path:    Path
    start_frame: int
    end_frame:   int
    trajectory:  Trajectory
    start_lat:   float = float("nan")   # NaN → auto-placed before rendering
    start_lon:   float = float("nan")   # NaN → auto-placed before rendering

    # Sizing: provide real_size_m for GSD-correct sizing, or scale as a
    # fraction of the frame short-edge.  real_size_m takes priority.
    real_size_m: Optional[float] = None   # physical width in metres
    scale:       float           = 0.05   # fallback: fraction of frame short-edge

    # Resolved at load time
    _img_rgba:   Optional[np.ndarray] = field(default=None, repr=False)

    # ── loader ────────────────────────────────────────────────────────────────
    def load_png(self) -> None:
        """
        Load the PNG as 4-channel BGRA, then auto-crop to the non-transparent
        content bounding box.

        Why: real_size_m should map to the VISIBLE animal, not the whole canvas.
        Many downloaded PNGs have large transparent margins around the subject.
        Without this crop, a 2 m cow set in a PNG whose canvas is 10× wider than
        the animal would render as 0.2 m visible content — exactly 10× too small.
        """
        raw = cv2.imread(str(self.png_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot load PNG: {self.png_path}")
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

        orig_h, orig_w = raw.shape[:2]

        # ── Auto-crop to non-transparent bounding box ─────────────────────────
        alpha = raw[:, :, 3]
        rows  = np.any(alpha > 10, axis=1)   # rows with any visible pixel
        cols  = np.any(alpha > 10, axis=0)   # cols with any visible pixel

        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            cropped = raw[r0 : r1 + 1, c0 : c1 + 1]
        else:
            cropped = raw   # fully opaque PNG — use as-is

        crop_h, crop_w = cropped.shape[:2]
        self._img_rgba = cropped

        if crop_w != orig_w or crop_h != orig_h:
            pct = 100.0 * crop_w * crop_h / max(1, orig_w * orig_h)
            print(f"    PNG auto-crop '{self.png_path.name}': "
                  f"{orig_w}×{orig_h} → {crop_w}×{crop_h} "
                  f"({pct:.0f}% of canvas was content)")

    # ── position ──────────────────────────────────────────────────────────────
    def geo_position_at_frame(
        self, fi: int, fps: float, rng: np.random.Generator
    ) -> tuple[float, float]:
        """
        Return (lat, lon) of the object centre at frame fi.

        Direction is resolved once per call for efficiency; callers should
        cache or compute from a pre-resolved direction (see resolve_direction).
        """
        raise NotImplementedError("use resolve_direction + position_at_frame_dt")

    def resolve_direction(self, rng: np.random.Generator) -> float:
        """Return the bearing in degrees, resolving None → random."""
        if self.trajectory.direction_deg is not None:
            return float(self.trajectory.direction_deg)
        return float(rng.uniform(0, 360))

    def position_at_dt(
        self,
        dt_s: float,
        direction_deg: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        """
        Compute (lat, lon) after dt_s seconds of linear motion from origin.

        Uses flat-earth approximation (valid for short trajectories < 2 km).
        """
        dist_m = self.trajectory.speed_ms * dt_s
        rad    = math.radians(direction_deg)
        # North component = cos(bearing), East component = sin(bearing)
        north_m = dist_m * math.cos(rad)
        east_m  = dist_m * math.sin(rad)
        lat = origin_lat + north_m / M_PER_DEG_LAT
        lon = origin_lon + east_m  / m_per_deg_lon(origin_lat)
        return lat, lon

    # ── pixel size ────────────────────────────────────────────────────────────
    def pixel_size(
        self,
        gsd_m_per_px: float,
        out_w: int,
        out_h: int,
    ) -> tuple[int, int]:
        """
        Return (width_px, height_px) for the object in the output frame.

        Priority: real_size_m (geo-correct) > scale (fraction of frame).
        Aspect ratio is always preserved from the source PNG.
        """
        src_h, src_w = self._img_rgba.shape[:2]
        aspect = src_h / src_w if src_w > 0 else 1.0

        if self.real_size_m and gsd_m_per_px > 0:
            w_px = max(4, int(self.real_size_m / gsd_m_per_px))
        else:
            w_px = max(4, int(self.scale * min(out_w, out_h)))

        h_px = max(2, int(w_px * aspect))
        return w_px, h_px

    # ── compositor ────────────────────────────────────────────────────────────
    def composite_onto(
        self,
        frame: np.ndarray,
        cx_px: int,
        cy_px: int,
        obj_w: int,
        obj_h: int,
    ) -> np.ndarray:
        """
        Alpha-blend the scaled PNG onto frame at centre (cx_px, cy_px).
        Handles partial out-of-bounds gracefully.
        """
        # Scale PNG to target size
        scaled = cv2.resize(self._img_rgba, (obj_w, obj_h),
                            interpolation=cv2.INTER_LANCZOS4)

        out_h, out_w = frame.shape[:2]
        x0 = cx_px - obj_w // 2
        y0 = cy_px - obj_h // 2
        x1 = x0 + obj_w
        y1 = y0 + obj_h

        # Clamp to frame bounds
        sx0 = max(0, -x0);       sy0 = max(0, -y0)
        fx0 = max(0,  x0);       fy0 = max(0,  y0)
        fx1 = min(out_w, x1);    fy1 = min(out_h, y1)
        sx1 = sx0 + (fx1 - fx0); sy1 = sy0 + (fy1 - fy0)

        if fx1 <= fx0 or fy1 <= fy0:
            return frame   # fully out of frame

        patch  = scaled[sy0:sy1, sx0:sx1]
        alpha  = patch[:, :, 3:4].astype(np.float32) / 255.0
        bgr    = patch[:, :, :3].astype(np.float32)

        roi    = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        blended = roi * (1.0 - alpha) + bgr * alpha
        frame  = frame.copy()
        frame[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader  (new)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Auto-parameter derivation  (new)
# ─────────────────────────────────────────────────────────────────────────────

# Default synthetic centre — Middle East / Mediterranean, visually neutral
_DEFAULT_CENTER_LAT = 32.000
_DEFAULT_CENTER_LON = 35.000


# How many camera footprints wide/tall the synthetic world is.
# 1.0 = bounds equal one footprint → drone immediately flies off-image.
# 3.0 = drone crosses three footprint-widths, camera always over real content
#       (reflected at image edges by BORDER_REFLECT_101 in crop_footprint).
_SYNTHETIC_BOUNDS_SCALE = 3.0


def synthetic_geo_bounds(
    img_w: int,
    img_h: int,
    alt_m: float,
    sensor_w_mm: float,
    sensor_h_mm: float,
    focal_mm:    float,
    center_lat:  float = _DEFAULT_CENTER_LAT,
    center_lon:  float = _DEFAULT_CENTER_LON,
    scale:       float = _SYNTHETIC_BOUNDS_SCALE,
) -> "GeoBounds":
    """
    Invent geo-bounds for an image that has no world file or GPS EXIF.

    The bounds are set to `scale` times the camera footprint (default 3×).
    This gives the drone a meaningful flight path — the camera traverses
    three footprint-widths of "virtual terrain".  Wherever the crop window
    extends beyond the actual image pixels, crop_footprint() fills using
    BORDER_REFLECT_101 (mirror), which looks natural and avoids stripes.

    All coordinates are self-consistent: SRT, pose_metadata.json and the
    ground-truth JSON share the same synthetic grid.  Absolute position is
    arbitrary but TAE's spatial pipeline (footprint, GSD, CLIP tiles) works.
    """
    fp_w_m = alt_m * (sensor_w_mm / focal_mm)
    fp_h_m = alt_m * (sensor_h_mm / focal_mm)

    # Virtual world is `scale` footprints wide/tall
    world_w_m = fp_w_m * scale
    world_h_m = fp_h_m * scale

    half_lat = (world_h_m / 2.0) / M_PER_DEG_LAT
    half_lon = (world_w_m / 2.0) / m_per_deg_lon(center_lat)

    bounds = GeoBounds(
        lat_min = center_lat - half_lat,
        lon_min = center_lon - half_lon,
        lat_max = center_lat + half_lat,
        lon_max = center_lon + half_lon,
    )
    print(f"  Synthetic bounds : lat [{bounds.lat_min:.6f}, {bounds.lat_max:.6f}]"
          f"  lon [{bounds.lon_min:.6f}, {bounds.lon_max:.6f}]")
    print(f"  Virtual area     : {world_w_m:.0f} m × {world_h_m:.0f} m"
          f"  ({scale:.0f}× one {fp_w_m:.0f}×{fp_h_m:.0f} m footprint,"
          f"  reflected at image edges)")
    return bounds


def derive_auto_params(
    img_w:       int,
    img_h:       int,
    alt_m:       float,
    sensor_w_mm: float = 6.3,
    sensor_h_mm: float = 4.7,
    focal_mm:    float = 4.5,
    targets:     list  = None,
) -> dict:
    """
    Compute optimal flight + video parameters from the image and altitude.

    Design targets
    ──────────────
    fps      : 10  (smooth, manageable file size)
    pattern  : straight  (one clean camera pass; ideal for tracking tests)
    overlap  : 0         (no multi-lane survey needed)
    speed    : chosen so the camera crosses the full image footprint in ~20 s
               → stationary object is visible for ~200 frames at 10 fps
    For moving objects: worst-case (fastest target moving toward the drone)
    is still visible for at least 30 frames — enough for any tracker.

    Returns a dict of generate() kwargs that should be used as defaults,
    overrideable by explicit YAML keys.
    """
    fp_w_m = alt_m * (sensor_w_mm / focal_mm)
    fp_h_m = alt_m * (sensor_h_mm / focal_mm)
    gsd    = fp_w_m / img_w                      # metres per output pixel

    fps         = 10
    cross_time  = 20.0                           # seconds to cross full footprint
    speed_ms    = fp_w_m / cross_time            # drone speed

    # If any target is fast, ensure it's still visible for ≥30 frames
    min_frames  = 30
    if targets:
        max_obj_speed = max(
            t.trajectory.speed_ms for t in targets
            if hasattr(t, "trajectory")
        )
        # Worst-case relative speed (head-on)
        rel_speed  = speed_ms + max_obj_speed
        t_visible  = fp_w_m / rel_speed          # seconds visible
        if t_visible * fps < min_frames:
            # Slow drone down so the object is visible long enough
            speed_ms = max(0.5, fp_w_m / (min_frames / fps) - max_obj_speed)

    params = {
        "fps":        fps,
        "speed":      round(speed_ms, 2),
        "pattern":    "straight",
        "overlap":    0.0,
        "gsd_cm_px":  round(gsd * 100, 2),
        "fp_w_m":     round(fp_w_m, 1),
        "fp_h_m":     round(fp_h_m, 1),
    }

    print(f"  Auto-params      : fps={params['fps']}  speed={params['speed']} m/s"
          f"  pattern={params['pattern']}  overlap={params['overlap']}%")
    print(f"  GSD              : {params['gsd_cm_px']} cm/px")
    print(f"  Footprint        : {params['fp_w_m']} m × {params['fp_h_m']} m")
    vis_frames = int((fp_w_m / speed_ms) * fps)
    print(f"  Stationary obj   : visible ~{vis_frames} frames per pass")
    return params


def load_yaml_config(config_path: Path) -> dict:
    """
    Load a YAML video-generation config.

    Minimal required keys:  image, altitude, tracking_targets[].png,
                            tracking_targets[].real_size_m,
                            tracking_targets[].speed_ms
    Everything else is auto-derived or has sensible defaults.
    """
    if not YAML_AVAILABLE:
        sys.exit("PyYAML is required for --config.  Install: pip install pyyaml")

    raw = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    targets: list[TrackingTarget] = []
    for i, spec in enumerate(raw.get("tracking_targets", [])):
        # ── only png is truly required ────────────────────────────────────
        if "png" not in spec:
            sys.exit(f"tracking_targets[{i}] missing required field: png")

        # speed_ms: accept at top level or nested in trajectory block
        traj_raw   = spec.get("trajectory", {})
        speed_ms   = float(
            spec.get("speed_ms")
            or traj_raw.get("speed_ms")
            or 2.0
        )
        direction  = (
            spec.get("direction_deg")
            or traj_raw.get("direction_deg")
            or None   # None → random
        )
        trajectory = Trajectory(
            direction_deg = float(direction) if direction is not None else None,
            speed_ms      = speed_ms,
        )

        # PNG resolution: CWD-relative first, then config-relative
        png_raw    = Path(spec["png"])
        cwd_path   = (Path.cwd() / png_raw).resolve()
        cfg_path_r = (config_path.parent / png_raw).resolve()
        if cwd_path.exists():
            png_path = cwd_path
        elif cfg_path_r.exists():
            png_path = cfg_path_r
        else:
            png_path = cwd_path   # will raise a clear error at load time

        # start/end frame: default to "entire video" (0 → will be set after
        # total_frames is known; we use sys.maxsize as sentinel for "until end")
        start_frame = int(spec["start_frame"]) if "start_frame" in spec else 0
        end_frame   = int(spec["end_frame"])   if "end_frame"   in spec else -1  # -1 = auto

        target = TrackingTarget(
            id          = spec.get("id", f"target_{i}"),
            png_path    = png_path,
            start_frame = start_frame,
            end_frame   = end_frame,           # clamped to total_frames-1 in generate()
            trajectory  = trajectory,
            start_lat   = float(spec["start_lat"]) if "start_lat" in spec else float("nan"),
            start_lon   = float(spec["start_lon"]) if "start_lon" in spec else float("nan"),
            real_size_m = float(spec["real_size_m"]) if "real_size_m" in spec else None,
            scale       = float(spec.get("scale", 0.05)),
        )
        targets.append(target)

    return {
        "image":           raw.get("image"),
        "bounds":          raw.get("bounds"),
        "altitude":        raw.get("altitude", 80.0),
        "center_lat":      raw.get("center_lat", _DEFAULT_CENTER_LAT),
        "center_lon":      raw.get("center_lon", _DEFAULT_CENTER_LON),
        # Flight params: None = auto-derive from image + altitude
        "speed":           raw.get("speed",   None),
        "fps":             raw.get("fps",     None),
        "pattern":         raw.get("pattern", None),
        "overlap":         raw.get("overlap", None),
        "width":           raw.get("width",   1920),
        "height":          raw.get("height",  1080),
        "sensor_w":        raw.get("sensor_w", 6.3),
        "sensor_h":        raw.get("sensor_h", 4.7),
        "focal":           raw.get("focal",    4.5),
        "objects":         raw.get("static_objects"),
        "extract_frames":  raw.get("extract_frames", False),
        "output":          raw.get("output", "output/sim_video"),
        "tracking_targets": targets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tracking-target renderer helpers  (new)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Auto-placement  (new)
# ─────────────────────────────────────────────────────────────────────────────

def auto_place_target(
    target:       "TrackingTarget",
    direction_deg: float,
    waypoints:    list["Waypoint"],
    bounds:       "GeoBounds",
    fp_w_m:       float,
    fp_h_m:       float,
    fps:          float,
    total_frames: int,
    rng:          np.random.Generator,
    min_visible:  int = 5,
    max_attempts: int = 50,
) -> bool:
    """
    Auto-assign start_lat / start_lon so the object is guaranteed to be
    visible for at least `min_visible` consecutive frames.

    Strategy
    ────────
    1. Collect all frames in [start_frame, end_frame] where the drone is
       actually within the image bounds (i.e. waypoints exist).
    2. Shuffle them and try each as an "anchor frame":
       a. Get drone (lat, lon) at the anchor frame.
       b. Pick a random offset within 40% of the half-footprint, away from
          the centre so the object has room to drift across the frame.
       c. Place the object AT that offset position at the anchor frame.
       d. Back-calculate start_lat/start_lon from the object's own velocity
          so that at time (anchor_frame), the object is at the offset position.
       e. Run a quick forward-scan to count consecutive visible frames
          around the anchor. Accept if ≥ min_visible.
    3. On failure after max_attempts, fall back to placing the object
       directly at the drone's position (always visible for at least 1 frame).

    Modifies target.start_lat and target.start_lon in-place.
    Returns True if a good placement was found, False if only fallback used.
    """
    import math as _math

    # Half-footprint in geo units (flat-earth approx, fine for small areas)
    center_lat = bounds.center_lat
    half_w_m   = fp_w_m * 0.5
    half_h_m   = fp_h_m * 0.5
    half_w_deg_lon = half_w_m / m_per_deg_lon(center_lat)
    half_h_deg_lat = half_h_m / M_PER_DEG_LAT

    # Candidate frames: every frame in the active window
    active_frames = [
        fi for fi in range(
            max(0, target.start_frame),
            min(total_frames, target.end_frame + 1),
        )
    ]
    if not active_frames:
        return False

    rng.shuffle(active_frames)
    candidates = active_frames[:max_attempts]

    for anchor_fi in candidates:
        t_s = anchor_fi / fps
        drone_lat, drone_lon, _ = interpolate_position(waypoints, t_s)

        # Skip if drone is outside image bounds (black-frame zone)
        if not (bounds.lat_min < drone_lat < bounds.lat_max and
                bounds.lon_min < drone_lon < bounds.lon_max):
            continue

        # Random offset: between 10% and 40% of half-footprint
        # (not at dead centre, gives the tracker something to follow)
        frac = rng.uniform(0.1, 0.40)
        angle = rng.uniform(0, 2 * _math.pi)
        offset_lat = frac * half_h_deg_lat * _math.cos(angle)
        offset_lon = frac * half_w_deg_lon * _math.sin(angle)

        anchor_lat = drone_lat + offset_lat
        anchor_lon = drone_lon + offset_lon

        # Back-calculate: where was the object at start_frame?
        dt_to_anchor = (anchor_fi - target.start_frame) / fps
        start_lat, start_lon = target.position_at_dt(
            -dt_to_anchor,   # negative dt = reverse
            direction_deg,
            anchor_lat,
            anchor_lon,
        )

        # Quick visibility scan around the anchor
        visible_count = 0
        for fi in range(
            max(target.start_frame, anchor_fi - 30),
            min(target.end_frame + 1, anchor_fi + 30),
        ):
            dt_s = (fi - target.start_frame) / fps
            obj_lat, obj_lon = target.position_at_dt(
                dt_s, direction_deg, start_lat, start_lon
            )
            t_s2 = fi / fps
            dlat, dlon, _ = interpolate_position(waypoints, t_s2)
            if (abs(obj_lat - dlat) <= half_h_deg_lat and
                    abs(obj_lon - dlon) <= half_w_deg_lon):
                visible_count += 1

        if visible_count >= min_visible:
            target.start_lat = start_lat
            target.start_lon = start_lon
            return True

    # Fallback: place directly below the drone at the first active frame
    fi0 = active_frames[0]
    target.start_lat, target.start_lon, _ = interpolate_position(waypoints, fi0 / fps)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Visibility estimator  (new)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_visibility_windows(
    targets:    list["TrackingTarget"],
    waypoints:  list["Waypoint"],
    bounds:     "GeoBounds",
    img_w:      int,
    img_h:      int,
    fp_w_px:    int,
    fp_h_px:    int,
    fps:        float,
    total_frames: int,
    rng:        np.random.Generator,
) -> list[tuple["TrackingTarget", float, list[tuple[int, int]]]]:
    """
    Pre-flight scan: for each tracking target compute which frames the drone
    footprint actually overlaps the object's geo-position.

    Returns list of (target, resolved_direction_deg, windows)
    where windows = list of (first_visible_frame, last_visible_frame) ranges.

    Prints a human-readable summary so the user can set meaningful
    start_frame / end_frame values in the YAML config.
    """
    results = []

    print("\n  Pre-flight visibility estimate:")
    print(f"  {'─'*62}")

    for target in targets:
        direction = target.resolve_direction(rng)

        # Collect every frame index where the object is inside the footprint
        visible_frames: list[int] = []

        for fi in range(total_frames):
            if not (target.start_frame <= fi <= target.end_frame):
                continue

            dt_s = (fi - target.start_frame) / fps
            obj_lat, obj_lon = target.position_at_dt(
                dt_s, direction, target.start_lat, target.start_lon
            )

            # Drone position at this frame
            t_s = fi / fps
            drone_lat, drone_lon, _ = interpolate_position(waypoints, t_s)

            # Object position in satellite-image pixels
            obj_sat_x, obj_sat_y = latlon_to_image_px(
                obj_lat, obj_lon, bounds, img_w, img_h
            )
            # Drone centre in satellite-image pixels
            drone_cx, drone_cy = latlon_to_image_px(
                drone_lat, drone_lon, bounds, img_w, img_h
            )

            # Half-extents of the footprint in satellite-image pixels
            hw = fp_w_px / 2
            hh = fp_h_px / 2

            if (abs(obj_sat_x - drone_cx) <= hw and
                    abs(obj_sat_y - drone_cy) <= hh):
                visible_frames.append(fi)

        # Compress to contiguous windows
        windows: list[tuple[int, int]] = []
        if visible_frames:
            run_start = visible_frames[0]
            run_end   = visible_frames[0]
            for f in visible_frames[1:]:
                if f == run_end + 1:
                    run_end = f
                else:
                    windows.append((run_start, run_end))
                    run_start = run_end = f
            windows.append((run_start, run_end))

        total_visible = len(visible_frames)
        total_active  = max(0, target.end_frame - target.start_frame + 1)

        print(f"  Target '{target.id}':")
        print(f"    Active window : frames {target.start_frame}–{target.end_frame}"
              f"  ({total_active} frames  ≈ {total_active/fps:.0f} s)")
        print(f"    Direction     : {direction:.1f}°   speed: {target.trajectory.speed_ms:.1f} m/s")

        if windows:
            print(f"    Visible in {len(windows)} pass(es), {total_visible} frames total:")
            for i, (a, b) in enumerate(windows):
                dur_s = (b - a + 1) / fps
                print(f"      pass {i+1}: frames {a}–{b}  ({b-a+1} frames, {dur_s:.1f} s)")
        else:
            print(f"    ⚠  NEVER VISIBLE — object never enters the drone footprint!")
            print(f"       Check start_lat/start_lon vs image bounds, or widen start/end frame range.")

        print(f"  {'─'*62}")
        results.append((target, direction, windows))

    return results


def _prepare_targets(
    targets: list[TrackingTarget],
    rng: np.random.Generator,
) -> list[tuple[TrackingTarget, float]]:
    """
    Load each target's PNG and resolve its random direction.
    Returns list of (target, resolved_direction_deg).
    """
    prepared = []
    for t in targets:
        t.load_png()
        direction = t.resolve_direction(rng)
        # Compute and show the expected rendered pixel size
        # so the user can immediately verify it looks right
        # (GSD at 80 m DJI Mini2 ≈ 5.8 cm/px → 2 m cow = ~34 px wide)
        if t.real_size_m and t._img_rgba is not None:
            # Use a placeholder GSD; caller will recompute per-frame
            # Just a sanity-check printout
            canvas_h, canvas_w = t._img_rgba.shape[:2]
            print(f"  Target '{t.id}'  frames [{t.start_frame}–{t.end_frame}]  "
                  f"dir {direction:.1f}°  speed {t.trajectory.speed_ms:.1f} m/s  "
                  f"real_size {t.real_size_m} m  "
                  f"PNG content {canvas_w}×{canvas_h} px  "
                  f"png {t.png_path.name}")
        else:
            print(f"  Target '{t.id}'  frames [{t.start_frame}–{t.end_frame}]  "
                  f"dir {direction:.1f}°  speed {t.trajectory.speed_ms:.1f} m/s  "
                  f"png {t.png_path.name}")
        prepared.append((t, direction))
    return prepared


def _render_targets_onto_frame(
    frame:      np.ndarray,
    fi:         int,
    fps:        float,
    drone_lat:  float,
    drone_lon:  float,
    bounds:     "GeoBounds",
    img_w:      int,
    img_h:      int,
    fp_w_px:    int,
    fp_h_px:    int,
    out_w:      int,
    out_h:      int,
    alt_m:      float,
    sensor_w_mm: float,
    focal_mm:   float,
    prepared:   list[tuple["TrackingTarget", float]],
) -> tuple[np.ndarray, list[dict]]:
    """
    For each active tracking target, compute its position and composite it
    onto the frame.  Returns the annotated frame and a list of per-frame
    ground-truth records (one per target, visible or not).
    """
    # GSD in metres per output pixel
    fp_w_m   = alt_m * (sensor_w_mm / focal_mm)
    gsd_m_px = fp_w_m / out_w

    # Camera footprint in satellite-image pixel space
    drone_cx, drone_cy = latlon_to_image_px(drone_lat, drone_lon, bounds, img_w, img_h)
    sat_x0 = drone_cx - fp_w_px / 2
    sat_y0 = drone_cy - fp_h_px / 2

    # Scale from satellite-image pixels to output pixels
    scale_x = out_w / fp_w_px
    scale_y = out_h / fp_h_px

    gt_records: list[dict] = []

    for target, direction in prepared:
        # Only active in [start_frame, end_frame]
        if not (target.start_frame <= fi <= target.end_frame):
            gt_records.append({
                "id":      target.id,
                "frame":   fi,
                "active":  False,
                "visible": False,
            })
            continue

        dt_s = (fi - target.start_frame) / fps
        obj_lat, obj_lon = target.position_at_dt(
            dt_s, direction, target.start_lat, target.start_lon
        )

        # Convert object geo-position to satellite-image pixels
        obj_sat_x, obj_sat_y = latlon_to_image_px(
            obj_lat, obj_lon, bounds, img_w, img_h
        )

        # Object position within the cropped satellite window
        local_x = obj_sat_x - sat_x0
        local_y = obj_sat_y - sat_y0

        # Scale to output frame pixels
        frame_cx = int(local_x * scale_x)
        frame_cy = int(local_y * scale_y)

        # Compute object pixel size
        obj_w, obj_h = target.pixel_size(gsd_m_px, out_w, out_h)

        # Visibility check: object centre must be within frame
        visible = (0 <= frame_cx < out_w) and (0 <= frame_cy < out_h)

        gt_records.append({
            "id":       target.id,
            "frame":    fi,
            "active":   True,
            "visible":  visible,
            "lat":      round(obj_lat, 7),
            "lon":      round(obj_lon, 7),
            "pixel_cx": frame_cx,
            "pixel_cy": frame_cy,
            "obj_w_px": obj_w,
            "obj_h_px": obj_h,
        })

        if visible:
            frame = target.composite_onto(frame, frame_cx, frame_cy, obj_w, obj_h)

    return frame, gt_records



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
    image, then resize to (out_w × out_h).

    When the crop window extends beyond the image edge, the border is filled
    using REPLICATE mode (nearest edge pixel) instead of black, eliminating
    the dark-bar artefact that appears at survey boundaries.
    """
    img_h, img_w = image.shape[:2]
    x0 = cx - fp_w_px // 2
    y0 = cy - fp_h_px // 2

    # Padding amounts needed on each side (zero when fully inside the image)
    pad_left  = max(0, -x0)
    pad_top   = max(0, -y0)
    pad_right = max(0, (x0 + fp_w_px) - img_w)
    pad_bot   = max(0, (y0 + fp_h_px) - img_h)

    if pad_left or pad_top or pad_right or pad_bot:
        # BORDER_REFLECT_101: mirrors the image at each edge.
        # Far superior to BORDER_REPLICATE for large overshots (e.g. when the
        # camera footprint extends past the satellite image boundary at the
        # start/end of a straight pass): reflected terrain looks natural rather
        # than producing the distinctive horizontal/vertical stripe artefact
        # caused by a single edge row/column being repeated across the padding.
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )
        nx0 = x0 + pad_left
        ny0 = y0 + pad_top
    else:
        padded = image
        nx0, ny0 = x0, y0

    crop = padded[ny0 : ny0 + fp_h_px, nx0 : nx0 + fp_w_px]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


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
            f"[rel_alt: {alt_var:.3f} abs_alt: {alt_var:.3f}] "
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
    image_path:       Path,
    bounds:           GeoBounds,
    out_dir:          Path,
    alt_m:            float = 80.0,
    speed_ms:         float = 8.0,
    pattern:          str   = "boustrophedon",
    overlap_pct:      float = 60.0,
    fps:              int   = 30,
    out_w:            int   = 1920,
    out_h:            int   = 1080,
    sensor_w_mm:      float = 6.3,
    sensor_h_mm:      float = 4.7,
    focal_mm:         float = 4.5,
    objects:          list[dict] | None = None,
    extract_frames:   bool  = False,
    tracking_targets: list[TrackingTarget] | None = None,
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

    # Apparent motion: how many output pixels the scene shifts per frame.
    # Rule of thumb for smooth video: aim for 1–5 % of frame width per frame.
    #   Too high → choppy / blurry   Too low  → barely moving
    # Levers:  ↑ altitude (bigger footprint), ↓ speed, ↑ fps
    pixels_per_frame = (speed_ms / fps) * (out_w / fp_w_m)
    pct_per_frame    = pixels_per_frame / out_w * 100
    advice = ("✓ smooth" if pct_per_frame < 5
              else "⚠ fast — raise altitude or lower speed or raise fps"
              if pct_per_frame < 15
              else "✗ very fast — raise altitude significantly or lower speed")
    print(f"  Motion     : {pixels_per_frame:.1f} px/frame  ({pct_per_frame:.1f}%% of width)  {advice}")

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

    print(f"  Duration   : {duration_s:.0f} s  →  video length "
          f"{total_frames/fps/60:.1f} min at {fps} fps")
    print(f"  Tip        : lower 'overlap' (e.g. 30%%) or use 'pattern: straight' "
          f"for a shorter video")

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

    # ── Clamp end_frame=-1 sentinel to total_frames-1 ────────────────────────
    for t in (tracking_targets or []):
        if t.end_frame < 0:
            t.end_frame = total_frames - 1

    # ── Prepare tracking targets + visibility estimate ─────────────────────────
    tracking_targets = tracking_targets or []
    prepared_targets: list[tuple[TrackingTarget, float]] = []

    if tracking_targets:
        print(f"\nTracking targets ({len(tracking_targets)}):")
        for t in tracking_targets:
            t.load_png()

        # ── Auto-placement for targets without explicit start_lat/start_lon ───
        placement_rng = np.random.default_rng(1)   # separate seed from frame rng
        for idx, t in enumerate(tracking_targets):
            if math.isnan(t.start_lat) or math.isnan(t.start_lon):
                # Resolve direction first so auto-placement uses the same one
                dir_rng   = np.random.default_rng(idx + 100)
                direction = t.resolve_direction(dir_rng)
                print(f"  Auto-placing '{t.id}' (no start_lat/start_lon in config)…")
                ok = auto_place_target(
                    target        = t,
                    direction_deg = direction,
                    waypoints     = waypoints,
                    bounds        = bounds,
                    fp_w_m        = fp_w_m,
                    fp_h_m        = fp_h_m,
                    fps           = fps,
                    total_frames  = total_frames,
                    rng           = placement_rng,
                )
                status = "✓ placed" if ok else "⚠ fallback placement"
                print(f"    {status}: start ({t.start_lat:.6f}, {t.start_lon:.6f})")

        # ── Visibility pre-scan ────────────────────────────────────────────────
        vis_results = estimate_visibility_windows(
            targets       = tracking_targets,
            waypoints     = waypoints,
            bounds        = bounds,
            img_w         = img_w,
            img_h         = img_h,
            fp_w_px       = fp_w_px,
            fp_h_px       = fp_h_px,
            fps           = fps,
            total_frames  = total_frames,
            rng           = np.random.default_rng(0),
        )

        for t, direction, _windows in vis_results:
            prepared_targets.append((t, direction))

    print(f"\nRendering {total_frames} video frames…")

    sample_idx  = 0
    all_gt:  list[dict] = []   # ground-truth records for every frame

    frame_iter = range(total_frames)
    if TQDM_AVAILABLE:
        frame_iter = _tqdm(
            frame_iter,
            desc   = "  Frames",
            unit   = "fr",
            ncols  = 72,
            colour = "green",
        )

    for fi in frame_iter:
        t_s = fi / fps
        lat, lon, bearing = interpolate_position(waypoints, t_s)

        # Convert to satellite image pixel
        cx, cy = latlon_to_image_px(lat, lon, bounds, img_w, img_h)

        # Crop + resize to output resolution
        frame = crop_footprint(sat_img, cx, cy, fp_w_px, fp_h_px, out_w, out_h)
        frame = add_jitter(frame, max_px=1)

        # ── Composite tracking targets ────────────────────────────────────────
        if prepared_targets:
            frame, gt_records = _render_targets_onto_frame(
                frame      = frame,
                fi         = fi,
                fps        = fps,
                drone_lat  = lat,
                drone_lon  = lon,
                bounds     = bounds,
                img_w      = img_w,
                img_h      = img_h,
                fp_w_px    = fp_w_px,
                fp_h_px    = fp_h_px,
                out_w      = out_w,
                out_h      = out_h,
                alt_m      = alt_m,
                sensor_w_mm= sensor_w_mm,
                focal_mm   = focal_mm,
                prepared   = prepared_targets,
            )
            # Only store frames where at least one target is visible
            if any(r.get("visible") for r in gt_records):
                all_gt.extend(gt_records)

        writer.write(frame)

        # Optional frame extraction — save with embedded DJI XMP so TAE's
        # _extract_dji_data can read lat/lon/altitude/gimbal from the JPEG.
        if extract_frames and fi % sample_every == 0:
            fname = frames_dir / f"frame_{sample_idx:05d}.jpg"
            _save_frame_with_xmp(
                fname, frame,
                lat=lat, lon=lon, alt_m=alt_m, gimbal_yaw=bearing,
            )
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

        if not TQDM_AVAILABLE and fi % (fps * 10) == 0:
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

    # ── tracking_ground_truth.json ────────────────────────────────────────────
    if all_gt:
        # Restructure: {target_id: {frame_idx: record, ...}, ...}
        gt_by_target: dict = {}
        for rec in all_gt:
            tid = rec["id"]
            fi  = rec["frame"]
            gt_by_target.setdefault(tid, {})[str(fi)] = {
                k: v for k, v in rec.items() if k not in ("id", "frame")
            }

        gt_path = out_dir / "tracking_ground_truth.json"
        gt_meta = {
            "fps":           fps,
            "total_frames":  total_frames,
            "video_w":       out_w,
            "video_h":       out_h,
            "targets": {
                tid: {
                    "start_frame":    t.start_frame,
                    "end_frame":      t.end_frame,
                    "start_lat":      t.start_lat,
                    "start_lon":      t.start_lon,
                    "speed_ms":       t.trajectory.speed_ms,
                    "real_size_m":    t.real_size_m,
                    "frames":         gt_by_target.get(tid, {}),
                }
                for t, _ in prepared_targets
                for tid in [t.id]
            },
        }
        gt_path.write_text(json.dumps(gt_meta, indent=2))
        visible_count = sum(1 for r in all_gt if r.get("visible"))
        print(f"  GT     → {gt_path}  ({visible_count} visible detections across all targets)")

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
    # ── Config file (new) ─────────────────────────────────────────────────────
    ap.add_argument("--config", "-c", default=None,
                    help="YAML config file (all CLI args can be set here too)")

    # ── Existing CLI args (all optional if --config is used) ──────────────────
    ap.add_argument("--image",    default=None, help="Satellite image (JPEG/PNG/GeoTIFF)")
    ap.add_argument("--bounds",   default=None,
                    help="lat_min,lon_min,lat_max,lon_max  (not needed for GeoTIFF or if .pgw/.jgw exists)")
    ap.add_argument("--altitude", type=float, default=None,  help="Flight altitude AGL in metres [80]")
    ap.add_argument("--speed",    type=float, default=None,   help="Flight speed in m/s [8]")
    ap.add_argument("--pattern",  choices=["boustrophedon","straight"], default=None)
    ap.add_argument("--overlap",  type=float, default=None,  help="Lane overlap %% [60]")
    ap.add_argument("--fps",      type=int,   default=None,    help="Output video FPS [30]")
    ap.add_argument("--width",    type=int,   default=None,  help="Output video width px [1920]")
    ap.add_argument("--height",   type=int,   default=None,  help="Output video height px [1080]")
    ap.add_argument("--sensor-w", type=float, default=None,   help="Camera sensor width mm [6.3 = DJI Mini 2]")
    ap.add_argument("--sensor-h", type=float, default=None,   help="Camera sensor height mm [4.7]")
    ap.add_argument("--focal",    type=float, default=None,   help="Focal length mm [4.5]")
    ap.add_argument("--objects",  default=None,
                    help='Synthetic targets: "lat,lon,color[,radius_m];…"  '
                         'Colors: red,blue,green,yellow,white,orange')
    ap.add_argument("--extract-frames", action="store_true",
                    help="Also extract sample JPEG frames + pose_metadata.json")
    ap.add_argument("--output",   default=None,
                    help="Output directory [output/sim_video]")
    args = ap.parse_args()

    # ── Merge config + CLI (CLI overrides YAML) ───────────────────────────────
    cfg: dict = {}
    tracking_targets: list[TrackingTarget] = []

    if args.config:
        cfg = load_yaml_config(Path(args.config))
        tracking_targets = cfg.pop("tracking_targets", [])

    # CLI overrides YAML (only if explicitly provided)
    def _cli(key, arg_val, default):
        if arg_val is not None:
            return arg_val
        val = cfg.get(key)
        return val if val is not None else default  # treat stored None as "not set"

    image_str = _cli("image", args.image, None)
    if not image_str:
        ap.error("--image is required (or set 'image' in --config)")
    image_path = Path(image_str)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    probe = cv2.imread(str(image_path))
    if probe is None:
        sys.exit(f"Cannot load image: {image_path}")
    img_h, img_w = probe.shape[:2]

    # ── Camera model ──────────────────────────────────────────────────────────
    sensor_w_mm = _cli("sensor_w", args.sensor_w, 6.3)
    sensor_h_mm = _cli("sensor_h", args.sensor_h, 4.7)
    focal_mm    = _cli("focal",    args.focal,    4.5)
    alt_m       = _cli("altitude", args.altitude, 80.0)

    # ── Geo bounds ────────────────────────────────────────────────────────────
    # Priority: explicit --bounds / bounds: in YAML  →  world file (.pgw/.jgw)
    #           →  synthetic bounds derived from altitude + camera model
    bounds_str = _cli("bounds", args.bounds, None)
    bounds = read_geo_bounds(image_path, img_w, img_h, bounds_str)
    if bounds is None:
        # No world file and no explicit bounds → synthesise
        center_lat = float(_cli("center_lat", None, _DEFAULT_CENTER_LAT))
        center_lon = float(_cli("center_lon", None, _DEFAULT_CENTER_LON))
        print(f"  No world file found — using synthetic geo-bounds "
              f"centred on ({center_lat}, {center_lon})")
        bounds = synthetic_geo_bounds(
            img_w, img_h, alt_m,
            sensor_w_mm, sensor_h_mm, focal_mm,
            center_lat, center_lon,
        )

    # ── Auto-derive flight params if not explicitly set ───────────────────────
    auto = derive_auto_params(
        img_w, img_h, alt_m,
        sensor_w_mm, sensor_h_mm, focal_mm,
        targets = tracking_targets,
    )
    # Explicit YAML / CLI values override auto-derived ones
    speed_ms    = _cli("speed",   args.speed,   auto["speed"])
    fps         = _cli("fps",     args.fps,     auto["fps"])
    pattern     = _cli("pattern", args.pattern, auto["pattern"])
    overlap_pct = _cli("overlap", args.overlap, auto["overlap"])

    objects_str = args.objects or cfg.get("objects")
    objects     = parse_objects(objects_str) if objects_str else []
    out_dir     = Path(_cli("output", args.output, "output/sim_video"))

    print("\n═══ TAE Synthetic Video Generator ═══\n")
    video_path, srt_path = generate(
        image_path       = image_path,
        bounds           = bounds,
        out_dir          = out_dir,
        alt_m            = alt_m,
        speed_ms         = speed_ms,
        pattern          = pattern,
        overlap_pct      = overlap_pct,
        fps              = fps,
        out_w            = _cli("width",  args.width,  1920),
        out_h            = _cli("height", args.height, 1080),
        sensor_w_mm      = sensor_w_mm,
        sensor_h_mm      = sensor_h_mm,
        focal_mm         = focal_mm,
        objects          = objects,
        extract_frames   = args.extract_frames or cfg.get("extract_frames", False),
        tracking_targets = tracking_targets,
    )

    print("\n═══ Done ═══")
    print(f"Upload to TAE:")
    print(f"  {video_path}")
    print(f"  {srt_path}")


if __name__ == "__main__":
    main()