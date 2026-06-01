import numpy as np
import cv2
from typing import Generator, Tuple
from config import settings


class SpatialEngine:
    """
    Handles all ground-plane geometry for UAV camera frames.

    Responsibilities:
      - compute_footprint: projects a full frame onto the ground (WGS84 quadrilateral)
      - tile_image: splits a frame into overlapping tiles
      - compute_tile_footprint: derives the ground footprint of a single tile
        from its parent frame footprint, using bilinear interpolation

    Flat-earth approximation is used throughout — valid for low-altitude
    small-area surveys (< 2 km²) as used in TAE.
    """

    def __init__(self, sensor_width_mm=None, sensor_height_mm=None, focal_length_mm=None):
        self.sensor_width  = float(sensor_width_mm  or settings.SENSOR_WIDTH_MM)
        self.sensor_height = float(sensor_height_mm or settings.SENSOR_HEIGHT_MM)
        self.focal_length  = float(focal_length_mm  or settings.FOCAL_LENGTH_MM)

    # ------------------------------------------------------------------
    # Frame footprint
    # ------------------------------------------------------------------

    def compute_footprint(self, lat, lon, alt_m, gimbal_yaw_deg, img_w_px, img_h_px) -> dict:
        """
        Projects the 4 image corners onto the ground plane.

        Parameters
        ----------
        lat, lon        : WGS84 center (GPS EXIF)
        alt_m           : AGL altitude in metres (RelativeAltitude XMP)
        gimbal_yaw_deg  : Camera yaw clockwise from North (GimbalYawDegree XMP)
        img_w_px        : Image width in pixels
        img_h_px        : Image height in pixels

        Returns
        -------
        dict with keys: ground_w_m, ground_h_m, gsd_cm_px,
                        center, nw, ne, se, sw, polygon_lonlat
        """
        if alt_m <= 0:
            raise ValueError(f"Invalid AGL altitude: {alt_m}. Must be > 0.")

        ground_w  = (alt_m * self.sensor_width)  / self.focal_length
        ground_h  = (alt_m * self.sensor_height) / self.focal_length
        gsd_cm_px = (alt_m * self.sensor_width)  / (self.focal_length * img_w_px) * 100

        hw, hh = ground_w / 2, ground_h / 2

        # Corners in local NED frame
        corners_ned = np.array([
            [ hh, -hw],  # NW: +north, -east  ← top-left   of image = north-west ✓
            [ hh,  hw],  # NE: +north, +east  ← top-right  of image = north-east ✓
            [-hh,  hw],  # SE: -north, +east  ← bottom-right        = south-east ✓
            [-hh, -hw],  # SW: -north, -east  ← bottom-left         = south-west ✓
        ])

        yaw_rad = np.radians(gimbal_yaw_deg)
        R = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                      [np.sin(yaw_rad),  np.cos(yaw_rad)]])
        corners_rot = (R @ corners_ned.T).T

        lat_rad = np.radians(lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(lat_rad)

        corners_wgs84 = [
            (lat + n / m_per_deg_lat, lon + e / m_per_deg_lon)
            for n, e in corners_rot
        ]
        nw, ne, se, sw = corners_wgs84

        return {
            "ground_w_m":     round(ground_w, 2),
            "ground_h_m":     round(ground_h, 2),
            "gsd_cm_px":      round(gsd_cm_px, 2),
            "center":         (lat, lon),
            "nw":             nw,
            "ne":             ne,
            "se":             se,
            "sw":             sw,
            "polygon_lonlat": [
                (nw[1], nw[0]), (ne[1], ne[0]),
                (se[1], se[0]), (sw[1], sw[0]),
                (nw[1], nw[0]),
            ]
        }

    # ------------------------------------------------------------------
    # Tiling
    # ------------------------------------------------------------------

    def tile_image(
        self,
        img: np.ndarray,
        tile_size: int = 640,
        overlap: float = 0.2
    ) -> Generator[Tuple[np.ndarray, int, int, int, int], None, None]:
        """
        Yields overlapping tiles from an image.

        Yields
        ------
        (tile_img, x_off, y_off, tile_w, tile_h)
            x_off, y_off : top-left corner of tile in parent image pixels
            tile_w, tile_h : actual tile dimensions (edge tiles may be smaller)
        """
        h, w = img.shape[:2]
        step = int(tile_size * (1 - overlap))

        for y in range(0, h, step):
            for x in range(0, w, step):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = img[y:y_end, x:x_end]
                tile_h, tile_w = tile.shape[:2]
                yield tile, x, y, tile_w, tile_h

    def tile_count(self, img_w: int, img_h: int, tile_size: int = 640, overlap: float = 0.2) -> int:
        """Returns the number of tiles that tile_image will produce for given image dimensions."""
        step = int(tile_size * (1 - overlap))
        cols = len(range(0, img_w, step))
        rows = len(range(0, img_h, step))
        return cols * rows

    # ------------------------------------------------------------------
    # Tile footprint
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_to_wgs84(
        x_px: float, y_px: float,
        img_w: int, img_h: int,
        nw: tuple, ne: tuple, se: tuple, sw: tuple
    ) -> tuple:
        """
        Bilinear interpolation of a pixel position within the parent frame footprint.

        u=0 → left edge, u=1 → right edge
        v=0 → top edge,  v=1 → bottom edge
        """
        u = x_px / img_w
        v = y_px / img_h
        lat = ((1 - v) * ((1 - u) * nw[0] + u * ne[0]) +
                     v  * ((1 - u) * sw[0] + u * se[0]))
        lon = ((1 - v) * ((1 - u) * nw[1] + u * ne[1]) +
                     v  * ((1 - u) * sw[1] + u * se[1]))
        return lat, lon

    def compute_tile_footprint(
        self,
        x_off: int, y_off: int,
        tile_w: int, tile_h: int,
        img_w: int, img_h: int,
        frame_footprint: dict
    ) -> dict:
        """
        Derives the WGS84 footprint of a tile from its parent frame footprint.

        The tile footprint is the bilinear projection of the tile's 4 pixel
        corners through the parent frame's ground quadrilateral.

        Parameters
        ----------
        x_off, y_off    : tile top-left corner in parent image pixels
        tile_w, tile_h  : tile dimensions in pixels
        img_w, img_h    : parent image dimensions in pixels
        frame_footprint : dict returned by compute_footprint()

        Returns
        -------
        dict with same corner keys as compute_footprint():
            nw, ne, se, sw, polygon_lonlat, gsd_cm_px
        """
        nw_f = frame_footprint['nw']
        ne_f = frame_footprint['ne']
        se_f = frame_footprint['se']
        sw_f = frame_footprint['sw']

        p2w = self._pixel_to_wgs84  # shorthand

        t_nw = p2w(x_off,          y_off,          img_w, img_h, nw_f, ne_f, se_f, sw_f)
        t_ne = p2w(x_off + tile_w,  y_off,          img_w, img_h, nw_f, ne_f, se_f, sw_f)
        t_se = p2w(x_off + tile_w,  y_off + tile_h, img_w, img_h, nw_f, ne_f, se_f, sw_f)
        t_sw = p2w(x_off,           y_off + tile_h, img_w, img_h, nw_f, ne_f, se_f, sw_f)

        return {
            "gsd_cm_px":      frame_footprint['gsd_cm_px'],  # GSD is frame-level, same for all tiles
            "nw":             t_nw,
            "ne":             t_ne,
            "se":             t_se,
            "sw":             t_sw,
            "polygon_lonlat": [
                (t_nw[1], t_nw[0]), (t_ne[1], t_ne[0]),
                (t_se[1], t_se[0]), (t_sw[1], t_sw[0]),
                (t_nw[1], t_nw[0]),
            ]
        }

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def log_footprint(self, footprint: dict, logger) -> None:
        logger.info(
            f"Footprint | "
            f"GSD: {footprint['gsd_cm_px']} cm/px | "
            f"NW: {footprint['nw'][0]:.6f},{footprint['nw'][1]:.6f} "
            f"SE: {footprint['se'][0]:.6f},{footprint['se'][1]:.6f}"
        )
