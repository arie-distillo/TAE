import numpy as np
from config import settings


class SpatialEngine:
    """
    Computes ground-plane footprints for UAV camera frames.
    
    Assumes a flat-earth projection — valid for the low-altitude, 
    small-area surveys TAE operates in (< 2 km²).
    """

    def __init__(self, sensor_width_mm=None, sensor_height_mm=None, focal_length_mm=None):
        self.sensor_width  = float(sensor_width_mm  or settings.SENSOR_WIDTH_MM)
        self.sensor_height = float(sensor_height_mm or settings.SENSOR_HEIGHT_MM)
        self.focal_length  = float(focal_length_mm  or settings.FOCAL_LENGTH_MM)

    def compute_footprint(self, lat, lon, alt_m, gimbal_yaw_deg, img_w_px, img_h_px):
        """
        Projects the 4 image corners onto the ground plane using a flat-earth
        approximation. Valid for near-nadir gimbal angles (pitch near -90°).

        Parameters
        ----------
        lat, lon        : float  — WGS84 center position (from GPS EXIF)
        alt_m           : float  — AGL altitude in meters (RelativeAltitude XMP)
        gimbal_yaw_deg  : float  — Camera yaw, clockwise from North (GimbalYawDegree XMP)
        img_w_px        : int    — Image width in pixels
        img_h_px        : int    — Image height in pixels

        Returns
        -------
        dict with keys:
            ground_w_m, ground_h_m  — footprint dimensions in meters
            gsd_cm_px               — ground sample distance in cm/pixel
            center                  — (lat, lon) tuple
            nw, ne, se, sw          — corner (lat, lon) tuples
            polygon_lonlat          — closed ring in (lon, lat) order for GeoJSON/Leaflet
        """
        if alt_m <= 0:
            raise ValueError(f"Invalid altitude: {alt_m}. Must be positive AGL meters.")

        # --- Ground dimensions in meters ---
        ground_w = (alt_m * self.sensor_width)  / self.focal_length
        ground_h = (alt_m * self.sensor_height) / self.focal_length
        gsd_cm_px = (alt_m * self.sensor_width) / (self.focal_length * img_w_px) * 100

        # --- 4 corners in local NED frame (North/East offsets in meters) ---
        # Before rotation: image is axis-aligned, centered at origin.
        # +North = up on image, +East = right on image.
        hw = ground_w / 2  # half-width  (East extent)
        hh = ground_h / 2  # half-height (North extent)

        corners_ned = np.array([
            [-hh, -hw],  # NW
            [-hh,  hw],  # NE
            [ hh,  hw],  # SE
            [ hh, -hw],  # SW
        ])  # shape (4, 2): [north_m, east_m]

        # --- Rotate by gimbal yaw (clockwise from North) ---
        yaw_rad = np.radians(gimbal_yaw_deg)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        R = np.array([[ cos_y, -sin_y],
                      [ sin_y,  cos_y]])
        corners_rotated = (R @ corners_ned.T).T  # shape (4, 2)

        # --- Convert NED meter offsets → WGS84 degrees ---
        # Flat-earth: valid when coverage << Earth radius
        lat_rad = np.radians(lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(lat_rad)

        corners_wgs84 = [
            (lat + north_m / m_per_deg_lat,
             lon  + east_m  / m_per_deg_lon)
            for north_m, east_m in corners_rotated
        ]

        nw, ne, se, sw = corners_wgs84

        return {
            "ground_w_m":    round(ground_w, 2),
            "ground_h_m":    round(ground_h, 2),
            "gsd_cm_px":     round(gsd_cm_px, 2),
            "center":        (lat, lon),
            "nw":            nw,
            "ne":            ne,
            "se":            se,
            "sw":            sw,
            # Closed ring in (lon, lat) order — GeoJSON / Leaflet convention
            "polygon_lonlat": [
                (nw[1], nw[0]), (ne[1], ne[0]),
                (se[1], se[0]), (sw[1], sw[0]),
                (nw[1], nw[0]),  # close the ring
            ]
        }

    def log_footprint(self, footprint, logger):
        """Convenience method for structured ingestion logging."""
        logger.info(
            f"Footprint | "
            f"Coverage: {footprint['ground_w_m']}m × {footprint['ground_h_m']}m | "
            f"GSD: {footprint['gsd_cm_px']} cm/px | "
            f"NW: {footprint['nw'][0]:.6f},{footprint['nw'][1]:.6f} "
            f"SE: {footprint['se'][0]:.6f},{footprint['se'][1]:.6f}"
        )