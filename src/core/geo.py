import math
import numpy as np

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in metres between two WGS84 points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def bbox_center_to_geo(
    bbox: list,
    tile_w: int, tile_h: int,
    fp_nw: tuple, fp_ne: tuple, fp_se: tuple, fp_sw: tuple
) -> tuple[float, float]:
    """
    Projects the pixel center of a bbox to WGS84 using bilinear
    interpolation across the tile's ground footprint.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    u = cx / tile_w   # 0=left,  1=right
    v = cy / tile_h   # 0=top,   1=bottom

    lat = ((1-v) * ((1-u)*fp_nw[0] + u*fp_ne[0]) +
               v  * ((1-u)*fp_sw[0] + u*fp_se[0]))
    lon = ((1-v) * ((1-u)*fp_nw[1] + u*fp_ne[1]) +
               v  * ((1-u)*fp_sw[1] + u*fp_se[1]))
    return lat, lon