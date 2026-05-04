from dataclasses import dataclass, field
from pathlib import Path

from core.geo import haversine_m, bbox_center_to_geo

import logging
logger = logging.getLogger(__name__)

GEO_NMS_THRESHOLD_M = 3.0    # detections within 3m = same object
MAX_ANGLES          = 3      # keep at most this many views per object instance

@dataclass
class Detection:
    """A single VLM detection, enriched with geo-coordinates."""
    filename:    str
    parent_path: str
    bbox:        list          # tile pixel coords [xmin,ymin,xmax,ymax]
    confidence:  float
    lat:         float         # geo-projected bbox center
    lon:         float
    tile_w:      int
    tile_h:      int
    candidate:   dict = field(repr=False)   # full LanceDB record

@dataclass
class ObjectInstance:
    """
    One physical object in the world, potentially seen from multiple frames.
    detections[0] is always the highest-confidence view.
    """
    instance_id:  int
    detections:   list[Detection]           # sorted by confidence desc
    is_multiangle: bool                     # True if >1 parent frame

    @property
    def best(self) -> Detection:
        return self.detections[0]

    @property
    def lat(self) -> float:
        return self.best.lat

    @property
    def lon(self) -> float:
        return self.best.lon


def enrich_detection(target: dict, cand: dict) -> Detection | None:
    """
    Converts a raw VLM target dict + LanceDB candidate into a Detection
    with geo-projected coordinates.
    """
    bbox = target.get('bbox', [])
    if len(bbox) != 4:
        return None

    lat, lon = bbox_center_to_geo(
        bbox=bbox,
        tile_w=cand['tile_w'], tile_h=cand['tile_h'],
        fp_nw=(cand['fp_nw_lat'], cand['fp_nw_lon']),
        fp_ne=(cand['fp_ne_lat'], cand['fp_ne_lon']),
        fp_se=(cand['fp_se_lat'], cand['fp_se_lon']),
        fp_sw=(cand['fp_sw_lat'], cand['fp_sw_lon']),
    )

    return Detection(
        filename=target.get('filename', Path(cand['image_path']).name),
        parent_path=cand['parent_path'],
        bbox=bbox,
        confidence=float(target.get('confidence', 1.0)),
        lat=lat,
        lon=lon,
        tile_w=cand['tile_w'],
        tile_h=cand['tile_h'],
        candidate=cand,
    )


def merge_detections(
    raw_targets: list[dict],
    candidates:  list[dict],
    threshold_m: float = GEO_NMS_THRESHOLD_M,
    max_angles:  int   = MAX_ANGLES,
) -> list[ObjectInstance]:
    """
    Groups VLM detections by geo-proximity and applies:
      - NMS within the same parent frame (keep highest confidence)
      - Multi-Angle Persistence across different parent frames (keep up to max_angles)

    Returns one ObjectInstance per distinct physical object found.
    """
    # Build lookup: filename → candidate record
    cand_by_name = {Path(c['image_path']).name: c for c in candidates}

    # Enrich all detections with geo-coordinates
    detections: list[Detection] = []
    for t in raw_targets:
        fname = t.get('filename', '')
        cand  = cand_by_name.get(fname)
        if cand is None:
            logger.warning(f"No candidate found for target filename: {fname}")
            continue
        det = enrich_detection(t, cand)
        if det:
            detections.append(det)

    if not detections:
        return []

    # Sort by confidence descending — greedy NMS picks best first
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Greedy geo-clustering
    instances: list[ObjectInstance] = []
    used = [False] * len(detections)

    for i, det in enumerate(detections):
        if used[i]:
            continue

        # Start a new instance with this (highest-confidence) detection
        cluster: list[Detection] = [det]
        used[i] = True

        for j, other in enumerate(detections):
            if used[j]:
                continue
            dist = haversine_m(det.lat, det.lon, other.lat, other.lon)
            if dist <= threshold_m:
                cluster.append(other)
                used[j] = True

        # Within cluster: apply NMS per parent frame
        # (keep only the best detection from each frame, up to max_angles frames)
        best_per_frame: dict[str, Detection] = {}
        for d in cluster:
            existing = best_per_frame.get(d.parent_path)
            if existing is None or d.confidence > existing.confidence:
                best_per_frame[d.parent_path] = d

        # Sort frames by confidence and cap at max_angles
        kept = sorted(best_per_frame.values(),
                      key=lambda d: d.confidence, reverse=True)[:max_angles]

        instances.append(ObjectInstance(
            instance_id=len(instances),
            detections=kept,
            is_multiangle=len(kept) > 1,
        ))

    return instances