"""
core/segment_viz.py — Segment visualisation for debugging
==========================================================
Two functions, both for human inspection only — not used in the query pipeline.

save_segment_overview()
    Called during Stage 2 ingestion after SAM2 segmentation.
    Draws all segment bounding boxes as coloured outlines on the full frame.
    Saved to: DETECTIONS_PATH/segments/{frame_stem}_segments.jpg

save_scored_crops()
    Called during anomaly query after CLIP scoring.
    Saves individual crops of the top-K segments with score info burned in.
    Saved to: DETECTIONS_PATH/segments/{frame_stem}_scored/{rank}_{score}.jpg
"""

import cv2
import numpy as np
from pathlib import Path

# 20 visually distinct BGR colours — high contrast against both dark and
# light backgrounds so they're readable on water, grass, tarmac, etc.
_PALETTE: list[tuple[int, int, int]] = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 234,  43), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


def save_segment_overview(
    frame_bgr: np.ndarray,
    segments:  list[dict],     # [{bbox, area, clip_vector}, ...]
    out_path:  Path,
) -> None:
    """
    Saves the full frame with every segment drawn as a coloured bounding-box
    outline (no filled rectangles — those obscure the underlying image content).

    Each box is labelled with its index and area in thousands of pixels,
    making it easy to identify which segment covers which object and whether
    the area filters are working correctly.

    Parameters
    ----------
    frame_bgr : full-resolution frame as loaded by cv2.imread (BGR)
    segments  : list of segment dicts from SegmentStore / SAM2 output
                must contain "bbox" ([xmin,ymin,xmax,ymax]) and "area" (int)
    out_path  : destination .jpg path — parent directories are created
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Work on a copy so the original is unmodified
    canvas = frame_bgr.copy()

    for i, seg in enumerate(segments):
        xmin, ymin, xmax, ymax = seg["bbox"]
        color = _PALETTE[i % len(_PALETTE)]

        # Bounding box outline — thickness 3px, clearly visible on any background
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 3)

        # Label: index + area in kpx (e.g. "#3 142k")
        area_k = seg.get("area", 0) // 1000
        label  = f"#{i} {area_k}k"

        # Small filled background behind text for readability
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        tx, ty = xmin + 4, ymin + th + 4
        cv2.rectangle(
            canvas,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline),
            color, -1,           # filled background in the segment's own colour
        )
        cv2.putText(
            canvas, label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0),           # black text on coloured background
            2, cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)


def save_scored_crops(
    frame_bgr:   np.ndarray,
    scored_segs: list,          # list[ScoredSegment], sorted by score desc
    out_dir:     Path,
    top_k:       int = 10,
) -> None:
    """
    Saves individual crops of the top-K anomaly-scored segments.

    Each crop has score info burned in:
        #rank  Δ=anomaly_score  ano=score_anomaly  exp=score_expected

    Files are named so they sort by rank:
        00_d0.187_523_768.jpg   ← rank 0, score 0.187, at x=523 y=768
        01_d0.091_...jpg

    Parameters
    ----------
    frame_bgr   : full-resolution frame (BGR)
    scored_segs : ScoredSegment list sorted descending by anomaly_score
    out_dir     : directory to write crops into
    top_k       : how many to save (default 10)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, seg in enumerate(scored_segs[:top_k]):
        xmin, ymin, xmax, ymax = seg.bbox
        crop = frame_bgr[ymin:ymax, xmin:xmax].copy()
        if crop.size == 0:
            continue

        # Score label — green on dark background
        label = (
            f"#{rank}  D={seg.anomaly_score:.3f}  "
            f"ano={seg.score_anomaly:.3f}  exp={seg.score_expected:.3f}"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(crop, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
        cv2.putText(
            crop, label,
            (4, th + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1, cv2.LINE_AA,
        )

        fname = out_dir / f"{rank:02d}_d{seg.anomaly_score:.3f}_{xmin}_{ymin}.jpg"
        cv2.imwrite(str(fname), crop)