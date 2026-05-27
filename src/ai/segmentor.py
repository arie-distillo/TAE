"""
ai/segmentor.py — Frame segmentation via Replicate (SAM2)
==========================================================
Two operating modes:

  Auto-segmentation (anomaly_detection)
  ─────────────────────────────────────
  .generate(frame_bgr)  →  list[dict{bbox, area, crop}]
  Uses a 16×16 uniform point grid to discover all regions in a frame.
  Expensive (~256 prompts per call). Runs at ingestion time; results
  cached in SegmentStore (SQLite) for reuse across queries.

  Box-prompted (object_detection)
  ────────────────────────────────
  .predict_box(tile_bgr, bbox)  →  np.ndarray boolean mask | None
  SAM refines a single YOLO bounding box into a precise pixel mask.
  Cheap (~1 prompt per call). Runs at query time in the detection pipeline.
  Returns a boolean mask in tile-local coordinates.

Configuration (.env)
--------------------
    REPLICATE_API_KEY=r8_...          # required
    SAM_REPLICATE_MODEL=meta/sam-2:fe97...  # optional override
    SAM_MAX_DIM=1024                  # long-edge resize before upload (0 = off)
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("TAE.Segmentor")

# ── Auto-segmentation constants ───────────────────────────────────────────────
POINTS_PER_SIDE: int   = 16      # NxN grid → 256 SAM prompts per frame
MIN_SEGMENT_AREA_PX    = 400     # < 20×20 px → noise, discard
MAX_SEGMENT_AREA_FRAC  = 0.40    # > 40% of frame → background, discard


def _make_point_grid(img_w: int, img_h: int, n: int) -> tuple[list, list]:
    xs = [int(img_w * (j + 0.5) / n) for j in range(n)]
    ys = [int(img_h * (i + 0.5) / n) for i in range(n)]
    points = [[x, y] for y in ys for x in xs]
    labels = [1] * len(points)
    return points, labels


def _encode_bgr_to_data_uri(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR numpy image as a base64 JPEG data URI."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _bbox_to_point_prompt(
    bbox: list[int],
    img_w: int,
    img_h: int,
) -> tuple[list[list[int]], list[int]]:
    """
    Convert a bounding box to a single foreground point at the bbox centre,
    plus 4 background points outside the box corners.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # Foreground: centre of the box
    points = [[cx, cy]]
    labels = [1]
    # Background: points just outside each corner
    margin = max(5, (x2 - x1 + y2 - y1) // 8)
    for px, py in [
        (max(0, x1 - margin), max(0, y1 - margin)),
        (min(img_w - 1, x2 + margin), max(0, y1 - margin)),
        (max(0, x1 - margin), min(img_h - 1, y2 + margin)),
        (min(img_w - 1, x2 + margin), min(img_h - 1, y2 + margin)),
    ]:
        points.append([px, py])
        labels.append(0)
    return points, labels


class SAM2Segmentor:
    """
    SAM2 via Replicate — supports both auto-segmentation and box-prompted modes.
    """

    def __init__(
        self,
        api_key:       str = "",
        model_version: str = "",
        max_dim:       int = 1024,
    ) -> None:
        self._model   = (
            model_version
            or "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"
        )
        self._max_dim = max_dim
        self._client  = None

        logger.info(
            f"Segmentor init | backend=replicate | "
            f"api_key={'SET' if api_key else 'MISSING'} | "
            f"model={self._model} | max_dim={max_dim}"
        )

        if not api_key:
            logger.warning(
                "REPLICATE_API_KEY not set — SAM2 segmentation disabled."
            )
            return

        try:
            import replicate
            self._client = replicate.Client(api_token=api_key)
            logger.info("SAM2 Replicate client ready")
        except ImportError:
            logger.warning("replicate package not installed — SAM2 disabled.")

    def is_available(self) -> bool:
        return self._client is not None

    # ──────────────────────────────────────────────────────────────────────────
    # Box-prompted mode (object_detection pipeline — Stage 3)
    # ──────────────────────────────────────────────────────────────────────────

    def predict_box(
        self,
        tile_bgr: np.ndarray,
        bbox: list[int],
    ) -> np.ndarray | None:
        """
        Run SAM2 with a single bounding-box prompt on a tile image.

        Returns a boolean mask in tile-local pixel coordinates, or None on failure.
        This is ~50× cheaper than auto-segmentation (1 prompt vs 256).

        Parameters
        ----------
        tile_bgr : BGR numpy array (the tile, not the full frame)
        bbox     : [x1, y1, x2, y2] in tile-local pixel coordinates
        """
        if self._client is None:
            return None

        h, w = tile_bgr.shape[:2]

        # Optional resize for API efficiency
        scale = 1.0
        img   = tile_bgr
        if self._max_dim > 0 and max(h, w) > self._max_dim:
            scale = self._max_dim / max(h, w)
            img   = cv2.resize(tile_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        inf_h, inf_w = img.shape[:2]
        data_uri = _encode_bgr_to_data_uri(img)

        # Scale bbox to inference resolution
        sx = inf_w / w; sy = inf_h / h
        s_bbox = [
            int(bbox[0] * sx), int(bbox[1] * sy),
            int(bbox[2] * sx), int(bbox[3] * sy),
        ]

        points, labels = _bbox_to_point_prompt(s_bbox, inf_w, inf_h)

        logger.debug(
            f"SAM2 box-prompt | {inf_w}×{inf_h}px | "
            f"bbox={s_bbox} | {len(points)} point(s)"
        )

        try:
            output = self._client.run(
                self._model,
                input={
                    "image":          data_uri,
                    "input_points":   json.dumps(points),
                    "input_labels":   json.dumps(labels),
                    "multimask_output":         False,
                    "pred_iou_thresh":          0.75,
                    "stability_score_thresh":   0.80,
                },
            )
        except Exception:
            import traceback
            logger.warning(f"SAM2 box-prompt API failed:\n{traceback.format_exc()}")
            return None

        if not output:
            return None

        # Take the first mask returned
        mask_src = output[0] if isinstance(output, list) else output
        mask_inf = self._load_mask(mask_src)
        if mask_inf is None:
            return None

        # Scale mask back to original tile resolution
        if scale != 1.0:
            mask_resized = cv2.resize(
                mask_inf.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = mask_inf

        # Clamp to bbox region (SAM should stay inside, but defensive)
        x1, y1, x2, y2 = bbox
        bounded = np.zeros((h, w), dtype=bool)
        bounded[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = \
            mask_resized[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if bounded.sum() == 0:
            logger.debug("SAM2 box-prompt returned empty mask after bounding")
            return None

        return bounded

    # ──────────────────────────────────────────────────────────────────────────
    # Auto-segmentation mode (anomaly_detection pipeline — ingestion stage)
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Segment a full drone frame using the 16×16 automatic point grid.
        Used for anomaly_detection ingestion (SegmentStore population).

        Returns list of dicts: {bbox, area, crop}
        """
        if self._client is None:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        frame_area = frame_h * frame_w
        max_area   = int(frame_area * MAX_SEGMENT_AREA_FRAC)

        # Resize for API efficiency
        scale = 1.0
        inf   = frame_bgr
        if self._max_dim > 0 and max(frame_h, frame_w) > self._max_dim:
            scale = self._max_dim / max(frame_h, frame_w)
            inf   = cv2.resize(frame_bgr, (int(frame_w * scale), int(frame_h * scale)),
                               interpolation=cv2.INTER_AREA)

        inf_h, inf_w = inf.shape[:2]
        data_uri = _encode_bgr_to_data_uri(inf)
        points, labels = _make_point_grid(inf_w, inf_h, POINTS_PER_SIDE)

        logger.info(
            f"SAM2 auto | {inf_w}×{inf_h}px | {len(points)} prompt points"
        )

        try:
            output = self._client.run(
                self._model,
                input={
                    "image":                    data_uri,
                    "input_points":             json.dumps(points),
                    "input_labels":             json.dumps(labels),
                    "multimask_output":         False,
                    "pred_iou_thresh":          0.80,
                    "stability_score_thresh":   0.90,
                },
            )
        except Exception:
            import traceback
            logger.error(f"SAM2 auto API failed:\n{traceback.format_exc()}")
            return []

        if not output:
            return []

        # Process masks in parallel threads (network fetch)
        mask_sources = output if isinstance(output, list) else [output]
        results: list[dict] = []

        def _process(src) -> dict | None:
            mask_inf = self._load_mask(src)
            if mask_inf is None:
                return None

            # Scale back to original frame resolution
            if scale != 1.0:
                mask_full = cv2.resize(
                    mask_inf.astype(np.uint8),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask_full = mask_inf

            ys, xs = np.where(mask_full)
            if len(xs) == 0:
                return None

            area = int(mask_full.sum())
            if area < MIN_SEGMENT_AREA_PX or area > max_area:
                return None

            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            crop = frame_bgr[y1:y2, x1:x2].copy()

            return {"bbox": [x1, y1, x2, y2], "area": area, "crop": crop}

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_process, src) for src in mask_sources]
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    results.append(r)

        logger.info(f"SAM2 auto | {len(mask_sources)} masks → {len(results)} valid segments")
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Shared helper — decode a mask from Replicate output
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_mask(src) -> np.ndarray | None:
        """Decode a mask from Replicate output (FileOutput, URL, bytes, or ndarray)."""
        try:
            if hasattr(src, "read"):
                raw = src.read()
            elif isinstance(src, str) and src.startswith("http"):
                with urllib.request.urlopen(src, timeout=30) as resp:
                    raw = resp.read()
            elif isinstance(src, bytes):
                raw = src
            elif isinstance(src, np.ndarray):
                return src.astype(bool)
            else:
                logger.debug(f"Unhandled mask source type: {type(src)}")
                return None

            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("cv2.imdecode returned None for SAM mask")
                return None
            return img > 127
        except Exception:
            import traceback
            logger.warning(f"Failed to load SAM mask:\n{traceback.format_exc()}")
            return None
