"""
ai/segmentor.py — Frame segmentation via Replicate (SAM2)
==========================================================
Runs Meta's SAM-2 model on the Replicate cloud platform — no local GPU
or model weights required.

Model
-----
    meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83

Configuration (.env)
--------------------
    REPLICATE_API_KEY=r8_...          # required
    SAM_REPLICATE_MODEL=meta/sam-2:fe97...  # optional override
    SAM_MAX_DIM=1024                  # long-edge resize before upload (0 = off)

How it works
------------
SAM-2 requires spatial prompts (points or boxes).  To produce automatic,
un-prompted segmentation we generate a uniform NxN grid of foreground points
that covers the image — the same strategy SAM-2's own AutomaticMaskGenerator
uses internally.  Each point "votes" for the segment it belongs to and SAM-2
deduplicates overlapping masks.  Grid density is controlled by POINTS_PER_SIDE
(default 16, giving 256 prompt points).

Pipeline per frame
------------------
    1. Resize to SAM_MAX_DIM on the long edge  (reduces API cost & latency)
    2. JPEG-encode → base64 data URI           (no disk write)
    3. POST to Replicate with point grid
    4. Replicate returns a list of mask PNGs   (one per detected segment)
    5. Each PNG decoded → binary numpy mask
    6. bbox + area computed from mask extent
    7. Area filters applied                    (noise + background)
    8. bbox scaled back to original resolution
    9. Crop extracted from original (unresized) frame → passed to CLIP encoder
"""

import base64
import logging
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import json
import numpy as np

logger = logging.getLogger("TAE.Segmentor")

# ── Grid prompting ────────────────────────────────────────────────────────────
# NxN evenly-spaced foreground points used to prompt SAM2.
# 16×16 = 256 points gives reasonable coverage without exploding API cost.
POINTS_PER_SIDE: int = 16

# ── Area filters ─────────────────────────────────────────────────────────────
MIN_SEGMENT_AREA_PX:   int   = 400    # < 20×20 px → noise, discard
MAX_SEGMENT_AREA_FRAC: float = 0.40   # > 40 % of frame area → background, discard


def _make_point_grid(img_w: int, img_h: int, n: int) -> tuple[list, list]:
    """
    Generate an n×n uniform grid of foreground points covering the image.
    Returns (points [[x,y],...], labels [1,...]) ready for the SAM2 API.
    """
    xs = [int(img_w * (j + 0.5) / n) for j in range(n)]
    ys = [int(img_h * (i + 0.5) / n) for i in range(n)]
    points = [[x, y] for y in ys for x in xs]
    labels = [1] * len(points)   # 1 = foreground
    return points, labels


class SAM2Segmentor:
    """
    Calls Replicate SAM-2 to segment a full drone frame into regions.

    Keeps the same .generate(frame_bgr) → list[dict] interface as the
    previous local-inference implementation so no other code changes.
    """

    def __init__(
        self,
        api_key:       str       = "",
        model_version: str       = "",
        max_dim:       int       = 1024,
    ) -> None:
        self._model  = (
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
                "REPLICATE_API_KEY not set — Stage 2 segmentation disabled. "
                "Add REPLICATE_API_KEY to .env to enable."
            )
            return

        try:
            import replicate
            self._client = replicate.Client(api_token=api_key)
            logger.info("Replicate client ready")
        except ImportError:
            logger.warning(
                "replicate package not installed — Stage 2 disabled. "
                "Run: pip install replicate"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Segment a full drone frame and return filtered region descriptors.

        Parameters
        ----------
        frame_bgr : np.ndarray  (BGR, as returned by cv2.imread)

        Returns
        -------
        list of dicts:
            bbox  : [xmin, ymin, xmax, ymax]  — full-frame pixel coordinates
            area  : int                        — region area in original px²
            crop  : np.ndarray (BGR)           — region crop for CLIP encoding
        """
        if self._client is None:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        frame_area = frame_h * frame_w
        max_area   = int(frame_area * MAX_SEGMENT_AREA_FRAC)

        # ── 1. Resize for API efficiency ──────────────────────────────────────
        scale          = 1.0
        inference_frame = frame_bgr

        if self._max_dim > 0 and max(frame_h, frame_w) > self._max_dim:
            scale  = self._max_dim / max(frame_h, frame_w)
            new_w  = int(frame_w * scale)
            new_h  = int(frame_h * scale)
            inference_frame = cv2.resize(
                frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            logger.debug(
                f"Resized {frame_w}×{frame_h} → {new_w}×{new_h} "
                f"(scale={scale:.3f}) before Replicate upload"
            )

        inf_h, inf_w = inference_frame.shape[:2]

        # ── 2. Encode as JPEG data URI (no disk write) ────────────────────────
        _, buf     = cv2.imencode(".jpg", inference_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
        data_uri   = (
            "data:image/jpeg;base64,"
            + base64.b64encode(buf.tobytes()).decode()
        )

        # ── 3. Build uniform point grid as SAM2 prompts ───────────────────────
        points, labels = _make_point_grid(inf_w, inf_h, POINTS_PER_SIDE)

        # ── 4. Call Replicate ─────────────────────────────────────────────────
        logger.info(
            f"Replicate SAM2 | uploading {inf_w}×{inf_h}px | "
            f"{len(points)} prompt points …"
        )
        try:
            output = self._client.run(
                self._model,
                input={
                    "image":         data_uri,
                    "input_points":  json.dumps(points),
                    "input_labels":  json.dumps(labels),
                    "multimask_output": False,   # one mask per point
                    "pred_iou_thresh":  0.80,
                    "stability_score_thresh": 0.90,
                },
            )
        except Exception:
            import traceback
            logger.error(
                f"Replicate API call failed:\n{traceback.format_exc()}"
            )
            return []

        logger.info(
            f"Replicate returned output type={type(output).__name__} "
            f"len={len(output) if hasattr(output, '__len__') else '?'}"
        )

        # ── 5-9. Parse masks → regions ────────────────────────────────────────
        return self._parse_output(
            output, frame_bgr, frame_h, frame_w, inf_h, inf_w,
            scale, max_area
        )

    # ── Output parsing ────────────────────────────────────────────────────────

    def _parse_output(
        self, output, frame_bgr, frame_h, frame_w,
        inf_h, inf_w, scale, max_area
    ) -> list[dict]:

        # ── Log structure (keep for diagnostics) ─────────────────────────────
        if isinstance(output, dict):
            logger.info(f"Replicate output keys: {list(output.keys())}")
            for k, v in output.items():
                logger.info(
                    f"  '{k}': type={type(v).__name__} "
                    f"is_list={isinstance(v,(list,tuple))} "
                    f"has_read={hasattr(v,'read')}"
                )

        # ── Strategy 1: individual_masks (preferred — one mask per object) ────
        if isinstance(output, dict) and "individual_masks" in output:
            ind          = output["individual_masks"]
            mask_sources = list(ind) if isinstance(ind, (list, tuple)) else [ind]
            logger.info(f"Using 'individual_masks' → {len(mask_sources)} object mask(s)")

            # Download all masks in parallel — each is a small PNG,
            # latency dominates so concurrency gives near-linear speedup.
            def _fetch(args):
                idx, src = args
                return idx, self._load_mask(src, None, None)

            masks_by_idx: dict[int, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=min(32, len(mask_sources))) as pool:
                futures = {pool.submit(_fetch, (i, src)): i
                        for i, src in enumerate(mask_sources)}
                for future in as_completed(futures):
                    idx, mask_np = future.result()
                    if mask_np is not None:
                        masks_by_idx[idx] = mask_np

            logger.info(
                f"Downloaded {len(masks_by_idx)}/{len(mask_sources)} masks "
                f"(parallel)"
            )

            regions:     list[dict] = []
            seen_bboxes: set[tuple] = set()

            # Process in original order for deterministic deduplication
            for idx in sorted(masks_by_idx):
                mask_np = masks_by_idx[idx]
                mask_h, mask_w = mask_np.shape
                sx = frame_w / mask_w
                sy = frame_h / mask_h

                row_idx = np.where(np.any(mask_np, axis=1))[0]
                col_idx = np.where(np.any(mask_np, axis=0))[0]
                if len(row_idx) == 0 or len(col_idx) == 0:
                    continue

                xmin = max(0,       int(col_idx[0]  * sx))
                ymin = max(0,       int(row_idx[0]  * sy))
                xmax = min(frame_w, int(col_idx[-1] * sx))
                ymax = min(frame_h, int(row_idx[-1] * sy))

                if xmax <= xmin or ymax <= ymin:
                    continue

                area = (xmax - xmin) * (ymax - ymin)
                if area < MIN_SEGMENT_AREA_PX or area > max_area:
                    continue

                bbox_key = (xmin // 20, ymin // 20, xmax // 20, ymax // 20)
                if bbox_key in seen_bboxes:
                    continue
                seen_bboxes.add(bbox_key)

                regions.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "area": area,
                    "crop": frame_bgr[ymin:ymax, xmin:xmax],
                })

            logger.info(f"individual_masks → {len(regions)} regions after filtering")
            return regions

        # ── Strategy 2: combined_mask — CC split (fallback) ───────────────────
        if isinstance(output, dict):
            src = output.get("combined_mask") or next(
                (v for v in output.values() if hasattr(v, "read")), None
            )
        else:
            src = output

        if src is None:
            logger.warning("No usable mask source found in Replicate output")
            return []

        mask_np = self._load_mask(src, None, None)
        if mask_np is None:
            return []

        mask_h, mask_w = mask_np.shape
        sx = frame_w / mask_w
        sy = frame_h / mask_h
        coverage = mask_np.sum() / (mask_h * mask_w)
        logger.info(
            f"combined_mask: {mask_w}×{mask_h} coverage={coverage:.1%} "
            f"→ CC splitting"
        )

        seen_bboxes: set[tuple] = set()
        return self._split_into_components(
            mask_np, frame_bgr, frame_h, frame_w, sx, sy, max_area, seen_bboxes
    )

    def _split_into_components(
        self, mask_np, frame_bgr, frame_h, frame_w,
        sx, sy, max_area, seen_bboxes
    ) -> list[dict]:
        mask_u8 = mask_np.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return []

        regions = []
        for lbl in range(1, num_labels):
            x_m = int(stats[lbl, cv2.CC_STAT_LEFT])
            y_m = int(stats[lbl, cv2.CC_STAT_TOP])
            w_m = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h_m = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            xmin = max(0,       int(x_m * sx))
            ymin = max(0,       int(y_m * sy))
            xmax = min(frame_w, int((x_m + w_m) * sx))
            ymax = min(frame_h, int((y_m + h_m) * sy))

            if xmax <= xmin or ymax <= ymin:
                continue

            area = (xmax - xmin) * (ymax - ymin)
            if area < MIN_SEGMENT_AREA_PX or area > max_area:
                continue

            bbox_key = (xmin // 20, ymin // 20, xmax // 20, ymax // 20)
            if bbox_key in seen_bboxes:
                continue
            seen_bboxes.add(bbox_key)

            regions.append({
                "bbox": [xmin, ymin, xmax, ymax],
                "area": area,
                "crop": frame_bgr[ymin:ymax, xmin:xmax],
            })

        logger.info(f"CC split: {num_labels-1} components → {len(regions)} regions")
        return regions

    @staticmethod
    def _load_mask(src, expected_h: int, expected_w: int) -> np.ndarray | None:
        """
        Load one mask from a Replicate FileOutput, URL string, or bytes.
        Returns a boolean numpy array (H, W) in inference-frame coordinates.
        expected_h, expected_w params kept for API compatibility but ignored — return at native resolution
        """
        try:
            # FileOutput object (replicate.helpers.FileOutput)
            if hasattr(src, "read"):
                raw = src.read()

            # Plain URL string
            elif isinstance(src, str) and src.startswith("http"):
                with urllib.request.urlopen(src, timeout=30) as resp:
                    raw = resp.read()

            # Already bytes
            elif isinstance(src, bytes):
                raw = src

            # Numpy array — already a mask
            elif isinstance(src, np.ndarray):
                return src.astype(bool)

            else:
                logger.debug(f"Unhandled mask source type: {type(src)}")
                return None

            # Decode PNG/JPEG → grayscale mask
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("cv2.imdecode returned None for mask")
                return None

            return img > 127   # threshold → boolean mask

        except Exception:
            import traceback
            logger.warning(f"Failed to load mask:\n{traceback.format_exc()}")
            return None