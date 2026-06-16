"""
ai/vlm.py — VLM service (TacticalAnalyst)
==========================================
Wraps Qwen2.5-VL (via OpenRouter) or a local Ollama model.

Two public methods
------------------
analyze_multiple_views(candidates, user_query)
    Legacy CLIP→VLM path used by the anomaly_detection pipeline.
    Accepts LanceDB tile records, runs the VLM on each, returns merged targets.

verify_detection(image, criteria, report_fields, ...)
    New method for Stage 4 of the object_detection pipeline (ai/detection_pipeline.py).
    Accepts a SAM-masked crop, structured criteria from ObjectDetectionParams,
    and returns a structured confirmation + report dict.
    Prompt is built dynamically — no hardcoded template per query type.

Renamed from: analyst.py / vlm.py
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential, retry_if_exception


logger = logging.getLogger("TacticalAnalyst")


# ─────────────────────────────────────────────────────────────────────────────
# Tile reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _load_tile_cv2(candidate: dict) -> np.ndarray | None:
    """
    Reconstruct a tile by cropping its parent frame.
    Tiles are never stored on disk — this is the single reconstruction point.
    """
    img = cv2.imread(candidate["parent_path"])
    if img is None:
        logger.warning("Cannot read parent frame: %s", candidate["parent_path"])
        return None
    x, y = candidate["tile_x"], candidate["tile_y"]
    w, h = candidate["tile_w"], candidate["tile_h"]
    return img[y: y + h, x: x + w]

#─────────────────────────────────────────────────────────────────────────────
# Helpers 
#─────────────────────────────────────────────────────────────────────────────

# returns (image, remapped_bboxes) 
def _context_crop_upscale(
    tile_img:      np.ndarray,
    bboxes:        list,
    pad_factor:    int   = 4,
    min_pad_px:    int   = 48,
    min_output_px: int   = 256,
) -> tuple[np.ndarray, list]:
    th, tw = tile_img.shape[:2]
    xs1 = [int(b[0]) for b in bboxes]; ys1 = [int(b[1]) for b in bboxes]
    xs2 = [int(b[2]) for b in bboxes]; ys2 = [int(b[3]) for b in bboxes]

    max_side = max(
        max(x2 - x1 for x1, x2 in zip(xs1, xs2)),
        max(y2 - y1 for y1, y2 in zip(ys1, ys2)),
    )
    if max_side > min(tw, th) * 0.25:
        return tile_img, bboxes      # no crop — bboxes are already correct

    pad  = max(min_pad_px, max_side * pad_factor)
    cx1  = max(0, min(xs1) - pad)
    cy1  = max(0, min(ys1) - pad)
    cx2  = min(tw, max(xs2) + pad)
    cy2  = min(th, max(ys2) + pad)
    crop = tile_img[int(cy1):int(cy2), int(cx1):int(cx2)]

    ch, cw = crop.shape[:2]
    scale  = 1.0
    if min(ch, cw) < min_output_px:
        scale = min_output_px / min(ch, cw)
        crop  = cv2.resize(
            crop,
            (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # Remap bboxes into cropped+scaled coordinate space
    remapped = [
        [
            max(0, int((b[0] - cx1) * scale)),
            max(0, int((b[1] - cy1) * scale)),
            max(0, int((b[2] - cx1) * scale)),
            max(0, int((b[3] - cy1) * scale)),
        ]
        for b in bboxes
    ]
    return crop, remapped

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates — analyze_multiple_views (legacy / anomaly path)
# ─────────────────────────────────────────────────────────────────────────────

_BBOX_RULES = (
    "Bounding box rules:\n"
    "- Return pixel coordinates [xmin, ymin, xmax, ymax]\n"
    "- 0,0 is the TOP-LEFT corner; image size is stated above\n"
    "- xmin < xmax and ymin < ymax always\n"
    "- Draw the TIGHTEST possible box around the object\n"
    "- If the object is not present, return an empty targets list\n"
    "- confidence: 0.0 (uncertain) to 1.0 (certain)\n"
)


def _build_prompt(
    user_query: str,
    filename:   str,
    img_w:      int,
    img_h:      int,
) -> tuple[str, bool]:
    """Returns (prompt_text, expects_bboxes)."""
    img_info = f"Image: {filename} ({img_w}×{img_h} pixels)"
    prompt = (
        f"You are analyzing UAV aerial imagery.\n"
        f"{img_info}\n\n"
        f"Find all instances of the following:\n"
        f"TARGET: {user_query}\n\n"
        f"AERIAL IMAGERY NOTE: Objects are seen from above at altitude. "
        f"Fine surface detail is not visible — judge by overall shape, approximate "
        f"size, colour, and shadow. A rough match to the query description is "
        f"sufficient to confirm. Do not reject because detail is absent.\n\n"
        "Return ONLY a valid JSON object:\n"
        "{\n"
        '  "report": "brief summary of findings",\n'
        '  "targets": [\n'
        '    {"filename": "<name>", "bbox": [xmin, ymin, xmax, ymax],'
        ' "confidence": 0.0, "description": "concise label"}\n'
        '  ]\n'
        "}\n\n"
        + _BBOX_RULES
        + "No markdown, no text outside the JSON object."
    )
    return prompt, True


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — verify_detection (object_detection pipeline Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def _build_verify_prompt(
    label_hint:    str,
    criteria:      str,
    report_fields: list[str],
    original_query: str,
    colour_hint:   str | None,
    size_qualifier: str | None,
    img_w: int,
    img_h: int,
) -> str:
    """
    Dynamic VLM verification prompt — adapts to any query type via
    criteria and report_fields from ObjectDetectionParams.
    No hardcoded templates; new object classes need no code changes.
    """
    field_schema = "\n".join(f'    "{f}": "<value>"' for f in report_fields)
    colour_line  = f"\nColour qualifier: {colour_hint}" if colour_hint else ""
    size_line    = f"\nSize qualifier: {size_qualifier}" if size_qualifier else ""

    return (
        f"You are analyzing a crop from a UAV nadir (top-down) aerial image.\n"
        f"The image shows a single candidate detection: {label_hint}.\n"
        f"Image size: {img_w}×{img_h} pixels.\n"
        f"Original operator query: \"{original_query}\""
        f"{colour_line}{size_line}\n\n"
        f"TASK: Determine whether this detection is genuine.\n\n"
        f"VERIFICATION CRITERIA\n{criteria}\n\n"
        f"If confirmed, populate the report fields below.\n"
        f"If rejected, explain why briefly.\n\n"
        "Return ONLY a valid JSON object in this exact format:\n"
        "{\n"
        '  "confirmed": true or false,\n'
        '  "reason": "brief explanation if rejected, else empty string",\n'
        '  "report": {\n'
        f"{field_schema}\n"
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- confirmed must be a boolean (true/false), not a string.\n"
        "- If confirmed is false, report fields may be empty strings.\n"
        "- No markdown, no text outside the JSON object.\n"
        "- Be strict: reject shadows, vegetation patterns, or ambiguous blobs."
    )

def _build_batch_verify_prompt(
    label_hints:           list[str],
    bboxes:                list[list],
    criteria:              str,
    report_fields:         list[str],
    original_query:        str,
    img_w: int, img_h:     int,
    confidences:           list[float] | None = None,
    caution_confirm_below: float | None = None,
    caution_reject_above:  float | None = None,
) -> str:
    candidates_desc = "\n".join(
        f"  {i}: label='{label_hints[i]}' bbox={bboxes[i]}"
        + (f" gdino_conf={confidences[i]:.2f}" if confidences else "")
        for i in range(len(bboxes))
    )
    field_schema = ", ".join(f'"{f}"' for f in report_fields)

    # Build calibration block only when both thresholds and confidence scores are available
    if confidences and caution_confirm_below is not None and caution_reject_above is not None:
        conf_guidance = (
            f"\nDETECTOR CONFIDENCE GUIDANCE\n"
            f"Each candidate carries gdino_conf: the score from Grounding DINO, a specialized object detector.\n"
            f"- gdino_conf < {caution_confirm_below:.2f} → weak detector signal; "
            f"apply extra scrutiny before confirming — do not confirm on ambiguous visual evidence alone.\n"
            f"- gdino_conf > {caution_reject_above:.2f} → strong detector signal; "
            f"require clear visual counter-evidence before rejecting.\n"
            f"- Otherwise → neutral; let visual evidence alone determine the decision.\n"
            f"This is a calibration prior. If visual evidence clearly contradicts the detector, trust your eyes.\n"
        )
    else:
        conf_guidance = ""

    return (
        f"You are analyzing a UAV nadir (top-down) aerial tile.\n"
        f"Image size: {img_w}×{img_h} pixels. 0,0 is top-left.\n"
        f"Original query: \"{original_query}\"\n\n"
        f"VERIFICATION CRITERIA\n{criteria}\n"
        f"{conf_guidance}\n"
        f"The following {len(bboxes)} candidate detection(s) are marked on this tile:\n"
        f"{candidates_desc}\n\n"
        f"For EACH candidate, decide: confirmed or rejected.\n"
        f"Return ONLY valid JSON — a list with one entry per candidate, in order:\n"
        f"[\n"
        f"  {{\"index\": 0, \"confirmed\": true/false, "
        f"\"detected_label\": \"the most accurate class name for what you see\", "
        f"\"confidence\": 0.0-1.0, "
        f"\"reason\": \"one sentence — ONLY when confirmed; omit key entirely if rejected\", "
        f"\"report\": {{{field_schema}: \"...\"}}}},\n"
        f"  ...\n"
        f"]\n"
        f"No markdown, no text outside the JSON array."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TacticalAnalyst
# ─────────────────────────────────────────────────────────────────────────────
def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 (rate-limit) and 403 (provider error) but not on 400/401."""
    msg = str(exc)
    return "429" in msg or "403" in msg

    
class TacticalAnalyst:

    def __init__(
        self,
        provider:   str        = "openrouter",
        model_name: str | None = None,
        api_key:    str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.api_key  = api_key
        logger.info("Analyst initialized. Provider: %s", self.provider)

        if self.provider == "openrouter":
            self.model_name = model_name or "qwen/qwen-2.5-vl-72b-instruct"
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/arie/TAE",
                    "X-Title": "TAE",
                },
            )
        else:
            self.model_name = model_name or "moondream"

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_multiple_views(
        self,
        candidates: list[dict],
        user_query: str,
    ) -> dict:
        """
        Legacy CLIP→VLM path — used by anomaly_detection pipeline.

        Analyses each candidate tile independently and merges results.
        Tiles are reconstructed on demand from parent frames; no disk I/O for tiles.
        """
        logger.info(
            "Query: %s | Tiles: %s",
            user_query,
            [Path(c["image_path"]).name for c in candidates],
        )

        merged_targets: list[dict] = []
        hit_reports:    list[str]  = []
        miss_count = 0

        for cand in candidates:
            filename = Path(cand["image_path"]).name
            tile_img = _load_tile_cv2(cand)
            if tile_img is None:
                logger.warning("Skipping %s — could not load parent frame", filename)
                continue

            img_h, img_w = tile_img.shape[:2]
            prompt, expects_bboxes = _build_prompt(user_query, filename, img_w, img_h)

            try:
                result  = self._call_vlm_with_validation(
                    tile_img, filename, prompt, expects_bboxes, img_w, img_h
                )
                targets = result.get("targets", [])
                if targets:
                    hit_reports.append(f"{filename}: {result.get('report', '')}")
                    merged_targets.extend(targets)
                else:
                    miss_count += 1
                    logger.info("No detection in %s (VLM confirmed absent)", filename)
            except Exception as e:
                logger.error("VLM failed for %s after retries: %s", filename, e)

        summary = (
            f"{len(hit_reports)} tiles with detections, "
            f"{miss_count} tiles confirmed empty"
        )
        report = (
            " | ".join(hit_reports)
            if hit_reports
            else f"No detections found. {summary}"
        )
        return {"report": report, "targets": merged_targets, "summary": summary}

    def verify_detection(
        self,
        image:          np.ndarray,
        criteria:       str,
        report_fields:  list[str],
        original_query: str,
        colour_hint:    str | None = None,
        size_qualifier: str | None = None,
        label_hint:     str        = "detected object",
    ) -> dict:
        """
        Stage 4 of the object_detection pipeline.

        Accepts the SAM-masked crop (background is neutral grey), structured
        criteria and report_fields from ObjectDetectionParams.

        Returns
        -------
        {
            "confirmed": bool,
            "reason":    str,   # why rejected (empty if confirmed)
            "report":    dict,  # report_fields populated by VLM
        }
        Never raises — returns {"confirmed": False, "reason": "error: ..."} on failure.
        """
        if image is None or image.size == 0:
            return {"confirmed": False, "reason": "empty image", "report": {}}

        h, w   = image.shape[:2]
        prompt = _build_verify_prompt(
            label_hint     = label_hint,
            criteria       = criteria,
            report_fields  = report_fields,
            original_query = original_query,
            colour_hint    = colour_hint,
            size_qualifier = size_qualifier,
            img_w          = w,
            img_h          = h,
        )

        filename = f"crop_{label_hint[:20]}.jpg"
        try:
            if self.provider == "openrouter":
                raw = self._analyze_openrouter(image, filename, prompt)
            else:
                raw = self._analyze_ollama(image, filename, prompt)
        except Exception as exc:
            logger.warning("VLM call failed in verify_detection: %s", exc)
            return {"confirmed": False, "reason": f"error: {exc}", "report": {}}

        try:
            clean  = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            result = json.loads(clean)
            return {
                "confirmed": bool(result.get("confirmed", False)),
                "reason":    result.get("reason", ""),
                "report":    result.get("report", {}),
            }
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse VLM verify response: %s\nRaw: %s", exc, raw[:200]
            )
            return {"confirmed": False, "reason": f"parse error: {exc}", "report": {}}

    def verify_detections_batch(
        self,
        tile_img:              np.ndarray,
        detections:            list[dict],
        criteria:              str,
        report_fields:         list[str],
        original_query:        str,
        colour_hint:           str | None = None,
        size_qualifier:        str | None = None,
        caution_confirm_below: float | None = None,
        caution_reject_above:  float | None = None,
    ) -> list[dict]:
        """
        Validate ALL detections on a tile in a single VLM call.
        Returns a list parallel to `detections`:
        [{"confirmed": bool, "reason": str, "report": dict}, ...]
        """
        if not detections or tile_img is None:
            return [{"confirmed": False, "reason": "empty", "report": {}} 
                    for _ in detections]

        h, w = tile_img.shape[:2]

        # Draw ALL bboxes on tile so VLM can see them
        annotated = tile_img.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (74, 222, 128), 2)
            cv2.putText(annotated, str(i), (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 222, 128), 1)

        # If objects are small relative to the tile, crop to context and upscale
        all_bboxes            = [det["bbox"] for det in detections]
        annotated, vlm_bboxes = _context_crop_upscale(annotated, all_bboxes)
        h, w                  = annotated.shape[:2]   # actual dims after crop+upscale

        confs = (
            [d["confidence"] for d in detections]
            if detections and "confidence" in detections[0]
            else None
        )
        prompt = _build_batch_verify_prompt(
            label_hints           = [d["label"] for d in detections],
            bboxes                = vlm_bboxes,
            criteria              = criteria,
            report_fields         = report_fields,
            original_query        = original_query,
            img_w = w, img_h = h,
            confidences           = confs,
            caution_confirm_below = caution_confirm_below,
            caution_reject_above  = caution_reject_above,
        )
        
        try:
            if self.provider == "openrouter":
                raw = self._analyze_openrouter(annotated, "tile_batch.jpg", prompt)
            else:
                raw = self._analyze_ollama(annotated, "tile_batch.jpg", prompt)
        except Exception as exc:
            logger.warning("Batch VLM failed: %s", exc)
            return [{"confirmed": False, "reason": f"error: {exc}", "report": {}}
                    for _ in detections]

        try:
            clean   = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            clean   = re.sub(r"^\s*>+\s*", "", clean)   # strip leading > emitted by some providers
            results = json.loads(clean)
            # Normalise: ensure one entry per detection, keyed by index
            out = [{"confirmed": False, "reason": "missing", "report": {}}] * len(detections)
            for r in results:
                idx = int(r.get("index", -1))
                if 0 <= idx < len(detections):
                    out[idx] = {
                        "confirmed":      bool(r.get("confirmed", False)),
                        "detected_label": r.get("detected_label", "").strip(),
                        "confidence":     float(r.get("confidence", 0.0)),
                        "reason":         r.get("reason", ""),   # empty string for rejections
                        "report":         r.get("report", {}),
                    }
            return out
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Batch VLM parse error: %s\nRaw: %s", exc, raw[:300])
            return [{"confirmed": False, "reason": f"parse error: {exc}", "report": {}}
                    for _ in detections]

    # ──────────────────────────────────────────────────────────────────────────
    # VLM call with retry
    # ──────────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(ValueError),
    )
    def _call_vlm_with_validation(
        self,
        tile_img:       np.ndarray,
        filename:       str,
        prompt:         str,
        expects_bboxes: bool,
        img_w:          int,
        img_h:          int,
    ) -> dict:
        if self.provider == "openrouter":
            res_text = self._analyze_openrouter(tile_img, filename, prompt)
        else:
            res_text = self._analyze_ollama(tile_img, filename, prompt)

        logger.debug("Raw VLM response: %s", res_text)
        clean  = re.sub(r"^```json\s*|\s*```$", "", res_text.strip(), flags=re.MULTILINE)
        result = json.loads(clean)

        if expects_bboxes and result.get("targets"):
            result["targets"] = self._normalize_targets(result["targets"], img_w, img_h)
            result["targets"] = self._clamp_targets(result["targets"], img_w, img_h)
            result["targets"] = self._filter_low_confidence_targets(result["targets"])
            if not self._validate_targets(result["targets"], img_w, img_h):
                raise ValueError(f"Invalid bboxes after normalization: {result['targets']}")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Provider backends
    # ──────────────────────────────────────────────────────────────────────────

    def _serialise_tile_for_api(self, tile_img: np.ndarray, filename: str) -> str:
        """Encode a numpy array as a base64 JPEG string."""
        ok, buf = cv2.imencode(".jpg", tile_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError(f"Failed to encode tile {filename}")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")
    

    def _analyze_openrouter(
        self,
        tile_img: np.ndarray,
        filename: str,
        prompt:   str,
    ) -> str:
        b64_image = self._serialise_tile_for_api(tile_img, filename)

        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        def _call() -> str:
            response = self.client.chat.completions.create(
                model    = self.model_name,
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens  = 1024,
                temperature = 0.1,
                extra_body = {
                    "provider": {
                        "order": ["Together", "NovitaAI", "Nebius Token Factory"],
                        "allow_fallbacks": True,
                    }
                },
            )
            return response.choices[0].message.content or ""

        return _call()


    def _analyze_ollama(
        self,
        tile_img: np.ndarray,
        filename: str,
        prompt:   str,
    ) -> str:
        b64_image = self._serialise_tile_for_api(tile_img, filename)
        response  = ollama.chat(
            model    = self.model_name,
            messages = [{
                "role":    "user",
                "content": prompt,
                "images":  [b64_image],
            }],
        )
        return response["message"]["content"]

    # ──────────────────────────────────────────────────────────────────────────
    # Bbox validation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_targets(
        self, targets: list, img_w: int, img_h: int
    ) -> list:
        """Convert 0-1000 normalized coords to pixels if the VLM used that scale."""
        normalized = []
        for t in targets:
            b = t.get("bbox")
            if not b or len(b) != 4:
                continue
            if all(0.0 <= v <= 1.0 for v in b):
                b = [b[0]*img_w, b[1]*img_h, b[2]*img_w, b[3]*img_h]
            elif max(b) <= 1000 and any(v > 1.0 for v in b):
                b = [b[0]/1000*img_w, b[1]/1000*img_h,
                     b[2]/1000*img_w, b[3]/1000*img_h]
            t = dict(t); t["bbox"] = [int(round(v)) for v in b]
            normalized.append(t)
        return normalized

    def _clamp_targets(
        self, targets: list, img_w: int, img_h: int
    ) -> list:
        clamped = []
        for t in targets:
            b = t.get("bbox")
            if not b or len(b) != 4:
                continue
            b = [
                max(0, min(int(b[0]), img_w - 1)),
                max(0, min(int(b[1]), img_h - 1)),
                max(0, min(int(b[2]), img_w)),
                max(0, min(int(b[3]), img_h)),
            ]
            if b[2] > b[0] and b[3] > b[1]:
                t = dict(t); t["bbox"] = b
                clamped.append(t)
        return clamped

    def _validate_targets(
        self, targets: list, img_w: int, img_h: int
    ) -> bool:
        for t in targets:
            b = t.get("bbox", [])
            if len(b) != 4:
                return False
            x1, y1, x2, y2 = b
            if x1 >= x2 or y1 >= y2:
                return False
            if x2 > img_w * 1.05 or y2 > img_h * 1.05:
                return False
        return True

    def _filter_low_confidence_targets(
        self, targets: list, threshold: float = 0.10
    ) -> list:
        return [
            t for t in targets
            if float(t.get("confidence", 1.0)) >= threshold
        ]
