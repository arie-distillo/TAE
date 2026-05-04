import re
import json
import base64
import logging
from pathlib import Path

import cv2
import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger("TacticalAnalyst")


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

QUERY_INTENT_KEYWORDS = {
    "count":            ["how many", "count", "number of", "quantity"],
    "area_description": ["what is", "describe", "what's in", "what do you see", "overview"],
    "change":           ["unusual", "anomaly", "anomalies", "out of place", "changed", "different"],
    "object_search":    [],  # default fallback
}

def classify_query(user_query: str) -> str:
    q = user_query.lower()
    for intent, keywords in QUERY_INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "object_search"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BBOX_RULES = (
    "Bounding box rules:\n"
    "- Return pixel coordinates [xmin, ymin, xmax, ymax]\n"
    "- 0,0 is the TOP-LEFT corner; image size is stated above\n"
    "- xmin < xmax and ymin < ymax always\n"
    "- Draw the TIGHTEST possible box around the object\n"
    "- If the object is not present, return an empty targets list\n"
    "- confidence: 0.0 (uncertain) to 1.0 (certain)\n"
)

def _build_prompt(user_query: str, filename: str, img_w: int, img_h: int) -> tuple[str, bool]:
    """
    Returns (prompt, expects_bboxes).
    filename and image dimensions are passed so the VLM knows the coordinate space.
    """
    intent = classify_query(user_query)
    img_info = f"Image: {filename} ({img_w}×{img_h} pixels)"

    if intent == "object_search":
        prompt = (
            f"You are analyzing UAV aerial imagery.\n"
            f"{img_info}\n\n"
            f"Find all instances of the following:\n"
            f"TARGET: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "brief summary of findings",\n'
            '  "targets": [\n'
            '    {"filename": "<name>", "bbox": [xmin, ymin, xmax, ymax], "confidence": 0.0}\n'
            '  ]\n'
            "}\n\n"
            + _BBOX_RULES +
            "No markdown, no text outside the JSON object."
        )
        return prompt, True

    elif intent == "count":
        prompt = (
            f"You are analyzing UAV aerial imagery.\n"
            f"{img_info}\n\n"
            f"Count and locate: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "total count and summary",\n'
            '  "count": 0,\n'
            '  "targets": [\n'
            '    {"filename": "<name>", "bbox": [xmin, ymin, xmax, ymax], "confidence": 0.0}\n'
            '  ]\n'
            "}\n\n"
            + _BBOX_RULES +
            "No markdown, no text outside the JSON object."
        )
        return prompt, True

    elif intent == "area_description":
        prompt = (
            f"You are analyzing UAV aerial imagery.\n"
            f"{img_info}\n\n"
            f"Answer the following question about this image:\n"
            f"QUERY: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "detailed answer based on what you observe",\n'
            '  "targets": []\n'
            "}\n\n"
            "No markdown, no text outside the JSON object."
        )
        return prompt, False

    elif intent == "change":
        prompt = (
            f"You are analyzing UAV aerial imagery for anomaly detection.\n"
            f"{img_info}\n\n"
            f"QUERY: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "description of anomalies found",\n'
            '  "targets": [\n'
            '    {"filename": "<name>", "bbox": [xmin, ymin, xmax, ymax], '
            '"confidence": 0.0, "reason": "why flagged"}\n'
            '  ]\n'
            "}\n\n"
            + _BBOX_RULES +
            "No markdown, no text outside the JSON object."
        )
        return prompt, True

    raise ValueError(f"Unhandled intent: {intent}")


# ---------------------------------------------------------------------------
# TacticalAnalyst
# ---------------------------------------------------------------------------

class TacticalAnalyst:
    def __init__(self, provider="openrouter", model_name=None, api_key=None):
        self.provider = provider.lower()
        self.api_key = api_key
        logger.info(f"Analyst initialized. Provider: {self.provider}")

        if self.provider == "openrouter":
            self.model_name = model_name or "qwen/qwen-2.5-vl-72b-instruct"
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/arie/TAE",
                    "X-Title": "TAE"
                }
            )
        else:
            self.model_name = model_name or "moondream"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_multiple_views(self, image_paths: list[str], user_query: str) -> dict:
        """
        Analyzes each tile independently and merges results.
        Each image is sent in a separate VLM call — no batching.
        """
        intent = classify_query(user_query)
        logger.info(
            f"Intent: {intent} | Query: {user_query} | "
            f"Tiles: {[Path(p).name for p in image_paths]}"
        )

        merged_targets = []
        merged_reports = []

        for path in image_paths:
            filename = Path(path).name

            # Read actual pixel dimensions for prompt and normalization
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Cannot read tile for analysis: {path}")
                continue
            img_h, img_w = img.shape[:2]

            prompt, expects_bboxes = _build_prompt(user_query, filename, img_w, img_h)

            try:
                result = self._call_vlm_with_validation(
                    path, prompt, expects_bboxes, img_w, img_h
                )
                merged_reports.append(f"{filename}: {result.get('report', '')}")
                merged_targets.extend(result.get('targets', []))
            except Exception as e:
                logger.error(f"VLM failed for {filename} after retries: {e}")
                merged_reports.append(f"{filename}: ERROR - {e}")

        return {
            "report":  " | ".join(merged_reports),
            "targets": merged_targets,
        }

    # ------------------------------------------------------------------
    # Validation and normalization
    # ------------------------------------------------------------------

    def _normalize_targets(self, targets: list, img_w: int, img_h: int) -> list:
        """
        Detects whether the VLM returned normalized (0-1000) or pixel coordinates
        and converts everything to tile pixel coordinates.

        Since we now pass actual image dimensions in the prompt, the VLM should
        return pixel coords — but this guards against models that ignore the prompt.
        """
        normalized = []
        for t in targets:
            bbox = t.get('bbox', [])
            if len(bbox) != 4:
                continue
            xmin, ymin, xmax, ymax = [float(v) for v in bbox]

            # Heuristic: if all values <= 1000 but image is larger, treat as 0-1000 normalized
            if max(xmin, ymin, xmax, ymax) <= 1000 and max(img_w, img_h) > 1000:
                xmin = round(xmin * img_w / 1000)
                ymin = round(ymin * img_h / 1000)
                xmax = round(xmax * img_w / 1000)
                ymax = round(ymax * img_h / 1000)
                logger.info(
                    f"Converted 0-1000 → pixel bbox: "
                    f"[{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}]"
                )

            normalized.append({**t, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]})
        return normalized

    def _validate_targets(self, targets: list, img_w: int, img_h: int) -> bool:
        """
        Validates bboxes in pixel coordinate space.
        Returns True if all targets are geometrically plausible.
        An empty list is valid (object not present in tile).
        """
        for t in targets:
            bbox = t.get('bbox', [])
            if len(bbox) != 4:
                logger.warning(f"Bad bbox length: {bbox}")
                return False
            xmin, ymin, xmax, ymax = bbox
            if not (xmax > xmin and ymax > ymin):
                logger.warning(f"Inverted bbox: {bbox}")
                return False
            if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                logger.warning(f"Bbox out of image bounds {img_w}×{img_h}: {bbox}")
                return False
            # Reject full-tile bbox — almost certainly a hallucination
            if (xmax - xmin) > 0.95 * img_w and (ymax - ymin) > 0.95 * img_h:
                logger.warning(f"Full-tile bbox (likely hallucination): {bbox}")
                return False
        return True

    def _filter_low_confidence_targets(
        self, targets: list, threshold: float = 0.4
    ) -> list:
        """
        Drops targets where the VLM expressed low confidence.
        Models that ignore the confidence field default to 1.0 (kept).
        """
        filtered = []
        for t in targets:
            conf = float(t.get('confidence', 1.0))
            if conf >= threshold:
                filtered.append(t)
            else:
                logger.info(
                    f"Dropped low-confidence target in {t.get('filename')} "
                    f"(conf={conf:.2f}, bbox={t.get('bbox')})"
                )
        return filtered

    # ------------------------------------------------------------------
    # VLM call with retry
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(ValueError)
    )
    def _call_vlm_with_validation(
        self,
        image_path: str,
        prompt: str,
        expects_bboxes: bool,
        img_w: int,
        img_h: int,
    ) -> dict:
        if self.provider == "openrouter":
            res_text = self._analyze_openrouter(image_path, prompt)
        else:
            res_text = self._analyze_ollama(image_path, prompt)

        logger.debug(f"Raw VLM response: {res_text}")
        clean = re.sub(r'^```json\s*|\s*```$', '', res_text.strip(), flags=re.MULTILINE)
        result = json.loads(clean)

        if expects_bboxes and result.get('targets'):
            result['targets'] = self._normalize_targets(result['targets'], img_w, img_h)
            result['targets'] = self._filter_low_confidence_targets(result['targets'])

            if not self._validate_targets(result['targets'], img_w, img_h):
                raise ValueError(f"Invalid bboxes after normalization: {result['targets']}")

        return result

    # ------------------------------------------------------------------
    # Provider backends — single image per call
    # ------------------------------------------------------------------

    def _analyze_openrouter(self, image_path: str, prompt: str) -> str:
        img_b64 = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
        size_kb = len(img_b64) * 3 / 4 / 1024  # approximate decoded size
        logger.info(f"Sending tile to VLM: {Path(image_path).name} (~{size_kb:.0f} KB)")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _analyze_ollama(self, image_path: str, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            format='json',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        return response['message']['content']
