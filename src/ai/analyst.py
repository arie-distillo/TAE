import re
import json
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger("TacticalAnalyst")


# ---------------------------------------------------------------------------
# Tile reconstruction
# ---------------------------------------------------------------------------

def _load_tile_cv2(candidate: dict) -> np.ndarray | None:
    """
    Reconstructs a tile by cropping its parent frame.
    Tiles are never stored on disk — this is the single reconstruction point
    used by the analyst whenever it needs the actual pixel data.
    """
    img = cv2.imread(candidate['parent_path'])
    if img is None:
        logger.warning(f"Cannot read parent frame: {candidate['parent_path']}")
        return None
    x, y = candidate['tile_x'], candidate['tile_y']
    w, h = candidate['tile_w'], candidate['tile_h']
    return img[y:y + h, x:x + w]


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
    Image dimensions are passed explicitly so the VLM knows the pixel space.
    """
    intent = "object_search"   # TODO PATCH FOR NOW
    img_info = f"Image: {filename} ({img_w}x{img_h} pixels)"

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
            + _BBOX_RULES
            + "No markdown, no text outside the JSON object."
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
            + _BBOX_RULES
            + "No markdown, no text outside the JSON object."
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
            + _BBOX_RULES
            + "No markdown, no text outside the JSON object."
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

    def analyze_multiple_views(self, candidates: list[dict], user_query: str) -> dict:
        """
        Analyzes each candidate tile independently and merges results.

        Accepts candidate dicts (from LanceDB) rather than file paths because
        tiles are not stored on disk — each tile is reconstructed on demand
        from its parent frame using the stored pixel offsets.

        Only tiles with confirmed detections appear in the final report.
        Tiles where the VLM found nothing are counted but not reported.
        """
        logger.info(
            f"Query: {user_query} | "
            f"Tiles: {[Path(c['image_path']).name for c in candidates]}"
        )

        merged_targets = []
        hit_reports    = []
        miss_count     = 0

        for cand in candidates:
            filename = Path(cand['image_path']).name

            # Reconstruct tile from parent frame — no file I/O for tiles
            tile_img = _load_tile_cv2(cand)
            if tile_img is None:
                logger.warning(f"Skipping {filename} — could not load parent frame")
                continue

            img_h, img_w = tile_img.shape[:2]
            prompt, expects_bboxes = _build_prompt(user_query, filename, img_w, img_h)

            try:
                result = self._call_vlm_with_validation(
                    tile_img, filename, prompt, expects_bboxes, img_w, img_h
                )
                targets = result.get('targets', [])
                if targets:
                    hit_reports.append(f"{filename}: {result.get('report', '')}")
                    merged_targets.extend(targets)
                else:
                    miss_count += 1
                    logger.info(f"No detection in {filename} (VLM confirmed absent)")

            except Exception as e:
                logger.error(f"VLM failed for {filename} after retries: {e}")

        summary = (
            f"{len(hit_reports)} tiles with detections, "
            f"{miss_count} tiles confirmed empty"
        )
        report = (
            " | ".join(hit_reports)
            if hit_reports
            else f"No detections found. {summary}"
        )

        return {
            "report":  report,
            "targets": merged_targets,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Validation and normalization
    # ------------------------------------------------------------------

    def _normalize_targets(self, targets: list, img_w: int, img_h: int) -> list:
        """
        Safety net: converts 0-1000 normalized coords to pixels if the VLM
        ignored the explicit pixel dimensions in the prompt.
        """
        normalized = []
        for t in targets:
            bbox = t.get('bbox', [])
            if len(bbox) != 4:
                continue
            xmin, ymin, xmax, ymax = [float(v) for v in bbox]

            if max(xmin, ymin, xmax, ymax) <= 1000 and max(img_w, img_h) > 1000:
                xmin = round(xmin * img_w / 1000)
                ymin = round(ymin * img_h / 1000)
                xmax = round(xmax * img_w / 1000)
                ymax = round(ymax * img_h / 1000)
                logger.info(
                    f"Converted 0-1000 to pixel bbox: "
                    f"[{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)}]"
                )

            normalized.append({**t, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]})
        return normalized

    def _validate_targets(self, targets: list, img_w: int, img_h: int) -> bool:
        """
        Validates bboxes in pixel space.
        Empty list is valid — means object not present in this tile.
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
                logger.warning(f"Bbox out of bounds {img_w}x{img_h}: {bbox}")
                return False
            if (xmax - xmin) > 0.95 * img_w and (ymax - ymin) > 0.95 * img_h:
                logger.warning(f"Full-tile bbox (likely hallucination): {bbox}")
                return False
        return True

    def _filter_low_confidence_targets(
        self, targets: list, threshold: float = 0.4
    ) -> list:
        """
        Drops targets below the confidence threshold.
        Models that omit the confidence field default to 1.0 (always kept).
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
    # Provider backends — encode numpy tile directly, no disk I/O
    # ------------------------------------------------------------------

    def _encode_tile(self, tile_img: np.ndarray, filename: str) -> str:
        """Encodes a numpy array as a base64 JPEG string."""
        _, buf = cv2.imencode('.jpg', tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        encoded = base64.b64encode(buf.tobytes()).decode('utf-8')
        size_kb = len(buf) / 1024
        logger.info(f"Encoding tile for VLM: {filename} (~{size_kb:.0f} KB)")
        return encoded

    def _analyze_openrouter(
        self, tile_img: np.ndarray, filename: str, prompt: str
    ) -> str:
        img_b64 = self._encode_tile(tile_img, filename)

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

    def _analyze_ollama(
        self, tile_img: np.ndarray, filename: str, prompt: str
    ) -> str:
        img_b64 = self._encode_tile(tile_img, filename)

        response = ollama.chat(
            model=self.model_name,
            format='json',
            messages=[{
                'role':    'user',
                'content': prompt,
                'images':  [img_b64]
            }]
        )
        return response['message']['content']