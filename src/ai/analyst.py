import re
import json
import base64
import logging
from pathlib import Path

import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger("TacticalAnalyst")


# ---------------------------------------------------------------------------
# Query intent classification
# ---------------------------------------------------------------------------

QUERY_INTENT_KEYWORDS = {
    "count":       ["how many", "count", "number of", "quantity"],
    "area_description": ["what is", "describe", "what's in", "what do you see", "overview"],
    "change":      ["unusual", "anomaly", "anomalies", "out of place", "changed", "different"],
    "object_search": [],  # default — catches everything else
}

def classify_query(user_query: str) -> str:
    """
    Returns the intent label for a user query.
    Checked in priority order; falls back to 'object_search'.
    """
    q = user_query.lower()
    for intent, keywords in QUERY_INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "object_search"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _build_prompt(user_query: str, filenames_str: str) -> tuple[str, bool]:
    """
    Selects and fills the correct prompt template based on query intent.

    Returns
    -------
    prompt : str
        The full prompt to send to the VLM.
    expects_bboxes : bool
        Whether the response schema includes 'targets' with bboxes.
        Used by the caller to decide whether to run _validate_targets.
    """
    intent = classify_query(user_query)

    # Shared coordinate rules — appended to any prompt that expects bboxes
    BBOX_RULES = (
        "Coordinate rules:\n"
        "- All values are integers in range 0–1000\n"
        "- 0,0 is the TOP-LEFT corner of the image\n"
        "- 1000,1000 is the BOTTOM-RIGHT corner\n"
        "- xmin < xmax and ymin < ymax always\n"
        "- Draw the tightest possible box around the object\n"
        "- If the object is not found in an image, omit that image from targets\n"
    )

    if intent == "object_search":
        prompt = (
            f"You are analyzing UAV aerial imagery. "
            f"Find all instances of the following in these images ({filenames_str}):\n"
            f"TARGET: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "brief summary of findings",\n'
            '  "targets": [\n'
            '    {"filename": "name.jpg", "bbox": [xmin, ymin, xmax, ymax]}\n'
            '  ]\n'
            "}\n\n"
            + BBOX_RULES +
            "No markdown, no explanation outside the JSON object."
        )
        return prompt, True

    elif intent == "count":
        prompt = (
            f"You are analyzing UAV aerial imagery. "
            f"Count and locate the following in these images ({filenames_str}):\n"
            f"QUERY: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "total count and summary",\n'
            '  "count": <integer>,\n'
            '  "targets": [\n'
            '    {"filename": "name.jpg", "bbox": [xmin, ymin, xmax, ymax]}\n'
            '  ]\n'
            "}\n\n"
            + BBOX_RULES +
            "No markdown, no explanation outside the JSON object."
        )
        return prompt, True

    elif intent == "area_description":
        prompt = (
            f"You are analyzing UAV aerial imagery. "
            f"Answer the following about these images ({filenames_str}):\n"
            f"QUERY: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "detailed answer to the query based on what you observe",\n'
            '  "targets": []\n'
            "}\n\n"
            "No markdown, no explanation outside the JSON object."
        )
        return prompt, False  # no bboxes expected

    elif intent == "change":
        prompt = (
            f"You are analyzing UAV aerial imagery for tactical anomaly detection. "
            f"Examine these images ({filenames_str}) and answer:\n"
            f"QUERY: {user_query}\n\n"
            "Return ONLY a valid JSON object:\n"
            "{\n"
            '  "report": "description of anomalies or unusual elements found",\n'
            '  "targets": [\n'
            '    {"filename": "name.jpg", "bbox": [xmin, ymin, xmax, ymax], '
            '"reason": "why this is flagged"}\n'
            '  ]\n'
            "}\n\n"
            + BBOX_RULES +
            "No markdown, no explanation outside the JSON object."
        )
        return prompt, True

    # Should never reach here given the fallback in classify_query
    raise ValueError(f"Unhandled intent: {intent}")


# ---------------------------------------------------------------------------
# Analyst
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

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def analyze_multiple_views(self, image_paths: list[str], user_query: str) -> dict:
        """
        Analyzes each candidate image independently and merges results.
        """
        filenames_str = ", ".join(Path(p).name for p in image_paths)
        intent = classify_query(user_query)
        logger.info(f"Intent: {intent} | Query: {user_query} | Images: {filenames_str}")

        merged_targets = []
        merged_reports = []

        for path in image_paths:
            filename = Path(path).name
            prompt, expects_bboxes = _build_prompt(user_query, filename)  # single filename

            try:
                result = self._call_vlm_with_validation([path], prompt, expects_bboxes)
                merged_reports.append(f"{filename}: {result.get('report', '')}")
                merged_targets.extend(result.get('targets', []))
            except Exception as e:
                logger.error(f"VLM failed for {filename} after retries: {e}")
                merged_reports.append(f"{filename}: ERROR - {e}")

        return {
            "report":  " | ".join(merged_reports),
            "targets": merged_targets
        }

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_targets(self, targets: list) -> bool:
        """
        Validates the targets list returned by the VLM.
        Returns True only if all bboxes are well-formed.
        An empty targets list is considered valid (object not found).
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
            if max(bbox) > 1000 or min(bbox) < 0:
                logger.warning(f"Out-of-range bbox: {bbox}")
                return False
            if (xmax - xmin) > 950 or (ymax - ymin) > 950:
                logger.warning(f"Full-image bbox (likely hallucination): {bbox}")
                return False
        return True

    # -----------------------------------------------------------------------
    # VLM call with retry
    # -----------------------------------------------------------------------
    def _normalize_targets(self, targets: list, img_w: int, img_h: int) -> list:
        """
        If VLM returned pixel coordinates instead of 0-1000 normalized,
        detect and convert. Leaves already-normalized coords untouched.
        """
        normalized = []
        for t in targets:
            bbox = t.get('bbox', [])
            if len(bbox) != 4:
                continue
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]

            if max(xmin, ymin, xmax, ymax) > 1000:
                # pixel coords — normalize to 0-1000
                xmin = round(xmin * 1000 / img_w)
                ymin = round(ymin * 1000 / img_h)
                xmax = round(xmax * 1000 / img_w)
                ymax = round(ymax * 1000 / img_h)
                logger.info(
                    f"Normalized pixel bbox {bbox} → [{xmin},{ymin},{xmax},{ymax}] "
                    f"(image: {img_w}×{img_h})"
                )

            normalized.append({**t, 'bbox': [xmin, ymin, xmax, ymax]})
        return normalized

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(ValueError)
    )
    def _call_vlm_with_validation(
        self, image_paths: list[str], prompt: str, expects_bboxes: bool
    ) -> dict:
        import cv2

        if self.provider == "openrouter":
            res_text = self._analyze_openrouter(image_paths, prompt)
        else:
            res_text = self._analyze_ollama(image_paths, prompt)

        logger.debug(f"Raw VLM response: {res_text}")
        clean = re.sub(r'^```json\s*|\s*```$', '', res_text.strip(), flags=re.MULTILINE)
        result = json.loads(clean)

        if expects_bboxes and result.get('targets'):
            # Read actual image dimensions for normalization
            # image_paths is always a single-element list at this point
            img = cv2.imread(image_paths[0])
            if img is not None:
                img_h, img_w = img.shape[:2]
                result['targets'] = self._normalize_targets(result['targets'], img_w, img_h)

            if not self._validate_targets(result['targets']):
                raise ValueError(f"Invalid bboxes after normalization: {result['targets']}")

        return result

    # -----------------------------------------------------------------------
    # Provider backends
    # -----------------------------------------------------------------------

    def _analyze_openrouter(self, image_paths: list[str], prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            img_b64 = base64.b64encode(open(path, "rb").read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _analyze_ollama(self, image_paths: list[str], prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            format='json',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': image_paths
            }]
        )
        return response['message']['content']