"""
ai/intent.py — TAE query intent classification + structured parameter extraction
================================================================================
Single LLM call produces a fully-structured query descriptor that every
downstream stage (YOLO, SAM, VLM) consumes directly — no stage re-interprets
free text.

Intent taxonomy (v2)
--------------------
  object_detection  — find ALL instances of a specified object class across
                      every tile; always followed by object tracking
  anomaly_detection — find what is out of place relative to a described normal scene

`object_search` and `moving_object` are retired.
Tracking is unconditional — not a separate intent.

Usage
-----
    from ai.intent import IntentClassifier, ObjectDetectionParams, AnomalyDetectionParams

    clf    = IntentClassifier(api_key="sk-or-...", model_name="...")
    result = clf.classify("find all animals in the field")

    match result.params.intent:
        case "object_detection":  handle_object_detection(result.params)
        case "anomaly_detection": handle_anomaly(result.params)
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

logger = logging.getLogger("IntentClassifier")


# ---------------------------------------------------------------------------
# Shape priors — altitude-normalised
# ---------------------------------------------------------------------------

class ShapePriors(BaseModel):
    """
    Expected visual appearance of the target from directly above (nadir view),
    expressed at a reference altitude of 100 m AGL.

    Area bounds are scaled by (actual_alt / 100)² at runtime.
    Fill ratio and aspect ratio are scale-invariant (dimensionless ratios).

    Typical values
    --------------
    Object          min_area  max_area  fill        aspect (h/w)
    Person          150       800       0.30–0.55   1.5–5.0
    Cow / horse     600       5000      0.40–0.70   0.4–3.0
    Small stone     30        300       0.65–0.92   0.6–1.4
    Car / truck     1500      10000     0.70–0.90   1.2–3.5
    Building        5000      200000   0.80–0.97   0.4–4.0
    Tree canopy     2000      40000    0.60–0.88   0.7–1.3
    """
    min_area_px:   int   = Field(description="Minimum expected footprint in px² at 100 m AGL")
    max_area_px:   int   = Field(description="Maximum expected footprint in px² at 100 m AGL")
    min_fill:      float = Field(description="Min mask-area / bbox-area ratio (compactness)")
    max_fill:      float = Field(description="Max mask-area / bbox-area ratio")
    min_aspect:    float = Field(description="Min height/width ratio")
    max_aspect:    float = Field(description="Max height/width ratio")
    expected_count: str  = Field(description="'one' | 'few' | 'many' | 'unknown'")


def scale_shape_priors(priors: ShapePriors, actual_alt_m: float) -> ShapePriors:
    """Scale area bounds for the actual flight altitude. Ratios are unchanged."""
    if actual_alt_m <= 0:
        return priors
    scale = (100.0 / actual_alt_m) ** 2
    return ShapePriors(
        min_area_px    = max(30,  int(priors.min_area_px * scale)),
        max_area_px    = max(200, int(priors.max_area_px * scale)),
        min_fill       = priors.min_fill,
        max_fill       = priors.max_fill,
        min_aspect     = priors.min_aspect,
        max_aspect     = priors.max_aspect,
        expected_count = priors.expected_count,
    )


# ---------------------------------------------------------------------------
# Difficulty → YOLO confidence mapping
# ---------------------------------------------------------------------------

YOLO_CONFIDENCE: dict[str, float] = {
    "easy":   0.15,   # large, visually distinctive objects (buildings, vehicles)
    "medium": 0.07,   # medium objects with moderate camouflage (cows in grass)
    "hard":   0.03,   # small or highly camouflaged (people, stones, prone animals)
}


# ---------------------------------------------------------------------------
# Intent parameter models
# ---------------------------------------------------------------------------

class ObjectDetectionParams(BaseModel):
    """
    Operator wants to find and track ALL instances of a specified object class.

    Downstream pipeline
    -------------------
    YOLO-World (all tiles, parallel)
      → Cross-tile NMS
        → SAM box-prompted refinement + shape filter
          → VLM verification on masked crop
            → Geo-location via tile footprint
              → Tracking across frames
    """
    intent: Literal["object_detection"] = "object_detection"

    # ── YOLO-World ────────────────────────────────────────────────────────────
    # Specific visually-grounded class names derived from the user query.
    # Limited to ≤8 terms to avoid diluting YOLO-World's text embeddings.
    # "find all animals" → ["cow", "horse", "sheep", "goat", "dog", "deer", "bird"]
    # "find red vehicle" → ["car", "truck", "van", "pickup"]
    # "find people"      → ["person", "human", "pedestrian"]
    yolo_classes: list[str] = Field(
        description=(
            "Specific, visually grounded YOLO-World class names derived from the query. "
            "Use concrete visual categories, not abstract collective nouns. "
            "Maximum 8 terms."
        )
    )

    # ── CLIP (semantic search, used for map queries and large-scale pre-filtering) ──
    clip_query: str = Field(
        description=(
            "Semantic search string for CLIP tile retrieval. "
            "May be broader or more descriptive than yolo_classes. "
            "Aerial context prefix will be added automatically."
        )
    )

    # ── VLM ───────────────────────────────────────────────────────────────────
    vlm_verification_criteria: str = Field(
        description=(
            "What the VLM should confirm to accept a detection. "
            "Example: 'confirm this is a living animal, not a shadow, rock, or vegetation pattern'."
        )
    )
    vlm_reporting_fields: list[str] = Field(
        description=(
            "Structured fields the VLM should populate for each confirmed detection. "
            "Examples: ['species', 'count', 'behaviour', 'health'], "
            "['vehicle_type', 'colour', 'orientation'], ['size_estimate', 'material']."
        )
    )

    # ── SAM / shape filtering ─────────────────────────────────────────────────
    shape_priors: ShapePriors

    # ── Confidence calibration ────────────────────────────────────────────────
    # "easy"   → YOLO confidence 0.15  (large, distinctive objects)
    # "medium" → YOLO confidence 0.07  (medium objects, some camouflage)
    # "hard"   → YOLO confidence 0.03  (small / highly camouflaged)
    expected_difficulty: Literal["easy", "medium", "hard"] = Field(
        description=(
            "Detection difficulty given object size, camouflage, and nadir view. "
            "Drives YOLO confidence threshold and tile coverage strategy."
        )
    )

    # ── Query modifiers ───────────────────────────────────────────────────────
    requires_counting: bool = Field(
        default=False,
        description="True when the user asked for a count ('how many', 'count all').",
    )
    spatial_relation: str | None = Field(
        default=None,
        description="Geographic constraint if specified, e.g. 'near the fence', 'inside the building'.",
    )
    colour_hint: str | None = Field(
        default=None,
        description="Colour qualifier if specified, e.g. 'red', 'white'. Used as VLM filter.",
    )
    size_qualifier: str | None = Field(
        default=None,
        description="Size qualifier if specified, e.g. 'large', 'small'. Modifies area thresholds.",
    )

    @property
    def yolo_confidence(self) -> float:
        return YOLO_CONFIDENCE[self.expected_difficulty]


class AnomalyDetectionParams(BaseModel):
    """
    Operator describes a NORMAL scene and asks what is out of place.

    Downstream pipeline (unchanged from v1)
    ----------------------------------------
    CLIP scene retrieval → SAM auto-segment → CLIP segment scoring → VLM verify
    """
    intent: Literal["anomaly_detection"] = "anomaly_detection"

    scene_context: str = Field(
        description=(
            "The expected/normal scene — what should be there. "
            "Examples: 'open grass field', 'wheat crop', 'empty tarmac', 'forest canopy'."
        )
    )
    anomaly_hint: str = Field(
        description=(
            "What makes something anomalous — colour, texture, shape, or foreign category. "
            "Examples: 'non-vegetation objects', 'discoloured patches', 'vehicles on footpath'."
        )
    )
    clip_query: str = Field(
        description="CLIP semantic search string for scene baseline retrieval."
    )
    vlm_verification_criteria: str = Field(
        description="What the VLM should confirm to accept an anomaly."
    )
    vlm_reporting_fields: list[str] = Field(
        default_factory=lambda: ["anomaly_type", "estimated_size", "severity"],
        description="Structured fields the VLM should return per confirmed anomaly.",
    )
    expected_difficulty: Literal["easy", "medium", "hard"] = "medium"


# Public type alias
IntentParams = Annotated[
    Union[ObjectDetectionParams, AnomalyDetectionParams],
    Field(discriminator="intent"),
]


class ClassifiedQuery(BaseModel):
    """Structured output of the intent classification agent."""
    params:     IntentParams
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning:  str


# ---------------------------------------------------------------------------
# Flat schema used inside the LLM agent (avoids discriminated-union issues)
# ---------------------------------------------------------------------------

class _FlatIntent(BaseModel):
    """
    Flat representation returned by the LLM agent.
    Reconstructed into ClassifiedQuery with proper discriminated params after.
    """
    # Common
    intent:             Literal["object_detection", "anomaly_detection"]
    confidence:         float = Field(ge=0.0, le=1.0)
    reasoning:          str

    # object_detection fields
    yolo_classes:               list[str] = Field(default_factory=list)
    clip_query:                 str       = ""
    vlm_verification_criteria:  str       = ""
    vlm_reporting_fields:       list[str] = Field(default_factory=list)
    expected_difficulty:        str       = "medium"  # "easy"|"medium"|"hard"
    requires_counting:          bool      = False
    spatial_relation:           str | None = None
    colour_hint:                str | None = None
    size_qualifier:             str | None = None

    # shape_priors fields (flattened)
    sp_min_area_px:    int   = 500
    sp_max_area_px:    int   = 50000
    sp_min_fill:       float = 0.30
    sp_max_fill:       float = 0.95
    sp_min_aspect:     float = 0.3
    sp_max_aspect:     float = 5.0
    sp_expected_count: str   = "unknown"

    # anomaly_detection fields
    scene_context: str = ""
    anomaly_hint:  str = ""


# ---------------------------------------------------------------------------
# LLM agent system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a tactical UAV query intent classifier and parameter extractor for TAE
(Tactical Awareness Engine). Operators query aerial drone imagery in natural language.

STEP 1 — Classify into EXACTLY ONE intent:

object_detection
  Find ALL instances of a specific object, animal, person, or structure.
  Cues: specific nouns, "find", "locate", "show me", "where is", "count", "all".
  Examples:
    "Find all animals" → object_detection
    "Where is the red truck?" → object_detection
    "Count the people near the fence" → object_detection
    "Find large rocks" → object_detection

anomaly_detection
  The operator describes a NORMAL scene and asks what is OUT OF PLACE.
  Cues: scene description + "unusual", "doesn't belong", "anomaly", "foreign",
        "out of place", "discoloured", "unexpected".
  Examples:
    "Find anything unusual on this crop field" → anomaly_detection
    "What doesn't belong in this forest?" → anomaly_detection
    "Detect anomalies on the runway" → anomaly_detection

STEP 2 — For object_detection, extract ALL fields:

yolo_classes
  Specific, visually-grounded class names for YOLO-World.
  NEVER use collective nouns like "animals" or "vehicles".
  Expand them to specific species/types the model can detect.
  "animals" → ["cow", "horse", "sheep", "goat", "dog", "deer", "bird"]
  "vehicles" → ["car", "truck", "van", "bus", "motorcycle"]
  "people" → ["person"]
  "livestock" → ["cow", "horse", "sheep", "goat", "pig"]
  Maximum 8 specific terms.

clip_query
  Broader semantic description for CLIP retrieval. Can be more descriptive.

vlm_verification_criteria
  One sentence telling the VLM what to CONFIRM about the detection.
  Example: "Confirm this is a living quadruped animal, not a shadow, rock, or tree stump."

vlm_reporting_fields
  List of structured fields the VLM should report per detection.
  For animals: ["species", "count", "behaviour", "approximate_size"]
  For vehicles: ["vehicle_type", "colour", "orientation", "moving_or_parked"]
  For people: ["count", "activity", "approximate_position"]
  For structures: ["structure_type", "dimensions_estimate", "condition"]

expected_difficulty
  "easy"   — large (>1m²), high-contrast, visually distinctive objects (buildings, large vehicles)
  "medium" — medium-sized objects with some natural camouflage (cows in grass, parked cars)
  "hard"   — small (<0.5m²) or highly camouflaged objects (people, stones, prone animals)

Shape priors (at 100 m AGL reference altitude):
  sp_min_area_px / sp_max_area_px — footprint in pixels
    Person:   150–800
    Animal (cow/horse): 600–5000
    Stone:    30–500
    Car/truck: 1500–10000
    Building: 5000–200000
  sp_min_fill / sp_max_fill — mask area / bbox area (compactness)
    Compact (stone, building): 0.65–0.95
    Animal: 0.40–0.75
    Person: 0.30–0.60
  sp_min_aspect / sp_max_aspect — height/width ratio
    Elongated (person walking): 1.5–5.0
    Roughly square (stone, crouching animal): 0.5–2.0
    Wide (lying animal, vehicle): 0.3–1.5

STEP 3 — For anomaly_detection, extract:
  scene_context — expected/normal scene description
  anomaly_hint  — what makes something anomalous
  clip_query    — for CLIP scene retrieval

ALWAYS populate clip_query and vlm_verification_criteria regardless of intent.

OUTPUT: Return the flat JSON schema exactly, no markdown, no extra keys.
"""


# ---------------------------------------------------------------------------
# Keyword fallback classifier
# ---------------------------------------------------------------------------

def _classify_keywords(user_query: str) -> ClassifiedQuery:
    q = user_query.lower()

    _KW_ANOMALY = frozenset({
        "unusual", "anomaly", "anomalies", "out of place", "doesn't belong",
        "foreign", "discoloured", "discolored", "shouldn't be there",
        "not supposed to", "unexpected",
    })

    if any(k in q for k in _KW_ANOMALY):
        params = AnomalyDetectionParams(
            scene_context             = "unknown scene",
            anomaly_hint              = user_query,
            clip_query                = user_query,
            vlm_verification_criteria = f"Confirm this is genuinely anomalous: {user_query}",
        )
        return ClassifiedQuery(
            params     = params,
            confidence = 0.55,
            reasoning  = "Anomaly keywords detected in query.",
        )

    # Default: object_detection
    params = ObjectDetectionParams(
        yolo_classes              = [user_query],
        clip_query                = user_query,
        vlm_verification_criteria = f"Confirm the target matches: {user_query}",
        vlm_reporting_fields      = ["description", "count", "location_in_frame"],
        expected_difficulty       = "medium",
        shape_priors              = ShapePriors(
            min_area_px    = 200,
            max_area_px    = 50000,
            min_fill       = 0.25,
            max_fill       = 0.97,
            min_aspect     = 0.2,
            max_aspect     = 6.0,
            expected_count = "unknown",
        ),
    )
    return ClassifiedQuery(
        params     = params,
        confidence = 0.5,
        reasoning  = "No anomaly keywords found; defaulting to object_detection.",
    )


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Wraps a pydantic-ai Agent that returns _FlatIntent.
    Falls back to keyword matching on any LLM error.
    """

    def __init__(self, api_key: str | None, model_name: str | None = None) -> None:
        self._agent: Agent | None = None

        if not api_key:
            logger.warning("IntentClassifier: no API key — keyword fallback only.")
            return

        try:
            _model = OpenAIChatModel(
                model_name or "anthropic/claude-haiku-4-5",
                provider=OpenAIProvider(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                ),
            )
            self._agent = Agent(_model, output_type=_FlatIntent, system_prompt=_SYSTEM_PROMPT)
            logger.info(f"IntentClassifier ready | model={model_name or 'default'}")
        except Exception as exc:
            logger.warning(f"IntentClassifier init failed ({exc}); keyword fallback only.")

    def classify(self, user_query: str) -> ClassifiedQuery:
        """
        Returns a ClassifiedQuery. Never raises.
        Falls back to keyword matching on any LLM error.
        """
        if self._agent is None:
            return self._keyword_fallback(user_query)

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            async def _run():
                r = await self._agent.run(user_query)
                return r.output

            with ThreadPoolExecutor(max_workers=1) as pool:
                flat: _FlatIntent = pool.submit(asyncio.run, _run()).result(timeout=30)

            result = self._reconstruct(flat)
            logger.info(
                f"Intent (LLM): {result.params.intent} "
                f"conf={result.confidence:.2f} | {result.reasoning}"
            )
            return result

        except Exception as exc:
            logger.warning(f"Intent agent failed ({exc}); keyword fallback.")
            return self._keyword_fallback(user_query)

    def _reconstruct(self, flat: _FlatIntent) -> ClassifiedQuery:
        """Reconstruct a ClassifiedQuery from the flat LLM output."""
        if flat.intent == "object_detection":
            diff = flat.expected_difficulty
            if diff not in ("easy", "medium", "hard"):
                diff = "medium"

            params = ObjectDetectionParams(
                yolo_classes              = flat.yolo_classes or [flat.clip_query],
                clip_query                = flat.clip_query,
                vlm_verification_criteria = flat.vlm_verification_criteria,
                vlm_reporting_fields      = flat.vlm_reporting_fields or ["description"],
                expected_difficulty       = diff,  # type: ignore[arg-type]
                requires_counting         = flat.requires_counting,
                spatial_relation          = flat.spatial_relation,
                colour_hint               = flat.colour_hint,
                size_qualifier            = flat.size_qualifier,
                shape_priors              = ShapePriors(
                    min_area_px    = flat.sp_min_area_px,
                    max_area_px    = flat.sp_max_area_px,
                    min_fill       = flat.sp_min_fill,
                    max_fill       = flat.sp_max_fill,
                    min_aspect     = flat.sp_min_aspect,
                    max_aspect     = flat.sp_max_aspect,
                    expected_count = flat.sp_expected_count,
                ),
            )
        else:
            params = AnomalyDetectionParams(
                scene_context             = flat.scene_context or "unknown",
                anomaly_hint              = flat.anomaly_hint or "",
                clip_query                = flat.clip_query,
                vlm_verification_criteria = flat.vlm_verification_criteria,
                expected_difficulty       = flat.expected_difficulty  # type: ignore[arg-type]
                                           if flat.expected_difficulty in ("easy","medium","hard")
                                           else "medium",
            )

        return ClassifiedQuery(
            params     = params,
            confidence = flat.confidence,
            reasoning  = flat.reasoning,
        )

    def _keyword_fallback(self, user_query: str) -> ClassifiedQuery:
        result = _classify_keywords(user_query)
        logger.info(
            f"Intent (keyword): {result.params.intent} "
            f"conf={result.confidence:.2f} | {result.reasoning}"
        )
        return result
