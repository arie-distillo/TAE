"""
ai/intent.py — TAE query intent classification
===============================================
Self-contained module: Pydantic models, LLM agent, keyword fallback.
Nothing here imports from analyst.py or main.py.

Usage
-----
    from ai.intent import IntentClassifier

    clf        = IntentClassifier(api_key="sk-or-...", model_name="meta-llama/...")
    classified = clf.classify("Find a white car")

    match classified.params.intent:
        case "object_search":   ...
        case "anomaly_detection": ...
        case "moving_object":   ...
"""

import logging
from typing import Literal, Union, Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

logger = logging.getLogger("IntentClassifier")


# ---------------------------------------------------------------------------
# Intent parameter models  (discriminated union on the 'intent' literal)
# ---------------------------------------------------------------------------

class ObjectSearchParams(BaseModel):
    """
    Operator is looking for a specific object, vehicle, person, or structure.

    Downstream flow
    ───────────────
    CLIP vector search  →  TacticalAnalyst.analyze_multiple_views()  →  bbox markers
    """
    intent: Literal["object_search"] = "object_search"
    target_description: str = Field(
        description=(
            "Verbatim description of what to find, preserving the operator's "
            "own wording. Examples: 'white car', 'helipad marked H', 'blue tent'."
        )
    )


class AnomalyDetectionParams(BaseModel):
    """
    Operator wants to find things that do not belong in an expected scene —
    foreign objects, colour / texture deviations, unusual formations.

    Downstream flow (not yet implemented)
    ──────────────────────────────────────
    Scene-embedding baseline  →  CLIP outlier filter  →  scene-aware VLM prompt
    """
    intent: Literal["anomaly_detection"] = "anomaly_detection"
    scene_context: str = Field(
        description=(
            "The expected / normal scene — what should be there. "
            "Examples: 'open ocean', 'wheat crop field', 'empty tarmac', "
            "'forest canopy'."
        )
    )
    anomaly_hint: str = Field(
        description=(
            "What makes something anomalous in this scene — colour, texture, "
            "shape, or foreign category. "
            "Examples: 'non-sea objects', 'discoloured crop patches', "
            "'vehicles on footpath'."
        )
    )


class MovingObjectParams(BaseModel):
    """
    Operator wants to detect or track objects in motion across consecutive
    frames or video.

    Downstream flow (not yet implemented)
    ──────────────────────────────────────
    Adjacent-frame pairs  →  optical flow / frame diff  →  motion-region VLM
    """
    intent: Literal["moving_object"] = "moving_object"
    motion_hint: str = Field(
        default="",
        description=(
            "Type of moving object if the operator specified one, else empty. "
            "Examples: 'vehicle', 'person', 'boat'."
        )
    )


# Public type alias for use in main.py routing
IntentParams = Annotated[
    Union[ObjectSearchParams, AnomalyDetectionParams, MovingObjectParams],
    Field(discriminator="intent"),
]


class ClassifiedQuery(BaseModel):
    """Structured output of the intent classification agent."""
    params:     IntentParams
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning:  str   = Field(
        description="One-sentence explanation of the classification decision."
    )


# ---------------------------------------------------------------------------
# LLM agent — system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a tactical UAV query intent classifier for an aerial intelligence system
called TAE (Tactical Awareness Engine). Operators query aerial drone imagery in
natural language. Classify each query into EXACTLY ONE intent.

INTENT DEFINITIONS
==================

object_search
  The operator wants to find, locate, or count a specific object, vehicle,
  person, or structure visible in the imagery.
  Cues: specific nouns ("car", "helipad", "tent"), "find", "locate", "show me",
        "where is", "count", "how many".
  Examples:
    - "Find a white car"
    - "Where is the helipad marked H?"
    - "Count the trucks in the parking area"
    - "Is there a boat near the pier?"

anomaly_detection
  The operator describes a NORMAL scene and asks what is OUT OF PLACE —
  foreign objects, colour / texture deviations, or unusual formations.
  Cues: scene context ("ocean", "crop field", "runway") + "unusual",
        "doesn't belong", "out of place", "discoloured", "anomaly", "foreign",
        "shouldn't be there".
  Examples:
    - "Find any object that is not sea in this ocean view"
    - "Detect discoloured patches in the crop field"
    - "Flag anything unusual on the runway"
    - "Find vehicles where there should only be pedestrians"

moving_object
  The operator wants to detect or track objects in motion across consecutive
  frames or video.
  Cues: "moving", "motion", "track", "what changed", "changed position",
        "in motion", "travelling".
  Examples:
    - "What is moving in the scene?"
    - "Track the moving vehicles"
    - "Find any object that changed position"
    - "Detect motion near the building"

DISAMBIGUATION RULES
====================
- Query names a SPECIFIC object type            → object_search
- Query describes a scene + asks for outliers   → anomaly_detection
- Motion / temporal change is primary concern   → moving_object
- Genuinely ambiguous                           → object_search

Always extract scene_context and anomaly_hint carefully for anomaly_detection —
these are passed directly to the downstream VLM prompt.
"""


# ---------------------------------------------------------------------------
# Keyword fallback  (used when LLM agent is unavailable or throws)
# ---------------------------------------------------------------------------

_KW_MOTION  = frozenset({
    "moving", "motion", "track", "tracking", "what changed",
    "changed position", "in motion", "travelling", "traveling",
})
_KW_ANOMALY = frozenset({
    "unusual", "anomaly", "anomalies", "out of place", "doesn't belong",
    "foreign", "discoloured", "discolored", "shouldn't be there",
    "not supposed to",
})


def _classify_keywords(user_query: str) -> ClassifiedQuery:
    """
    Fast keyword-based fallback.
    Returns a ClassifiedQuery with lower confidence than the LLM path.
    """
    q = user_query.lower()

    if any(kw in q for kw in _KW_MOTION):
        return ClassifiedQuery(
            params=MovingObjectParams(motion_hint=""),
            confidence=0.55,
            reasoning="Keyword match: motion / temporal-change terms detected.",
        )
    if any(kw in q for kw in _KW_ANOMALY):
        return ClassifiedQuery(
            params=AnomalyDetectionParams(
                scene_context="unknown",
                anomaly_hint=user_query,
            ),
            confidence=0.55,
            reasoning="Keyword match: anomaly / out-of-place terms detected.",
        )
    return ClassifiedQuery(
        params=ObjectSearchParams(target_description=user_query),
        confidence=0.60,
        reasoning="Default fallback: no motion or anomaly keywords found.",
    )


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Wraps a Pydantic AI agent for intent classification.
    Falls back to keyword matching if the agent cannot be initialised
    or if any individual call raises.

    Parameters
    ----------
    api_key:
        OpenRouter API key.
    model_name:
        Any OpenRouter text-model slug.  Defaults to a cheap, fast model —
        intent classification is structured output, not vision.
    """

    def __init__(
        self,
        api_key:    str        = "",
        model_name: str | None = None,
    ) -> None:
        self._agent: Agent | None = None
        _slug = model_name or "meta-llama/llama-3.1-8b-instruct"

        if api_key:
            try:
                _model = OpenAIModel(
                    model_name=_slug,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                self._agent = Agent(
                    model=_model,
                    result_type=ClassifiedQuery,
                    system_prompt=_SYSTEM_PROMPT,
                )
                logger.info(f"Intent agent ready: {_slug} via OpenRouter")
            except Exception as exc:
                logger.warning(
                    f"Could not initialise intent agent ({exc}); "
                    "keyword fallback will be used."
                )
        else:
            logger.info("No API key — intent classifier using keyword fallback only.")

    def classify(self, user_query: str) -> ClassifiedQuery:
        """
        Classify a natural-language operator query.

        Returns a ClassifiedQuery whose `params.intent` is always one of:
            'object_search' | 'anomaly_detection' | 'moving_object'

        Never raises — falls back to keyword matching on any LLM error.
        """
        if self._agent is None:
            result = _classify_keywords(user_query)
            logger.info(
                f"Intent (keyword): {result.params.intent} "
                f"conf={result.confidence:.2f} | {result.reasoning}"
            )
            return result

        try:
            run    = self._agent.run_sync(user_query)
            result = run.data
            logger.info(
                f"Intent (LLM): {result.params.intent} "
                f"conf={result.confidence:.2f} | {result.reasoning}"
            )
            return result

        except Exception as exc:
            logger.warning(
                f"Intent agent failed ({exc}); falling back to keyword matching."
            )
            result = _classify_keywords(user_query)
            logger.info(
                f"Intent (keyword fallback): {result.params.intent} "
                f"conf={result.confidence:.2f} | {result.reasoning}"
            )
            return result
