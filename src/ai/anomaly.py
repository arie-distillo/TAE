"""
ai/anomaly.py — Anomaly detection pipeline components
=======================================================
Three responsibilities:

1. VocabularyBuilder  — expands scene_context + anomaly_hint into lists of
                        CLIP-searchable terms using a cheap LLM call.

2. score_segments()   — scores stored SAM2 segment CLIP vectors against the
                        vocabulary using cosine similarity.  Pure numpy,
                        microseconds for hundreds of segments.

3. ScoredSegment      — dataclass carrying a flagged segment and its scores,
                        ready to be converted into a VLM candidate.

Intended call sequence (in main.py _handle_anomaly_query):
    vocab   = VocabularyBuilder(client, model).build(scene_context, anomaly_hint)
    e_vecs  = [lib.encode_text(t) for t in vocab.expected_terms]
    a_vecs  = [lib.encode_text(t) for t in vocab.anomaly_terms]
    scored  = score_segments(segments_by_frame, e_vecs, a_vecs)
    top_k   = scored[:TOP_K_FOR_VLM]
    → hand top_k to _segment_to_candidate() + VLM confirmation in main.py
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("TAE.Anomaly")

# ── Tuning constants ──────────────────────────────────────────────────────────
ANOMALY_SCORE_MARGIN: float = 0.03   # anomaly must outscore expected by this much
TOP_K_FOR_VLM:        int   = 5      # top-N candidates sent to VLM for confirmation


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Vocabulary:
    """CLIP text terms describing the expected scene and the target anomaly."""
    expected_terms: list[str]   # what SHOULD be in this scene
    anomaly_terms:  list[str]   # what would be anomalous


@dataclass
class ScoredSegment:
    """
    A SAM2 segment flagged as potentially anomalous by CLIP scoring.
    All coordinates are in full-frame pixels.
    """
    frame_path:     str
    bbox:           list[int]     # [xmin, ymin, xmax, ymax]
    area:           int
    clip_vector:    np.ndarray
    anomaly_score:  float         # score_anomaly - score_expected  (higher = more anomalous)
    score_anomaly:  float
    score_expected: float


# ── Vocabulary builder ────────────────────────────────────────────────────────

class VocabularyBuilder:
    """
    Uses an LLM (via OpenRouter) to expand operator-supplied intent params
    into CLIP-searchable term lists.

    Keeps a simple in-process cache keyed on (scene_context, anomaly_hint) so
    repeated queries with the same params don't incur extra LLM calls.
    """

    _SYSTEM_PROMPT = """\
You are a computer vision vocabulary builder for aerial UAV drone imagery
analyzed by a CLIP model.  CLIP was trained on internet photos — you MUST
compensate for its ground-level bias by using aerial/top-down descriptive
phrases.

Given a scene description and anomaly hint, produce two lists of phrases
that describe what things look like FROM ABOVE at drone altitude.

expected_terms : what the normal scene looks like top-down from a drone.
anomaly_terms  : what the anomaly looks like top-down from a drone.

Rules:
- Always add "aerial view", "top-down view", "drone footage", or "seen from above"
  to each term — CLIP needs this perspective cue to match correctly.
- Use visual texture and color cues, not object names alone.
- 5-8 terms per list.
- Return ONLY valid JSON, no markdown.

{
  "expected_terms": ["term1", "term2", "..."],
  "anomaly_terms":  ["term1", "term2", "..."]
}
"""

    def __init__(self, client, model_name: str) -> None:
        self._client = client
        self._model  = model_name
        self._cache: dict[tuple, Vocabulary] = {}

    def build(self, scene_context: str, anomaly_hint: str) -> Vocabulary:
        """
        Build or retrieve cached vocabulary for this (scene, anomaly) pair.
        Falls back to trivial vocabulary on LLM error — scoring still works,
        just with lower discrimination.
        """
        key = (scene_context.strip().lower(), anomaly_hint.strip().lower())
        if key in self._cache:
            logger.info("Vocabulary: cache hit")
            return self._cache[key]

        prompt = (
            f"Scene (what should be there): {scene_context}\n"
            f"Anomaly to detect: {anomaly_hint}"
        )
        try:
            response = self._client.chat.completions.create(
                model    = self._model,
                messages = [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                max_tokens      = 250,
                temperature     = 0.2,
            )
            raw   = response.choices[0].message.content or ""
            clean = re.sub(r'^```json\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
            data  = json.loads(clean)
            vocab = Vocabulary(
                expected_terms = data.get("expected_terms") or [scene_context],
                anomaly_terms  = data.get("anomaly_terms")  or [anomaly_hint],
            )
            logger.info(
                f"Vocabulary built | expected={vocab.expected_terms} | "
                f"anomaly={vocab.anomaly_terms}"
            )
        except Exception as exc:
            logger.warning(
                f"Vocabulary LLM call failed ({exc}) — using fallback terms"
            )
            vocab = Vocabulary(
                expected_terms = [
                    scene_context,
                    f"aerial view of {scene_context}",
                    "normal",
                ],
                anomaly_terms = [
                    anomaly_hint,
                    f"aerial view of {anomaly_hint}",
                    "anomaly",
                    "unusual",
                ],
            )

        self._cache[key] = vocab
        return vocab


# ── Segment scoring ───────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_segments(
    segments_by_frame: dict[str, list[dict]],
    expected_vecs:     list[np.ndarray],
    anomaly_vecs:      list[np.ndarray],
    margin:            float = ANOMALY_SCORE_MARGIN,
) -> list[ScoredSegment]:
    """
    Score every stored segment against the vocabulary.

    For each segment, compute:
        score_expected = max cosine similarity against expected_vecs
        score_anomaly  = max cosine similarity against anomaly_vecs
        anomaly_score  = score_anomaly - score_expected

    Segments where anomaly_score > margin are returned sorted by
    anomaly_score descending.  All other segments are silently discarded.

    This is O(N_segments × N_vocab_terms) dot products — negligible cost.
    """
    if not expected_vecs or not anomaly_vecs:
        logger.warning("score_segments: empty vocabulary — returning no candidates")
        return []

    candidates: list[ScoredSegment] = []
    total = 0

    all_scored_debug: list[tuple] = []   # (frame_path, seg_idx, diff, ano, exp)
    for frame_path, segments in segments_by_frame.items():
        for seg in segments:
            total += 1
            vec = seg["clip_vector"]

            s_exp = max((_cosine(vec, ev) for ev in expected_vecs), default=0.0)
            s_ano = max((_cosine(vec, av) for av in anomaly_vecs), default=0.0)

            # Use ratio if both scores are non-zero, fall back to difference
            if s_exp > 0.01:
                diff = (s_ano - s_exp) / s_exp   # relative margin
            else:
                diff = s_ano - s_exp            
            all_scored_debug.append((frame_path, 0, diff, s_ano, s_exp))
            
            if diff > margin:
                candidates.append(ScoredSegment(
                    frame_path     = frame_path,
                    bbox           = seg["bbox"],
                    area           = seg["area"],
                    clip_vector    = vec,
                    anomaly_score  = diff,
                    score_anomaly  = s_ano,
                    score_expected = s_exp,
                ))

    candidates.sort(key=lambda s: s.anomaly_score, reverse=True)

    # Always log top-5 scores for diagnostics regardless of threshold
    if candidates:
        logger.info(
            f"Segment scoring | {total} total | "
            f"{len(candidates)} above margin={margin:.2f} | "
            f"top score={candidates[0].anomaly_score:.3f}"
        )
    else:
        # Log best scores even when nothing cleared the threshold
        all_scored = sorted(all_scored_debug, key=lambda x: x[2], reverse=True)
        top5 = all_scored[:5]
        logger.info(
            f"Segment scoring | {total} total | 0 above margin={margin:.2f} | "
            f"top-5 (frame, diff, ano, exp): "
            + " | ".join(
                f"{Path(fp).name} Δ={diff:.3f} a={ano:.3f} e={exp:.3f}"
                for fp, _, diff, ano, exp in top5
            )
        )
    return candidates

