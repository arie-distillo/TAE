from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunOutput:
    scores:     dict[str, np.ndarray]            # scorer name → score per crop (higher = more "yes")
    embeddings: np.ndarray | None = None         # (N, D) L2-normalised, when the backend has them
    pre_ms:     float = 0.0                      # total preprocessing time for the batch pass
    model_ms:   float = 0.0                      # total model time for the batch pass
    text_ms:    float = 0.0                      # one-off text/prompt encoding time
    notes:      list[str] = field(default_factory=list)


class Verifier:
    """A local verifier: crops in, one score per crop out."""
    name: str = "verifier"
    has_embeddings: bool = False

    def load(self) -> None:                      # heavy init, timed by the evaluator
        pass

    def run(self, crops: list[np.ndarray], cands: list, queries: dict,
            batch: int = 16) -> RunOutput:
        raise NotImplementedError

    def time_single(self, crops: list[np.ndarray], cands: list, queries: dict) -> list[float]:
        """Per-crop latency in ms at batch size 1 (preprocess + model), one entry per crop."""
        raise NotImplementedError
