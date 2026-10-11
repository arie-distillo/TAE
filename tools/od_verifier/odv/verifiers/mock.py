"""Deterministic stand-in used by the tests: scores a crop by how red its centre is."""
from __future__ import annotations

import time

import numpy as np

from .base import RunOutput, Verifier


def _redness(crop: np.ndarray) -> float:
    h, w = crop.shape[:2]
    c = crop[h // 3: 2 * h // 3 + 1, w // 3: 2 * w // 3 + 1].astype(np.float32)
    b, g, r = c[..., 0].mean(), c[..., 1].mean(), c[..., 2].mean()
    return float((r - (g + b) / 2.0) / 255.0)


def _features(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    c = crop[h // 3: 2 * h // 3 + 1, w // 3: 2 * w // 3 + 1].reshape(-1, 3).astype(np.float32) / 255.0
    f = np.concatenate([c.mean(axis=0), c.std(axis=0), crop.reshape(-1, 3).mean(axis=0) / 255.0])
    return f / (np.linalg.norm(f) + 1e-9)


class MockVerifier(Verifier):
    name = "mock"
    has_embeddings = True

    def run(self, crops, cands, queries, batch: int = 16) -> RunOutput:
        t0 = time.perf_counter()
        s = np.array([_redness(c) for c in crops], np.float32)
        emb = np.stack([_features(c) for c in crops]) if crops else np.zeros((0, 9), np.float32)
        return RunOutput(scores={"redness": s}, embeddings=emb,
                         model_ms=(time.perf_counter() - t0) * 1e3)

    def time_single(self, crops, cands, queries) -> list[float]:
        out = []
        for c in crops:
            t0 = time.perf_counter()
            _redness(c)
            out.append((time.perf_counter() - t0) * 1e3)
        return out
