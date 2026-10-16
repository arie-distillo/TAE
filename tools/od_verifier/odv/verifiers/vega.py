"""
Vega (vegaml) — LOCAL typed-decision verifier.

A frozen Qwen3.5 backbone (0.8B or 4B) with a small decision head.  Images go
through the backbone's own vision tower (`decide_image`); the head was trained
on text features only, so the image path is zero-shot on the author's side.
One `boolean` question per crop; score = P(true).  The `abstain` flag is
returned as a second scorer (1 − abstain) so the cascade can use it.

Runs on CPU (device="cpu").  The cost is the full backbone forward pass over
the image tokens, not just the head.

    pip install 'vegaml[vision]'     # downloads Qwen3.5-0.8B from the Hub on first use
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import RunOutput, Verifier


def build_question(q: dict) -> dict:
    classes = ", ".join(q.get("object_classes") or [q.get("canonical_label") or "the target object"])
    qual = " ".join(x for x in (q.get("size_qualifier"), q.get("colour_hint")) if x)
    target = f"{qual} {classes}".strip()
    crit = (q.get("criteria") or "").strip()
    return {"match": {
        "type": "boolean",
        "instructions": f"The image is a top-down (nadir) drone crop. "
                        f"Is the object at the centre a {target}?",
        "criteria": {
            "true": crit or f"a real {target} is visible at the centre",
            "false": "something else: ground, rooftop, shadow, vegetation, rock, road marking, or empty",
        },
    }}


class VegaVerifier(Verifier):
    has_embeddings = False

    def __init__(self, size: str = "", device: str = "cpu", threads: int = 0) -> None:
        self.size = size or "0.8b"
        self.device = device
        self.threads = threads
        self.name = f"vega/{self.size}"

    def load(self) -> None:
        import torch
        import vegaml
        if self.threads:
            torch.set_num_threads(self.threads)
        # vegaml.load() takes the size alias ("4b") or a repo/path; default repo is the 0.8B.
        self._v = vegaml.load(None if self.size == "0.8b" else self.size, device=self.device)
        self._qcache: dict[str, dict] = {}

    def _score(self, crop: np.ndarray, qid: str, q: dict) -> tuple[float, float]:
        from PIL import Image
        qs = self._qcache.get(qid)
        if qs is None:
            qs = self._qcache[qid] = build_question(q)
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out = self._v.decide_image(img, qs)
        a = out["answers"]["match"]
        return float(a["boolean"]), 0.0 if a.get("abstain") else 1.0

    def run(self, crops, cands, queries, batch: int = 1) -> RunOutput:
        p = np.zeros(len(crops), np.float32); conf = np.zeros(len(crops), np.float32)
        t0 = time.perf_counter()
        for i, (c, k) in enumerate(zip(crops, cands)):
            p[i], conf[i] = self._score(c, k.query_id, queries.get(k.query_id, {}))
        return RunOutput(scores={"p_true": p, "p_true_confident": np.where(conf > 0, p, 0.5)},
                         model_ms=(time.perf_counter() - t0) * 1e3)

    def time_single(self, crops, cands, queries) -> list[float]:
        out = []
        for c, k in zip(crops, cands):
            t0 = time.perf_counter()
            self._score(c, k.query_id, queries.get(k.query_id, {}))
            out.append((time.perf_counter() - t0) * 1e3)
        return out
