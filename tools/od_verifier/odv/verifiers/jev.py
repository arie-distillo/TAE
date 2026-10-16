"""
Jev (TypeSafe AI) — REMOTE typed-decision verifier.

One POST /v1/systemone per crop with the crop as an inline base64 asset and a
single `noul` question; the answer is P(yes) in [0, 1].  Calls run in parallel
threads, as TAE does for Qwen.  Needs TYPESAFE_API_KEY in the environment.

This is not a local option: it replaces the Qwen call with a cheaper, typed
one.  Latency reported by the evaluator is round-trip time per crop.

Image input and the `TypeSafe-Danger-Zone-Images` header are experimental on
TypeSafe's side (Jev 1.14); image sides must be 280–1568 px, so crops are
upscaled to JEV_MIN_SIDE before sending.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .base import RunOutput, Verifier

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MIN_SIDE = 280
JEV_MAX_SIDE = 1568


def build_question(q: dict) -> dict:
    classes = ", ".join(q.get("object_classes") or [q.get("canonical_label") or "the target object"])
    qual = " ".join(x for x in (q.get("size_qualifier"), q.get("colour_hint")) if x)
    target = f"{qual} {classes}".strip()
    crit = (q.get("criteria") or "").strip()
    return {
        "type": "noul",
        "instructions": f"`crop` is a top-down (nadir) drone image. "
                        f"Is the object at the centre of `crop` a {target}?",
        "criteria": {
            "true": crit or f"a real {target} is visible at the centre",
            "false": "something else: ground, rooftop, shadow, vegetation, rock, road marking, or empty",
        },
    }


def _encode(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    s = max(1.0, JEV_MIN_SIDE / min(h, w))
    s = min(s, JEV_MAX_SIDE / max(h, w))
    if s != 1.0:
        crop = cv2.resize(crop, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class JevVerifier(Verifier):
    has_embeddings = False

    def __init__(self, model: str = "jev-1.14.0", workers: int = 8, timeout_s: float = 60.0,
                 api_key: str | None = None) -> None:
        self.model = model
        self.workers = workers
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.name = f"jev/{model}"
        self.tokens_in = 0

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")

    # ── one call ─────────────────────────────────────────────────────────────
    def _call(self, crop: np.ndarray, q: dict, attempts: int = 4) -> tuple[float, float]:
        """Returns (P(yes), round-trip ms); NaN when every attempt failed."""
        body = json.dumps({
            "model": self.model,
            "state": {"crop": "crop-asset"},
            "assets": {"crop-asset": {"type": "image", "format": "jpg", "data_base64": _encode(crop)}},
            "questions": {"match": build_question(q)},
        }).encode("utf-8")
        req = urllib.request.Request(JEV_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        })
        delay = 2.0
        t0 = time.perf_counter()
        for i in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                self.tokens_in += int(out.get("usage", {}).get("input_tokens", 0))
                return float(out["answers"]["match"]["noul"]), (time.perf_counter() - t0) * 1e3
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 529, 500, 502, 503) or i == attempts - 1:
                    return float("nan"), (time.perf_counter() - t0) * 1e3
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
                if i == attempts - 1:
                    return float("nan"), (time.perf_counter() - t0) * 1e3
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        return float("nan"), (time.perf_counter() - t0) * 1e3

    # ── API ──────────────────────────────────────────────────────────────────
    def run(self, crops, cands, queries, batch: int = 16) -> RunOutput:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            res = list(pool.map(lambda a: self._call(a[0], queries.get(a[1].query_id, {})),
                                zip(crops, cands)))
        s = np.array([r[0] for r in res], np.float32)
        out = RunOutput(scores={"noul": s}, model_ms=(time.perf_counter() - t0) * 1e3)
        n_fail = int(np.isnan(s).sum())
        if n_fail:
            out.notes.append(f"{n_fail} of {len(s)} Jev calls failed (scored NaN)")
        return out

    def time_single(self, crops, cands, queries) -> list[float]:
        # sequential round trips: what one crop costs when nothing else is in flight
        return [self._call(c, queries.get(k.query_id, {}))[1] for c, k in zip(crops, cands)]
