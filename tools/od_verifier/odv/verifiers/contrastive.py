"""
Contrastive verifier (CLIP / SigLIP family through open_clip).

One image-tower forward pass per crop; no text generation.  Produces three
zero-shot scores from the same embedding plus the embedding itself, which the
evaluator uses to train a distilled probe head.

Zero-shot scoring
-----------------
"X" vs "not X" prompts are deliberately NOT used: contrastive models handle
negation poorly.  The target phrases are instead contrasted against a list of
concrete things the detector typically confuses on aerial imagery
(DEFAULT_DISTRACTORS, extendable per query).

    zs_softmax   softmax over [targets + distractors], summed over targets
    zs_margin    best target cosine − best distractor cosine
    zs_cos       best target cosine

Engines
-------
    torch      plain PyTorch CPU
    openvino   image tower converted to OpenVINO (CPU, or the Intel iGPU with
               --ov-device GPU).  Text tower stays in PyTorch: it runs once
               per query, not per crop.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import RunOutput, Verifier

# Same domain prefix TAE uses (core/app_state._CLIP_AERIAL_CTX) plus one variant.
TEMPLATES = [
    "aerial drone nadir overhead view: {}",
    "a top-down aerial photo of {}",
]

DEFAULT_DISTRACTORS = [
    "bare ground", "a paved road", "a rooftop", "a tree", "bushes and vegetation",
    "a shadow", "a rock", "water", "road markings", "a pile of rubble",
    "an empty field", "a fence", "a patch of dirt",
]


def _article(phrase: str) -> str:
    p = phrase.strip()
    if p.split(" ", 1)[0].lower() in ("a", "an", "the"):
        return p
    return ("an " if p[:1].lower() in "aeiou" else "a ") + p


def target_phrases(q: dict) -> list[str]:
    classes = list(q.get("object_classes") or [])
    canon = q.get("canonical_label") or ""
    if canon and canon not in classes:
        classes.append(canon)
    if not classes:
        classes = [q.get("query", "object")]
    qual = " ".join(x for x in (q.get("size_qualifier"), q.get("colour_hint")) if x)
    return [_article(f"{qual} {c}".strip()) for c in classes]


def distractor_phrases(q: dict, extra: list[str] | None = None) -> list[str]:
    words = set()
    for c in (q.get("object_classes") or []) + [q.get("canonical_label") or ""]:
        words.update(w.lower().rstrip("s") for w in c.split() if len(w) > 2)
    out = []
    for d in list(q.get("distractors") or []) + list(extra or []) + DEFAULT_DISTRACTORS:
        toks = {w.lower().rstrip("s") for w in d.split()}
        if toks & words:
            continue                       # never use the target itself as a distractor
        a = _article(d)
        if a not in out:
            out.append(a)
    return out


class OpenClipVerifier(Verifier):
    has_embeddings = True

    def __init__(self, arch: str, pretrained: str = "", engine: str = "torch",
                 threads: int = 0, ov_device: str = "CPU",
                 distractors: dict | None = None) -> None:
        self.arch = arch
        self.pretrained = pretrained
        self.engine = engine
        self.threads = threads
        self.ov_device = ov_device
        self.extra_distractors = distractors or {}
        tag = pretrained.replace("\\", "/").rsplit("/", 1)[-1] or "none"
        self.name = f"{arch}/{tag}" + ("" if engine == "torch" else f"@{engine}-{ov_device}")
        self._text_cache: dict[str, np.ndarray] = {}

    # ── init ─────────────────────────────────────────────────────────────────
    def load(self) -> None:
        import torch
        import open_clip
        if self.threads:
            torch.set_num_threads(self.threads)
        pre = None if self.pretrained.lower() in ("", "none", "random") else self.pretrained
        model, _, preprocess = open_clip.create_model_and_transforms(self.arch, pretrained=pre)
        self._torch = torch
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.arch)
        self._scale = float(model.logit_scale.exp().item())
        self._ov = None
        if self.engine == "openvino":
            self._compile_openvino()
        elif self.engine != "torch":
            raise ValueError(f"unknown engine '{self.engine}'")

    def _compile_openvino(self) -> None:
        import openvino as ov
        torch = self._torch
        model = self._model

        class _Tower(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.encode_image(x)

        size = self._model.visual.image_size
        h, w = (size, size) if isinstance(size, int) else size
        with torch.no_grad():
            ov_model = ov.convert_model(_Tower(model), example_input=torch.zeros(1, 3, h, w))
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if self.threads and self.ov_device == "CPU":
            cfg["INFERENCE_NUM_THREADS"] = self.threads
        self._ov = core.compile_model(ov_model, self.ov_device, cfg)

    # ── embedding ────────────────────────────────────────────────────────────
    def _prep(self, crops: list[np.ndarray]):
        from PIL import Image
        return self._torch.stack([
            self._preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops
        ])

    def _forward(self, x) -> np.ndarray:
        if self._ov is not None:
            feats = np.asarray(self._ov(x.numpy())[0], dtype=np.float32)
        else:
            with self._torch.no_grad():
                feats = self._model.encode_image(x).float().numpy()
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    def _embed_text(self, phrase: str) -> np.ndarray:
        v = self._text_cache.get(phrase)
        if v is None:
            toks = self._tokenizer([t.format(phrase) for t in TEMPLATES])
            with self._torch.no_grad():
                f = self._model.encode_text(toks).float().numpy()
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
            v = f.mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-9)
            self._text_cache[phrase] = v
        return v

    # ── API ──────────────────────────────────────────────────────────────────
    def run(self, crops, cands, queries, batch: int = 16) -> RunOutput:
        out = RunOutput(scores={})
        embs = []
        for i in range(0, len(crops), batch):
            t0 = time.perf_counter()
            x = self._prep(crops[i:i + batch])
            t1 = time.perf_counter()
            embs.append(self._forward(x))
            t2 = time.perf_counter()
            out.pre_ms += (t1 - t0) * 1e3
            out.model_ms += (t2 - t1) * 1e3
        emb = np.concatenate(embs) if embs else np.zeros((0, 1), np.float32)
        out.embeddings = emb

        n = len(cands)
        soft = np.zeros(n, np.float32); margin = np.zeros(n, np.float32); cos = np.zeros(n, np.float32)
        t0 = time.perf_counter()
        qvecs = {}
        for qid in {c.query_id for c in cands}:
            q = queries.get(qid, {})
            pos = np.stack([self._embed_text(p) for p in target_phrases(q)])
            neg = np.stack([self._embed_text(p)
                            for p in distractor_phrases(q, self.extra_distractors.get(qid))])
            qvecs[qid] = (pos, neg)
        out.text_ms = (time.perf_counter() - t0) * 1e3

        for qid, (pos, neg) in qvecs.items():
            idx = np.array([i for i, c in enumerate(cands) if c.query_id == qid])
            e = emb[idx]
            cp = e @ pos.T                       # (n, P)
            cn = e @ neg.T                       # (n, N)
            logits = self._scale * np.concatenate([cp, cn], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            soft[idx] = p[:, :pos.shape[0]].sum(axis=1)
            margin[idx] = cp.max(axis=1) - cn.max(axis=1)
            cos[idx] = cp.max(axis=1)
        out.scores = {"zs_softmax": soft, "zs_margin": margin, "zs_cos": cos}
        return out

    def time_single(self, crops, cands, queries) -> list[float]:
        times = []
        for c in crops:
            t0 = time.perf_counter()
            self._forward(self._prep([c]))
            times.append((time.perf_counter() - t0) * 1e3)
        return times
