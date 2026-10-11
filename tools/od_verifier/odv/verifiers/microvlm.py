"""
Small generative VLM used as a Yes/No classifier (SmolVLM-class models through
Hugging Face transformers).

No text is generated: one forward pass, then the logits at the first answer
position are read for the Yes / No tokens and turned into

    P(yes) = exp(z_yes) / (exp(z_yes) + exp(z_no))

Note that this saves the decoding loop only.  The image encoder and the prompt
prefill still run in full, and they are most of the cost for these models.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import RunOutput, Verifier


def build_question(q: dict) -> str:
    classes = ", ".join(q.get("object_classes") or [q.get("canonical_label") or "the target object"])
    qual = " ".join(x for x in (q.get("size_qualifier"), q.get("colour_hint")) if x)
    target = f"{qual} {classes}".strip()
    criteria = (q.get("criteria") or "").strip()
    text = (
        "This is a crop from a top-down (nadir) drone image. "
        f"Target: {target}. "
    )
    if criteria:
        text += f"Criteria: {criteria} "
    text += "Is the object at the centre of the image a match? Answer Yes or No."
    return text


class HFYesNoVerifier(Verifier):
    has_embeddings = False

    def __init__(self, model_id: str, threads: int = 0) -> None:
        self.model_id = model_id
        self.threads = threads
        self.name = model_id.rsplit("/", 1)[-1]

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except ImportError:                                  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoModel
        if self.threads:
            torch.set_num_threads(self.threads)
        self._torch = torch
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = _AutoModel.from_pretrained(self.model_id, torch_dtype=torch.float32).eval()
        # One sub-image only: the crop is small, tiling it would multiply the encoder cost.
        ip = getattr(self._proc, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = False
        tok = self._proc.tokenizer
        self._yes = self._first_ids(tok, ["Yes", " Yes", "yes", " yes"])
        self._no = self._first_ids(tok, ["No", " No", "no", " no"])

    @staticmethod
    def _first_ids(tok, words: list[str]) -> list[int]:
        ids = []
        for w in words:
            enc = tok(w, add_special_tokens=False)["input_ids"]
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    def _score(self, crop: np.ndarray, q: dict) -> float:
        from PIL import Image
        torch = self._torch
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": build_question(q)}]}]
        prompt = self._proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._proc(text=prompt, images=[img], return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1].float()
        z_yes = torch.logsumexp(logits[self._yes], dim=0)
        z_no = torch.logsumexp(logits[self._no], dim=0)
        return float(torch.sigmoid(z_yes - z_no))

    def run(self, crops, cands, queries, batch: int = 1) -> RunOutput:
        out = RunOutput(scores={})
        s = np.zeros(len(crops), np.float32)
        t0 = time.perf_counter()
        for i, (crop, c) in enumerate(zip(crops, cands)):
            s[i] = self._score(crop, queries.get(c.query_id, {}))
        out.model_ms = (time.perf_counter() - t0) * 1e3
        out.scores = {"yesno": s}
        return out

    def time_single(self, crops, cands, queries) -> list[float]:
        times = []
        for crop, c in zip(crops, cands):
            t0 = time.perf_counter()
            self._score(crop, queries.get(c.query_id, {}))
            times.append((time.perf_counter() - t0) * 1e3)
        return times
