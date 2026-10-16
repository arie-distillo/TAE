"""
Verifier backends.

Spec strings accepted by make_verifier():

    openclip:<arch>:<pretrained>   contrastive image/text model through open_clip
                                   e.g. openclip:ViT-B-32-quickgelu:openai   (the CLIP TAE already loads)
                                        openclip:ViT-B-16-SigLIP:webli
                                        openclip:ViT-B-32:/path/to/checkpoint.pt
                                        openclip:ViT-B-32:none               (random weights — latency only)
    hfvlm:<hf model id>            small generative VLM, first-token Yes/No logits
                                   e.g. hfvlm:HuggingFaceTB/SmolVLM-256M-Instruct
    jev[:<model>]                  REMOTE: TypeSafe Jev typed decision (noul), needs TYPESAFE_API_KEY
                                   e.g. jev            (= jev-1.14.0)
    vega[:<size>]                  LOCAL: vegaml decision model on CPU, 0.8b (default) or 4b
    mock                           deterministic colour heuristic (tests only)
"""
from __future__ import annotations

from .base import RunOutput, Verifier


def make_verifier(spec: str, engine: str = "torch", threads: int = 0,
                  ov_device: str = "CPU", distractors: dict | None = None) -> Verifier:
    kind, _, rest = spec.partition(":")
    if kind == "openclip":
        arch, _, pretrained = rest.partition(":")
        if not arch:
            raise ValueError("openclip spec needs an architecture: openclip:<arch>:<pretrained>")
        from .contrastive import OpenClipVerifier
        return OpenClipVerifier(arch, pretrained, engine=engine, threads=threads,
                                ov_device=ov_device, distractors=distractors)
    if kind == "hfvlm":
        if not rest:
            raise ValueError("hfvlm spec needs a model id: hfvlm:<hf model id>")
        from .microvlm import HFYesNoVerifier
        return HFYesNoVerifier(rest, threads=threads)
    if kind == "jev":
        from .jev import JevVerifier
        return JevVerifier(model=rest or "jev-1.14.0")
    if kind == "vega":
        from .vega import VegaVerifier
        return VegaVerifier(size=rest, device=ov_device.lower() if ov_device.lower() in ("cpu", "cuda", "mps") else "cpu",
                            threads=threads)
    if kind == "mock":
        from .mock import MockVerifier
        return MockVerifier()
    raise ValueError(f"unknown verifier spec '{spec}'")


__all__ = ["make_verifier", "Verifier", "RunOutput"]
