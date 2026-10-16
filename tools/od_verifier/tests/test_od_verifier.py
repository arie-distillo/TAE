"""
Tests for the od_verifier experiment.  No network, no API keys, no model weights.

    python -m pytest tools/od_verifier/tests -q

The record test drives the REAL TAE stage code (run_detector_stage,
cross_tile_nms, vlm_verify_stage) with the two network calls stubbed, and is
skipped when TAE's own dependencies are not installed.  The open_clip tests
use randomly initialised weights: they check the code path and timing, not
quality.
"""
from __future__ import annotations

import base64
import json
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import evaluate                                    # noqa: E402
import make_synthetic                              # noqa: E402
import review                                      # noqa: E402
from odv import metrics as M                       # noqa: E402
from odv.crops import make_crop                    # noqa: E402
from odv.dataset import Dataset                    # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    make_synthetic.build(out, frames=12)
    return out


# ── crops ────────────────────────────────────────────────────────────────────

def test_crops_are_square_inside_tile_and_contain_box():
    tile = np.zeros((640, 500, 3), np.uint8)
    tile[100:120, 480:500] = 255                     # object touching the right edge
    for strat in ("tight20", "ctx1", "ctx4", "ctx4box"):
        c = make_crop(tile, [480, 100, 500, 120], strat)
        assert c.shape[0] == c.shape[1] >= 16
        assert c.max() == 255                        # object still inside the window
    assert make_crop(tile, [480, 100, 500, 120], "ctx4").shape[0] == 20 + 2 * 80
    assert make_crop(tile, [0, 0, 400, 600], "ctx4").shape[0] == 500      # clamped to tile
    with pytest.raises(ValueError):
        make_crop(tile, [0, 0, 10, 10], "wide")


# ── thresholds ───────────────────────────────────────────────────────────────

def test_thresholds():
    y = np.array([1] * 100 + [0] * 100)
    s = np.r_[np.linspace(0.5, 1.0, 100), np.linspace(0.0, 0.6, 100)]
    t = M.thr_for_recall(y, s, 0.95)
    assert (s[y == 1] >= t).mean() >= 0.95
    assert (s[y == 1] >= t).mean() < 0.97            # and not needlessly permissive
    hi = M.thr_for_precision(y, s, 0.95)
    keep = s >= hi
    assert y[keep].mean() >= 0.95
    assert M.thr_for_precision(np.zeros(5, int), np.arange(5.0), 0.9) == float("inf")


def test_evaluate_run_perfect_and_random():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.4).astype(int)
    groups = np.array([f"f{i % 30}" for i in range(n)])
    qids = np.array(["q"] * n); sides = rng.integers(8, 100, n)
    cfg = M.EvalConfig()
    perfect = M.evaluate_run(y, y + rng.random(n) * 0.1, groups, qids, sides, cfg)
    assert perfect["auc"] == 1.0
    assert perfect["op"]["specificity"] > 0.99 and perfect["op"]["escalate"] < 0.05
    assert not perfect["in_sample"]
    rand = M.evaluate_run(y, rng.random(n), groups, qids, sides, cfg)
    assert 0.4 < rand["auc"] < 0.6
    assert rand["op"]["specificity"] < 0.25          # a useless score filters almost nothing
    assert rand["op"]["escalate"] > 0.8              # …and a cascade escalates nearly everything


# ── end to end on synthetic data ─────────────────────────────────────────────

def test_evaluate_end_to_end_mock(synth):
    res = evaluate.main(["--data", str(synth), "--backends", "mock", "--crops", "tight20", "ctx2"])
    runs = {r["label"]: r for r in res["runs"]}
    assert runs["(no verifier) · – · detector_conf"]["baseline"]
    red = runs["mock · tight20 · redness"]
    assert red["auc"] > 0.9                          # bounded by the 5 % label noise
    assert red["op"]["n"] == res["dataset"]["n"]     # every candidate scored out of fold
    probe = runs["mock · tight20 · probe"]
    assert probe["auc"] > 0.85
    assert probe["learning_curve"]
    assert red["latency"]["batched_ms_per_crop"] > 0
    out = synth / "eval"
    saved = json.loads((out / "results.json").read_text())
    assert "_pred" not in saved["runs"][0]
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Largest disagreements" in html and "NaN" not in html
    header = (out / "scores.csv").read_text().splitlines()[0]
    assert "mock · tight20 · redness" in header


def test_review_roundtrip(synth, tmp_path):
    review.main(["--data", str(synth), "--limit", "20"])
    page = (synth / "review.html").read_text(encoding="utf-8")
    assert page.count("class='c") == 20
    ds = Dataset.load(synth)
    flip = ds.candidates[0]
    g = tmp_path / "gold.json"
    g.write_text(json.dumps({flip.id: not flip.teacher, "unknown-id": True}))
    review.main(["--data", str(synth), "--apply", str(g)])
    ds2 = Dataset.load(synth)
    assert ds2.candidates[0].gold == (not flip.teacher)
    cands, y = ds2.labelled("best")
    assert y[0] == int(not flip.teacher)
    assert len(ds2.labelled("gold")[0]) == 1
    (synth / "gold.json").unlink()


# ── recorder through the real TAE stages ─────────────────────────────────────

def test_record_through_tae_stages(tmp_path, monkeypatch):
    pytest.importorskip("replicate"); pytest.importorskip("openai")
    pytest.importorskip("pydantic_ai"); pytest.importorskip("pydantic_settings")
    pytest.importorskip("ollama"); pytest.importorskip("tenacity")
    monkeypatch.setenv("REPLICATE_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.chdir(tmp_path)                      # no stray .env

    import record
    record._setup_tae(None)
    import replicate
    from ai.intent import IntentClassifier
    from ai.vlm import TacticalAnalyst

    class FakeClient:
        def __init__(self, api_token=None): pass
        def run(self, model, input):
            assert input["query"] and input["image"].read(2) == b"\xff\xd8"   # a JPEG was sent
            return {"detections": [
                {"bbox": [100, 100, 140, 130], "confidence": 0.5, "label": "car"},
                {"bbox": [300, 300, 360, 350], "confidence": 0.3, "label": "car"},
                {"bbox": [10, 10, 14, 14], "confidence": 0.9, "label": "car"},     # below 8 px → dropped
            ]}

    def fake_vlm(self, tile_img, filename, prompt):
        n = int(prompt.split("The following ")[1].split(" ")[0])
        return json.dumps([{"index": i, "confirmed": i % 2 == 0, "confidence": 0.8,
                            "detected_label": "sedan", **({"reason": "clear car"} if i % 2 == 0 else {})}
                           for i in range(n)])

    monkeypatch.setattr(replicate, "Client", FakeClient)
    monkeypatch.setattr(TacticalAnalyst, "_analyze_openrouter", fake_vlm)
    monkeypatch.setattr(IntentClassifier, "classify", lambda self, q: self._keyword_fallback(q))

    frames = tmp_path / "frames"; frames.mkdir()
    rng = np.random.default_rng(1)
    for i in range(2):
        cv2.imwrite(str(frames / f"f{i}.jpg"), rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))

    out = tmp_path / "ds"
    ds = record.main(["--frames", str(frames), "--query", "find cars", "--out", str(out)])
    # 1280×720 → 3×2 tiles of 640 px with 64 px overlap; 2 surviving boxes per tile
    assert len(ds.candidates) == 2 * 6 * 2
    assert all(c.teacher_valid for c in ds.candidates)
    assert sum(c.teacher for c in ds.candidates) == 12
    assert {c.label for c in ds.candidates} == {"car"}            # detector label preserved
    assert {c.teacher_label for c in ds.candidates if c.teacher} == {"sedan"}
    assert all((out / c.tile).exists() for c in ds.candidates)
    assert ds.meta["timing"]["frames"] == 2 and ds.meta["timing"]["tiles"] == 12

    again = record.main(["--frames", str(frames), "--query", "find cars", "--out", str(out)])
    assert len(again.candidates) == 24                              # resume: nothing re-recorded

    # a failed VLM call must not be recorded as a "No"
    def broken_vlm(self, tile_img, filename, prompt):
        raise RuntimeError("503")
    monkeypatch.setattr(TacticalAnalyst, "_analyze_openrouter", broken_vlm)
    bad = record.main(["--frames", str(frames), "--query", "find trucks", "--out", str(out)])
    trucks = [c for c in bad.candidates if c.query_id == "find-trucks"]
    assert trucks and not any(c.teacher_valid for c in trucks)
    assert len(Dataset.load(out).labelled("teacher")[0]) == 24


# ── real backend code path with random weights ───────────────────────────────

@pytest.mark.parametrize("engine", ["torch", "openvino"])
def test_openclip_backend_random_weights(synth, engine):
    pytest.importorskip("open_clip")
    if engine == "openvino":
        pytest.importorskip("openvino")
    res = evaluate.main(["--data", str(synth), "--backends", "openclip:ViT-B-32:none",
                         "--crops", "ctx2", "--engine", engine, "--limit", "60",
                         "--latency-samples", "6", "--out", str(synth / f"eval_{engine}")])
    scorers = {r["scorer"] for r in res["runs"]}
    assert {"zs_softmax", "zs_margin", "zs_cos", "probe"} <= scorers
    r = next(r for r in res["runs"] if r["scorer"] == "zs_softmax")
    assert r["n"] == 60 and r["latency"]["single_p50_ms"] > 0


def test_prompt_building_never_uses_target_as_distractor():
    from odv.verifiers.contrastive import distractor_phrases, target_phrases
    q = {"object_classes": ["tree", "rooftop antenna"], "canonical_label": "structure",
         "colour_hint": "white", "size_qualifier": None}
    assert target_phrases(q)[0] == "a white tree"
    d = distractor_phrases(q, ["a parked car"])
    assert "a tree" not in d and "a rooftop" not in d and "a parked car" in d and "a shadow" in d


# ── Jev (remote) and Vega (local) backends, stubbed ──────────────────────────

class _FakeJevResponse:
    def __init__(self, body: bytes): self._b = body
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _fake_jev_server(monkeypatch, fail_first=0, record=None):
    """urlopen stub: answers noul = redness of the sent crop; optionally fails the first N calls with 429."""
    import urllib.request
    import urllib.error
    from odv.verifiers.mock import _redness
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        assert req.get_header("Typesafe-danger-zone-images") == "true"
        assert req.get_header("Authorization") == "Bearer k"
        body = json.loads(req.data)
        assert body["questions"]["match"]["type"] == "noul"
        assert body["assets"]["crop-asset"]["format"] == "jpg"
        img = cv2.imdecode(np.frombuffer(base64.b64decode(body["assets"]["crop-asset"]["data_base64"]),
                                         np.uint8), cv2.IMREAD_COLOR)
        assert min(img.shape[:2]) >= 280                       # upscaled to Jev's minimum side
        if record is not None:
            record.append(body)
        if calls["n"] <= fail_first:
            raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, None)
        p = float(np.clip(0.5 + _redness(img), 0, 1))
        return _FakeJevResponse(json.dumps({"model": body["model"],
                                            "answers": {"match": {"type": "noul", "noul": p}},
                                            "usage": {"input_tokens": 300, "output_tokens": 20}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return calls


def test_jev_backend_stubbed(synth, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    sent = []
    calls = _fake_jev_server(monkeypatch, fail_first=2, record=sent)
    res = evaluate.main(["--data", str(synth), "--backends", "jev", "--crops", "tight20",
                         "--limit", "40", "--latency-samples", "4", "--out", str(synth / "eval_jev")])
    r = next(r for r in res["runs"] if r["backend"] == "jev/jev-1.14.0")
    assert r["scorer"] == "noul" and r["n"] == 40
    assert r["auc"] > 0.8                                      # redness tracks the synthetic labels
    assert r["latency"]["single_p50_ms"] > 0
    assert calls["n"] == 4 + 40 + 2                            # timing + batch + the two retried 429s
    assert sent[0]["model"] == "jev-1.14.0"
    assert "car, van" in sent[0]["questions"]["match"]["instructions"]


def test_jev_backend_needs_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from odv.verifiers import make_verifier
    with pytest.raises(RuntimeError):
        make_verifier("jev").load()


def test_vega_backend_stubbed(synth, monkeypatch):
    """vegaml is not installed here: a fake module with the documented decide_image() shape."""
    import types
    from odv.verifiers.mock import _redness

    class FakeVega:
        def decide_image(self, img, questions):
            assert set(questions) == {"match"} and questions["match"]["type"] == "boolean"
            arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            p = float(np.clip(0.5 + _redness(arr), 0, 1))
            return {"answers": {"match": {"type": "boolean", "boolean": p, "choice": "true" if p > .5 else "false",
                                          "abstain": abs(p - 0.5) < 0.05, "readout": "engine"}}}

    loads = []
    fake = types.ModuleType("vegaml")
    fake.load = lambda repo=None, **kw: loads.append((repo, kw)) or FakeVega()
    monkeypatch.setitem(sys.modules, "vegaml", fake)

    res = evaluate.main(["--data", str(synth), "--backends", "vega", "vega:4b", "--crops", "tight20",
                         "--limit", "40", "--latency-samples", "4", "--out", str(synth / "eval_vega")])
    assert loads == [(None, {"device": "cpu"}), ("4b", {"device": "cpu"})]
    runs = {(r["backend"], r["scorer"]) for r in res["runs"]}
    assert ("vega/0.8b", "p_true") in runs and ("vega/4b", "p_true_confident") in runs
    r = next(r for r in res["runs"] if r["backend"] == "vega/0.8b" and r["scorer"] == "p_true")
    assert r["auc"] > 0.8 and r["latency"]["batched_ms_per_crop"] > 0


def test_record_with_jev_through_tae_stages(tmp_path, monkeypatch):
    pytest.importorskip("replicate"); pytest.importorskip("openai")
    pytest.importorskip("pydantic_ai"); pytest.importorskip("pydantic_settings")
    pytest.importorskip("ollama"); pytest.importorskip("tenacity")
    monkeypatch.setenv("REPLICATE_API_KEY", "test"); monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    import record
    record._setup_tae(None)
    import replicate
    from ai.intent import IntentClassifier
    from ai.vlm import TacticalAnalyst

    class FakeClient:
        def __init__(self, api_token=None): pass
        def run(self, model, input):
            return {"detections": [{"bbox": [100, 100, 140, 130], "confidence": 0.5, "label": "car"}]}

    monkeypatch.setattr(replicate, "Client", FakeClient)
    monkeypatch.setattr(TacticalAnalyst, "_analyze_openrouter", lambda self, img, fn, prompt: json.dumps(
        [{"index": 0, "confirmed": True, "confidence": 0.8, "detected_label": "car", "reason": "ok"}]))
    monkeypatch.setattr(IntentClassifier, "classify", lambda self, q: self._keyword_fallback(q))
    calls = _fake_jev_server(monkeypatch)

    frames = tmp_path / "frames"; frames.mkdir()
    cv2.imwrite(str(frames / "f0.jpg"), np.full((720, 1280, 3), 90, np.uint8))
    ds = record.main(["--frames", str(frames), "--query", "find cars", "--out", str(tmp_path / "ds"), "--jev"])
    assert len(ds.candidates) == 6 and calls["n"] == 6
    assert all(c.jev_p is not None for c in ds.candidates)
    assert ds.meta["timing"]["jev_frames"] == 1 and ds.meta["jev_model"] == "jev-1.14.0"

    ds2 = Dataset.load(tmp_path / "ds")
    assert all(c.jev_p is not None for c in ds2.candidates)   # persisted and reloaded
    # evaluate picks the recorded verdicts up as a baseline without calling Jev again
    res = evaluate.main(["--data", str(tmp_path / "ds"), "--backends", "mock", "--crops", "ctx2",
                         "--no-probe", "--latency-samples", "2"])
    base = next(r for r in res["runs"] if r["backend"].startswith("jev/") and r["baseline"])
    assert base["crop"] == "ctx2" and base["latency"]["batched_ms_per_crop"] >= 0
    assert calls["n"] == 6
