#!/usr/bin/env python
"""
evaluate.py — quality and CPU latency of local verifiers on a recorded dataset.

    python tools/od_verifier/evaluate.py --data tools/od_verifier/data/run1 \
        --backends openclip:ViT-B-32-quickgelu:openai openclip:ViT-B-16-SigLIP:webli \
        --crops tight20 ctx2 ctx4

For every backend × crop strategy it reports, against the recorded remote-VLM
verdicts (or human-reviewed labels where present):

  quality   AUC / AP, a recall-targeted single threshold, and a two-threshold
            cascade (accept / reject locally, escalate the uncertain band)
  latency   ms per crop at batch 1 (p50 / p95) and batched, load time, memory,
            and the resulting seconds per frame at the dataset's candidate load

Embedding backends additionally get a distilled probe head (logistic
regression on frozen embeddings, trained out-of-fold on the reference labels).

Outputs <out>/results.json, <out>/scores.csv and <out>/report.html.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odv import metrics as M                      # noqa: E402
from odv.crops import make_crop, parse as parse_crop   # noqa: E402
from odv.dataset import Dataset                   # noqa: E402
from odv.report import render, thumb_b64          # noqa: E402
from odv.verifiers import make_verifier           # noqa: E402


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return None


def _machine() -> str:
    import os
    cpu = platform.processor() or platform.machine()
    try:
        import psutil
        cores = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
    except Exception:
        cores = f"{os.cpu_count()} threads"
    return f"{cpu} · {cores} · {platform.system()} {platform.release()} · Python {platform.python_version()}"


def _clean(o):
    """JSON-safe copy: drops private keys, turns NaN/inf into None."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset directory written by record.py")
    ap.add_argument("--backends", nargs="+", required=True,
                    help="verifier specs, e.g. openclip:ViT-B-32-quickgelu:openai  hfvlm:HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--crops", nargs="+", default=["tight20", "ctx2", "ctx4"])
    ap.add_argument("--engine", choices=["torch", "openvino"], default="torch",
                    help="inference engine for openclip backends")
    ap.add_argument("--ov-device", default="CPU", help="OpenVINO device: CPU, GPU (Intel iGPU) or AUTO")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--labels", choices=["best", "gold", "teacher"], default="best")
    ap.add_argument("--out", default=None, help="output directory (default <data>/eval)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N labelled candidates")
    ap.add_argument("--latency-samples", type=int, default=40, help="crops timed at batch size 1")
    ap.add_argument("--no-probe", action="store_true", help="skip the distilled probe head")
    ap.add_argument("--probe-c", type=float, default=1.0, help="inverse L2 strength of the probe")
    ap.add_argument("--distractors", default=None,
                    help="JSON file {query_id: [phrases]} of extra confusers for zero-shot scoring")
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--cascade-recall", type=float, default=0.98)
    ap.add_argument("--cascade-precision", type=float, default=0.95)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    for c in args.crops:
        parse_crop(c)                                   # fail early on typos

    ds = Dataset.load(args.data)
    cands, y = ds.labelled(args.labels)
    if args.limit:
        cands, y = cands[:args.limit], y[:args.limit]
    if len(cands) == 0:
        raise SystemExit("no labelled candidates in the dataset")
    out_dir = Path(args.out) if args.out else ds.root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = M.EvalConfig(args.recall, args.cascade_recall, args.cascade_precision, args.folds)
    groups = np.array([c.frame for c in cands])
    qids = np.array([c.query_id for c in cands])
    sides = np.array([c.min_side for c in cands])
    det_conf = np.array([c.det_conf for c in cands], np.float32)
    extras = np.stack([det_conf, np.log2(np.maximum(1, [c.max_side for c in cands]))], axis=1)
    distractors = json.loads(Path(args.distractors).read_text(encoding="utf-8")) if args.distractors else {}

    per_frame = {}
    for c in ds.candidates:
        per_frame[c.frame] = per_frame.get(c.frame, 0) + 1
    counts = np.array(list(per_frame.values()), float)

    print(f"dataset: {len(cands)} labelled candidates, {int(y.sum())} confirmed, "
          f"{len(set(groups.tolist()))} frames, {len(set(qids.tolist()))} query(ies)")

    runs: list[dict] = []
    score_cols: dict[str, np.ndarray] = {}

    def add_run(backend: str, crop: str, scorer: str, s: np.ndarray, latency: dict | None,
                baseline: bool = False, extra: dict | None = None) -> dict:
        r = M.evaluate_run(y, s.astype(np.float64), groups, qids, sides, cfg)
        r.update(backend=backend, crop=crop, scorer=scorer, baseline=baseline,
                 label=f"{backend} · {crop} · {scorer}", latency=latency, **(extra or {}))
        runs.append(r)
        score_cols[r["label"]] = s
        op = r["op"]
        print(f"  {r['label']:<58} AUC {r['auc']:.3f}  rejects removed {op['specificity']:.1%} "
              f"@ recall {op['recall']:.1%}  escalated {op['escalate']:.1%}")
        return r

    # Baseline that needs no verifier at all: rank by detector confidence.
    add_run("(no verifier)", "–", "detector_conf", det_conf, None, baseline=True)

    # Jev verdicts recorded by record.py --jev: a remote baseline that costs no new calls here.
    if any(c.jev_p is not None for c in cands):
        s = np.array([np.nan if c.jev_p is None else c.jev_p for c in cands], np.float32)
        jt = ds.meta.get("timing", {})
        lat = None
        if jt.get("jev_frames"):
            n_scored = sum(1 for c in ds.candidates if c.jev_p is not None)
            lat = {"single_p50_ms": float("nan"), "single_p95_ms": float("nan"),
                   "batched_ms_per_crop": jt["jev_s"] * 1e3 / max(1, n_scored)}
        add_run(f"jev/{ds.meta.get('jev_model', '?')} (recorded)", ds.meta.get("jev_crop", "?"),
                "noul", s, lat, baseline=True)

    backends_info = []
    for spec in args.backends:
        print(f"\n[{spec}] loading …")
        ver = make_verifier(spec, engine=args.engine, threads=args.threads,
                            ov_device=args.ov_device, distractors=distractors)
        t0 = time.perf_counter()
        ver.load()
        info = {"name": ver.name, "spec": spec, "load_s": time.perf_counter() - t0,
                "rss_mb": _rss_mb(), "text_ms": None}
        backends_info.append(info)
        print(f"[{spec}] loaded in {info['load_s']:.1f}s")

        for crop in args.crops:
            crops = [make_crop(ds.tile_image(c), c.bbox, crop) for c in cands]
            # Batch-1 timing first: it doubles as warm-up for the batched pass.
            k = min(args.latency_samples, len(crops))
            sel = np.linspace(0, len(crops) - 1, k).astype(int)
            single = ver.time_single([crops[i] for i in sel], [cands[i] for i in sel], ds.queries)
            out = ver.run(crops, cands, ds.queries, batch=args.batch)
            lat = M.latency_summary(single, out.pre_ms + out.model_ms, len(crops))
            lat["pre_ms_per_crop"] = out.pre_ms / len(crops)
            if out.text_ms and info["text_ms"] is None:
                info["text_ms"] = out.text_ms

            for scorer, s in out.scores.items():
                add_run(ver.name, crop, scorer, s, lat)

            if out.embeddings is not None and not args.no_probe:
                s = M.probe_oof(out.embeddings, y, groups, qids, extras, cfg.folds, args.probe_c)
                if np.isnan(s).all():
                    print("  probe skipped: not enough labelled data per query / fewer than 2 frames")
                else:
                    lc = M.probe_learning_curve(out.embeddings, y, qids, extras, args.probe_c)
                    add_run(ver.name, crop, "probe", s, lat, extra={"learning_curve": lc})
        del ver

    # ── gallery for the best non-baseline run ────────────────────────────────
    real = [r for r in runs if not r["baseline"] and not math.isnan(r["auc"])]
    gallery = None
    if real:
        best = max(real, key=lambda r: r["auc"])
        s = score_cols[best["label"]]
        ok = ~np.isnan(s)

        def items(idx):
            return [{"img": thumb_b64(make_crop(ds.tile_image(cands[i]), cands[i].bbox, "ctx4box")),
                     "label": cands[i].label, "side": cands[i].min_side,
                     "score": float(s[i]), "det_conf": cands[i].det_conf} for i in idx]

        neg = np.where(ok & (y == 0))[0]; pos = np.where(ok & (y == 1))[0]
        gallery = {"label": best["label"],
                   "false_accepts": items(neg[np.argsort(-s[neg])][:12]),
                   "misses": items(pos[np.argsort(s[pos])][:12])}

    rem = ds.meta.get("timing") or {}
    nfr = max(1, rem.get("frames", 0))
    res = {
        "machine": _machine(),
        "config": {"recall_target": cfg.recall_target, "cascade_recall": cfg.cascade_recall,
                   "cascade_precision": cfg.cascade_precision, "folds": cfg.folds,
                   "engine": args.engine, "ov_device": args.ov_device, "threads": args.threads,
                   "batch": args.batch, "probe_c": args.probe_c},
        "dataset": {
            "path": str(ds.root), "label_source": args.labels,
            "n": len(cands), "pos_rate": float(y.mean()),
            "n_frames": len(set(groups.tolist())),
            "n_gold": sum(1 for c in cands if c.gold is not None),
            "cands_per_frame": {"p50": float(np.percentile(counts, 50)),
                                "p95": float(np.percentile(counts, 95)),
                                "max": float(counts.max())},
            "remote": ({"detector_s_per_frame": rem["detector_s"] / nfr,
                        "vlm_s_per_frame": rem["vlm_s"] / nfr} if rem.get("frames") else {}),
        },
        "backends": backends_info,
        "runs": runs,
    }

    (out_dir / "results.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        labels = list(score_cols)
        w.writerow(["id", "frame", "query_id", "label", "min_side", "det_conf", "reference"] + labels)
        for i, c in enumerate(cands):
            w.writerow([c.id, c.frame, c.query_id, c.label, c.min_side, f"{c.det_conf:.4f}", int(y[i])]
                       + [("" if np.isnan(score_cols[l][i]) else f"{score_cols[l][i]:.5f}") for l in labels])
    (out_dir / "report.html").write_text(render(res, gallery), encoding="utf-8")
    print(f"\nwrote {out_dir / 'report.html'}")
    return res


if __name__ == "__main__":
    main()
