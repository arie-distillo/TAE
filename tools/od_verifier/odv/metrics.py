"""
Quality metrics for a verifier's scores against reference labels.

Everything threshold-dependent is measured OUT OF FOLD: thresholds are chosen
on some frames and applied to others (folds are grouped by source frame, per
query), so the reported operating points are what a deployed threshold would
give, not a best case fitted to the same data.

Two operating modes are reported:

  single threshold   keep everything the reference keeps with recall ≥ target;
                     quality = how many reference rejections are also rejected
                     (specificity = the verifier's filtering power).

  cascade            two thresholds.  score ≥ hi → accept locally,
                     score < lo → reject locally, in between → escalate to the
                     remote VLM.  Reported: share escalated (= remote calls
                     still needed), precision of local accepts, share of true
                     objects lost by local rejects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE_BUCKETS = [(0, 16, "<16 px"), (16, 32, "16–32 px"), (32, 64, "32–64 px"), (64, 10**9, "≥64 px")]


@dataclass
class EvalConfig:
    recall_target:     float = 0.95   # single-threshold mode
    cascade_recall:    float = 0.98   # lo: at most (1 − this) of true objects fall below it
    cascade_precision: float = 0.95   # hi: local accepts are at least this precise
    folds:             int   = 5


# ── basic scores ─────────────────────────────────────────────────────────────

def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(y) == 0 or y.max() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def thr_for_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Largest threshold t such that recall of (s ≥ t) is at least `target`."""
    pos = np.sort(s[y == 1])
    if len(pos) == 0:
        return float("-inf")
    k = int(np.floor((1.0 - target) * len(pos) + 1e-9))      # positives allowed below t
    return float(pos[min(k, len(pos) - 1)])


def thr_for_precision(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Smallest threshold t such that precision of (s ≥ t) is at least `target` (+inf if none)."""
    if len(y) == 0:
        return float("inf")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    # only cut between distinct score values
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]
    ok = np.where((prec >= target) & last_of_tie)[0]
    return float(ss[ok[-1]]) if len(ok) else float("inf")


def group_folds(groups: np.ndarray, k: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
    """Grouped K-fold index pairs. Returns (folds, in_sample) — in_sample when <2 groups."""
    from sklearn.model_selection import GroupKFold
    n_groups = len(set(groups.tolist()))
    idx = np.arange(len(groups))
    if n_groups < 2:
        return [(idx, idx)], True
    gkf = GroupKFold(n_splits=min(k, n_groups))
    return [(tr, te) for tr, te in gkf.split(idx, groups=groups)], False


# ── run evaluation ───────────────────────────────────────────────────────────

def evaluate_run(y: np.ndarray, s: np.ndarray, groups: np.ndarray, qids: np.ndarray,
                 sides: np.ndarray, cfg: EvalConfig) -> dict:
    n = len(y)
    pred = np.full(n, -1, np.int8)        # single threshold: 1 keep / 0 reject / −1 unscored
    zone = np.full(n, -1, np.int8)        # cascade: 2 accept / 1 escalate / 0 reject
    in_sample = False
    by_query = []

    valid = ~np.isnan(s)
    for q in sorted(set(qids.tolist())):
        qi = np.where((qids == q) & valid)[0]
        yq, sq = y[qi], s[qi]
        row = {"query_id": q, "n": int(len(qi)), "n_pos": int(yq.sum()),
               "auc": auc(yq, sq), "ap": average_precision(yq, sq)}
        by_query.append(row)
        if len(qi) == 0 or yq.min() == yq.max():
            continue                                         # nothing to separate
        folds, ins = group_folds(groups[qi], cfg.folds)
        in_sample |= ins
        for tr, te in folds:
            ytr, strn = yq[tr], sq[tr]
            if ytr.min() == ytr.max():
                continue
            t = thr_for_recall(ytr, strn, cfg.recall_target)
            lo = thr_for_recall(ytr, strn, cfg.cascade_recall)
            hi = max(thr_for_precision(ytr, strn, cfg.cascade_precision), lo)
            pred[qi[te]] = (sq[te] >= t).astype(np.int8)
            zone[qi[te]] = np.where(sq[te] >= hi, 2, np.where(sq[te] >= lo, 1, 0)).astype(np.int8)

    def _op(mask: np.ndarray) -> dict:
        m = mask & (pred >= 0)
        yy, pp, zz = y[m], pred[m], zone[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
        tn = int(((pp == 0) & (yy == 0)).sum())
        acc = zz == 2; rej = zz == 0; esc = zz == 1
        return {
            "n": int(m.sum()), "n_pos": npos,
            "recall":      tp / npos if npos else float("nan"),
            "precision":   tp / (tp + fp) if (tp + fp) else float("nan"),
            "specificity": tn / nneg if nneg else float("nan"),
            "escalate":    float(esc.mean()) if m.sum() else float("nan"),
            "auto_accept": float(acc.mean()) if m.sum() else float("nan"),
            "auto_reject": float(rej.mean()) if m.sum() else float("nan"),
            "accept_precision": float(yy[acc].mean()) if acc.sum() else float("nan"),
            "pos_lost":    float((rej & (yy == 1)).sum() / npos) if npos else float("nan"),
        }

    overall = _op(np.ones(n, bool))
    weights = np.array([r["n"] for r in by_query if not np.isnan(r["auc"])], float)
    aucs = np.array([r["auc"] for r in by_query if not np.isnan(r["auc"])], float)
    aps = np.array([r["ap"] for r in by_query if not np.isnan(r["auc"])], float)

    by_size = []
    for lo_px, hi_px, name in SIZE_BUCKETS:
        m = (sides >= lo_px) & (sides < hi_px)
        if m.sum():
            by_size.append({"bucket": name, **_op(m)})

    # PR curve for the report: the largest separable query
    pr = []
    cand_q = [r for r in by_query if not np.isnan(r["auc"])]
    pr_query = ""
    if cand_q:
        pr_query = max(cand_q, key=lambda r: r["n"])["query_id"]
        qi = np.where((qids == pr_query) & valid)[0]
        pr = pr_curve(y[qi], s[qi])

    return {
        "n": int(valid.sum()), "n_pos": int(y[valid].sum()),
        "auc": float(np.average(aucs, weights=weights)) if len(aucs) else float("nan"),
        "ap":  float(np.average(aps, weights=weights)) if len(aps) else float("nan"),
        "op": overall, "by_size": by_size, "by_query": by_query,
        "in_sample": bool(in_sample), "pr": pr, "pr_query": pr_query,
        "_pred": pred, "_zone": zone,
    }


def pr_curve(y: np.ndarray, s: np.ndarray, max_points: int = 120) -> list[list[float]]:
    if len(y) == 0 or y.max() == 0:
        return []
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / ys.sum()
    step = max(1, len(ys) // max_points)
    pts = [[float(rec[i]), float(prec[i])] for i in range(0, len(ys), step)]
    pts.append([float(rec[-1]), float(prec[-1])])
    return pts


# ── distilled probe ──────────────────────────────────────────────────────────

def _probe_fit_predict(Xtr, ytr, Xte, etr, ete, C: float) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    if etr is not None:
        mu, sd = etr.mean(axis=0), etr.std(axis=0) + 1e-6
        Xtr = np.hstack([Xtr, (etr - mu) / sd])
        Xte = np.hstack([Xte, (ete - mu) / sd])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1].astype(np.float32)


def probe_oof(emb: np.ndarray, y: np.ndarray, groups: np.ndarray, qids: np.ndarray,
              extras: np.ndarray | None, folds: int = 5, C: float = 1.0,
              min_train: int = 20) -> np.ndarray:
    """
    Out-of-fold scores of a per-query logistic-regression head on frozen
    embeddings, trained on the reference labels (= distillation of the remote
    VLM into a head that costs one dot product at inference).
    NaN where no prediction could be made (too little data for that query).
    """
    out = np.full(len(y), np.nan, np.float32)
    for q in sorted(set(qids.tolist())):
        qi = np.where(qids == q)[0]
        if len(qi) < min_train or y[qi].min() == y[qi].max():
            continue
        fl, in_sample = group_folds(groups[qi], folds)
        if in_sample:
            continue                                          # cannot hold frames out
        for tr, te in fl:
            if len(tr) < min_train or y[qi[tr]].min() == y[qi[tr]].max():
                continue
            out[qi[te]] = _probe_fit_predict(
                emb[qi[tr]], y[qi[tr]], emb[qi[te]],
                None if extras is None else extras[qi[tr]],
                None if extras is None else extras[qi[te]], C)
    return out


def probe_learning_curve(emb: np.ndarray, y: np.ndarray, qids: np.ndarray,
                         extras: np.ndarray | None, C: float = 1.0,
                         sizes: tuple = (25, 50, 100, 200, 400, 800, 1600)) -> list[dict]:
    """
    Train on the FIRST n candidates of a query (recording order = mission
    order), test on all later ones.  Answers: how many remote-VLM verdicts
    does a mission need before the local head can take over?
    """
    rows = []
    for q in sorted(set(qids.tolist())):
        qi = np.where(qids == q)[0]
        for n in sizes:
            if n > len(qi) - 20:
                break
            tr, te = qi[:n], qi[n:]
            if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
                continue
            s = _probe_fit_predict(emb[tr], y[tr], emb[te],
                                   None if extras is None else extras[tr],
                                   None if extras is None else extras[te], C)
            rows.append({"query_id": q, "n_train": int(n), "n_test": int(len(te)),
                         "auc": auc(y[te], s)})
    return rows


# ── latency ──────────────────────────────────────────────────────────────────

def latency_summary(single_ms: list[float], batch_total_ms: float, n: int,
                    warmup: int = 3) -> dict:
    a = np.asarray(single_ms[warmup:] if len(single_ms) > warmup + 2 else single_ms, float)
    return {
        "single_p50_ms": float(np.percentile(a, 50)) if len(a) else float("nan"),
        "single_p95_ms": float(np.percentile(a, 95)) if len(a) else float("nan"),
        "batched_ms_per_crop": batch_total_ms / n if n else float("nan"),
    }
