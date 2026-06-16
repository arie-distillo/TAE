"""
core/confidence_stats.py — Pipeline confidence statistics accumulator
======================================================================
Collects GDINO and VLM confidence metrics at each pipeline stage and
writes them to DATA_DIR/<mission>/detections/ as:

  confidence_stats.json       — all runs this session (append mode)
  confidence_stats_<run>.md   — human-readable report per run

Usage (called from run_detection_pipeline in detection_pipeline.py):

    from core import confidence_stats as _cstats
    _cstats.start_run(original_query, params.object_confidence)
    ...
    _cstats.record_gdino_detections(raw_dets)
    _cstats.record_nms_result(raw_dets, nms_kept_dets)
    _cstats.record_size_filter(nms_kept_dets, size_filtered_dets)
    _cstats.record_vlm_result(pre_vlm_candidates, vlm_confirmed)
    _cstats.finalize_run(paths.detections)
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("TAE.ConfidenceStats")

_lock    = threading.Lock()
_session: dict | None = None   # active pipeline run accumulator


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_stats(values: list[float]) -> dict:
    """Return descriptive stats dict for a list of floats."""
    if not values:
        return {"count": 0}
    srt = sorted(values)
    n   = len(srt)

    def pct(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        return round(srt[lo] + (srt[hi] - srt[lo]) * (idx - lo), 4)

    return {
        "count": n,
        "min":   round(min(srt), 4),
        "p25":   pct(25),
        "p50":   pct(50),
        "mean":  round(sum(srt) / n, 4),
        "p75":   pct(75),
        "p90":   pct(90),
        "max":   round(max(srt), 4),
    }


def _fmt_stats(d: dict) -> str:
    """Render a stats dict as a compact inline Markdown string."""
    if not d or d.get("count", 0) == 0:
        return "_no data_"
    return (
        f"n={d['count']}  "
        f"min={d['min']:.3f}  "
        f"p25={d['p25']:.3f}  "
        f"p50={d['p50']:.3f}  "
        f"mean={d['mean']:.3f}  "
        f"p75={d['p75']:.3f}  "
        f"p90={d['p90']:.3f}  "
        f"max={d['max']:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called from detection_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def _new_session_dict(run_id: str, query: str, gdino_threshold: float) -> dict:
    """Create a blank session accumulator dict."""
    return {
        "run_id":          run_id,
        "query":           query,
        "ts":              datetime.now(timezone.utc).isoformat(),
        "gdino_threshold": gdino_threshold,
        # raw tuples (confidence, label) from all GDINO detections
        "_gdino_raw":      [],
        # confidence of NMS-kept and NMS-suppressed
        "_nms_kept":       [],
        "_nms_suppressed": [],
        # size-filter reduction counts (accumulated with += across streaming frames)
        "_size_pre":       0,
        "_size_post":      0,
        "_vlm_c_gdino":    [],   # confirmed GDINO conf
        "_vlm_c_vlm":      [],   # confirmed VLM conf
        "_vlm_r_gdino":    [],   # rejected  GDINO conf
        "_vlm_r_vlm":      [],   # rejected  VLM conf
    }


def start_run(query: str, gdino_threshold: float) -> str:
    """
    Open a new accumulator for one pipeline run.
    Returns the run_id (used in the Markdown filename).
    Must be called before any record_* function.
    """
    global _session
    run_id = uuid.uuid4().hex[:8]
    with _lock:
        _session = _new_session_dict(run_id, query, gdino_threshold)
    logger.debug("ConfidenceStats: new run %s | query=%r", run_id, query)
    return run_id


def open_stream_run(query: str, gdino_threshold: float) -> None:
    """
    Open a streaming stats session, creating one only if none is currently active.

    Call on every streaming frame (foreground and background). The first call
    creates the session; subsequent calls are no-ops, so stats accumulate
    across all frames of the stream. Call finalize_run() from stream_stop()
    to close the session and write the reports.
    """
    global _session
    with _lock:
        if _session is None:
            run_id   = uuid.uuid4().hex[:8]
            _session = _new_session_dict(run_id, query, gdino_threshold)
            logger.debug("ConfidenceStats: stream session %s started | query=%r",
                         run_id, query)


def record_gdino_detections(detections) -> None:
    """
    Record raw GDINO detections (before NMS).
    Pass the full list returned by run_detector_stage().
    """
    with _lock:
        if _session is None:
            return
        for det in detections:
            _session["_gdino_raw"].append((det.confidence, det.label))


def record_nms_result(pre_nms: list, post_nms: list) -> None:
    """
    Record NMS impact.

    pre_nms  — raw GDINO detections before cross_tile_nms()
    post_nms — kept detections returned by cross_tile_nms()

    NMS selects the highest-confidence box per overlapping group, so
    suppressed detections are almost always lower-confidence duplicates.
    Comparing kept vs suppressed confidence distributions confirms this.
    """
    with _lock:
        if _session is None:
            return
        kept_ids = {id(d) for d in post_nms}
        for d in pre_nms:
            if id(d) in kept_ids:
                _session["_nms_kept"].append(d.confidence)
            else:
                _session["_nms_suppressed"].append(d.confidence)


def record_size_filter(pre: list, post: list) -> None:
    """
    Record the bbox-size filter reduction (after NMS, before SAM/VLM).
    pre  — candidates before size filter
    post — candidates after size filter

    Uses += so that multiple calls (one per streaming frame) accumulate
    correctly. Single-run (batch) calls also work because the session
    initialises both counters to 0.
    """
    with _lock:
        if _session is None:
            return
        _session["_size_pre"]  += len(pre)
        _session["_size_post"] += len(post)


def record_vlm_result(candidates: list, confirmed: list) -> None:
    """
    Record VLM verification outcome.

    candidates — list entering vlm_verify_stage() (after SAM refine)
    confirmed  — list returned by vlm_verify_stage()

    Identity comparison (id()) identifies which candidates were rejected.
    Both GDINO confidence (det.confidence) and VLM confidence
    (det.vlm_confidence) are recorded separately for confirmed vs rejected.
    """
    with _lock:
        if _session is None:
            return
        c_ids = {id(d) for d in confirmed}
        for det in candidates:
            if id(det) in c_ids:
                _session["_vlm_c_gdino"].append(det.confidence)
                _session["_vlm_c_vlm"].append(getattr(det, "vlm_confidence", 0.0))
            else:
                _session["_vlm_r_gdino"].append(det.confidence)
                _session["_vlm_r_vlm"].append(getattr(det, "vlm_confidence", 0.0))


def finalize_run(output_dir: Path | str) -> None:
    """
    Build the stats dict, append to confidence_stats.json, write per-run .md + histogram PNG.
    Clears the session accumulator on exit.
    """
    global _session
    with _lock:
        if _session is None:
            return
        sess     = _session
        _session = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Capture raw lists before building summary stats ───────────────────────
    # These are needed by the histogram renderer and not stored in the JSON.
    raw = {
        "gdino_all":      [c for c, _ in sess["_gdino_raw"]],
        "nms_kept":       list(sess["_nms_kept"]),
        "nms_suppressed": list(sess["_nms_suppressed"]),
        "vlm_c_gdino":    list(sess["_vlm_c_gdino"]),
        "vlm_c_vlm":      list(sess["_vlm_c_vlm"]),
        "vlm_r_gdino":    list(sess["_vlm_r_gdino"]),
        "vlm_r_vlm":      list(sess["_vlm_r_vlm"]),
    }

    # ── Build structured stats ────────────────────────────────────────────────
    gdino_confs = raw["gdino_all"]
    by_label: dict[str, list[float]] = {}
    for c, lbl in sess["_gdino_raw"]:
        by_label.setdefault(lbl, []).append(c)

    total_vlm = len(sess["_vlm_c_gdino"]) + len(sess["_vlm_r_gdino"])
    rej_count = len(sess["_vlm_r_gdino"])
    rej_rate  = round(rej_count / total_vlm, 4) if total_vlm > 0 else 0.0

    stats = {
        "run_id":          sess["run_id"],
        "query":           sess["query"],
        "ts":              sess["ts"],
        "gdino_threshold": sess["gdino_threshold"],

        "gdino_stage": {
            "total":                len(gdino_confs),
            "per_label":            {lbl: len(v) for lbl, v in by_label.items()},
            "confidence":           _pct_stats(gdino_confs),
            "per_label_confidence": {lbl: _pct_stats(v) for lbl, v in by_label.items()},
        },

        "nms_stage": {
            "pre_nms_count":         len(sess["_nms_kept"]) + len(sess["_nms_suppressed"]),
            "post_nms_count":        len(sess["_nms_kept"]),
            "suppressed_count":      len(sess["_nms_suppressed"]),
            "iou_threshold":         0.50,
            "kept_confidence":       _pct_stats(sess["_nms_kept"]),
            "suppressed_confidence": _pct_stats(sess["_nms_suppressed"]),
        },

        "size_filter_stage": {
            "pre_count":     sess["_size_pre"],
            "post_count":    sess["_size_post"],
            "dropped_count": sess["_size_pre"] - sess["_size_post"],
            "min_bbox_px":   _get_min_bbox_px(),
        },

        "vlm_stage": {
            "total_candidates": total_vlm,
            "confirmed_count":  len(sess["_vlm_c_gdino"]),
            "rejected_count":   rej_count,
            "rejection_rate":   rej_rate,
            "confirmed": {
                "gdino_confidence": _pct_stats(sess["_vlm_c_gdino"]),
                "vlm_confidence":   _pct_stats(sess["_vlm_c_vlm"]),
            },
            "rejected": {
                "gdino_confidence": _pct_stats(sess["_vlm_r_gdino"]),
                "vlm_confidence":   _pct_stats(sess["_vlm_r_vlm"]),
            },
        },
    }

    # ── Generate histogram PNG ────────────────────────────────────────────────
    hist_fname = _render_histograms(stats, raw, output_dir)

    # ── Append to session JSON ────────────────────────────────────────────────
    json_path = output_dir / "confidence_stats.json"
    runs: list = []
    if json_path.exists():
        try:
            runs = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            runs = []
    if not isinstance(runs, list):
        runs = []
    runs.append(stats)
    try:
        json_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        logger.info("ConfidenceStats JSON → %s (%d run(s) total)", json_path, len(runs))
    except Exception as exc:
        logger.warning("ConfidenceStats: could not write JSON: %s", exc)

    # ── Write per-run Markdown ────────────────────────────────────────────────
    md_path = output_dir / f"confidence_stats_{sess['run_id']}.md"
    try:
        md_path.write_text(_render_md(stats, hist_fname), encoding="utf-8")
        logger.info("ConfidenceStats Markdown → %s", md_path)
    except Exception as exc:
        logger.warning("ConfidenceStats: could not write Markdown: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_min_bbox_px() -> int:
    try:
        from config import Settings
        return Settings().DETECTION_MIN_BBOX_PX
    except Exception:
        return 0


def _render_histograms(
    stats:      dict,
    raw:        dict,
    output_dir: Path,
) -> str | None:
    """
    Generate a two-panel confidence histogram PNG and save it to output_dir.

    Left panel  — GDINO confidence: confirmed (green) vs rejected (red),
                  with a vertical dashed line at the detection threshold.
    Right panel — VLM confidence:   confirmed (green) vs rejected (red).

    Returns the filename (relative, for embedding in Markdown), or None if
    matplotlib is not available or there is no data to plot.
    """
    if not raw["vlm_c_gdino"] and not raw["vlm_r_gdino"]:
        return None   # nothing to plot

    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive; must be set before pyplot import
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        logger.warning("ConfidenceStats: matplotlib not available — histogram skipped")
        return None

    CONFIRMED_COLOR = "#4ade80"   # green — matches TAE map markers
    REJECTED_COLOR  = "#f87171"   # red
    THRESHOLD_COLOR = "#facc15"   # yellow
    BINS = np.linspace(0.0, 1.0, 26)   # 25 bins of width 0.04

    fig, (ax_gdino, ax_vlm) = plt.subplots(
        1, 2,
        figsize     = (11, 4),
        facecolor   = "#1a1b26",   # dark background consistent with TAE UI
        tight_layout= True,
    )
    fig.suptitle(
        f"Confidence Distributions — Run {stats['run_id']}  |  {stats['query'][:60]}",
        fontsize  = 10,
        color     = "#c0caf5",
        y         = 1.01,
    )

    def _style_ax(ax, title: str) -> None:
        ax.set_facecolor("#16161e")
        ax.set_title(title, color="#c0caf5", fontsize=10, pad=6)
        ax.set_xlabel("Confidence", color="#565f89", fontsize=9)
        ax.set_ylabel("Count",      color="#565f89", fontsize=9)
        ax.tick_params(colors="#565f89", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2b3d")
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", color="#2a2b3d", linewidth=0.6, linestyle="--")

    # ── Left: GDINO confidence ────────────────────────────────────────────────
    _style_ax(ax_gdino, "GDINO Confidence (entering VLM)")

    if raw["vlm_c_gdino"]:
        ax_gdino.hist(
            raw["vlm_c_gdino"], bins=BINS,
            color=CONFIRMED_COLOR, alpha=0.75, edgecolor="#111",  linewidth=0.4,
            label=f"Confirmed  n={len(raw['vlm_c_gdino'])}",
        )
    if raw["vlm_r_gdino"]:
        ax_gdino.hist(
            raw["vlm_r_gdino"], bins=BINS,
            color=REJECTED_COLOR,  alpha=0.65, edgecolor="#111",  linewidth=0.4,
            label=f"Rejected   n={len(raw['vlm_r_gdino'])}",
        )
    ax_gdino.axvline(
        stats["gdino_threshold"],
        color=THRESHOLD_COLOR, linewidth=1.5, linestyle="--",
        label=f"Threshold  {stats['gdino_threshold']:.3f}",
    )
    ax_gdino.legend(
        fontsize=8, framealpha=0.25,
        labelcolor="#c0caf5", facecolor="#1a1b26", edgecolor="#2a2b3d",
    )

    # ── Right: VLM confidence ─────────────────────────────────────────────────
    _style_ax(ax_vlm, "VLM Confidence")

    if raw["vlm_c_vlm"]:
        ax_vlm.hist(
            raw["vlm_c_vlm"], bins=BINS,
            color=CONFIRMED_COLOR, alpha=0.75, edgecolor="#111", linewidth=0.4,
            label=f"Confirmed  n={len(raw['vlm_c_vlm'])}",
        )
    if raw["vlm_r_vlm"]:
        ax_vlm.hist(
            raw["vlm_r_vlm"], bins=BINS,
            color=REJECTED_COLOR,  alpha=0.65, edgecolor="#111", linewidth=0.4,
            label=f"Rejected   n={len(raw['vlm_r_vlm'])}",
        )
    ax_vlm.legend(
        fontsize=8, framealpha=0.25,
        labelcolor="#c0caf5", facecolor="#1a1b26", edgecolor="#2a2b3d",
    )

    fname = f"confidence_hist_{stats['run_id']}.png"
    try:
        fig.savefig(
            str(output_dir / fname),
            dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        logger.info("ConfidenceStats histogram → %s", output_dir / fname)
    except Exception as exc:
        logger.warning("ConfidenceStats: could not save histogram: %s", exc)
        fname = None
    finally:
        plt.close(fig)

    return fname


def _render_md(s: dict, hist_fname: str | None = None) -> str:
    """Render a stats dict as a human-readable Markdown report."""
    g  = s["gdino_stage"]
    n  = s["nms_stage"]
    sf = s["size_filter_stage"]
    v  = s["vlm_stage"]

    lines = [
        f"# TAE Confidence Stats — Run `{s['run_id']}`",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Query | `{s['query']}` |",
        f"| Timestamp | {s['ts']} |",
        f"| GDINO threshold | **{s['gdino_threshold']}** |",
        "",
        "---",
        "",
        "## 1  GDINO Detection Stage",
        "",
        f"Total raw detections: **{g['total']}**",
        "",
        "### 1.1  Per-label counts",
        "",
        "| Label | Count |",
        "|-------|------:|",
    ]
    for lbl, cnt in sorted(g["per_label"].items()):
        lines.append(f"| {lbl} | {cnt} |")

    lines += [
        "",
        "### 1.2  Overall confidence distribution",
        "",
        f"`{_fmt_stats(g['confidence'])}`",
        "",
        "### 1.3  Per-label confidence",
        "",
        "| Label | Distribution |",
        "|-------|-------------|",
    ]
    for lbl, st in sorted(g["per_label_confidence"].items()):
        lines.append(f"| {lbl} | {_fmt_stats(st)} |")

    lines += [
        "",
        "---",
        "",
        "## 2  NMS Stage",
        "",
        f"IoU threshold: **{n['iou_threshold']}**  ",
        f"Pre-NMS: **{n['pre_nms_count']}** → Post-NMS: **{n['post_nms_count']}**"
        f" (suppressed: **{n['suppressed_count']}**)",
        "",
        "| Group | Confidence distribution |",
        "|-------|------------------------|",
        f"| Kept       | {_fmt_stats(n['kept_confidence'])} |",
        f"| Suppressed | {_fmt_stats(n['suppressed_confidence'])} |",
        "",
        "> **Interpretation:** NMS keeps the highest-confidence box per overlapping group,",
        "> so suppressed detections should have lower confidence than kept ones.",
        "> A large mean gap (kept − suppressed) confirms NMS is selecting well.",
        "> A small or negative gap suggests overlapping, non-duplicate objects —",
        "> consider lowering the IoU threshold.",
        "",
        "---",
        "",
        "## 2a  Bbox Size Filter",
        "",
        f"Min side length: **{sf['min_bbox_px']} px**  ",
        f"After NMS: **{sf['pre_count']}** → After size filter: **{sf['post_count']}**"
        f" (dropped: **{sf['dropped_count']}**)",
        "",
        "---",
        "",
        "## 3  VLM Verification Stage",
        "",
        f"Candidates in: **{v['total_candidates']}**  |  "
        f"Confirmed: **{v['confirmed_count']}**  |  "
        f"Rejected: **{v['rejected_count']}**  |  "
        f"Rejection rate: **{v['rejection_rate']:.1%}**",
        "",
        "### 3.1  GDINO confidence (entering VLM)",
        "",
        "| Outcome | Distribution |",
        "|---------|-------------|",
        f"| Confirmed | {_fmt_stats(v['confirmed']['gdino_confidence'])} |",
        f"| Rejected  | {_fmt_stats(v['rejected']['gdino_confidence'])} |",
        "",
        "> If confirmed GDINO confidence is consistently higher than rejected,",
        "> the GDINO threshold is roughly calibrated. If they overlap heavily,",
        "> the threshold may need raising to pre-filter obvious noise.",
        "",
        "### 3.2  VLM confidence",
        "",
        "| Outcome | Distribution |",
        "|---------|-------------|",
        f"| Confirmed | {_fmt_stats(v['confirmed']['vlm_confidence'])} |",
        f"| Rejected  | {_fmt_stats(v['rejected']['vlm_confidence'])} |",
        "",
        "> VLM confidence for confirmed detections should cluster near 1.0.",
        "> Confirmed with low VLM confidence are worth manual review.",
        "> Rejected with high VLM confidence may indicate the rejection criteria",
        "> are too strict, or that a genuine object was missed.",
    ]

    if hist_fname:
        lines += [
            "",
            "---",
            "",
            "## 4  Confidence Histograms",
            "",
            f"![Confidence histograms]({hist_fname})",
        ]

    return "\n".join(lines) + "\n"