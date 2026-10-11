"""Self-contained HTML report for an evaluation run."""
from __future__ import annotations

import base64
import html
import math

import cv2
import numpy as np

_CSS = """
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--line:#dddcd6;--card:#fff;--acc:#1f6f5c;--bad:#b3432b;--warn:#9a6b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9c9b94;--line:#33332f;--card:#1e1e1c;--acc:#5fc2a8;--bad:#e8876f;--warn:#e0b352}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 8px}
p.sub{color:var(--mut);margin:0 0 14px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.fact{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.fact b{display:block;font-size:18px}.fact span{color:var(--mut);font-size:12px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600;background:var(--card);position:sticky;top:0}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}
tr.base td{color:var(--mut)}tr.best td{background:color-mix(in srgb,var(--acc) 10%,transparent)}
.note{color:var(--mut);font-size:12.5px;margin:8px 0 0}
.warn{color:var(--warn)}.good{color:var(--acc)}.bad{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(124px,1fr));gap:8px}
.th{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px;font-size:11px;color:var(--mut)}
.th img{width:100%;display:block;border-radius:3px;image-rendering:auto}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
svg{max-width:100%;height:auto}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 4px}
"""


def _f(x, d=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return f"{100 * x:.{max(d - 2, 0)}f}%" if pct else f"{x:.{d}f}"


def _ms(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def thumb_b64(img: np.ndarray, size: int = 112) -> str:
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _pr_svg(runs: list[dict]) -> str:
    W, H, L, B = 420, 300, 44, 34
    pw, ph = W - L - 12, H - B - 12
    colors = ["#1f6f5c", "#b3432b", "#3b62b8", "#9a6b00", "#7a4aa8", "#4d8a2a"]
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-label="Precision-recall curves">']
    for i in range(6):
        v = i / 5
        x = L + v * pw; y = 12 + (1 - v) * ph
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="currentColor" opacity=".12"/>')
        parts.append(f'<text x="{L - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="currentColor" opacity=".6">{v:.1f}</text>')
        parts.append(f'<text x="{x:.1f}" y="{H - B + 14}" font-size="10" text-anchor="middle" fill="currentColor" opacity=".6">{v:.1f}</text>')
    parts.append(f'<text x="{L + pw / 2}" y="{H - 4}" font-size="11" text-anchor="middle" fill="currentColor">recall</text>')
    parts.append(f'<text x="12" y="{12 + ph / 2}" font-size="11" text-anchor="middle" fill="currentColor" transform="rotate(-90 12 {12 + ph / 2})">precision</text>')
    legend = []
    for k, r in enumerate(runs):
        pts = r.get("pr") or []
        if not pts:
            continue
        c = colors[k % len(colors)]
        d = " ".join(f"{L + p[0] * pw:.1f},{12 + (1 - p[1]) * ph:.1f}" for p in pts)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="1.8"/>')
        legend.append(f'<span style="color:{c}">■</span> {html.escape(r["label"])}')
    parts.append("</svg>")
    return "".join(parts) + '<p class="note">' + " &nbsp; ".join(legend) + "</p>"


def render(res: dict, gallery: dict | None = None) -> str:
    runs = res["runs"]
    cfg = res["config"]
    ds = res["dataset"]
    real = [r for r in runs if not r.get("baseline")]
    best = max(real, key=lambda r: (-1 if math.isnan(r["auc"]) else r["auc"]), default=None)
    load = ds.get("cands_per_frame", {})

    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Local verifier evaluation</title><style>{_CSS}</style></head><body><main>"]
    h.append("<h1>Local verifier evaluation</h1>")
    h.append(f"<p class='sub'>Dataset <code>{html.escape(ds['path'])}</code> · reference labels: "
             f"<b>{html.escape(ds['label_source'])}</b> · {html.escape(res['machine'])}</p>")

    h.append("<div class='facts'>")
    facts = [
        (ds["n"], "labelled candidates"),
        (_f(ds["pos_rate"], 3, True), "confirmed by reference"),
        (ds["n_frames"], "frames"),
        (ds["n_gold"], "human-reviewed"),
        (f"{load.get('p50', 0):.0f} / {load.get('p95', 0):.0f}", "candidates per frame (p50 / p95)"),
    ]
    rem = ds.get("remote") or {}
    if rem.get("vlm_s_per_frame") is not None:
        facts.append((f"{rem['detector_s_per_frame']:.1f}s + {rem['vlm_s_per_frame']:.1f}s",
                      "recorded remote detector + VLM per frame"))
    for v, k in facts:
        h.append(f"<div class='fact'><b>{v}</b><span>{html.escape(k)}</span></div>")
    h.append("</div>")

    if ds["n_gold"] == 0:
        h.append("<p class='note warn'>No human-reviewed labels: quality below is agreement with the "
                 "remote VLM, which is itself imperfect. Review a sample with <code>review.py</code> "
                 "before drawing conclusions.</p>")
    if any(r.get("in_sample") for r in runs):
        h.append("<p class='note warn'>Some thresholds were fitted and applied on the same frames "
                 "(fewer than 2 frames for a query) — those operating points are optimistic.</p>")

    # ── summary table ────────────────────────────────────────────────────────
    h.append("<h2>Summary</h2><div class='scroll'><table><thead><tr>"
             "<th class='l'>Backend</th><th class='l'>Crop</th><th class='l'>Scorer</th>"
             "<th>n</th><th>AUC</th><th>AP</th>"
             f"<th>Recall</th><th>Rejects removed</th>"
             "<th>Escalated</th><th>Accept precision</th><th>Objects lost</th>"
             "<th>ms/crop b=1 p50</th><th>p95</th><th>batched</th>"
             "<th>s/frame p50</th><th>p95</th></tr></thead><tbody>")
    for r in runs:
        op = r["op"]; lat = r.get("latency") or {}
        cls = "base" if r.get("baseline") else ("best" if r is best else "")
        b = lat.get("batched_ms_per_crop")
        sf50 = b * load.get("p50", 0) / 1000 if b is not None and not math.isnan(b) else None
        sf95 = b * load.get("p95", 0) / 1000 if b is not None and not math.isnan(b) else None
        h.append(
            f"<tr class='{cls}'><td class='l'>{html.escape(r['backend'])}</td>"
            f"<td class='l'>{html.escape(r['crop'])}</td><td class='l'>{html.escape(r['scorer'])}</td>"
            f"<td>{r['n']}</td><td>{_f(r['auc'])}</td><td>{_f(r['ap'])}</td>"
            f"<td>{_f(op['recall'], 3, True)}</td><td>{_f(op['specificity'], 3, True)}</td>"
            f"<td>{_f(op['escalate'], 3, True)}</td><td>{_f(op['accept_precision'], 3, True)}</td>"
            f"<td>{_f(op['pos_lost'], 3, True)}</td>"
            f"<td>{_ms(lat.get('single_p50_ms'))}</td><td>{_ms(lat.get('single_p95_ms'))}</td>"
            f"<td>{_ms(b)}</td><td>{_f(sf50, 2)}</td><td>{_f(sf95, 2)}</td></tr>")
    h.append("</tbody></table></div>")
    h.append(
        f"<p class='note'><b>Recall / Rejects removed</b>: single threshold tuned for recall ≥ "
        f"{cfg['recall_target']:.0%} on other frames — share of reference-confirmed objects kept, and "
        f"share of reference-rejected candidates the local verifier also rejects. "
        f"<b>Escalated / Accept precision / Objects lost</b>: two-threshold cascade "
        f"(local accepts ≥ {cfg['cascade_precision']:.0%} precise, local rejects lose ≤ "
        f"{1 - cfg['cascade_recall']:.0%} of objects on the tuning frames); escalated = share still "
        f"sent to the remote VLM. <b>s/frame</b>: batched ms/crop × candidates per frame. "
        f"Grey rows are baselines that need no verifier model.</p>")

    # ── per-run detail ───────────────────────────────────────────────────────
    h.append("<h2>Precision–recall and object size</h2><div class='cols'><div>")
    top = sorted(real, key=lambda r: (-1 if math.isnan(r["auc"]) else r["auc"]), reverse=True)[:5]
    base = [r for r in runs if r.get("baseline")][:1]
    h.append(_pr_svg(top + base))
    if top and top[0].get("pr_query"):
        h.append(f"<p class='note'>Top runs by AUC, query <code>{html.escape(top[0]['pr_query'])}</code>.</p>")
    h.append("</div><div>")
    if best:
        h.append(f"<p class='note'>By detector box size (shorter side), for "
                 f"<b>{html.escape(best['label'])}</b>:</p>")
        h.append("<div class='scroll'><table><thead><tr><th class='l'>Box</th><th>n</th><th>objects</th>"
                 "<th>Recall</th><th>Rejects removed</th><th>Escalated</th><th>Objects lost</th>"
                 "</tr></thead><tbody>")
        for s in best["by_size"]:
            h.append(f"<tr><td class='l'>{s['bucket']}</td><td>{s['n']}</td><td>{s['n_pos']}</td>"
                     f"<td>{_f(s['recall'], 3, True)}</td><td>{_f(s['specificity'], 3, True)}</td>"
                     f"<td>{_f(s['escalate'], 3, True)}</td><td>{_f(s['pos_lost'], 3, True)}</td></tr>")
        h.append("</tbody></table></div>")
    h.append("</div></div>")

    # ── learning curves ──────────────────────────────────────────────────────
    lc = [(r["label"], r["learning_curve"]) for r in runs if r.get("learning_curve")]
    if lc:
        h.append("<h2>Probe learning curve</h2><p class='note'>Probe trained on the first <i>n</i> "
                 "candidates of a query in recording order, tested on all later ones: how many "
                 "remote verdicts a mission needs before the local head is usable.</p>"
                 "<div class='scroll'><table><thead><tr><th class='l'>Run</th><th class='l'>Query</th>"
                 "<th>n train</th><th>n test</th><th>AUC</th></tr></thead><tbody>")
        for label, rows in lc:
            for row in rows:
                h.append(f"<tr><td class='l'>{html.escape(label)}</td>"
                         f"<td class='l'>{html.escape(row['query_id'])}</td><td>{row['n_train']}</td>"
                         f"<td>{row['n_test']}</td><td>{_f(row['auc'])}</td></tr>")
        h.append("</tbody></table></div>")

    # ── backends ─────────────────────────────────────────────────────────────
    h.append("<h2>Backends</h2><div class='scroll'><table><thead><tr><th class='l'>Backend</th>"
             "<th>Load s</th><th>Process RSS MB</th><th>Text encode ms (once per query)</th>"
             "</tr></thead><tbody>")
    for b in res["backends"]:
        h.append(f"<tr><td class='l'>{html.escape(b['name'])}</td><td>{_f(b['load_s'], 1)}</td>"
                 f"<td>{_ms(b.get('rss_mb'))}</td><td>{_ms(b.get('text_ms'))}</td></tr>")
    h.append("</tbody></table></div>")

    # ── gallery ──────────────────────────────────────────────────────────────
    if gallery:
        h.append(f"<h2>Largest disagreements — {html.escape(gallery['label'])}</h2>")
        for title, key in (("Reference rejected, local score high (would be false accepts)", "false_accepts"),
                           ("Reference confirmed, local score low (would be lost objects)", "misses")):
            items = gallery.get(key) or []
            if not items:
                continue
            h.append(f"<p class='note'>{title}. Check these by eye: some will be reference errors.</p>"
                     "<div class='grid'>")
            for it in items:
                h.append(f"<div class='th'><img alt='' src='data:image/jpeg;base64,{it['img']}'>"
                         f"{html.escape(it['label'])} · {it['side']}px<br>score {it['score']:.3f} · "
                         f"det {it['det_conf']:.2f}</div>")
            h.append("</div>")

    h.append("</main></body></html>")
    return "".join(h)
