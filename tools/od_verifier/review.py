#!/usr/bin/env python
"""
review.py — human review of the reference labels.

The remote VLM's verdicts are a convenient reference but not ground truth.
This writes a single HTML page of candidate crops; click a card to flip its
verdict, then "Export" to download gold.json and merge it back:

    python tools/od_verifier/review.py --data DIR --limit 300
    python tools/od_verifier/review.py --data DIR --apply ~/Downloads/gold.json

With --scores/--run (from evaluate.py) the page shows the candidates where the
chosen local verifier and the reference disagree most — the quickest way to
find out which of the two is wrong.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odv.crops import make_crop                     # noqa: E402
from odv.dataset import Dataset                     # noqa: E402
from odv.report import thumb_b64                    # noqa: E402

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Verifier label review</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--line:#dddcd6;--card:#fff;--yes:#1f6f5c;--no:#b3432b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9c9b94;--line:#33332f;--card:#1e1e1c;--yes:#5fc2a8;--no:#e8876f}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:2}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
main{padding:14px 16px 60px}h2{font-size:15px;margin:22px 0 4px}p{color:var(--mut);margin:0 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.c{background:var(--card);border:3px solid var(--no);border-radius:8px;padding:4px;cursor:pointer;font-size:11.5px;color:var(--mut);user-select:none}
.c.y{border-color:var(--yes)}.c img{width:100%;display:block;border-radius:4px}
.c b{color:var(--no)}.c.y b{color:var(--yes)}.c.ch{outline:2px dashed var(--fg);outline-offset:1px}
</style></head><body>
<header><b>Label review</b><span id="stat"></span>
<button id="exp">Export gold.json</button>
<span style="color:var(--mut)">Click a card to flip it. Every card on this page is exported as reviewed.</span></header>
<main>__BODY__</main>
<script>
const cards=[...document.querySelectorAll('.c')];
function stat(){const ch=cards.filter(c=>c.classList.contains('ch')).length;
 document.getElementById('stat').textContent=cards.length+' candidates, '+ch+' changed';}
cards.forEach(c=>c.addEventListener('click',()=>{c.classList.toggle('y');
 c.classList.toggle('ch',c.classList.contains('y')!==(c.dataset.init==='1'));
 c.querySelector('b').textContent=c.classList.contains('y')?'YES':'NO';stat();}));
document.getElementById('exp').addEventListener('click',()=>{const o={};
 cards.forEach(c=>o[c.dataset.id]=c.classList.contains('y'));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{type:'application/json'}));
 a.download='gold.json';a.click();});
stat();
</script></body></html>"""


def _pick(ds: Dataset, limit: int, scores: str | None, run: str | None, seed: int) -> list:
    cands = [c for c in ds.candidates if c.teacher_valid or c.gold is not None]
    if scores:
        with open(scores, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = [c for c in rd.fieldnames if c not in
                    ("id", "frame", "query_id", "label", "min_side", "det_conf", "reference")]
            col = run or cols[-1]
            if col not in cols:
                raise SystemExit(f"run '{col}' not in {scores}. Available:\n  " + "\n  ".join(cols))
            sc = {r["id"]: float(r[col]) for r in rd if r[col] != ""}
        cands = [c for c in cands if c.id in sc]
        if not cands:
            return []
        s = np.array([sc[c.id] for c in cands])
        rank = (np.argsort(np.argsort(s)) + 0.5) / len(s)         # 0..1, query-agnostic
        ref = np.array([c.gold if c.gold is not None else c.teacher for c in cands], bool)
        disagreement = np.where(ref, 1.0 - rank, rank)            # high = verifier contradicts reference
        order = np.argsort(-disagreement)
        return [cands[i] for i in order[:limit]]
    rng = np.random.default_rng(seed)
    yes = [c for c in cands if c.teacher]; no = [c for c in cands if not c.teacher]
    rng.shuffle(yes); rng.shuffle(no)
    half = limit // 2
    take = yes[:half] + no[:limit - min(half, len(yes))]
    return take[:limit]


def build_page(ds: Dataset, picked: list) -> str:
    by_q: dict[str, list] = {}
    for c in picked:
        by_q.setdefault(c.query_id, []).append(c)
    parts = []
    for qid, items in by_q.items():
        q = ds.queries.get(qid, {})
        parts.append(f"<h2>{html.escape(q.get('query', qid))}</h2>"
                     f"<p>{html.escape(q.get('criteria', ''))}</p><div class='grid'>")
        for c in items:
            cur = c.gold if c.gold is not None else c.teacher
            img = thumb_b64(make_crop(ds.tile_image(c), c.bbox, "ctx4box"), 160)
            parts.append(
                f"<div class='c{' y' if cur else ''}' data-id='{c.id}' data-init='{int(cur)}'>"
                f"<img alt='' src='data:image/jpeg;base64,{img}'>"
                f"<b>{'YES' if cur else 'NO'}</b> · {html.escape(c.label)} · {c.min_side}px · "
                f"det {c.det_conf:.2f}</div>")
        parts.append("</div>")
    return _PAGE.replace("__BODY__", "".join(parts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=300, help="candidates on the page")
    ap.add_argument("--scores", default=None, help="eval/scores.csv — show strongest disagreements first")
    ap.add_argument("--run", default=None, help="column of scores.csv to compare against (default: last)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", default=None, help="merge an exported gold.json into the dataset")
    args = ap.parse_args(argv)

    ds = Dataset.load(args.data)
    if args.apply:
        new = json.loads(Path(args.apply).read_text(encoding="utf-8"))
        ids = {c.id for c in ds.candidates}
        gold = json.loads(ds.gold_path.read_text(encoding="utf-8")) if ds.gold_path.exists() else {}
        gold.update({k: bool(v) for k, v in new.items() if k in ids})
        ds.gold_path.write_text(json.dumps(gold, indent=1), encoding="utf-8")
        flipped = sum(1 for c in ds.candidates if c.id in new and bool(new[c.id]) != c.teacher)
        print(f"{ds.gold_path}: {len(gold)} reviewed labels "
              f"({flipped} of the {len(new)} just applied differ from the remote VLM)")
        return

    picked = _pick(ds, args.limit, args.scores, args.run, args.seed)
    if not picked:
        raise SystemExit("nothing to review")
    out = ds.root / "review.html"
    out.write_text(build_page(ds, picked), encoding="utf-8")
    print(f"wrote {out} ({len(picked)} candidates) — open it in a browser")


if __name__ == "__main__":
    main()
