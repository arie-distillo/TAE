#!/usr/bin/env python
"""
make_synthetic.py — a small fake dataset for checking the install.

Red blobs on textured ground are "confirmed", grey blobs are "rejected", with
a little label noise.  It exercises every code path of evaluate.py and
review.py without API keys or footage.  It says NOTHING about real quality.

    python tools/od_verifier/make_synthetic.py --out tools/od_verifier/data/synthetic
    python tools/od_verifier/evaluate.py --data tools/od_verifier/data/synthetic --backends mock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odv.dataset import Candidate, Dataset          # noqa: E402


def build(out: Path | str, frames: int = 12, per_tile: int = 6, noise: float = 0.05,
          seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    ds = Dataset(out)
    ds.tiles_dir.mkdir(parents=True, exist_ok=True)
    qid = "find-red-cars"
    ds.queries[qid] = {
        "query": "find red cars", "object_classes": ["car", "van"], "detector_classes": ["car", "van"],
        "canonical_label": "vehicle", "criteria": "Confirm this is a red vehicle, not a rooftop or shadow.",
        "report_fields": ["vehicle_type"], "colour_hint": "red", "size_qualifier": None,
        "difficulty": "medium", "detector_threshold": 0.2,
    }
    rows, k = [], 0
    for f in range(frames):
        for t in range(2):
            base = rng.integers(70, 130, 3)
            tile = (rng.normal(0, 14, (640, 640, 3)) + base).clip(0, 255).astype(np.uint8)
            tile = cv2.GaussianBlur(tile, (5, 5), 0)
            name = f"frame{f:03d}_t{t * 576}_0.jpg"
            for _ in range(per_tile):
                w, h = int(rng.integers(10, 70)), int(rng.integers(10, 70))
                x, y = int(rng.integers(8, 632 - w)), int(rng.integers(8, 632 - h))
                is_obj = bool(rng.random() < 0.4)
                col = (rng.integers(20, 60), rng.integers(20, 60), rng.integers(170, 240)) if is_obj \
                    else tuple(int(v) for v in rng.integers(40, 200, 1).repeat(3))
                cv2.rectangle(tile, (x, y), (x + w, y + h), tuple(int(c) for c in col), -1)
                teacher = is_obj ^ bool(rng.random() < noise)
                rows.append(Candidate(
                    id=f"syn{k:05d}", frame=f"frame{f:03d}.jpg", tile=f"tiles/{name}",
                    tile_x=t * 576, tile_y=0, tile_w=640, tile_h=640, bbox=[x, y, x + w, y + h],
                    label="car", det_conf=float(np.clip(rng.normal(0.45 if is_obj else 0.3, 0.12), 0.03, 0.95)),
                    query_id=qid, teacher=teacher, teacher_conf=0.9, teacher_label="car" if teacher else "",
                ))
                k += 1
            cv2.imwrite(str(ds.tiles_dir / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ds.append(rows)
    ds.save_queries()
    ds.meta = {"synthetic": True}
    ds.save_meta()
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=12)
    a = ap.parse_args()
    d = build(a.out, a.frames)
    print(f"wrote {len(d.candidates)} synthetic candidates to {a.out}")
