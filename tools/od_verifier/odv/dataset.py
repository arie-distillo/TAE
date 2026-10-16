"""
Dataset schema and I/O.

A dataset is a directory:

    candidates.jsonl   one detector candidate per line (see Candidate)
    queries.json       {query_id: {...intent parameters...}}
    meta.json          recording metadata + measured remote-stage timings
    tiles/             tile images referenced by Candidate.tile
    gold.json          optional human corrections {candidate_id: true|false}

Tiles are stored (not just frame paths) so the dataset is self-contained and
every crop strategy can be re-applied offline without the original footage.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Candidate:
    id:             str
    frame:          str          # source frame name — the grouping key for CV folds
    tile:           str          # tile image path relative to the dataset dir
    tile_x:         int
    tile_y:         int
    tile_w:         int
    tile_h:         int
    bbox:           list         # [x1, y1, x2, y2] tile-local pixels
    label:          str          # detector label
    det_conf:       float        # detector confidence
    query_id:       str
    teacher:        bool         # remote VLM verdict
    teacher_conf:   float = 0.0
    teacher_label:  str   = ""
    teacher_reason: str   = ""
    teacher_valid:  bool  = True   # False when the VLM call errored / did not answer
    jev_p:          float | None = None   # P(yes) from Jev when record.py ran with --jev
    gold:           bool | None = field(default=None)

    @property
    def min_side(self) -> int:
        return int(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def max_side(self) -> int:
        return int(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


class Dataset:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.candidates: list[Candidate] = []
        self.queries: dict[str, dict] = {}
        self.meta: dict = {}
        self._tile_cache: dict[str, np.ndarray] = {}

    # ── paths ────────────────────────────────────────────────────────────────
    @property
    def cand_path(self) -> Path:  return self.root / "candidates.jsonl"
    @property
    def query_path(self) -> Path: return self.root / "queries.json"
    @property
    def meta_path(self) -> Path:  return self.root / "meta.json"
    @property
    def gold_path(self) -> Path:  return self.root / "gold.json"
    @property
    def tiles_dir(self) -> Path:  return self.root / "tiles"

    # ── load / save ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, root: Path | str) -> "Dataset":
        ds = cls(root)
        if not ds.cand_path.exists():
            raise FileNotFoundError(f"{ds.cand_path} not found — run record.py first")
        fields = set(Candidate.__dataclass_fields__)
        with ds.cand_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    ds.candidates.append(Candidate(**{k: v for k, v in row.items() if k in fields}))
        if ds.query_path.exists():
            ds.queries = json.loads(ds.query_path.read_text(encoding="utf-8"))
        if ds.meta_path.exists():
            ds.meta = json.loads(ds.meta_path.read_text(encoding="utf-8"))
        if ds.gold_path.exists():
            gold = json.loads(ds.gold_path.read_text(encoding="utf-8"))
            for c in ds.candidates:
                if c.id in gold:
                    c.gold = bool(gold[c.id])
        return ds

    def append(self, cands: list[Candidate]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.cand_path.open("a", encoding="utf-8") as fh:
            for c in cands:
                row = asdict(c)
                row.pop("gold", None)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.candidates.extend(cands)

    def save_queries(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.query_path.write_text(json.dumps(self.queries, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    def save_meta(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False),
                                  encoding="utf-8")

    # ── tiles ────────────────────────────────────────────────────────────────
    def tile_image(self, cand: Candidate) -> np.ndarray:
        img = self._tile_cache.get(cand.tile)
        if img is None:
            img = cv2.imread(str(self.root / cand.tile))
            if img is None:
                raise FileNotFoundError(f"tile image missing: {self.root / cand.tile}")
            if len(self._tile_cache) > 512:
                self._tile_cache.clear()
            self._tile_cache[cand.tile] = img
        return img

    # ── labels ───────────────────────────────────────────────────────────────
    def labelled(self, source: str = "best") -> tuple[list[Candidate], np.ndarray]:
        """
        Return (candidates, y) for candidates that have a usable label.

        source = "gold"    only human-reviewed candidates
                 "teacher" remote VLM verdicts (valid calls only)
                 "best"    gold where present, otherwise teacher
        """
        out, y = [], []
        for c in self.candidates:
            if source in ("gold", "best") and c.gold is not None:
                out.append(c); y.append(int(c.gold)); continue
            if source in ("teacher", "best") and c.teacher_valid:
                out.append(c); y.append(int(c.teacher))
        return out, np.asarray(y, dtype=np.int64)
