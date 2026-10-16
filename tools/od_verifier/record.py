#!/usr/bin/env python
"""
record.py — build a verifier dataset from real footage.

Runs the CURRENT production stages (imported from src/, not re-implemented)
on a folder of frames or a video:

    tiles → Grounding DINO (Replicate) → cross-tile NMS → size filter
          → per-tile cap → Qwen2.5-VL batch verification (OpenRouter)

and stores every candidate the VLM was asked about, with its verdict, plus the
tile images.  The verdicts are the reference a local verifier is measured
against; the per-stage wall times are the baseline it has to beat.

    python tools/od_verifier/record.py --frames path/to/frames_or_video \
        --query "find all cars" --out tools/od_verifier/data/run1

Needs the same .env TAE uses (REPLICATE_API_KEY, OPENROUTER_API_KEY).
This spends real API calls: one detector call per tile and one VLM call per
tile that has candidates.  Use --max-frames to bound it; re-running with the
same --out resumes and skips frames already recorded.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from odv.dataset import Candidate, Dataset          # noqa: E402

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_UNVISITED = "__unvisited__"
_INVALID_PREFIXES = ("error:", "parse error", "missing", "empty", _UNVISITED)


def _setup_tae(env_file: str | None) -> None:
    """Put src/ on the path and load TAE's .env before config.py is imported."""
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from dotenv import load_dotenv
        for p in ([Path(env_file)] if env_file else [Path.cwd() / ".env", src / ".env", REPO / ".env"]):
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "query"


def collect_frames(source: Path, out_frames: Path, every_s: float, max_frames: int) -> list[Path]:
    """Image folder → sorted image paths.  Video file → frames sampled every `every_s` seconds."""
    if source.is_dir():
        frames = sorted(p for p in source.iterdir() if p.suffix.lower() in _IMG_EXT)
    elif source.suffix.lower() in _IMG_EXT:
        frames = [source]
    else:
        out_frames.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(every_s * fps)))
        frames, idx = [], 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    p = out_frames / f"{source.stem}_{int(idx / fps * 1000):08d}.jpg"
                    if not p.exists():
                        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frames.append(p)
                    if max_frames and len(frames) >= max_frames:
                        break
            idx += 1
        cap.release()
    return frames[:max_frames] if max_frames else frames


def frame_tiles(frame_path: Path, img) -> list[dict]:
    """Tile dicts in the shape the pipeline stages expect (same tiling as ingestion)."""
    from core.ingestion import TILE_OVERLAP, TILE_SIZE, _iter_tiles
    tiles = []
    for _, tx, ty, tw, th in _iter_tiles(img, TILE_SIZE, TILE_OVERLAP):
        tiles.append({
            "parent_path": str(frame_path),
            "tile_x": int(tx), "tile_y": int(ty), "tile_w": int(tw), "tile_h": int(th),
            # footprints are irrelevant for verification; zeros keep the stage code happy
            "fp_nw_lat": 0.0, "fp_nw_lon": 0.0, "fp_ne_lat": 0.0, "fp_ne_lon": 0.0,
            "fp_se_lat": 0.0, "fp_se_lon": 0.0, "fp_sw_lat": 0.0, "fp_sw_lon": 0.0,
        })
    return tiles


def main(argv: list[str] | None = None) -> Dataset:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of images, a single image, or a video file")
    ap.add_argument("--query", required=True, help="mission definition / query, as typed in TAE")
    ap.add_argument("--out", required=True, help="dataset directory (appended to if it exists)")
    ap.add_argument("--every", type=float, default=3.0, help="video sampling interval, seconds")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--max-per-tile", type=int, default=None,
                    help="candidate cap per tile (default: MAX_CANDIDATES_PER_TILE or 8, as in streaming)")
    ap.add_argument("--env-file", default=None, help="path to TAE's .env (default: ./.env, src/.env, repo/.env)")
    ap.add_argument("--jev", nargs="?", const="jev-1.14.0", default=None, metavar="MODEL",
                    help="also score every candidate with TypeSafe Jev (remote, needs TYPESAFE_API_KEY) "
                         "and time it per frame alongside Qwen; crops use --jev-crop")
    ap.add_argument("--jev-crop", default="ctx2", help="crop strategy sent to Jev (default ctx2)")
    args = ap.parse_args(argv)

    _setup_tae(args.env_file)
    from config import settings
    from ai.detection_pipeline import cross_tile_nms, run_detector_stage, vlm_verify_stage
    from ai.intent import IntentClassifier
    from ai.vlm import TacticalAnalyst

    out = Path(args.out)
    ds = Dataset.load(out) if (out / "candidates.jsonl").exists() else Dataset(out)
    if not ds.meta and ds.meta_path.exists():
        import json
        ds.meta = json.loads(ds.meta_path.read_text(encoding="utf-8"))
    ds.tiles_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(Path(args.frames), out / "frames", args.every, args.max_frames)
    if not frames:
        raise SystemExit(f"no frames found in {args.frames}")

    # ── intent: identical to what the pipeline derives from the mission definition ──
    clf = IntentClassifier(api_key=settings.OPENROUTER_API_KEY,
                           model_name=getattr(settings, "INTENT_MODEL", None))
    params = clf.classify(args.query).params
    if getattr(params, "intent", "") != "object_detection":
        raise SystemExit(f"query classified as '{params.intent}', not object_detection")
    det_classes = params.object_classes
    if getattr(settings, "GDINO_ENRICHED_QUERY", True) and getattr(params, "gdino_classes", []):
        det_classes = params.gdino_classes
    qid = _slug(args.query)
    ds.queries[qid] = {
        "query": args.query,
        "object_classes": list(params.object_classes),
        "detector_classes": list(det_classes),
        "canonical_label": params.canonical_label,
        "criteria": params.vlm_verification_criteria,
        "report_fields": list(params.vlm_reporting_fields),
        "colour_hint": params.colour_hint,
        "size_qualifier": params.size_qualifier,
        "difficulty": params.expected_difficulty,
        "detector_threshold": params.object_confidence,
    }
    ds.save_queries()

    analyst = TacticalAnalyst(settings.AI_PROVIDER, settings.VLM_MODEL, settings.OPENROUTER_API_KEY)
    jev = None
    if args.jev:
        from odv.crops import make_crop
        from odv.verifiers.jev import JevVerifier
        jev = JevVerifier(model=args.jev)
        jev.load()                                   # fails early without TYPESAFE_API_KEY
        ds.meta["jev_model"] = args.jev
        ds.meta["jev_crop"] = args.jev_crop
    max_per_tile = args.max_per_tile or getattr(settings, "MAX_CANDIDATES_PER_TILE", 8)
    min_px = settings.DETECTION_MIN_BBOX_PX

    done = set(ds.meta.setdefault("frames_done", {}).setdefault(qid, []))
    timing = ds.meta.setdefault("timing", {"frames": 0, "tiles": 0, "detector_s": 0.0, "vlm_s": 0.0})
    timing.setdefault("jev_s", 0.0); timing.setdefault("jev_frames", 0)
    ds.meta["detector_model"] = settings.DETECTOR_REPLICATE_MODEL
    ds.meta["vlm_model"] = settings.VLM_MODEL

    todo = [f for f in frames if f.name not in done]
    print(f"query '{args.query}' → classes {det_classes} | difficulty {params.expected_difficulty} "
          f"| detector threshold {params.object_confidence}")
    print(f"{len(todo)} frame(s) to record ({len(frames) - len(todo)} already done)")

    for n, fpath in enumerate(todo, 1):
        img = cv2.imread(str(fpath))
        if img is None:
            print(f"  [{n}/{len(todo)}] {fpath.name}: unreadable, skipped")
            continue
        tiles = frame_tiles(fpath, img)

        t0 = time.perf_counter()
        raw = run_detector_stage(
            tiles=tiles, classes=det_classes, confidence=params.object_confidence,
            api_key=settings.REPLICATE_API_KEY, model_version=settings.DETECTOR_REPLICATE_MODEL,
            timeout_s=getattr(settings, "DETECTOR_TIMEOUT_S", 180),
        )
        det_s = time.perf_counter() - t0

        cands = [d for d in cross_tile_nms(raw)
                 if (d.bbox_tile[2] - d.bbox_tile[0]) >= min_px
                 and (d.bbox_tile[3] - d.bbox_tile[1]) >= min_px] if raw else []
        buckets = defaultdict(list)
        for d in cands:
            buckets[(d.tile_x, d.tile_y)].append(d)
        cands = []
        for b in buckets.values():
            b.sort(key=lambda d: d.confidence, reverse=True)
            cands.extend(b[:max_per_tile])

        vlm_s = jev_s = 0.0
        rows: list[Candidate] = []
        if cands:
            for d in cands:
                d.vlm_reason = _UNVISITED
                d._det_label = d.label      # the stage overwrites label with the VLM's on confirmation
            t0 = time.perf_counter()
            vlm_verify_stage(cands, params, args.query, analyst)
            vlm_s = time.perf_counter() - t0

            for d in cands:
                tile_name = f"{fpath.stem}_t{d.tile_x}_{d.tile_y}.jpg"
                tile_path = ds.tiles_dir / tile_name
                if not tile_path.exists():
                    cv2.imwrite(str(tile_path),
                                img[d.tile_y:d.tile_y + d.tile_h, d.tile_x:d.tile_x + d.tile_w],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                reason = d.vlm_reason or ""
                valid = bool(d.confirmed) or not reason.lower().startswith(_INVALID_PREFIXES)
                rows.append(Candidate(
                    id=uuid.uuid4().hex[:12], frame=fpath.name, tile=f"tiles/{tile_name}",
                    tile_x=d.tile_x, tile_y=d.tile_y, tile_w=d.tile_w, tile_h=d.tile_h,
                    bbox=[int(v) for v in d.bbox_tile],
                    label=d._det_label, det_conf=float(d.confidence), query_id=qid,
                    teacher=bool(d.confirmed), teacher_conf=float(d.vlm_confidence),
                    teacher_label=d.label if d.confirmed else "",
                    teacher_reason="" if reason == _UNVISITED else reason,
                    teacher_valid=valid,
                ))
            if jev is not None:
                crops = [make_crop(img[r.tile_y:r.tile_y + r.tile_h, r.tile_x:r.tile_x + r.tile_w],
                                   r.bbox, args.jev_crop) for r in rows]
                t0 = time.perf_counter()
                jout = jev.run(crops, rows, ds.queries)
                jev_s = time.perf_counter() - t0
                for r, pj in zip(rows, jout.scores["noul"]):
                    r.jev_p = None if pj != pj else float(pj)       # NaN → None
                timing["jev_s"] += jev_s; timing["jev_frames"] += 1
            ds.append(rows)

        timing["frames"] += 1; timing["tiles"] += len(tiles)
        timing["detector_s"] += det_s; timing["vlm_s"] += vlm_s
        ds.meta["frames_done"][qid].append(fpath.name)
        ds.save_meta()
        n_yes = sum(r.teacher for r in rows); n_bad = sum(not r.teacher_valid for r in rows)
        print(f"  [{n}/{len(todo)}] {fpath.name}: {len(tiles)} tiles, {len(raw)} raw → {len(rows)} "
              f"candidates, {n_yes} confirmed" + (f", {n_bad} VLM failures" if n_bad else "")
              + f" | detector {det_s:.1f}s, VLM {vlm_s:.1f}s"
              + (f", Jev {jev_s:.1f}s" if jev is not None and cands else ""))

    valid = [c for c in ds.candidates if c.teacher_valid]
    print(f"\ndataset {out}: {len(ds.candidates)} candidates, {len(valid)} with a valid verdict, "
          f"{sum(c.teacher for c in valid)} confirmed")
    if timing["frames"]:
        print(f"remote stages, mean per frame: detector {timing['detector_s'] / timing['frames']:.1f}s, "
              f"VLM {timing['vlm_s'] / timing['frames']:.1f}s"
              + (f", Jev {timing['jev_s'] / timing['jev_frames']:.1f}s "
                 f"(over {timing['jev_frames']} frames with candidates)" if timing.get("jev_frames") else ""))
    return ds


if __name__ == "__main__":
    main()
