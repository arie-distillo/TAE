"""
clip_benchmark.py
=================
Compares encoding throughput of different CLIP implementations
on the same batch of synthetic tiles (CPU and GPU if available).

Implementations tested:
  1. openai/CLIP          — the original, used in current TAE
  2. open_clip            — community fork, more models, often faster
  3. clip-retrieval       — wrapper around CLIP with batching utilities

Run:
    pip install git+https://github.com/openai/CLIP.git open-clip-torch clip-retrieval
    python clip_benchmark.py

Optional: point --image_dir at your actual tile folder for realistic data.
"""

import argparse
import time
import sys
import importlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZES   = [8, 16, 32, 48]   # tile counts to test per implementation
TILE_SIZE_PX  = 640               # synthetic tile size (matches TAE default)
N_WARMUP      = 2                 # warmup passes (not timed)
N_REPEATS     = 3                 # timed repetitions — median is reported
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_QUERIES  = [
    "helipad marked with H",
    "military vehicle",
    "person on rooftop",
]

# Model configurations per implementation
OPENAI_MODELS   = ["RN50", "ViT-B/32", "ViT-B/16"]
OPENCLIP_MODELS = [
    ("RN50",          "openai"),
    ("ViT-B-32",      "openai"),
    ("ViT-B-16",      "openai"),
    ("ViT-B-32-quickgelu", "laion400m_e32"),  # community model
]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_tiles(n: int, size: int = TILE_SIZE_PX) -> list[np.ndarray]:
    """Generates n random BGR uint8 tiles — realistic in dtype and shape."""
    return [
        np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    implementation: str
    model:          str
    device:         str
    batch_size:     int
    median_ms:      float
    ms_per_tile:    float
    tiles_per_sec:  float
    dim:            int
    error:          str = ""


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------

def timed_batch(fn: Callable, tiles: list, n_repeats: int) -> float:
    """Returns median wall-clock time in ms over n_repeats calls."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(tiles)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Implementation 1: openai/CLIP
# ---------------------------------------------------------------------------

def bench_openai_clip(model_name: str, results: list[BenchmarkResult]):
    try:
        import clip
    except ImportError:
        print("  [SKIP] openai/CLIP not installed — pip install git+https://github.com/openai/CLIP.git")
        return

    try:
        model, preprocess = clip.load(model_name, device=DEVICE)
        model.eval()

        def encode(cv2_imgs):
            inputs = []
            for img in cv2_imgs:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                inputs.append(preprocess(pil))
            batch = torch.stack(inputs).to(DEVICE)
            with torch.no_grad():
                return model.encode_image(batch).cpu().numpy()

        # Warmup
        warmup_tiles = make_synthetic_tiles(8)
        for _ in range(N_WARMUP):
            encode(warmup_tiles)

        # Get output dim
        dim = encode(make_synthetic_tiles(1)).shape[1]

        for bs in BATCH_SIZES:
            tiles = make_synthetic_tiles(bs)
            ms = timed_batch(encode, tiles, N_REPEATS)
            results.append(BenchmarkResult(
                implementation="openai/CLIP",
                model=model_name,
                device=DEVICE,
                batch_size=bs,
                median_ms=round(ms, 1),
                ms_per_tile=round(ms / bs, 1),
                tiles_per_sec=round(bs / (ms / 1000), 1),
                dim=dim,
            ))
            print(f"    batch={bs:2d} → {ms:.0f}ms total | {ms/bs:.1f}ms/tile")

    except Exception as e:
        results.append(BenchmarkResult(
            implementation="openai/CLIP", model=model_name,
            device=DEVICE, batch_size=0, median_ms=0,
            ms_per_tile=0, tiles_per_sec=0, dim=0, error=str(e)
        ))
        print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Implementation 2: open_clip
# ---------------------------------------------------------------------------

def bench_open_clip(model_name: str, pretrained: str, results: list[BenchmarkResult]):
    try:
        import open_clip
    except ImportError:
        print("  [SKIP] open_clip not installed — pip install open-clip-torch")
        return

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=DEVICE
        )
        model.eval()

        def encode(cv2_imgs):
            inputs = []
            for img in cv2_imgs:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                inputs.append(preprocess(pil))
            batch = torch.stack(inputs).to(DEVICE)
            with torch.no_grad():
                return model.encode_image(batch).cpu().numpy()

        # Warmup
        for _ in range(N_WARMUP):
            encode(make_synthetic_tiles(8))

        dim = encode(make_synthetic_tiles(1)).shape[1]

        for bs in BATCH_SIZES:
            tiles = make_synthetic_tiles(bs)
            ms = timed_batch(encode, tiles, N_REPEATS)
            results.append(BenchmarkResult(
                implementation="open_clip",
                model=f"{model_name}/{pretrained}",
                device=DEVICE,
                batch_size=bs,
                median_ms=round(ms, 1),
                ms_per_tile=round(ms / bs, 1),
                tiles_per_sec=round(bs / (ms / 1000), 1),
                dim=dim,
            ))
            print(f"    batch={bs:2d} → {ms:.0f}ms total | {ms/bs:.1f}ms/tile")

    except Exception as e:
        results.append(BenchmarkResult(
            implementation="open_clip", model=f"{model_name}/{pretrained}",
            device=DEVICE, batch_size=0, median_ms=0,
            ms_per_tile=0, tiles_per_sec=0, dim=0, error=str(e)
        ))
        print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Implementation 3: clip-retrieval (uses openai/CLIP under the hood but
# exposes a ClipClient and local inference via ClipModel wrapper)
# ---------------------------------------------------------------------------

def bench_clip_retrieval(results: list[BenchmarkResult]):
    try:
        from clip_retrieval.clip_client import ClipModel
    except ImportError:
        print("  [SKIP] clip-retrieval not installed — pip install clip-retrieval")
        return

    try:
        # clip-retrieval's local inference wrapper
        clip_model = ClipModel(model_name="ViT-B/32", use_jit=True, device=DEVICE)

        def encode(cv2_imgs):
            pils = [
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for img in cv2_imgs
            ]
            return clip_model.encode_image(pils)

        for _ in range(N_WARMUP):
            encode(make_synthetic_tiles(8))

        dim = encode(make_synthetic_tiles(1)).shape[1]

        for bs in BATCH_SIZES:
            tiles = make_synthetic_tiles(bs)
            ms = timed_batch(encode, tiles, N_REPEATS)
            results.append(BenchmarkResult(
                implementation="clip-retrieval",
                model="ViT-B/32 (jit)",
                device=DEVICE,
                batch_size=bs,
                median_ms=round(ms, 1),
                ms_per_tile=round(ms / bs, 1),
                tiles_per_sec=round(bs / (ms / 1000), 1),
                dim=dim,
            ))
            print(f"    batch={bs:2d} → {ms:.0f}ms total | {ms/bs:.1f}ms/tile")

    except Exception as e:
        results.append(BenchmarkResult(
            implementation="clip-retrieval", model="ViT-B/32",
            device=DEVICE, batch_size=0, median_ms=0,
            ms_per_tile=0, tiles_per_sec=0, dim=0, error=str(e)
        ))
        print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Text encoding benchmark (same implementations, query→vector)
# ---------------------------------------------------------------------------

def bench_text_encoding():
    """
    Separately benchmarks text query encoding — only matters at query time,
    not ingestion, but worth knowing.
    """
    print("\n=== TEXT ENCODING (query time only) ===")
    try:
        import clip
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        model.eval()
        tokens = clip.tokenize(TEXT_QUERIES).to(DEVICE)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.encode_text(tokens)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  openai/CLIP ViT-B/32: {np.median(times):.1f}ms for {len(TEXT_QUERIES)} queries")
    except Exception as e:
        print(f"  [SKIP] {e}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[BenchmarkResult]):
    print("\n" + "=" * 90)
    print("CLIP BENCHMARK RESULTS")
    print(f"Device: {DEVICE} | Tile size: {TILE_SIZE_PX}px | Repeats: {N_REPEATS}")
    print("=" * 90)

    # Group by batch size
    for bs in BATCH_SIZES:
        batch_results = [r for r in results if r.batch_size == bs and not r.error]
        if not batch_results:
            continue

        print(f"\nBatch size: {bs} tiles")
        print(f"  {'Implementation':<20} {'Model':<30} {'Total ms':>9} {'ms/tile':>9} {'tiles/s':>9} {'dim':>6}")
        print(f"  {'-'*20} {'-'*30} {'-'*9} {'-'*9} {'-'*9} {'-'*6}")

        # Sort by ms/tile ascending
        for r in sorted(batch_results, key=lambda x: x.ms_per_tile):
            print(
                f"  {r.implementation:<20} {r.model:<30} "
                f"{r.median_ms:>9.0f} {r.ms_per_tile:>9.1f} "
                f"{r.tiles_per_sec:>9.1f} {r.dim:>6}"
            )

    # Errors
    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors:")
        for r in errors:
            print(f"  {r.implementation} / {r.model}: {r.error}")

    # TAE-specific recommendation
    print("\n" + "=" * 90)
    print("TAE RECOMMENDATION")
    print("=" * 90)

    # Find fastest at batch=48 (closest to current tile count)
    target_bs = max(BATCH_SIZES)
    candidates = [r for r in results if r.batch_size == target_bs and not r.error]
    if candidates:
        best = min(candidates, key=lambda r: r.ms_per_tile)
        frames_per_min = 60 / (best.ms_per_tile * 48 / 1000)
        print(
            f"Fastest at batch={target_bs}: {best.implementation} / {best.model}\n"
            f"  {best.ms_per_tile:.1f}ms/tile → "
            f"~{best.ms_per_tile * 48 / 1000:.1f}s/frame (48 tiles) → "
            f"~{frames_per_min:.0f} frames/min at current tile count\n"
            f"  Embedding dim: {best.dim} (update CLIP_DIM in config.py if switching)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLIP implementation benchmark for TAE")
    parser.add_argument("--skip-openai",      action="store_true")
    parser.add_argument("--skip-openclip",    action="store_true")
    parser.add_argument("--skip-retrieval",   action="store_true")
    parser.add_argument("--models",           nargs="+",
                        help="Override openai/CLIP models to test (e.g. RN50 ViT-B/32)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: Running on CPU — results will be slow. GPU is strongly recommended.")
    print()

    results: list[BenchmarkResult] = []

    # --- openai/CLIP ---
    if not args.skip_openai:
        models = args.models or OPENAI_MODELS
        for m in models:
            print(f"[openai/CLIP] {m}")
            bench_openai_clip(m, results)

    # --- open_clip ---
    if not args.skip_openclip:
        for model_name, pretrained in OPENCLIP_MODELS:
            print(f"[open_clip] {model_name} / {pretrained}")
            bench_open_clip(model_name, pretrained, results)

    # --- clip-retrieval ---
    if not args.skip_retrieval:
        print("[clip-retrieval] ViT-B/32 with JIT")
        bench_clip_retrieval(results)

    # --- text encoding ---
    bench_text_encoding()

    # --- report ---
    print_report(results)


if __name__ == "__main__":
    main()
