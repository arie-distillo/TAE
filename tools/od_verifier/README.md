# od_verifier — local CPU verifier experiment

Question this answers: **can the remote Qwen2.5-VL verification stage be replaced
(fully or mostly) by a model that runs on the local CPU, and at what cost in
quality?**

Nothing in `src/` is modified. The recorder imports the production stage
functions, so candidates and verdicts are exactly what the pipeline produces.

## Workflow

```
1. record    footage → Grounding DINO candidates + Qwen verdicts → dataset
2. evaluate  dataset → quality and latency per local verifier → report.html
3. review    correct the reference labels by eye, re-evaluate
```

### 0. Install and self-check (no keys, no footage)

```
pip install -r tools/od_verifier/requirements.txt
python -m pytest tools/od_verifier/tests -q
python tools/od_verifier/make_synthetic.py --out tools/od_verifier/data/synthetic
python tools/od_verifier/evaluate.py --data tools/od_verifier/data/synthetic --backends mock
```

### 1. Record a dataset

```
python tools/od_verifier/record.py --frames path/to/video_or_frames --query "find all cars" --out tools/od_verifier/data/city --every 3 --max-frames 40
```

- Uses TAE's `.env` (`REPLICATE_API_KEY`, `OPENROUTER_API_KEY`). **Spends API
  calls**: one detector call per tile, one VLM call per tile with candidates.
- Re-running with the same `--out` resumes. Run again with another `--query`
  to add a second query to the same dataset.
- Prints and stores the measured wall time of the two remote stages per frame.
- `--jev` additionally scores every candidate with Jev right after Qwen and
  times it per frame, so the Qwen-vs-Jev latency comparison is measured on the
  same frames. The verdicts are stored in the dataset and show up in
  `evaluate.py` as a baseline row without further calls.
- Aim for a few hundred candidates per query, from more than one flight, with
  at least 30 confirmed and 30 rejected. Fewer than that and the numbers are noise.

### 2. Evaluate

```
python tools/od_verifier/evaluate.py --data tools/od_verifier/data/city --backends openclip:ViT-B-32-quickgelu:openai openclip:ViT-B-16-SigLIP:webli --crops tight20 ctx2 ctx4 ctx4box --engine openvino
```

Backends:

| Spec | What it is |
|---|---|
| `openclip:ViT-B-32-quickgelu:openai` | The CLIP weights TAE already loads for indexing. Baseline, no extra model in memory. |
| `openclip:ViT-B-16-SigLIP:webli` | SigLIP base. `ViT-B-16-SigLIP2:webli` also works with a recent open_clip. |
| `openclip:<arch>:<path.pt>` | Any open_clip-compatible checkpoint, e.g. a remote-sensing CLIP. |
| `hfvlm:HuggingFaceTB/SmolVLM-256M-Instruct` | Small generative VLM read as first-token Yes/No logits. |
| `vega`, `vega:4b` | **Local.** vegaml decision model (frozen Qwen3.5 0.8B / 4B + small head) on CPU; score = P(true), plus a variant that neutralises abstained answers. `pip install 'vegaml[vision]'`. |
| `jev`, `jev:jev-1.14.0` | **Remote.** TypeSafe Jev typed decision (`noul`), one HTTP call per crop, needs `TYPESAFE_API_KEY`. Not a local option: a cheaper, typed alternative to Qwen. |
| `mock` | Colour heuristic, for the tests. |

`--engine openvino` converts the image tower to OpenVINO (`pip install openvino`);
add `--ov-device GPU` to try the Intel iGPU. `--threads N` pins CPU threads.

Each contrastive backend yields four scorers per crop strategy:

- `zs_softmax`, `zs_margin`, `zs_cos` — zero-shot, target phrases against a
  list of concrete aerial confusers (not "not X": contrastive models handle
  negation badly). Extend the confusers per query with `--distractors file.json`.
- `probe` — logistic regression on the frozen embedding plus detector
  confidence and box size, trained on the reference verdicts, scored out of
  fold. This is the remote VLM distilled into a head that costs one dot product.

### 3. Review labels

```
python tools/od_verifier/review.py --data tools/od_verifier/data/city --scores tools/od_verifier/data/city/eval/scores.csv --run "ViT-B-16-SigLIP/webli@openvino-CPU · ctx2 · probe"
python tools/od_verifier/review.py --data tools/od_verifier/data/city --apply path/to/gold.json
```

Opens the strongest disagreements between a local verifier and the remote VLM
first. Until some labels are reviewed, "quality" means agreement with Qwen,
not correctness.

## Three-way comparison (Qwen / Jev / Vega)

```
python tools/od_verifier/record.py --frames VIDEO --query "find all cars" --out DATA --jev
python tools/od_verifier/evaluate.py --data DATA --backends vega openclip:ViT-B-16-SigLIP:webli --crops tight20 ctx2
```

Qwen is the reference column; the `jev/... (recorded)` row and the `vega/...`
rows are measured against it in the same table as the contrastive models.
Where the three disagree, `review.py --scores` shows the crops so a person can
decide who was right.

## Reading the report

| Column | Meaning |
|---|---|
| AUC / AP | Ranking quality, threshold-free. The `(no verifier)` row ranks by detector confidence alone: a verifier that does not beat it adds nothing. |
| Recall / Rejects removed | One threshold, tuned on other frames for recall ≥ 95 %. "Rejects removed" is the share of reference-rejected candidates the local verifier also rejects. |
| Escalated / Accept precision / Objects lost | Two-threshold cascade: accept and reject locally where confident, send the band in between to the remote VLM. "Escalated" is the share of remote calls that remain. |
| ms/crop, s/frame | Batch-1 p50/p95 and batched cost per crop; s/frame multiplies by the dataset's candidates per frame. |

All thresholds are fitted on some frames and applied to others (grouped
K-fold per query), so the operating points are not fitted to the data they
are reported on. The by-size table matters most: expect the smallest boxes to
be where a local verifier fails.

The probe learning curve trains on the first *n* candidates in recording
order and tests on the rest. It tells you how many remote verdicts a mission
needs before a local head can take over.

## What has and has not been run

Run in development (no network to model hubs, no footage, no API keys):

- All tests pass, including the recorder driven through the real
  `run_detector_stage` / `cross_tile_nms` / `vlm_verify_stage` with the two
  network calls stubbed.
- The open_clip backend end to end on PyTorch and OpenVINO **with random
  weights**: code path and timing only.

Not run, so treat as unverified until you run it:

- Any pretrained weights, hence any real quality number.
- `record.py` against the live Replicate and OpenRouter APIs.
- The `hfvlm` backend (written against the transformers API, never executed).
- The `jev` and `vega` backends against the real service / package. Both are
  tested only against stubs that follow the documented request and response
  shapes (Jev: `POST /v1/systemone`, `noul`; vegaml 0.8.0: `decide_image`,
  `answers[..]["boolean"]`).
- `--ov-device GPU`.
