### Operations
- [] add detailed logging at each step
- [x] skip ingestion when it's already done

### Misc
- [] YOLO-World / Grounding DINO on each tile   ← fast, local, zero-shot
- [x] Multi-Angle Persistence - how is it manifesetd in top 3 candidates? (see Gemini)
- [x] Merge detections → NMS to remove duplicates from overlapping tiles

### non-generative Visual models
- [x] learn CLIP
- [x] is it useful to use Dino / Yolo?

### VLM
- [x] would tiling help? Consider a full cycle, starting from indexing
- [x] response quality - sometimes model returns bad "bbox" in "targets" like `[0, 0, 1000, 600] [1000, 600, 2000, 1200] [2000, 0, 3000, 600]` need to detect such casees and retry
- [x] handle cases of a different intents, beyond finding a specific object
- [] add context
  
### Python libraries to manipulate / seacrh geo object
- [`on hold`] see Gemini chat

### Jetson
- [] what one to purchase - dev kit?

### UI
- [] images / video upload
- [] chat widget
- [] map with markers

### Deployment
- [] Runpod