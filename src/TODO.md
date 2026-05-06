### Operations
- [] add detailed logging at each step
- [x] c

### Misc
- [] YOLO-World / Grounding DINO on each tile   ← fast, local, zero-shot
- [x] Multi-Angle Persistence - how is it manifesetd in top 3 candidates? (see Gemini)
- [x] Merge detections → NMS to remove duplicates from overlapping tiles
- [] Improve startup message if something already uploaded `_msg("TAE ready. Upload imagery to build the theater index, ""then query in natural language.", "sys"),`
- [] Investigate bad detections
- [] Split into two services - on-edge server and cloud server

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
- [x] images upload
- [] video upload
- [x] chat widget
- [x] map with markers
- [] remove Span("Intelligence", cls="brand-sub")


### Deployment
- [] Runpod