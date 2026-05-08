### Operations
- [] add detailed logging at each step
- [1] audit

### Misc
- [] YOLO-World / Grounding DINO on each tile   ← fast, local, zero-shot
- [v] Multi-Angle Persistence - how is it manifesetd in top 3 candidates? (see Gemini)
- [v] Merge detections → NMS to remove duplicates from overlapping tiles
- [3] Improve startup message if something already uploaded `_msg("TAE ready. Upload imagery to build the theater index, ""then query in natural language.", "sys"),`
- [] Investigate bad detections

### Refactoring
- [] Split into two services - on-edge server and cloud server
- [] Split `main.py` into UI and Orchestrator

### non-generative Visual models
- [v] learn CLIP
- [v] is it useful to use Dino / Yolo?

### VLM
- [v] would tiling help? Consider a full cycle, starting from indexing
- [v] response quality - sometimes model returns bad "bbox" in "targets" like `[0, 0, 1000, 600] [1000, 600, 2000, 1200] [2000, 0, 3000, 600]` need to detect such casees and retry
- [v] handle cases of a different intents, beyond finding a specific object
- [] add context
- [] agentic intent detection 
  - [v] object search intent
  - [] anomaly detection intent
  - [] moving object intent
- [] persistent queries
  
### Python libraries to manipulate / seacrh geo object
- [>] see Gemini chat

### Jetson
- [] what one to purchase - dev kit?

### UI
- [v] images upload
- [] video upload
- [v] chat widget
- [v] map with markers
- [] remove Span("Intelligence", cls="brand-sub")
- [] frames with detection on a timeline

### Deployment
- [] Runpod