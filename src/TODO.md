### Operations
- [] add detailed logging at each step
- [1] audit
- [] Streaming a file from a local disk - what is the right behaviour at the end of the file? Can TAE stop at the end of a file?

### Misc
- [v] YOLO-World / Grounding DINO on each tile   ← fast, local, zero-shot  [YOLO26](https://docs.ultralytics.com/tasks#detection)
- [v] Multi-Angle Persistence - how is it manifesetd in top 3 candidates? (see Gemini)
- [v] Merge detections → NMS to remove duplicates from overlapping tiles
- [3] Improve startup message if something already uploaded `_msg("TAE ready. Upload imagery to build the theater index, ""then query in natural language.", "sys"),`
- [v] Investigate bad detections
- [v] Missions
  - [v] mission CRUD in UI
  - [v] databases / directories per mission
- [] The Intent model extands a user query to addional terms (e.g `car` into [`vehicle`, `car`, `truck`, `SUV`). As a result, G-Dino may detect same object on different frames once as a `car` and once as a `SUV` which hurts tracking

### Refactoring
- [-] Split into two services - on-edge server and cloud server
- [v] Refactor `main.py` into UI and business logic

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
  - [v] anomaly detection intent
  - [] moving object intent
- [v] persistent queries
  
### Python libraries to manipulate / seacrh geo object
- [>] see Gemini chat

### Jetson
- [-] which one to purchase - dev kit?

### UI
- [v] images upload
- [v] video upload
- [v] chat widget
- [v] map with markers
- [v] show time / frame when pointed on a marker on a map, bypass inetrmediate pop-up
- [v] frames with detection on a timeline
- [v] video panel vs. chat panel vs. image panel - inconsistency
- [x] have panels resizeable (and movable?). By default a large video panel takes a central place

### Deployment
- [] Runpod

### Defects
- [v] Delete mission doesn't work
- [v] Mission directories under DATA_DIR should be named after mission name, not mission id. The latter only as a fallback if name not defined
- [v] video playback doesn't work
- [x] polyline missing - last message in Claude `Synthetic drone video generation and streaming simulation`
- [] trace - last message in Claude `Drone video detection and mapping issues`
- [x] video panel stall after ingestion / processing
- [] update the anomaly path to be in sync with all the changes in object detection path

### Cleanup
- [] code duplication in `_bg_detect_callback` `_stream_on_frame_telem` and `run_detection_pipeline`
- [] replace all `yolo_` with `detector_`
- [] migrate the map renderer to read from detections.json (as you suggested) so we can reitire tracks.json
- [] two different usages of term   `segment` - (1) image segmentation with SAM2, and (2) video segmentations. Confusing


### Geospatial
- [x] Need to have retry on detector (Grounding DINO via Replicate) calls - fix for _run_detector_on_tile in ai/detection_pipeline.py  (see Claude)


