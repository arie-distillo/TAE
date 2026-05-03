### Operations
- [] add detailed logging at each step
- [] skip ingestion when it's already done

### Misc
 - [] Multi-Angle Persistence - how is it manifesetd in top 3 candidates? (see Gemini chat)

### non-generative Visual models
- [] learn CLIP
- [] is it useful to use Dino / Yolo?

### VLM
- [] would tiling help? Consider a full cycle, starting from indexing
- [] response quality - sometimes model returns bad "bbox" in "targets" like `[0, 0, 1000, 600] [1000, 600, 2000, 1200] [2000, 0, 3000, 600]` need to detect such casees and retry
- [] Prompt `f"Locate all instances of '{user_query}' in these images: {filenames_str}\n"` - handle cases of a different type of query, beyond finding a specific object

### Python libraries to manipulate / seacrh geo object
- [] see Gemini chat

### Jetson
- [] what one to purchase - dev kit?
