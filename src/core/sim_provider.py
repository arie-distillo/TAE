import json
import cv2
from pathlib import Path

class SimD3Environment:
    def __init__(self, metadata_file):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.frame_names = list(self.metadata.keys())

    def __iter__(self):
        for name in self.frame_names:
            entry = self.metadata[name]
            # We use the 'full_path' we added to the JSON earlier
            img_path = entry['full_path']
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image at {img_path}")
                continue
                
            yield img, entry