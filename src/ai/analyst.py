import ollama
from pathlib import Path

class TacticalAnalyst:
    def __init__(self, model_name="qwen2.5vl"):
        # We no longer load the model into memory here; 
        # Ollama manages the lifecycle.
        self.model_name = model_name

    def analyze_multiple_views(self, image_paths, user_query):
        prompt = (
            f"You are looking at multiple tactical views of the same target area. "
            f"Synthesize the information from all images to answer: {user_query}. "
            "Report only confirmed intelligence."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': image_paths # Ollama supports a list of image paths
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"MAP Reasoning Error: {e}"
