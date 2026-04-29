import ollama
from openai import OpenAI
from pathlib import Path
import base64
import logging
import json
import re

logger = logging.getLogger("TacticalAnalyst")

class TacticalAnalyst:
    def __init__(self, provider="openrouter", model_name=None, api_key=None):
        self.provider = provider.lower()
        self.api_key = api_key
        logger.info(f"Analyst initialized. Provider: {self.provider}")
        
        if self.provider == "openrouter":
            self.model_name = model_name or "qwen/qwen-2.5-vl-72b-instruct"
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={"HTTP-Referer": "https://github.com/arie/TAE", "X-Title": "TAE"}
            )
        else:
            self.model_name = model_name or "moondream"

    def analyze_multiple_views(self, image_paths, user_query):
        logger.info(f"Analyzing {len(image_paths)} images for grounding: {user_query}")
        
        filenames_str = ", ".join([Path(p).name for p in image_paths])

        # Using [ymin, xmin, ymax, xmax] standard for higher grounding accuracy
        prompt = (
            f"Locate all instances of '{user_query}' in these images: {filenames_str}\n"
            "Return ONLY a JSON object with this schema:\n"
            "{\n"
            "  \"report\": \"brief summary\",\n"
            "  \"targets\": [\n"
            "    {\"filename\": \"name.jpg\", \"bbox\": [ymin, xmin, ymax, xmax]}\n"
            "  ]\n"
            "}\n"
            "Coordinates must be integers 0-1000. Provide the tightest possible box."
        )

        try:
            if self.provider == "openrouter":
                res_text = self._analyze_openrouter(image_paths, prompt)
            else:
                res_text = self._analyze_ollama(image_paths, prompt)
            
            logger.debug(f"Raw VLM Response: {res_text}")
            
            # Clean response if model adds markdown backticks
            clean_json = re.sub(r'^```json\s*|\s*```$', '', res_text.strip(), flags=re.MULTILINE)
            logger.info(f"VLM Response: {res_text}") 
            return json.loads(clean_json)
            
        except Exception as e:
            logger.error(f"Failed to parse VLM response: {e}")
            return {"report": f"Error: {e}", "targets": []}

    def _analyze_openrouter(self, image_paths, prompt):
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            img_b64 = base64.b64encode(open(path, "rb").read()).decode('utf-8')
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _analyze_ollama(self, image_paths, prompt):
        response = ollama.chat(
            model=self.model_name,
            format='json',
            messages=[{'role': 'user', 'content': prompt, 'images': image_paths}]
        )
        return response['message']['content']