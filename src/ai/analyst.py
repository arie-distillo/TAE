import ollama
from openai import OpenAI
from pathlib import Path
import base64
import logging
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TacticalAnalyst")

class TacticalAnalyst:
    def __init__(self, provider="openrouter", model_name=None, api_key=None):
        self.provider = provider.lower()
        self.api_key = api_key
        
        logger.info(f"Initializing Analyst with provider: {self.provider}")
        
        if self.provider == "openrouter":
            self.model_name = model_name or "qwen/qwen-2.5-vl-72b-instruct"
            if not self.api_key:
                logger.error("OpenRouter API Key is missing!")
            
            # Explicitly setting the client with the API key
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                # OpenRouter sometimes requires these additional headers
                default_headers={
                    "HTTP-Referer": "https://github.com/arie/TAE", 
                    "X-Title": "TAE Tactical Engine",
                }
            )
            logger.info(f"OpenRouter client ready. Model: {self.model_name}")
        else:
            self.model_name = model_name or "moondream"
            logger.info(f"Local Ollama provider ready. Model: {self.model_name}")

    def _encode_image(self, image_path):
        """Encodes local image to base64 with logging."""
        try:
            start_time = time.time()
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logger.debug(f"Encoded {Path(image_path).name} in {time.time()-start_time:.2f}s")
                return encoded
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def analyze_multiple_views(self, image_paths, user_query):
        logger.info(f"Starting Multi-Angle Persistence (MAP) analysis for {len(image_paths)} images.")
        
        prompt = (
            f"You are looking at multiple tactical views of the same target area. "
            f"Synthesize the information from all images to answer: {user_query}. "
            "Report only confirmed intelligence."
        )

        try:
            start_time = time.time()
            if self.provider == "openrouter":
                res = self._analyze_openrouter(image_paths, prompt)
            else:
                res = self._analyze_ollama(image_paths, prompt)
            
            logger.info(f"Analysis complete in {time.time()-start_time:.2f}s")
            return res
        except Exception as e:
            logger.error(f"MAP Reasoning Failed: {str(e)}")
            return f"Analyst Error ({self.provider}): {e}"

    def _analyze_openrouter(self, image_paths, prompt):
        logger.info(f"Sending request to OpenRouter ({self.model_name})...")
        content = [{"type": "text", "text": prompt}]
        
        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}]
        )
        return completion.choices[0].message.content

    def _analyze_ollama(self, image_paths, prompt):
        logger.info(f"Invoking local Ollama inference ({self.model_name})...")
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': image_paths
            }]
        )
        return response['message']['content']