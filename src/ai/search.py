import torch
import clip # Ensure this is installed via the git link in requirements
import cv2
from PIL import Image

class SearchLibrarian:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_image(self, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
        return features.cpu().numpy().flatten()

    # --- ADD THIS METHOD ---
    def encode_text(self, text_query):
        """Converts a text string into a 512-dim CLIP vector."""
        text_tokens = clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.cpu().numpy().flatten()