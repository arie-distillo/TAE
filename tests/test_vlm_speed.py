import io
import os
import time
import base64
import re
import requests
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "qwen/qwen-vl-plus"
IMAGE_PATH = "DJI_20221020150359_0031_W.JPG"
OBJECT_NAME = "helipad"
TEST_SIZES = [448, 672, 896, 1120, 1344, 1568, 1792, 2016, 2464, 3360]

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def center_crop(img, crop_size):
    width, height = img.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    return img.crop((left, top, right, bottom))

def parse_bbox(text):
    """
    Extracts numbers from VLM response. 
    Qwen usually returns [ymin, xmin, ymax, xmax] in normalized (0-1000) scale.
    """
    nums = re.findall(r"(\d+)", text)
    if len(nums) >= 4:
        return [int(n) for n in nums[:4]]
    return None

def is_bbox_valid(bbox):
    """
    Validates if the bbox contains the center of the image.
    Normalized center is (500, 500).
    """
    if not bbox: return False
    ymin, xmin, ymax, xmax = bbox
    # Check if the center point (500, 500) falls within the box
    return xmin <= 500 <= xmax and ymin <= 500 <= ymax

def query_vlm_with_validation(image_base64, object_name):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Find the {object_name} and return the bounding box [ymin, xmin, ymax, xmax]."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
    }
    
    start_time = time.time()
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        bbox = parse_bbox(content)
        valid = is_bbox_valid(bbox)
        return (end_time - start_time), valid, content
    else:
        return None, False, response.text

def run_analysis():
    original_img = Image.open(IMAGE_PATH)
    times = []
    valid_count = 0

    print(f"Analyzing {MODEL_NAME} on OpenRouter...")
    
    for size in TEST_SIZES:
        cropped = center_crop(original_img, size)
        img_b64 = encode_image(cropped)
        
        duration, is_valid, raw_text = query_vlm_with_validation(img_b64, OBJECT_NAME)
        
        if duration:
            status = "PASS" if is_valid else "FAIL (Off-center)"
            print(f"Size {size:4d}: {duration:.2f}s | Grounding: {status} | Response: {raw_text[:30]}...")
            times.append(duration)
            if is_valid: valid_count += 1
        else:
            print(f"Size {size:4d}: API Error")

    # Final Summary
    print("-" * 40)
    print(f"Accuracy: {valid_count/len(TEST_SIZES)*100:.1f}% | Avg Speed: {sum(times)/len(times):.2f}s")

    plt.plot(TEST_SIZES, times, marker='x', color='red')
    plt.title("Performance vs Resolution (OpenRouter)")
    plt.xlabel("Image Size (px)")
    plt.ylabel("Inference Time (s)")
    plt.show()

if __name__ == "__main__":
    run_analysis()