import os
import base64
from PIL import Image
import io

def get_fal_key():
    key = os.environ.get("FAL_KEY")
    if key:
        return key
    key_file = ".fal_key"
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key = f.read().strip()
            if key:
                return key
    raise RuntimeError("Fal API key not found. Set FAL_KEY env or create .fal_key file.")

def image_to_data_uri(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_str}" 