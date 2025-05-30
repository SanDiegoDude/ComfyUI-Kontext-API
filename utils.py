import os
import base64
from PIL import Image
import io

# Import DEBUG flag
try:
    from .nodes import DEBUG
except ImportError:
    DEBUG = True

def get_fal_key():
    """
    Get the Fal API key from environment variable or .fal_key file.
    The .fal_key file should be in the same directory as this module.
    """
    if DEBUG:
        print("[get_fal_key] Looking for FAL API key...")
    
    # First, check environment variable
    key = os.environ.get("FAL_KEY")
    if key:
        if DEBUG:
            print("[get_fal_key] Found FAL_KEY in environment")
        return key
    
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    key_file_path = os.path.join(module_dir, ".fal_key")
    
    if DEBUG:
        print(f"[get_fal_key] Looking for .fal_key file at: {key_file_path}")
    
    if os.path.exists(key_file_path):
        try:
            with open(key_file_path, "r") as f:
                key = f.read().strip()
                if key:
                    if DEBUG:
                        print("[get_fal_key] Found FAL_KEY in .fal_key file")
                    return key
                else:
                    raise ValueError(".fal_key file is empty")
        except Exception as e:
            raise RuntimeError(f"Error reading .fal_key file: {e}")
    
    # If we get here, no key was found
    raise RuntimeError(
        "\n\n" + "="*60 + "\n" +
        "FAL API KEY NOT FOUND!\n" + 
        "="*60 + "\n\n" +
        "To use the Kontext API node, you need to provide your Fal API key.\n\n" +
        "Option 1: Create a .fal_key file\n" +
        f"  1. Create a file named '.fal_key' in: {module_dir}\n" +
        "  2. Put your Fal API key in this file (just the key, no quotes or extra text)\n" +
        "  3. The file should contain only your API key, like: fal_1234567890abcdef\n\n" +
        "Option 2: Set an environment variable\n" +
        "  Set FAL_KEY environment variable with your API key\n\n" +
        "To get a Fal API key:\n" +
        "  1. Sign up at https://fal.ai/\n" +
        "  2. Go to your dashboard\n" +
        "  3. Generate an API key\n\n" +
        "Note: The .fal_key file is already in .gitignore for security.\n" +
        "="*60 + "\n"
    )

def image_to_data_uri(img: Image.Image, format: str = "PNG") -> str:
    if DEBUG:
        print(f"[image_to_data_uri] Converting PIL image to data URI, format: {format}")
        print(f"[image_to_data_uri] Image size: {img.size}, mode: {img.mode}")
    
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/{format.lower()};base64,{base64_str}"
    
    if DEBUG:
        print(f"[image_to_data_uri] Generated data URI length: {len(data_uri)}")
    
    return data_uri 