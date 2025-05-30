import requests
from .utils import get_fal_key, image_to_data_uri

# TODO: import fal_client if available in the ComfyUI environment
try:
    import fal_client
except ImportError:
    fal_client = None

def call_kontext_api(prompt, image, aspect_ratio, num_images, seed, guidance_scale, output_format, raw, image_prompt_strength, num_inference_steps, safety_tolerance):
    """
    Call the Fal Kontext API and return output images and info string.
    """
    # Convert image to data URI
    image_url = image_to_data_uri(image, format='PNG')
    payload = {
        "prompt": prompt,
        "image_url": image_url,
        "safety_tolerance": str(safety_tolerance),
        "seed": int(seed),
        "guidance_scale": float(guidance_scale),
        "num_images": int(num_images),
        "output_format": output_format,
        "raw": bool(raw),
        "image_prompt_strength": float(image_prompt_strength),
        "num_inference_steps": int(num_inference_steps)
    }
    if aspect_ratio and aspect_ratio != "Match input image":
        payload["aspect_ratio"] = aspect_ratio
    # TODO: Actually call the API using fal_client and return images/info
    # For now, return stub
    return [image], f"Stub: payload={payload}" 