import os
import requests
from .utils import get_fal_key, image_to_data_uri
from PIL import Image
import io
import base64
import torch
import numpy as np

# Import DEBUG flag from nodes
try:
    from .nodes import DEBUG
except ImportError:
    DEBUG = True

# ComfyUI tensor to PIL conversion
def tensor2pil(img_tensor):
    """Convert ComfyUI tensor (B, H, W, C) to PIL image."""
    # Handle list of tensors
    if isinstance(img_tensor, list):
        img_tensor = img_tensor[0]
    
    if not torch.is_tensor(img_tensor):
        raise TypeError(f"Input is not a torch tensor: {type(img_tensor)}")
    
    # Clone and detach to avoid modifying original
    img = img_tensor.detach().cpu().clamp(0, 1)
    
    # Debug info
    if DEBUG:
        print(f"[tensor2pil] Input shape: {img.shape}, dtype: {img.dtype}")
    
    # Handle different tensor formats
    shape = img.shape
    
    # Remove batch dimension if present
    if len(shape) == 4:
        # Assuming (B, H, W, C) format
        if shape[0] == 1:
            img = img[0]  # Remove batch dimension
        else:
            # Take first image from batch
            img = img[0]
            if DEBUG:
                print(f"[tensor2pil] Taking first image from batch of {shape[0]}")
    
    # Now we should have (H, W, C)
    if len(img.shape) == 3:
        # Check if it's (C, H, W) format instead
        if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # Likely (C, H, W) format, transpose to (H, W, C)
            img = img.permute(1, 2, 0)
        
        # Handle different channel counts
        if img.shape[2] == 1:  # Grayscale
            img = img.repeat(1, 1, 3)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]  # Drop alpha
    elif len(img.shape) == 2:
        # Grayscale image, add channel dimension
        img = img.unsqueeze(-1).repeat(1, 1, 3)
    
    # Convert to numpy and then PIL
    img_np = (img * 255).byte().numpy()
    pil_img = Image.fromarray(img_np)
    
    if DEBUG:
        print(f"[tensor2pil] Output PIL size: {pil_img.size}, mode: {pil_img.mode}")
    
    return pil_img

# TODO: import fal_client if available in the ComfyUI environment
try:
    import fal_client
    from fal_client import submit, InProgress
    # Set up fal_client with API key
    from .utils import get_fal_key
    try:
        fal_key = get_fal_key()
        os.environ["FAL_KEY"] = fal_key
        if DEBUG:
            print(f"[fal_client] FAL_KEY set successfully")
    except RuntimeError as e:
        # Re-raise with the detailed error message
        print(str(e))
        fal_client = None
        raise e
    except Exception as e:
        print(f"Warning: Could not set FAL_KEY: {e}")
        fal_client = None
except ImportError as e:
    fal_client = None
    print(f"Warning: fal_client not installed or import error: {e}")
    print("Please install fal-client with: pip install fal-client")
except RuntimeError:
    # Key error already printed above
    fal_client = None

def call_kontext_api(prompt, image, aspect_ratio, num_images, seed, guidance_scale, output_format, raw, image_prompt_strength, num_inference_steps, safety_tolerance):
    """
    Call the Fal Kontext API and return output images and info string.
    """
    if fal_client is None:
        # Try to import and get key again to show the proper error message
        try:
            from .utils import get_fal_key
            get_fal_key()
        except RuntimeError as e:
            # This will show the detailed key error message
            raise e
        except:
            pass
        # If we're here, it's a different issue
        raise ImportError("fal_client is not installed. Please install fal-client in your environment with: pip install fal-client")

    # Convert tensor to PIL image if needed
    pil_image = tensor2pil(image) if not isinstance(image, Image.Image) else image
    image_url = image_to_data_uri(pil_image, format='PNG')
    
    if DEBUG:
        print(f"[call_kontext_api] Input PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        print(f"[call_kontext_api] Generated image_url: (base64 image)")
    
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
        "num_inference_steps": int(num_inference_steps),
        "sync_mode": True  # Get images directly without CDN
    }
    if aspect_ratio and aspect_ratio != "Match input image":
        payload["aspect_ratio"] = aspect_ratio

    if DEBUG:
        debug_payload = payload.copy()
        debug_payload["image_url"] = "(base64 image)"
        print(f"[call_kontext_api] Payload: {debug_payload}")

    logs = []
    def on_queue_update(update):
        if hasattr(update, 'logs'):
            for log in update.logs:
                log_msg = log.get("message", str(log)) if isinstance(log, dict) else str(log)
                logs.append(log_msg)
                if DEBUG:
                    print(f"[call_kontext_api] Queue log: {log_msg}")

    try:
        if DEBUG:
            print(f"[call_kontext_api] Calling Fal API...")
            # Debug: Check what's available in fal_client
            print(f"[call_kontext_api] fal_client type: {type(fal_client)}")
            print(f"[call_kontext_api] fal_client attributes: {dir(fal_client)}")
        
        # Try different methods based on what's available
        if hasattr(fal_client, 'subscribe'):
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext",
                arguments=payload,
                with_logs=True,
                on_queue_update=on_queue_update,
            )
        elif hasattr(fal_client, 'run'):
            # Try the run method as an alternative
            result = fal_client.run(
                "fal-ai/flux-pro/kontext",
                arguments=payload,
            )
        elif hasattr(fal_client, 'submit'):
            # Try submit method - this might return a different structure
            handle = fal_client.submit(
                "fal-ai/flux-pro/kontext",
                arguments=payload,
            )
            # Wait for completion
            result = handle.get()
        else:
            # Try as a callable module
            result = fal_client(
                "fal-ai/flux-pro/kontext",
                **payload
            )
        
        if DEBUG:
            print(f"[call_kontext_api] API call completed")
            print(f"[call_kontext_api] Result type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"[call_kontext_api] Result attributes: {list(result.__dict__.keys())}")
    except Exception as e:
        error_msg = f"Fal API call failed: {str(e)}"
        print(f"[call_kontext_api] ERROR: {error_msg}")
        import traceback
        if DEBUG:
            print(f"[call_kontext_api] Full traceback:\n{traceback.format_exc()}")
        return [], error_msg

    # Parse result
    output_images = []
    info_lines = []
    passed_nsfw_filtering = True  # Default to True (not blocked)
    
    # Extract request ID
    request_id = getattr(result, 'request_id', None)
    if not request_id and hasattr(result, '__getitem__'):
        try:
            request_id = result.get('request_id')
        except:
            pass
    
    if request_id:
        info_lines.append(f"Request ID: {request_id}")
    
    # Extract seed
    result_seed = None
    if hasattr(result, 'seed'):
        result_seed = result.seed
    elif hasattr(result, '__getitem__'):
        try:
            result_seed = result.get('seed')
        except:
            pass
    elif hasattr(result, 'data') and isinstance(result.data, dict):
        result_seed = result.data.get('seed')
    
    if result_seed is not None:
        info_lines.append(f"Seed: {result_seed}")
    
    if DEBUG:
        print(f"[call_kontext_api] Parsing result...")
        print(f"[call_kontext_api] Result content: {str(result)[:500]}...")  # First 500 chars
    
    # Try to extract images from various possible response formats
    images_data = None
    
    # Method 1: Direct attribute access
    if hasattr(result, 'images'):
        images_data = result.images
        if DEBUG:
            print(f"[call_kontext_api] Found images via result.images")
    # Method 2: Dictionary-style access
    elif hasattr(result, '__getitem__'):
        try:
            images_data = result.get('images') or result.get('output')
            if DEBUG:
                print(f"[call_kontext_api] Found images via dict access")
        except:
            pass
    # Method 3: Data attribute
    elif hasattr(result, 'data'):
        if isinstance(result.data, dict):
            images_data = result.data.get('images') or result.data.get('output')
            if DEBUG:
                print(f"[call_kontext_api] Found images via result.data")
        else:
            images_data = result.data
            if DEBUG:
                print(f"[call_kontext_api] Using result.data directly")
    # Method 4: Direct dict
    elif isinstance(result, dict):
        images_data = result.get('images') or result.get('output')
        if DEBUG:
            print(f"[call_kontext_api] Result is dict, found images")
    
    # Extract NSFW status
    has_nsfw_concepts = None
    if hasattr(result, 'has_nsfw_concepts'):
        has_nsfw_concepts = result.has_nsfw_concepts
    elif hasattr(result, '__getitem__'):
        try:
            has_nsfw_concepts = result.get('has_nsfw_concepts')
        except:
            pass
    elif hasattr(result, 'data') and isinstance(result.data, dict):
        has_nsfw_concepts = result.data.get('has_nsfw_concepts')
    
    if has_nsfw_concepts is not None:
        if isinstance(has_nsfw_concepts, list) and len(has_nsfw_concepts) > 0:
            # has_nsfw_concepts[0] is True when NSFW detected, so we invert it
            passed_nsfw_filtering = not bool(has_nsfw_concepts[0])
        else:
            # Invert the boolean
            passed_nsfw_filtering = not bool(has_nsfw_concepts)
        
        if passed_nsfw_filtering:
            info_lines.append("✓ Content passed safety check")
        else:
            info_lines.append("⚠️ NSFW content detected and blocked")
    
    if images_data is None:
        error_msg = f"Could not find images in result. Result type: {type(result)}"
        print(f"[call_kontext_api] ERROR: {error_msg}")
        if DEBUG and hasattr(result, '__dict__'):
            print(f"[call_kontext_api] Result dict: {result.__dict__}")
        info_lines.append(f"Error: {error_msg}")
        images_data = []
    elif DEBUG:
        print(f"[call_kontext_api] Found {len(images_data) if hasattr(images_data, '__len__') else 'unknown number of'} images")
    
    # Process images
    for i, img_data in enumerate(images_data):
        try:
            if DEBUG:
                print(f"[call_kontext_api] Processing image {i+1}/{len(images_data)}")
            
            img_url = None  # Track the URL for info output
            
            if isinstance(img_data, dict):
                # Handle dict with url field
                img_url = img_data.get("url") or img_data.get("image_url")
                if img_url:
                    img_data = img_url
            
            if isinstance(img_data, str):
                if img_data.startswith("data:image"):
                    # base64 data URI
                    if DEBUG:
                        print(f"[call_kontext_api] Image {i+1} is base64 data URI")
                    header, base64_data = img_data.split(",", 1)
                    img_bytes = io.BytesIO(base64.b64decode(base64_data))
                    pil_img = Image.open(img_bytes)
                    output_images.append(pil_img)
                    info_lines.append(f"Image {i+1}: Base64 encoded ({pil_img.size[0]}x{pil_img.size[1]})")
                elif img_data.startswith("http"):
                    # URL, try to fetch
                    img_url = img_data
                    if DEBUG:
                        print(f"[call_kontext_api] Image {i+1} is URL: {img_data}")
                    resp = requests.get(img_data)
                    resp.raise_for_status()
                    pil_img = Image.open(io.BytesIO(resp.content))
                    output_images.append(pil_img)
                    info_lines.append(f"Image {i+1}: {img_url}")
                else:
                    error_msg = f"Unknown image format for image {i+1}"
                    print(f"[call_kontext_api] ERROR: {error_msg}")
                    info_lines.append(f"Error: {error_msg}")
            elif isinstance(img_data, Image.Image):
                # Already a PIL Image
                if DEBUG:
                    print(f"[call_kontext_api] Image {i+1} is already PIL Image")
                output_images.append(img_data)
                info_lines.append(f"Image {i+1}: PIL Image ({img_data.size[0]}x{img_data.size[1]})")
        except Exception as e:
            error_msg = f"Failed to process image {i+1}: {str(e)}"
            print(f"[call_kontext_api] ERROR: {error_msg}")
            info_lines.append(f"Error: {error_msg}")
    
    if DEBUG:
        print(f"[call_kontext_api] Successfully processed {len(output_images)} images")
        print(f"[call_kontext_api] Passed NSFW filtering: {passed_nsfw_filtering}")
    
    if not output_images:
        # Fallback to input image
        error_msg = "No output images from API, returning input image"
        print(f"[call_kontext_api] WARNING: {error_msg}")
        info_lines.append(f"Warning: {error_msg}")
        output_images = [tensor2pil(image) if not isinstance(image, Image.Image) else image]
    
    # Add logs to info
    if logs:
        info_lines.append("API Logs:\n" + "\n".join(logs))
    
    # Check for errors in result
    if hasattr(result, 'error') and result.error:
        error_msg = f"API Error: {result.error}"
        print(f"[call_kontext_api] ERROR: {error_msg}")
        info_lines.append(f"Error: {error_msg}")
    
    info = "\n".join(info_lines)
    
    return output_images, info, passed_nsfw_filtering 