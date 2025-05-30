# nodes.py for ComfyUI-Kontext-API
"""
Custom ComfyUI nodes for the Fal Kontext API.
"""

import os
from typing import Any, Dict
import torch
import numpy as np
from PIL import Image
from .api import call_kontext_api

# Debug flag - set to False to disable debug output
DEBUG = True

# ComfyUI imports - use proper error handling
def pil2tensor(image):
    """Convert PIL image to ComfyUI tensor format."""
    # Convert PIL to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Drop alpha channel
    
    # Convert to tensor in ComfyUI format (H, W, C)
    tensor = torch.from_numpy(img_array)
    
    if DEBUG:
        print(f"[pil2tensor] Output shape: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min():.3f}, max: {tensor.max():.3f}")
    
    return tensor

class KontextAPINode:
    """
    Main node for calling the Fal Kontext API.
    Inputs: prompt, image, seed, disable prompt enhancement.
    Outputs: single output image, info, passed_nsfw_filtering.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE", {}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "disable_prompt_enhancement": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "info", "passed_nsfw_filtering")

    FUNCTION = "execute"

    CATEGORY = "image/generation"

    def execute(self, prompt, image, seed, disable_prompt_enhancement=False):
        if DEBUG:
            print(f"[KontextAPINode] Input image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[KontextAPINode] Prompt: {prompt}")
            print(f"[KontextAPINode] Seed: {seed}")
        
        # Validate inputs
        if not prompt.strip():
            return (image, "Error: Prompt cannot be empty", True)  # Default to True when error
        
        # Handle random seed
        if seed == -1:
            import random
            seed = random.randint(0, 2147483647)
            if DEBUG:
                print(f"[KontextAPINode] Generated random seed: {seed}")
        
        # Set hidden/default values for API call
        aspect_ratio = None  # Always match input image
        guidance_scale = 3.5  # Default
        output_format = "jpeg"  # Not used, but required by API
        raw = disable_prompt_enhancement
        image_prompt_strength = 0.1  # Default
        num_inference_steps = 28  # Default
        safety_tolerance = 6  # Max safety
        num_images = 1  # Always generate only 1 image
        
        try:
            output_images, info, passed_nsfw_filtering = call_kontext_api(
                prompt, image, aspect_ratio, num_images, seed, guidance_scale, 
                output_format, raw, image_prompt_strength, num_inference_steps, safety_tolerance
            )
            
            # Convert PIL image to tensor for ComfyUI
            if output_images and len(output_images) > 0:
                # Take only the first image
                output_image = output_images[0]
                if DEBUG:
                    print(f"[KontextAPINode] Got {len(output_images)} images from API, using first one")
                    print(f"[KontextAPINode] Output PIL image size: {output_image.size}, mode: {output_image.mode}")
                    print(f"[KontextAPINode] Passed NSFW filtering: {passed_nsfw_filtering}")
                
                # Convert to tensor
                output_tensor = pil2tensor(output_image)
                
                # Add batch dimension for ComfyUI
                output_tensor = output_tensor.unsqueeze(0)
                
                if DEBUG:
                    print(f"[KontextAPINode] Final output tensor shape: {output_tensor.shape}")
                
                return (output_tensor, info, passed_nsfw_filtering)
            else:
                # Return original image if no output
                print(f"[KontextAPINode] ERROR: No images returned from API")
                return (image, f"Error: No images returned from API\n{info}", True)
                
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"[KontextAPINode] ERROR: {error_msg}")
            return (image, error_msg, True)

# Add these for ComfyUI node discovery
NODE_CLASS_MAPPINGS = {
    "KontextAPINode": KontextAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KontextAPINode": "Fal Kontext API",
} 