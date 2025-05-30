# nodes.py for ComfyUI-Kontext-API
"""
Custom ComfyUI nodes for the Fal Kontext API.
"""

import os
from typing import Any, Dict
from .api import call_kontext_api

# ComfyUI node API imports (assume available in ComfyUI runtime)
try:
    from comfy.nodes import Node, register_node
except ImportError:
    # For type checking or dev outside ComfyUI
    Node = object
    def register_node(cls):
        return cls

class KontextAPINode(Node):
    """
    Main node for calling the Fal Kontext API.
    Inputs: prompt, image, aspect ratio, etc.
    Outputs: output image(s), info, etc.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE", {}),
                "aspect_ratio": ("STRING", {"default": "Match input image"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0}),
                "output_format": ("STRING", {"default": "jpeg"}),
                "raw": ("BOOLEAN", {"default": False}),
                "image_prompt_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "num_inference_steps": ("INT", {"default": 28, "min": 10, "max": 100}),
                "safety_tolerance": ("INT", {"default": 5, "min": 1, "max": 6}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_images", "info")

    FUNCTION = "run"

    CATEGORY = "Fal/Editing"

    def run(self, prompt, image, aspect_ratio, num_images, seed, guidance_scale, output_format, raw, image_prompt_strength, num_inference_steps, safety_tolerance):
        output_images, info = call_kontext_api(
            prompt, image, aspect_ratio, num_images, seed, guidance_scale, output_format, raw, image_prompt_strength, num_inference_steps, safety_tolerance
        )
        return (output_images, info)

def register_kontext_nodes():
    register_node(KontextAPINode) 