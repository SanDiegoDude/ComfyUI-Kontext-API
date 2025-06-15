"""
ComfyUI-Kontext-API: Custom nodes for integrating the Fal Kontext API with ComfyUI.
"""

print("[ComfyUI-Kontext-API] __init__.py loaded.")

try:
    from .nodes import KontextAPINode
    from .api import FalKontextMaxMultiImageNode
    print("[ComfyUI-Kontext-API] KontextAPINode and FalKontextMaxMultiImageNode imported.")
except Exception as e:
    print(f"[ComfyUI-Kontext-API] Failed to import nodes: {e}")

NODE_CLASS_MAPPINGS = {
    "KontextAPINode": KontextAPINode,
    "FalKontextMaxMultiImageNode": FalKontextMaxMultiImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KontextAPINode": "Fal Kontext API",
    "FalKontextMaxMultiImageNode": "Fal Kontext[Max] Multi-Image API",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 