"""
ComfyUI-Kontext-API: Custom nodes for integrating the Fal Kontext API with ComfyUI.
"""

print("[ComfyUI-Kontext-API] __init__.py loaded.")

try:
    from .nodes import KontextAPINode
    print("[ComfyUI-Kontext-API] KontextAPINode imported.")
except Exception as e:
    print(f"[ComfyUI-Kontext-API] Failed to import KontextAPINode: {e}")

NODE_CLASS_MAPPINGS = {
    "KontextAPINode": KontextAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KontextAPINode": "Fal Kontext API",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 