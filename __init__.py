"""
ComfyUI-Kontext-API: Custom nodes for integrating the Fal Kontext API with ComfyUI.
"""

# Node registration will be handled here

def register_nodes():
    # This function will be called by ComfyUI to register all custom nodes
    from .nodes import register_kontext_nodes
    register_kontext_nodes() 