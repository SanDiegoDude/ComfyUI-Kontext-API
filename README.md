# ComfyUI-Kontext-API

A custom ComfyUI node for integrating with the Fal Kontext API for advanced image editing and generation.

![image](https://github.com/user-attachments/assets/dd414939-49cc-4f78-a804-2497aa413a23)


## Features

- **Image-to-Image Generation**: Transform images based on text prompts using Fal's Kontext model
- **Single Image Output**: Generates one image per request for optimal ComfyUI compatibility
- **Seed Control**: Use specific seeds for reproducible results or -1 for random
- **Prompt Enhancement**: Optional AI prompt enhancement (can be disabled)
- **Multi-Image Support**: Process multiple images (1 required, up to 3 optional) with the Kontext Max model
- **Automatic Image Resizing**: Large images are automatically resized to fit under 1.25MP

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SanDiegoDude/ComfyUI-Kontext-API.git
```

2. Install required dependencies:
```bash
cd ComfyUI-Kontext-API
pip install -r requirements.txt
```

3. Set up your Fal API key (REQUIRED):

### Getting a Fal API Key
1. Sign up at [https://fal.ai/](https://fal.ai/)
2. Go to your dashboard
3. Generate an API key

### Setting up the API Key
Choose one of these methods:

**Option 1: Create a .fal_key file (Recommended)**
1. Create a file named `.fal_key` in the `ComfyUI-Kontext-API` directory
2. Put your Fal API key in this file (just the key, nothing else)
3. The file should contain only your API key, for example:
   ```
   fal_1234567890abcdef
   ```
4. Note: The `.fal_key` file is already in `.gitignore` for security

**Option 2: Set environment variable**
```bash
export FAL_KEY="your_fal_api_key_here"
```

## Usage

1. Find either node in the `image/generation` category:
   - **Fal Kontext API**: For single image processing
   - **Fal Kontext[Max] Multi-Image API**: For processing multiple images
2. Connect your input image(s):
   - Single node: Connect one image to the `image` input
   - Multi node: Connect 1-4 images to `image1` through `image4` inputs
3. Enter your prompt describing the desired transformation
4. Adjust settings as needed

## Node Inputs

### Fal Kontext API (Single Image)
- **prompt** (STRING): Text description of the desired image transformation
- **image** (IMAGE): Input image to transform
- **seed** (INT): Random seed (-1 for random, default: -1)
- **disable_prompt_enhancement** (BOOLEAN): Disable AI prompt enhancement (default: False)

### Fal Kontext[Max] Multi-Image API
- **prompt** (STRING): Text description of the desired image transformation
- **image1** (IMAGE): Required input image
- **image2** (IMAGE, optional): Additional input image
- **image3** (IMAGE, optional): Additional input image
- **image4** (IMAGE, optional): Additional input image
- **aspect_ratio** (COMBO): Output image aspect ratio
- **num_images** (INT): Number of images to generate (1-4)
- **seed** (INT): Random seed (-1 for random, default: -1)
- **guidance_scale** (FLOAT): How closely to follow the prompt (default: 3.5)
- **output_format** (COMBO): Output image format (jpeg/png)
- **disable_prompt_enhancement** (BOOLEAN): Disable AI prompt enhancement (default: False)
- **image_prompt_strength** (FLOAT): Strength of image influence (0-1, default: 0.1)
- **num_inference_steps** (INT): Number of generation steps (default: 30)
- **safety_tolerance** (COMBO): NSFW filter tolerance (1-6, default: 6)

## Node Outputs

Both nodes provide:
- **image** (IMAGE): Generated image(s) as tensor(s)
- **info** (STRING): API response information including:
  - Request ID
  - Seed used for generation
  - Safety check status (✓ passed or ⚠️ blocked)
  - Any error messages or warnings
- **passed_nsfw_filtering** (BOOLEAN): True if content passed safety checks, False if blocked
  - Designed to work with "save on true" nodes in ComfyUI
  - Returns True for safe content that should be saved
  - Returns False for blocked content that should be skipped

## Example Prompts

- "Change the car color to red"
- "Convert to pencil sketch with natural graphite lines"
- "Transform to oil painting with visible brushstrokes"
- "It's now snowing, everything is covered in snow"
- "Using this style, a bunny, a dog and a cat are having a tea party"

## Troubleshooting

### Debug Mode
The node includes detailed debug logging to help diagnose issues. To disable debug output once everything is working:

1. Open `nodes.py`
2. Change `DEBUG = True` to `DEBUG = False` at the top of the file

### Image Handling Errors
The node properly handles ComfyUI's tensor format (B, H, W, C) and converts between PIL images and tensors automatically.

### API Key Issues
Ensure your Fal API key is properly set either as an environment variable or in the `.fal_key` file.

If you see an error like "FAL API KEY NOT FOUND!", follow these steps:
1. Make sure you have created the `.fal_key` file in the `ComfyUI-Kontext-API` directory (not in ComfyUI root)
2. Check that the file contains only your API key with no extra spaces or quotes
3. Verify the key is valid by testing it on [fal.ai](https://fal.ai/)

### Dependencies
Make sure you have installed all required packages:
```bash
pip install fal-client Pillow numpy torch
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
