# ComfyUI-Kontext-API

Custom ComfyUI nodes for the [Fal Kontext API](https://fal.ai/models/fal-ai/flux-pro/kontext/api?platform=python).

## Features
- Edit images using the Fal Kontext API from within ComfyUI workflows
- Supports prompt, image input, aspect ratio, batch size, seed, guidance scale, output format, raw mode, prompt strength, inference steps, and safety tolerance
- Reads API key from `FAL_KEY` environment variable or `.fal_key` file (same as the Gradio UI)
- Output is a processed image and info string

## Installation

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Place this folder in your ComfyUI custom nodes directory, or add it to your Python path.

## API Key Setup
- The node will look for the `FAL_KEY` environment variable, or a `.fal_key` file in the same directory.
- If neither is found, it will raise an error.

## Usage
- Add the "Kontext API" node to your ComfyUI workflow.
- Connect an image and set your prompt and other parameters.
- The node will call the Fal Kontext API and return the output image(s) and info string.

See the main repo for more details and the original Gradio UI implementation. 