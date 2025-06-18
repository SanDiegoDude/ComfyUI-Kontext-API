#!/usr/bin/env python3
"""
Test script to verify Fal upload functionality works correctly.
This script tests the upload_image_to_fal function from the api module.
"""

import os
import sys
import tempfile
from PIL import Image
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api import upload_image_to_fal, tensor2pil
    print("✓ Successfully imported api module")
except ImportError as e:
    print(f"✗ Failed to import api module: {e}")
    sys.exit(1)

def create_test_image(width=512, height=512):
    """Create a simple test image."""
    # Create a gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img_array[y, x] = [
                int(255 * x / width),  # Red gradient
                int(255 * y / height), # Green gradient
                128  # Fixed blue
            ]
    return Image.fromarray(img_array)

def test_upload_functionality():
    """Test the upload functionality."""
    print("\n=== Testing Fal Upload Functionality ===\n")
    
    # Check if FAL_KEY is available
    if not os.environ.get('FAL_KEY'):
        print("⚠️  FAL_KEY not found in environment")
        print("   Please set FAL_KEY environment variable or create .fal_key file")
        print("   Skipping upload test...")
        return False
    
    print("✓ FAL_KEY found in environment")
    
    # Create a test image
    print("Creating test image...")
    test_img = create_test_image(256, 256)  # Smaller image for faster upload
    print(f"✓ Created test image: {test_img.size} pixels")
    
    try:
        # Test upload
        print("Uploading image to Fal storage...")
        url = upload_image_to_fal(test_img)
        print(f"✓ Successfully uploaded image to: {url}")
        
        # Verify URL format
        if url.startswith('http'):
            print("✓ URL format looks correct")
            return True
        else:
            print("✗ URL format looks incorrect")
            return False
            
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False

def test_tensor_conversion():
    """Test tensor to PIL conversion."""
    print("\n=== Testing Tensor Conversion ===\n")
    
    # Create a test image
    test_img = create_test_image(100, 100)
    print(f"✓ Created test image: {test_img.size}")
    
    # Convert to numpy array and then to tensor format
    img_array = np.array(test_img).astype(np.float32) / 255.0
    tensor = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(f"✓ Created tensor with shape: {tensor.shape}")
    
    try:
        # Test conversion back to PIL
        pil_img = tensor2pil(tensor)
        print(f"✓ Successfully converted tensor back to PIL: {pil_img.size}")
        return True
    except Exception as e:
        print(f"✗ Tensor conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("ComfyUI-Kontext-API Upload Test")
    print("=" * 40)
    
    # Test tensor conversion
    tensor_test_passed = test_tensor_conversion()
    
    # Test upload functionality
    upload_test_passed = test_upload_functionality()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Tensor Conversion: {'✓ PASSED' if tensor_test_passed else '✗ FAILED'}")
    print(f"Upload Functionality: {'✓ PASSED' if upload_test_passed else '✗ FAILED'}")
    
    if tensor_test_passed and upload_test_passed:
        print("\n🎉 All tests passed! The upload functionality should work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 