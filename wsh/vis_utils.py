import gradio as gr
import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

def resize_image_with_max_side_length(image, max_side_length):
    """Resize an image while maintaining aspect ratio."""
    width, height = image.size
    if width > height:
        new_width = max_side_length
        new_height = int(height * (max_side_length / width))
    else:
        new_height = max_side_length
        new_width = int(width * (max_side_length / height))
    resized_image = image.resize((new_width, new_height))
    return resized_image

def get_image_from_url(image_url: str):
    """Fetch image from URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error loading image from URL {image_url}: {e}")
        # Create a simple error image
        error_img = Image.new('RGB', (300, 200), color=(240, 240, 240))
        return error_img

def local_url_to_image(local_url):
    """Load an image from a local path."""
    try:
        image = Image.open(local_url)
        return image
    except Exception as e:
        print(f"Problem with image {local_url}: {e}")
        # Create a blank white image as a fallback
        image = np.ones([256, 256, 3], dtype=np.uint8) * 255
        return Image.fromarray(image)