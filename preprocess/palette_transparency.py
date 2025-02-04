

# Handle Palette Transparency
import numpy as np

def rgba_to_rgb(image):
    """Convert RGBA images to RGB by alpha compositing on white background"""
    if image.shape[-1] == 4:
        rgb = image[..., :3]  # Extract RGB channels
        alpha = image[..., 3:]  # Extract alpha channel
        
        # Convert alpha to 0-1 range if needed
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        
        # Composite on white background
        white_background = np.ones_like(rgb)
        return white_background * (1 - alpha) + rgb * alpha
    return image[..., :3]  # Return first 3 channels for RGB images