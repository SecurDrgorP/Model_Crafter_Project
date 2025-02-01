import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, logger

def validate_images():
    """Check for corrupt/non-image files in dataset"""
    invalid_files = []
    
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            file_path = Path(root) / file
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Check valid extensions
                if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    invalid_files.append(str(file_path))
            except (IOError, OSError, Image.DecompressionBombError) as e:
                invalid_files.append(str(file_path))
                logger.warning(f"Invalid image file: {file_path} - {str(e)}")
    
    # Save list of invalid files
    with open('invalid_files.txt', 'w') as f:
        f.write('\n'.join(invalid_files))
    
    logger.info(f"Found {len(invalid_files)} problematic files. List saved to invalid_files.txt")
    return invalid_files

