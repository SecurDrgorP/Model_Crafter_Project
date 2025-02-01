import os
from pathlib import Path
from PIL import Image

def convert_images(input_file: str = "invalid_files.txt"):
    """Convert all files listed in invalid_files.txt to JPG format."""
    try:
        # Read the list of invalid files
        with open(input_file, 'r') as f:
            files_to_convert = [line.strip() for line in f.readlines()]
        
        # Convert each file
        for file_path in files_to_convert:
            file_path = Path(file_path)
            if file_path.exists():
                # Open the image
                with Image.open(file_path) as img:
                    # Convert to RGB mode (required for JPG)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPG with the same name
                    new_path = file_path.with_suffix('.jpg')
                    img.save(new_path, 'JPEG', quality=95)
                    print(f"Converted: {file_path} -> {new_path}")
                    
                    # Optionally, remove the original file
                    file_path.unlink()
                    print(f"Removed original: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        print("Conversion complete!")
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


def convert_entire_dataset(root_dir):
    """Convert ALL images to JPEG/RGB regardless of format"""
    for root, _, files in os.walk(root_dir):
        for file in files:
            src = Path(root) / file
            if src.suffix.lower() not in ['.jpg', '.jpeg']:
                try:
                    with Image.open(src) as img:
                        rgb_img = img.convert('RGB')
                        new_path = src.with_suffix('.jpg')
                        rgb_img.save(new_path, 'JPEG', quality=95)
                    src.unlink()  # Remove original
                except Exception as e:
                    print(f"Failed {src}: {e}")
                    src.unlink()


