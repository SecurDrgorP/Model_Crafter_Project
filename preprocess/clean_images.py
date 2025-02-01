import os
from pathlib import Path
from PIL import Image
import shutil

class ImageCleaner:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        self.backup_dir = self.data_dir.parent / 'corrupted_backup'
        self.issue_count = 0

    def _create_backup(self):
        self.backup_dir.mkdir(exist_ok=True)

    def clean_dataset(self):
        self._create_backup()
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(root) / file
                self._process_file(file_path)
        print(f"Cleaning complete. Found {self.issue_count} issues.")

    def _process_file(self, file_path):
        # Check extension validity
        ext = file_path.suffix.lower()
        if ext not in self.valid_extensions:
            self._handle_invalid(file_path, f"Invalid extension: {ext}")
            return

        # Verify image content
        try:
            # Use context manager for proper resource handling
            with Image.open(file_path) as img:
                img.verify()
                
                # Additional check for unsupported modes
                if img.mode not in ['1', 'L', 'P', 'RGB', 'RGBA']:
                    raise ValueError(f"Unsupported color mode: {img.mode}")
                    
        except Exception as e:
            self._handle_invalid(file_path, f"Invalid image: {str(e)}")

    def _handle_invalid(self, file_path, reason):
        self.issue_count += 1
        backup_path = self.backup_dir / file_path.relative_to(self.data_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Removing problematic file: {file_path} | Reason: {reason}")
        shutil.move(str(file_path), str(backup_path))
