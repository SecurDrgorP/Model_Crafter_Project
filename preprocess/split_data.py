import splitfolders
from pathlib import Path
from config import RAW_DATA_DIR, SPLIT_DATA_DIR, SEED, logger

def split_dataset():
    """Split dataset into train/val/test sets"""
    try:
        logger.info("Starting dataset splitting...")
        
        splitfolders.ratio(
            input=RAW_DATA_DIR,
            output=SPLIT_DATA_DIR,
            seed=SEED,
            ratio=(0.7, 0.15, 0.15),
            group_prefix=None,
            move=False
        )
        
        logger.info(f"Dataset successfully split and saved to {SPLIT_DATA_DIR}")
    
    except Exception as e:
        logger.error(f"Dataset splitting failed: {str(e)}")
        raise