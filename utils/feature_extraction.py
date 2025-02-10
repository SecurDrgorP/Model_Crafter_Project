
import numpy as np
from config import logger


def extract_features(generator):
    """Extract features from image generator with progress tracking"""
    X, y = [], []
    total_batches = len(generator)
    
    for batch_idx in range(total_batches):
        images, labels = next(generator)
        X.append(images.reshape(images.shape[0], -1))
        y.append(np.argmax(labels, axis=1))
        
        # Show progress every 10%
        if (batch_idx + 1) % (total_batches // 10) == 0:
            logger.info(f"Processed {batch_idx+1}/{total_batches} batches")
    
    logger.info("Feature extraction completed")
    return np.vstack(X), np.concatenate(y)