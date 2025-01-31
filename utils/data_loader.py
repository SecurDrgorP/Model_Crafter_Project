from pathlib import Path
import tensorflow as tf
from config import SPLIT_DATA_DIR, IMG_SIZE, BATCH_SIZE, logger

class DataLoader:
    def __init__(self):
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2)
        ])

    def load_datasets(self):
        try:
            # Load datasets from split directories
            train_ds = tf.keras.utils.image_dataset_from_directory(
                SPLIT_DATA_DIR / "train",
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical'
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                SPLIT_DATA_DIR / "val",
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical'
            )

            test_ds = tf.keras.utils.image_dataset_from_directory(
                SPLIT_DATA_DIR / "test",
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                label_mode='categorical'
            )

            # Apply preprocessing
            normalization = tf.keras.layers.Rescaling(1./255)
            return (
                train_ds.map(lambda x, y: (self.augmentation(normalization(x)), y)),
                val_ds.map(lambda x, y: (normalization(x), y)),
                test_ds.map(lambda x, y: (normalization(x), y))
            )
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise