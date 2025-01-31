import numpy as np
import tensorflow as tf
from typing import Tuple
from config import logger

class FeatureExtractor:
    @staticmethod
    def extract_features(dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Convert TF Dataset to numpy arrays for traditional ML"""
        try:
            images, labels = [], []
            for img, label in dataset.unbatch():
                images.append(img.numpy().flatten())
                labels.append(np.argmax(label.numpy()))
            return np.array(images), np.array(labels)
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    @staticmethod
    def create_hybrid_features(cnn_model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
        """Create CNN-based features for hybrid approach"""
        try:
            feature_model = tf.keras.Model(
                inputs=cnn_model.input,
                outputs=cnn_model.layers[-2].output  # Get penultimate layer
            )
            return feature_model.predict(dataset)
        except Exception as e:
            logger.error(f"Hybrid feature creation failed: {str(e)}")
            raise