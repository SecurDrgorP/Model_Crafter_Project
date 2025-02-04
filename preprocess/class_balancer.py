# utils/class_balancer.py
import numpy as np
from collections import Counter
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from config import SEED, logger

class ClassBalancer:
    def __init__(self, train_gen):
        self.train_gen = train_gen
        self.X, self.y = self._extract_data()
        self.class_counts = Counter(self.y)
        self.class_indices = train_gen.class_indices
        self.class_indices_inv = {v: k for k, v in self.class_indices.items()}
        self.num_classes = train_gen.num_classes
        self.original_shape = self.X.shape[1:]  # Keep original image shape

    def _extract_data(self):
        """Convert generator data to numpy arrays and flatten for SMOTE"""
        self.train_gen.reset()
        X, y = [], []
        
        for _ in range(len(self.train_gen)):
            x_batch, y_batch = next(self.train_gen)
            # Flatten images for SMOTE compatibility
            X.append(x_batch.reshape(x_batch.shape[0], -1))  
            y.append(np.argmax(y_batch, axis=1))
        
        return np.concatenate(X), np.concatenate(y)

    def analyze_balance(self):
        """Log class distribution analysis"""
        logger.info("Class distribution analysis:")
        for class_idx, count in self.class_counts.items():
            class_name = self.class_indices_inv[class_idx]
            logger.info(f"{class_name}: {count} samples")

    def apply_balance(self, strategy='augmentation'):
        """Return balanced data generator"""
        if strategy == 'augmentation':
            return self._balanced_generator()
        elif strategy == 'smote':
            return self._balance_via_smote()
        raise ValueError(f"Invalid strategy: {strategy}")

    def _balanced_generator(self):
        """Create balanced batch generator with proper reshaping"""
        base_gen = BalancedBatchGenerator(
            self.X,
            self.y,
            sampler=SMOTE(),
            batch_size=self.train_gen.batch_size,
            keep_sparse=True,
            random_state=SEED
        )
        return self.ReshapingWrapper(base_gen)

    def _balance_via_smote(self):
        """Apply SMOTE and reshape back to original format"""
        smote = SMOTE(random_state=SEED)
        X_res, y_res = smote.fit_resample(self.X, self.y)
        
        # Reshape back to original image dimensions
        X_res = X_res.reshape(-1, *self.original_shape)
        return X_res, tf.keras.utils.to_categorical(y_res, num_classes=self.num_classes)

    class ReshapingWrapper:
        """Wrapper to reshape flattened SMOTE output back to images"""
        def __init__(self, base_gen):
            self.base_gen = base_gen
            self.batch_size = base_gen.batch_size
            
        def __len__(self):
            return len(self.base_gen)
        
        def __getitem__(self, idx):
            x_flat, y = self.base_gen[idx]
            # Reshape to original image dimensions
            x = x_flat.reshape(-1, *self.base_gen.sampler_.original_shape)
            return x, tf.keras.utils.to_categorical(y, num_classes=self.base_gen.sampler_.num_classes)
        
        def on_epoch_end(self):
            self.base_gen.on_epoch_end()