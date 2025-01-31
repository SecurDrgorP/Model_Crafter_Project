import tensorflow as tf
from typing import Tuple
from config import MODEL_DIR, IMG_SIZE, EPOCHS, logger

class CNNTrainer:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        except Exception as e:
            logger.error(f"Model build failed: {str(e)}")
            raise

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> tf.keras.Model:
        try:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_DIR / 'best_cnn.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )

            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=[early_stop, checkpoint]
            )

            self.model.save(str(MODEL_DIR / 'final_cnn.h5'))
            return history
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise