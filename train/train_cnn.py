# train/train_cnn.py
import tensorflow as tf
from config import BATCH_SIZE, EPOCHS, MODEL_DIR

# train/train_cnn.py
def train_model(model, train_gen, val_gen, class_weights=None):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        batch_size=BATCH_SIZE,  # FIXED: Removed bitwise OR
        callbacks=[checkpoint],
        class_weight=class_weights  # Added for imbalance handling
    )
    return history