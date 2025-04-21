# train/train_cnn.py
import tensorflow as tf
from config import BATCH_SIZE, EPOCHS, MODEL_DIR

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.best_diff = float('inf')
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        if train_acc is not None and val_acc is not None:
            diff = abs(train_acc - val_acc)
            if diff < self.best_diff or val_loss < self.best_loss:
                self.best_diff = diff
                self.best_loss = val_loss
                self.model.save(self.save_path)
                print(f"Model saved at epoch {epoch + 1} with train-val accuracy diff: {diff:.4f} and val_loss: {val_loss:.4f}")


def train_model(model, train_gen, val_gen, class_weights=None, use_custom_checkpoint=False):
    if use_custom_checkpoint:
        print("Using custom checkpoint logic.")
        checkpoint = CustomModelCheckpoint(str(MODEL_DIR / 'best_model.keras'))
    else:
        print("Using default checkpoint logic.")
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
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        class_weight=class_weights
    )
    return history