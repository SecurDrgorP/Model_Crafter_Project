# train/train_cnn.py
import tensorflow as tf
from config import BATCH_SIZE, EPOCHS, MODEL_DIR

def train_model(model, train_gen, val_gen):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen, # it is used to train the model
        epochs=EPOCHS,
        validation_data=val_gen, # it is used to evaluate the model after each epoch
        batch_size=BATCH_SIZE | 32,
        callbacks=[checkpoint, early_stop]
    )
    
    return history