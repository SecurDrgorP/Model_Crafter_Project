# utils/data_loader.py
import tensorflow as tf
from config import SPLIT_DATA_DIR, IMG_SIZE, BATCH_SIZE

def create_data_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,          # Reduced from 40
        width_shift_range=0.1,      # Reduced from 0.2
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: x[:, :, :3]  # Force RGB format
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = test_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator