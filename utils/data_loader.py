import numpy as np
import tensorflow as tf
from config import COLOR_MODEL, SEED, SPLIT_DATA_DIR, IMG_SIZE, BATCH_SIZE
from preprocess.palette_transparency import rgba_to_rgb
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(generator):
    classes = generator.classes
    return compute_class_weight('balanced', classes=np.unique(classes), y=classes)

def create_data_generators():
    # Enhanced augmentation for fruit/vegetable defects
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7,1.3],
        preprocessing_function=rgba_to_rgb
    )

    # Add test-time augmentation for better evaluation
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=rgba_to_rgb
    )

    # Create generators with optimized caching
    train_gen = train_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode=COLOR_MODEL,
        shuffle=True,
        seed=SEED
    )

    val_gen = test_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode=COLOR_MODEL,
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        SPLIT_DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode=COLOR_MODEL,
        shuffle=False
    )

    return train_gen, val_gen, test_gen