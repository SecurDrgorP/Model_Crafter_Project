import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from utils.split_data import split_data

def load_data(data_dir, img_size, batch_size):

    split_data()

    train_ds = image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    val_ds = image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    test_ds = image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    return train_ds, val_ds, test_ds