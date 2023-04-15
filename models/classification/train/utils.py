import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_data(train_path, val_path, test_path):
    batch_size = 16
    image_size = (224,224)

    train_ds = image_dataset_from_directory(
        train_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
    )

    val_ds = image_dataset_from_directory(
        val_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
    )

    test_ds = image_dataset_from_directory(
        test_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
    )
    return train_ds, val_ds, test_ds






