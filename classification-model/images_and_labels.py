import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def get_images_and_labels(path_to_images, path_to_labels_csv, number_output_layers):
    # Read data/test_labels.csv
    csv = pd.read_csv(path_to_labels_csv)
    # Read the images using the test_labels dataframe
    images = csv['Image'].apply(lambda x: plt.imread(path_to_images + x))
    # Make the images grayscale
    images = [np.dot(image[..., :3], [0.299, 0.587, 0.114]) for image in images]
    # Normalize the images
    images = [image / 255 for image in images]
    # Make into np array
    images = np.array(images)

    # Split images into training and validation sets manually
    train_images = images[:7000]
    val_images = images[7000:]
    # Convert the images to tensors
    train_images_list = [tf.convert_to_tensor(image, dtype=tf.float32) for image in train_images]
    train_images_tensor = tf.stack(train_images_list)
    val_images_list = [tf.convert_to_tensor(image, dtype=tf.float32) for image in val_images]
    val_images_tensor = tf.stack(val_images_list)

    # Get labels
    chars_list = [csv[f"char_{i}"].apply(lambda x: x) for i in range(1, number_output_layers + 1)]
    # Converting the labels into categorical to fit the model
    labels = []
    for i, char_list in enumerate(chars_list):
        one_colum_labels = tf.keras.utils.to_categorical(char_list, num_classes=27)
        labels.append(one_colum_labels)
    # Splitting the labels into training and validation sets
    train_labels = []
    val_labels = []
    for label in labels:
        train_labels.append(np.array(label[:7000]))
        val_labels.append(np.array(label[7000:]))
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    return train_images_tensor, val_images_tensor, train_labels, val_labels


