from images_and_labels import get_images_and_labels
import tensorflow as tf
from models import *

number_output_layers = 20
train_images, val_images, train_labels, val_labels = get_images_and_labels('../data/test/', '../data/alt_test_labels.csv', number_output_layers)
print(train_images.shape)
print(val_images.shape)
print(train_labels.shape)
print(val_labels.shape)


def create_test_conv_model():
    # Create a model
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    pool2 = tf.keras.layers.MaxPooling2D(2, 2)(pool1)
    pool3 = tf.keras.layers.MaxPooling2D(2, 2)(pool2)
    flatten = tf.keras.layers.Flatten()(pool3)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    outputs = [tf.keras.layers.Dense(27, activation='softmax', name='output{}'.format(i))(dense1) for i in range(1, number_output_layers+1)]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_test_conv_model()
# Compile the model
loss = ['categorical_crossentropy'] * number_output_layers
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
# Summary of the model
model.summary()

# Model fit with the val images and labels too
model.fit(train_images, [label for label in train_labels], epochs=1, batch_size=16, validation_data=(val_images, [label for label in val_labels]))




