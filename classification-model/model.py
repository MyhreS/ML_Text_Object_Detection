import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

number_outputs = 20

# TODO: Make images grayscale and normalize them.
# Read data/test_labels.csv
csv = pd.read_csv('../data/alt_test_labels.csv')
# Read the images using the test_labels dataframe
images = csv['Image'].apply(lambda x: plt.imread('../data/test/' + x))
# Split images into training and validation sets manually
train_images = images[:7000]
val_images = images[7000:]
#Convert the images to tensors
train_images_list = [tf.convert_to_tensor(image, dtype=tf.float32) for image in train_images]
train_images_tensor = tf.stack(train_images_list)
val_images_list = [tf.convert_to_tensor(image, dtype=tf.float32) for image in val_images]
val_images_tensor = tf.stack(val_images_list)


# Get labels
chars_list = [csv[f"char_{i}"].apply(lambda x: x) for i in range(1, number_outputs+1)]
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

print(train_labels.shape)
print(val_labels.shape)
print(train_images.shape)
print(val_images.shape)

# Create a model
inputs = tf.keras.Input(shape=(50, 250, 3))
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(inputs)
pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
pool2 = tf.keras.layers.MaxPooling2D(2, 2)(pool1)
pool3 = tf.keras.layers.MaxPooling2D(2, 2)(pool2)
flatten = tf.keras.layers.Flatten()(pool3)
dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
outputs = [tf.keras.layers.Dense(27, activation='softmax', name='output{}'.format(i))(dense1) for i in range(1, number_outputs+1)]
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Compile the model
loss = ['categorical_crossentropy'] * number_outputs
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
# Summary of the model
model.summary()

#model.fit(train_images_tensor, [train_labels[0], train_labels[1], train_labels[2], train_labels[3], train_labels[4], train_labels[5], train_labels[6], train_labels[7], train_labels[8], train_labels[9], train_labels[10], train_labels[11], train_labels[12], train_labels[13], train_labels[14], train_labels[15], train_labels[16], train_labels[17], train_labels[18], train_labels[19]], epochs=1, batch_size=16)
model.fit(train_images_tensor, [label for label in train_labels], epochs=1, batch_size=16)




