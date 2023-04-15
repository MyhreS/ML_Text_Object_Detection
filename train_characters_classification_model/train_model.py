import datetime
import os

import utils
import tensorflow as tf
from tensorflow.keras import layers

# Load data
train_ds, val_ds, test_ds = utils.load_data('../data/dataset-8-emnist-raw/train', '../data/dataset-8-emnist-raw/val', '../data/dataset-8-emnist-raw/test')

# Create a ResNet101 model
def create_resnet101_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet101(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Fine-tuning the model (if desired)
    base_model.trainable = True

    # Create a new model
    model = tf.keras.Sequential([
        base_model,
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


input_shape = (224, 224, 3)
num_classes = len(train_ds.class_names)
resnet101_model = create_resnet101_model(input_shape, num_classes)
resnet101_model.summary()

# Create a directory for saved models
saved_models_dir = '../saved_models/classification'
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Train the model
epochs = 50

history = resnet101_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, tensorboard_callback]
)

# Save the model
resnet101_model.save(saved_models_dir + '/resnet101_model')

# Evaluate the model
test_loss, test_acc = resnet101_model.evaluate(test_ds)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)



