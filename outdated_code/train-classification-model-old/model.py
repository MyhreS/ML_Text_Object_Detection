from images_and_labels import get_images_and_labels
import tensorflow as tf
from test_conv_model import create_test_conv_model
from test_residual_model import create_test_residual_model
from residual_model_1 import create_residual_model_1

number_output_layers = 20
train_images, train_labels = get_images_and_labels('../../datasets/dataset-4-words/train/', '../../datasets/dataset-4-words/alt_train_labels.csv', number_output_layers)
val_images, val_labels = get_images_and_labels('../../datasets/dataset-4-words/valid/', '../../datasets/dataset-4-words/alt_val_labels.csv', number_output_layers)
print(train_images.shape)
print(val_images.shape)
print(train_labels.shape)
print(val_labels.shape)

model = create_residual_model_1(number_output_layers)
# Compile the model
loss = ['categorical_crossentropy'] * number_output_layers
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
# Summary of the model
model.summary()

# Add tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/resnet-1', histogram_freq=1)

# Model fit with the val images and labels too
model.fit(train_images, [label for label in train_labels], epochs=1, batch_size=16, validation_data=(val_images, [label for label in val_labels]), callbacks=[tensorboard_callback])

# Save the model
model.save('../saved_models/resnet-1/model.h5')



