from images_and_labels import get_images_and_labels
import tensorflow as tf
from test_conv_model import create_test_conv_model
from test_residual_model import create_test_residual_model

number_output_layers = 20
train_images, val_images, train_labels, val_labels = get_images_and_labels('../data/test/', '../data/alt_test_labels.csv', number_output_layers)
print(train_images.shape)
print(val_images.shape)
print(train_labels.shape)
print(val_labels.shape)

#model = create_test_conv_model(number_output_layers)
model = create_test_residual_model(number_output_layers)
# Compile the model
loss = ['categorical_crossentropy'] * number_output_layers
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
# Summary of the model
model.summary()

# Model fit with the val images and labels too
model.fit(train_images, [label for label in train_labels], epochs=1, batch_size=16, validation_data=(val_images, [label for label in val_labels]))




