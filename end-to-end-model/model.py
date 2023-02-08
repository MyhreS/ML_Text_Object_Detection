import numpy as np
import tensorflow as tf
from images_and_labels import get_images_and_labels
from residual_block import ResidualBlock

test_tensor, test_labels = get_images_and_labels('../data/dataset-4-words/test/', '../data/dataset-4-words/alt_test_labels.csv', 20)
print(test_tensor.shape)
print(test_labels.shape)

test_tensor = test_tensor[:5]
test_labels = test_labels[:5]

# Load tensorflow model
model = tf.keras.models.load_model('../saved_models/resnet-1/model.h5', custom_objects={'ResidualBlock': ResidualBlock})

# Predict
predictions = model.predict(test_tensor)

# make a dictionary of the char_mapping the other way
char_mapping = {0:'_',1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6:'f', 7:'g', 8:'h', 9:'i', 10:'j', 11:'k', 12:'l', 13:'m', 14:'n', 15:'o', 16:'p', 17:'q', 18:'r', 19:'s', 20:'t', 21:'u', 22:'v', 23:'w', 24:'x', 25:'y', 26:'z'}

prediction_rows = predictions[:20]

for row in zip(*prediction_rows):
    print(*[char_mapping[np.argmax(char_prediction)] for char_prediction in row])


#model.evaluate(test_tensor, [label for label in test_labels])


