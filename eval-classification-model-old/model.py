import numpy as np
import tensorflow as tf
from images_and_labels import get_images_and_labels
from residual_block import ResidualBlock

test_tensor, test_labels = get_images_and_labels('../data/dataset-4-words/test/', '../data/dataset-4-words/alt_test_labels.csv', 20)
test_tensor = test_tensor
# Load tensorflow model
model = tf.keras.models.load_model('../saved_models/resnet-1/model.h5', custom_objects={'ResidualBlock': ResidualBlock})

# Predict
predictions = model.predict(test_tensor)
def decode(label_or_prediction):
    # make a dictionary of the char_mapping the other way
    char_mapping = {0: '_', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k',
                    12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
                    23: 'w', 24: 'x', 25: 'y', 26: 'z'}

    decoded = []
    for row in zip(*label_or_prediction):
        decoded.append(''.join([char_mapping[np.argmax(char_prediction)] for char_prediction in row]))
    return decoded

actual = decode(test_labels)
predicted = decode(predictions)

def visualize(actual, predicted, amount=5):
    if amount > len(predicted):
        amount = len(predicted)
    for i in range(amount):
        print(f'Actual: {actual[i]}')
        print(f'Predicted: {predicted[i]}')

visualize(actual, predicted)

def calculate_accuracy(actual, predicted, skip_underscores=False):
    correct = 0
    total_chars = 0
    for actual_word, predicted_word in zip(actual, predicted):
        if skip_underscores:
            # Find the index of the first underscore in actual. Cut both actual_word and predicted_word at that index
            underscore_index = actual_word.find('_')
            actual_word = actual_word[:underscore_index]
            predicted_word = predicted_word[:underscore_index]
        for actual_char, predicted_char in zip(actual_word, predicted_word):
            if actual_char == predicted_char:
                correct += 1
            total_chars += 1
    return correct / total_chars


print(f'Skip underscore accuracy: {calculate_accuracy(actual, predicted, skip_underscores=True)}')
print(f'No skip underscore Accuracy: {calculate_accuracy(actual, predicted)}')





