import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Read data/test_labels.csv
test_labels = pd.read_csv('../data/test_labels.csv')

# Read the images using the test_labels dataframe
images = test_labels['Image'].apply(lambda x: plt.imread('../data/test/' + x))

# Extracts the labels from the dataframe.
labels = test_labels['Word'].apply(lambda x:x)

# Convert the images to a numpy array
images = np.array(images)
labels = np.array(labels)

# Split into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)



