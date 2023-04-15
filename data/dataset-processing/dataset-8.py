import os

import cv2
import pandas as pd
from matplotlib import pyplot as plt

# Read the data from the CSV file
df_train = pd.read_csv('../data/dataset-8-emnist-raw/emnist-letters-train.csv')
df_test = pd.read_csv('../data/dataset-8-emnist-raw/emnist-letters-test.csv')
print(df_train.shape)
print(df_test.shape)

df_test.rename(columns={'1': '23'}, inplace=True)

# Count each class in df_train in column "23"
classes = df_train['23'].value_counts()
print(classes)

for directory in ['train', 'val', 'test']:
    # Iterate through the unique classes and create a folder for each class
    for class_label in df_train['23'].unique():
        class_folder = os.path.join("../../datasets/dataset-8-emnist-raw/"+directory, str(class_label))
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)

# Split df_train into train and validation
df_val, df_train = df_train[:10000], df_train[10000:]
# How many rows in each dataframe?
print(df_train.shape)
print(df_val.shape)

print(df_train.head())


for dataframe, name in zip([df_train, df_val, df_test], ['train', 'val', 'test']):
    i = 0
    for index, row in dataframe.iterrows():
        # Get the label
        label = row['23']
        # Get the image
        image = row.drop('23')
        # Reshape the image
        image = image.values.reshape(28, 28)
        # Convert the image data type to uint8
        image = image.astype('uint8')
        # Resize the image to 224x224
        image_resized = cv2.resize(image, (224, 224))
        # Convert the image to 3 channels
        image_3channels = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
        # Save the image
        image_name = str(index) + '-' + str(label) + '.png'
        image_path = '../../datasets/dataset-8-emnist-raw/' + name + '/' + str(label) + '/' + image_name
        plt.imsave(image_path, image_3channels)
        i += 1
        if i % 100 == 0:
            print(image_name)


