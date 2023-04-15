import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.patches import Rectangle

"""
This script visualizes the dataset-7-raw-test-dataset-4-with-bb
"""

# Load annotations.csv
annotations = pd.read_csv("../../datasets/dataset-7-raw-test-dataset-4-with-bb/annotations.csv")
print(annotations.head())

# Get all the image paths
image_paths = glob.glob("../../datasets/dataset-7-raw-test-dataset-4-with-bb/*.png")

i = 0
for image_path in image_paths:
    if i == 8:
        # Get image filename
        image_name = image_path.split("\\")[-1]

        # Get annotations for this image
        annotations_for_image = annotations[annotations["image_path"] == image_name]

        # Read and display the image
        img = plt.imread(image_path)
        plt.imshow(img)

        print(annotations_for_image.head())

        # Draw bounding boxes
        for idx, row in annotations_for_image.iterrows():
            x, y, width, height = row[["x", "y", "width", "height"]]
            bbox = Rectangle((x, y), width, height, linewidth=1, edgecolor="r", facecolor="none")
            plt.gca().add_patch(bbox)

        # Show the image with bounding boxes
        plt.show()
    i += 1

