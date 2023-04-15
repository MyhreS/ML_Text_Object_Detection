import pandas as pd

"""
This script converts the predictions.csv file to the annotations.csv file.
"""

# Load predictions.csv
predictions = pd.read_csv("../../datasets/dataset-7-raw-test-dataset-4-with-bb/predictions.csv")

# Choose which columns to keep
predictions = predictions[["x", "y", "width", "height", "class", "image_path"]]

# Fix the image_path column
predictions["image_path"] = predictions["image_path"].apply(lambda path: path.split("/")[-1])

print(predictions.head())

# Save as a csv
predictions.to_csv("../../datasets/dataset-7-raw-test-dataset-4-with-bb/annotations.csv", index=False)
