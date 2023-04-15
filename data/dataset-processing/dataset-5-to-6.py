from PIL import Image, ImageOps
import glob

"""
This script inverts the colors of the images in the dataset-5-words-with-characters-bb folder to create the dataset-6-words-with-characters-bb folder.
"""
def invert_image(input_file, output_folder):
    # Open the image file
    img = Image.open(input_file)

    # Invert the colors
    inverted_img = ImageOps.invert(img)

    # Save the inverted image to the output folder
    output_file = output_folder + '/' + input_file.split('\\')[-1]
    inverted_img.save(output_file)


train_images = glob.glob("../../datasets/dataset-5-words-with-characters-bb/train/*.jpg")
test_images = glob.glob("../../datasets/dataset-5-words-with-characters-bb/test/*.jpg")
val_images = glob.glob("../../datasets/dataset-5-words-with-characters-bb/valid/*.jpg")

for train_image in train_images:
    invert_image(train_image, "../../datasets/dataset-6-words-with-characters-bb/train")

for test_image in test_images:
    invert_image(test_image, "../../datasets/dataset-6-words-with-characters-bb/test")

for val_image in val_images:
    invert_image(val_image, "../../datasets/dataset-6-words-with-characters-bb/valid")

