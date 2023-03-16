import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import utils

"""
This script tests the saved model by drawing bounding boxes around the detected words from a single form.
"""

saved_model_path = '../saved_models/object-detectors/faster-rcnn-training-v2/inference_graph/saved_model'

# Load the model
model = tf.saved_model.load(saved_model_path)

image_np = utils.load_image_into_numpy_array('../data/dataset-test-words/1.jpg')
output_dict = utils.run_inference_for_single_image(model, image_np)

detection_boxes = output_dict['detection_boxes']
detection_scores = output_dict['detection_scores']

# This creates an image with the bounding boxes drawn on it
def draw_boxes_on_image(image, boxes, scores, line_width=2):
    """Draw bounding boxes on an image.

    Args:
    - image: numpy array representing the image
    - boxes: list of bounding box coordinates, in the format of [[xmin, ymin, xmax, ymax]]
    - color: tuple representing the color of the boxes in (B, G, R) format
    - line_width: integer representing the width of the lines

    Returns:
    - image: numpy array representing the image with the bounding boxes drawn on it
    """
    for box, score in zip(boxes, scores):
        if score < 0.2:
            continue
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin * image.shape[1])
        ymin = int(ymin * image.shape[0])
        xmax = int(xmax * image.shape[1])
        ymax = int(ymax * image.shape[0])
        cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (0, 255, 0), line_width)
    return image

new_image = draw_boxes_on_image(image_np, detection_boxes, detection_scores)
plt.imshow(new_image)
plt.show()