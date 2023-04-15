import matplotlib.pyplot as plt
import tensorflow as tf
import utils

"""
This script tests the saved model by cutting out the first detected word from the form image.
"""

saved_model_path = '../saved_models/faster-rcnn-training-v2-for-forms/inference_graph/saved_model'

# Load the model
model = tf.saved_model.load(saved_model_path)

# Load image
image_np = utils.load_image_into_numpy_array('../../datasets/dataset-test-words/1.jpg')
output_dict = utils.run_inference_for_single_image(model, image_np)

detection_boxes = output_dict['detection_boxes']
detection_scores = output_dict['detection_scores']

images = utils.cut_boxes_from_image(image_np, detection_boxes, detection_scores)

plt.imshow(images[1])
plt.show()