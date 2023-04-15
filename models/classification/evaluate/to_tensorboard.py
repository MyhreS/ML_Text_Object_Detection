import tensorflow as tf
from datetime import datetime

# Define the arrays
epoch_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
loss_values = [0.2748, 0.1680, 0.1321, 0.1103, 0.0900, 0.0752, 0.0661, 0.0577, 0.0497, 0.0444, 0.0391, 0.0327, 0.0298, 0.0237]
accuracy_values = [0.9113, 0.9414, 0.9524, 0.9589, 0.9652, 0.9698, 0.9731, 0.9763, 0.9793, 0.9814, 0.9843, 0.9873, 0.9889, 0.9912]
val_loss_values = [0.1783, 0.1773, 0.1625, 0.1597, 0.1633, 0.1616, 0.1653, 0.1842, 0.1939, 0.1996, 0.2070, 0.2101, 0.2101, 0.2252]
val_accuracy_values = [0.9388, 0.9424, 0.9492, 0.9482, 0.9465, 0.9495, 0.9513, 0.9483, 0.9472, 0.9447, 0.9449, 0.9464, 0.9414, 0.9468]

# Create a summary writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/scalars/{current_time}"
writer = tf.summary.create_file_writer(log_dir)

# Write the values to the summary writer
with writer.as_default():
    for epoch in range(len(epoch_values)):
        tf.summary.scalar("Loss", loss_values[epoch], step=epoch_values[epoch])
        tf.summary.scalar("Accuracy", accuracy_values[epoch], step=epoch_values[epoch])
        tf.summary.scalar("Validation Loss", val_loss_values[epoch], step=epoch_values[epoch])
        tf.summary.scalar("Validation Accuracy", val_accuracy_values[epoch], step=epoch_values[epoch])
