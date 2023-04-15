import tensorflow as tf
from residual_block import ResidualBlock
def create_test_residual_model(number_output_layers):
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
    resblock1 = ResidualBlock(32, (3,3))(conv1)
    resblock2 = ResidualBlock(32, (3,3))(resblock1)
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(resblock2)
    pool2 = tf.keras.layers.MaxPooling2D(2, 2)(pool1)
    flatten = tf.keras.layers.Flatten()(pool2)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    outputs = [tf.keras.layers.Dense(27, activation='softmax', name='output{}'.format(i))(dense1) for i in
               range(1, number_output_layers + 1)]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




