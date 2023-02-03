import tensorflow as tf
def create_test_conv_model():
    # Create a model
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    pool2 = tf.keras.layers.MaxPooling2D(2, 2)(pool1)
    pool3 = tf.keras.layers.MaxPooling2D(2, 2)(pool2)
    flatten = tf.keras.layers.Flatten()(pool3)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    outputs = [tf.keras.layers.Dense(27, activation='softmax', name='output{}'.format(i))(dense1) for i in range(1, number_output_layers+1)]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model