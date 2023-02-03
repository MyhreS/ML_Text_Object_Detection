import tensorflow as tf
from residual_block import ResidualBlock
def create_residual_model_1(number_output_layers):
    def create_lane(resblock1, lane_number):
        conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(resblock1)
        batchnorm = tf.keras.layers.BatchNormalization()(conv)
        resblock = ResidualBlock(32, (3, 3))(batchnorm)
        pool = tf.keras.layers.MaxPooling2D(2, 2)(resblock)
        resblock = ResidualBlock(32, (3, 3))(pool)
        pool = tf.keras.layers.MaxPooling2D(2, 2)(resblock)
        conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool)
        flatten = tf.keras.layers.Flatten()(conv)
        dense = tf.keras.layers.Dense(128, activation='relu')(flatten)
        dense_output = tf.keras.layers.Dense(27, activation='softmax', name=f'output{lane_number}')(dense)
        return dense_output

    # Feature extraction
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)
    resblock1 = ResidualBlock(32, (3,3))(batchnorm1)

    outputs = []
    for i in range(1, number_output_layers+1):
        outputs.append(create_lane(resblock1, i))

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




