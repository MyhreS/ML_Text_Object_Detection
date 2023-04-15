import tensorflow as tf
from residual_block import ResidualBlock
def create_residual_model_1(number_output_layers):
    def create_lane(resblock1, lane_number):
        conv_x_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(resblock1)
        batchnorm_x_1 = tf.keras.layers.BatchNormalization()(conv_x_1)
        resblock_x_1 = ResidualBlock(32, (3, 3))(batchnorm_x_1)
        pool_x_1 = tf.keras.layers.MaxPooling2D(2, 2)(resblock_x_1)
        resblock_x_2 = ResidualBlock(32, (3, 3))(pool_x_1)
        pool_x_2 = tf.keras.layers.MaxPooling2D(2, 2)(resblock_x_2)
        conv_x_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(pool_x_2)
        flatten_x = tf.keras.layers.Flatten()(conv_x_2)
        dense_x_1 = tf.keras.layers.Dense(128, activation='relu')(flatten_x)
        dense_output = tf.keras.layers.Dense(27, activation='softmax', name=f'output{lane_number}')(dense_x_1)
        return dense_output

    # Feature extraction
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(pool1)
    resblock1 = ResidualBlock(32, (3,3))(batchnorm1)



    outputs = []
    for i in range(1, number_output_layers+1):
        outputs.append(create_lane(resblock1, i))

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model




