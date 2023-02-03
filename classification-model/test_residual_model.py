import tensorflow as tf

def create_test_residual_model(number_output_layers):
    inputs = tf.keras.Input(shape=(50, 250, 1))
    conv1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(inputs)
    resblock1 = residual_block(conv1, 32, (3,3))
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(resblock1)
    pool2 = tf.keras.layers.MaxPooling2D(2, 2)(pool1)
    flatten = tf.keras.layers.Flatten()(pool2)
    dense1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    outputs = [tf.keras.layers.Dense(27, activation='softmax', name='output{}'.format(i))(dense1) for i in
               range(1, number_output_layers + 1)]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def residual_block(inputs, filters, kernel_size):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', strides=1)(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=1)(inputs)
    x = tf.keras.layers.Add()([x, shortcut])
    return x
