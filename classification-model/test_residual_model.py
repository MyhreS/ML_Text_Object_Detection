import tensorflow as tf

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


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same', strides=1)
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(self.filters, (1, 1), padding='same', strides=1)
        self.add = tf.keras.layers.Add()
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        shortcut = self.conv3(inputs)
        x = self.add([x, shortcut])
        return x

