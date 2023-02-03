import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same', strides=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(self.filters, (1, 1), padding='same', strides=1)
        self.add = tf.keras.layers.Add()
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        shortcut = self.conv3(inputs)
        x = self.add([x, shortcut])
        x = self.relu(x)
        return x