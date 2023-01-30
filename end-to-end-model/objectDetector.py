import tensorflow as tf

class ScaleLayer(tf.keras.layers.Layer):
  def __init__(self, scale, **kwargs):
    super(ScaleLayer, self).__init__(**kwargs)
    self.scale = scale

  def call(self, inputs):
    return inputs * self.scale