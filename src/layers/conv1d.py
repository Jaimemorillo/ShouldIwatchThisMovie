import tensorflow as tf


class MyConv1D(tf.keras.layers.Conv1D):

    def __init__(self, kernel_size, filters=10, strides=1, act='relu', reg=None):
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.acr = act
        self.reg = reg

        super().__init__(filters,kernel_size)

    def build(self, input_shape):
        self.fc = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='valid',
                                         strides=self.strides, kernel_initializer='he_uniform',
                                         kernel_regularizer=self.reg)
        self.fc.build(input_shape)
        self._trainable_weights = self.fc.trainable_weights
        super().build(input_shape)

    def call(self, input):
        c1d = self.fc(input)
        c1d = tf.keras.layers.Activation(self.act)(c1d)
        c1d = tf.keras.layers.BatchNormalization()(c1d)
        return c1d
