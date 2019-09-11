import tensorflow as tf
import numpy as np


class MyLayers:

    @staticmethod
    def my_conv1d(kernel_size, layer, filters=10, strides=1, act='relu', reg=None):
        c1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', strides=strides,
                                     kernel_initializer='he_uniform', kernel_regularizer=reg)(layer)
        c1d = tf.keras.layers.Activation(act)(c1d)
        c1d = tf.keras.layers.BatchNormalization()(c1d)

        return c1d

    @staticmethod
    def my_dense(neurons, ant, act='sigmoid', ini='glorot_uniform', reg=None):
        dense = tf.keras.layers.Dense(neurons, kernel_initializer=ini, kernel_regularizer=reg)(ant)
        dense = tf.keras.layers.Activation(act)(dense)
        dense = tf.keras.layers.BatchNormalization()(dense)

        return dense

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + tf.keras.backend.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.keras.backend.pow(x, 3))))

    @staticmethod
    def custom_sigmoid(x):
        return tf.keras.backend.sigmoid(x) + tf.keras.backend.epsilon()
