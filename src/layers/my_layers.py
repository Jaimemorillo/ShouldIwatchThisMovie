from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Convolution1D, MaxPooling1D, Input, Bidirectional
from tensorflow.keras.layers import Embedding, CuDNNLSTM, SpatialDropout1D, Reshape, Flatten, BatchNormalization, CuDNNGRU
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam, Adagrad


class MyLayers:

    @staticmethod
    def my_conv1d(kernel_size, layer, filters=10, strides=1, act='relu', reg=None):
        c1d = Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', strides=strides,
                     kernel_initializer='he_uniform', kernel_regularizer=reg)(layer)
        c1d = Activation(act)(c1d)
        c1d = BatchNormalization()(c1d)

        return c1d

    @staticmethod
    def my_dense(neurons, ant, act='sigmoid', ini='glorot_uniform', reg=None):
        dense = Dense(neurons, kernel_initializer=ini, kernel_regularizer=reg)(ant)
        dense = Activation(act)(dense)
        dense = BatchNormalization()(dense)

        return dense
