import tensorflow as tf
from layers.my_layers import MyLayers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.python.client import device_lib
from keras.layers import CuDNNGRU, GRU
from keras.models import Model
from keras.layers import Dropout, MaxPooling1D, Input
from keras.layers import Embedding, SpatialDropout1D, BatchNormalization
from keras.layers import GlobalMaxPooling1D


class Modelling:

    def __init__(self, vocab_size, max_len=60, model_path='models/'):

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model = self.like_model_cpu(embedding_size=150, dropout=0.5, filters=64,
                                         kernel=3, maxp=2, gnup=32, neurons=8,
                                         act='gelu')
        self.model.load_weights(model_path + 'my_model_movie_like.h5')
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def like_model_gpu(self, embedding_size, dropout, filters, kernel, maxp, gnup, neurons, act):
        words = Input(shape=(self.max_len,))
        em = Embedding(input_dim=self.vocab_size, output_dim=embedding_size)(words)
        em = SpatialDropout1D(dropout)(em)

        c = MyLayers.my_conv1d(kernel, em, filters=filters, act=act)
        m = MaxPooling1D(pool_size=maxp)(c)
        d = Dropout(dropout)(m)

        gru = CuDNNGRU(gnup, kernel_initializer='glorot_uniform', return_sequences=True)(d)
        gru = BatchNormalization()(gru)
        gru = GlobalMaxPooling1D()(gru)
        drop = Dropout(dropout)(gru)

        pred = MyLayers.my_dense(neurons=neurons, ant=drop, act=act)
        pred = MyLayers.my_dense(1, pred)

        model = Model(inputs=words, outputs=pred)

        return model

    def like_model_cpu(self, embedding_size, dropout, filters, kernel, maxp, gnup, neurons, act):
        words = Input(shape=(self.max_len,))
        em = Embedding(input_dim=self.vocab_size, output_dim=embedding_size)(words)
        em = SpatialDropout1D(dropout)(em)

        c = MyLayers.my_conv1d(kernel, em, filters=filters, act=act)
        m = MaxPooling1D(pool_size=maxp)(c)
        d = Dropout(dropout)(m)

        gru = GRU(gnup, kernel_initializer='glorot_uniform', activation='tanh',
                  recurrent_activation='sigmoid', return_sequences=True, reset_after=True)(d)
        gru = BatchNormalization()(gru)
        gru = GlobalMaxPooling1D()(gru)
        drop = Dropout(dropout)(gru)

        pred = MyLayers.my_dense(neurons=neurons, ant=drop, act=act)
        pred = MyLayers.my_dense(1, pred)

        model = Model(inputs=words, outputs=pred)

        return model

    def get_available_gpus(self):
        local_device = device_lib.list_local_devices()
        return [x.name for x in local_device if x.device_type == 'GPU']

    def fit_model(self, X_train, y_train, epochs, batch_size):

        self.model.fit(X_train, y_train, epochs=epochs, verbose=True, batch_size=batch_size)

    def predict(self, X_test):

        y_score = self.model.predict(X_test, verbose=1)
        threshold = 0.5

        y_pred = y_score.copy()
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0

        return y_pred, y_score

    def evaluate(self, y_test, y_pred=None):

        if y_pred is None:
            y_pred = self.y_pred

        print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_pred)))
        print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
        print("Kappa: {:.4f}".format(cohen_kappa_score(y_test, y_pred)))
        print("Precision: {:.4f}".format(precision_score(y_test, y_pred, pos_label=1)))
        print("Recall: {:.4f}".format(recall_score(y_test, y_pred, pos_label=1)))
        print("F1: {:.4f}".format(f1_score(y_test, y_pred, pos_label=1)))
        print("Auc: {:.4f}".format(roc_auc_score(y_test, y_pred)))

        return "Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred))


