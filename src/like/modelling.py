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


class Modelling:

    def __init__(self, max_len, vocab_size, model=None):

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model = model

    def taste_model(self, embedding_size, dropout, filters1,
                    filters2, kernel, maxp, gnup, act):

        words = tf.keras.layers.Input(shape=(self.max_len,))
        em = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_size)(words)
        em = tf.keras.layers.SpatialDropout1D(dropout)(em)

        c = MyLayers.my_conv1d(1, em, filters=filters1, act=act)
        d = tf.keras.layers.Dropout(dropout)(c)

        c = MyLayers.my_conv1d(kernel, d, filters=filters2, act=act)
        m = tf.keras.layers.MaxPooling1D(pool_size=maxp)(c)
        d = tf.keras.layers.Dropout(dropout)(m)

        gru = tf.keras.layers.CuDNNGRU(gnup, kernel_initializer='glorot_uniform', return_sequences=True)(d)
        gru = tf.keras.layers.BatchNormalization()(gru)
        gru = tf.keras.layers.GlobalMaxPooling1D()(gru)
        drop = tf.keras.layers.Dropout(dropout)(gru)

        pred = MyLayers.my_dense(1, drop)

        model = tf.keras.models.Model(inputs=words, outputs=pred)
        self.model = model

        return model

    def get_available_gpus(self):
        local_device = device_lib.list_local_devices()
        return [x.name for x in local_device if x.device_type == 'GPU']

    def fit_model(self, X_train, y_train, X_dev, y_dev):
        # Fit model just can be use with gpu
        gpus = self.get_available_gpus()
        if len(gpus) < 1:
            print("No gpu is available")

        else:
            model = self.model

            model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                          metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=150, verbose=True,
                      validation_data=(X_dev, y_dev), batch_size=64)

            self.model = model

    def predict(self, X_test):
        y_score = self.model.predict(X_test, verbose=1)
        threshold = 0.5

        y_pred = y_score.copy()
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0

        self.y_pred = y_pred
        return y_pred

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


