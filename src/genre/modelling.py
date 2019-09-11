import tensorflow as tf
from layers.my_layers import MyLayers
from layers.attention import AttentionWithContext
from layers.bert import Bert
from tensorflow.keras import backend as K

from sklearn.metrics import jaccard_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Modelling:

    tokenizer = Bert.create_tokenizer_from_hub_module()
    sess = tf.Session()

    def genre_model(self, max_seq_length, gru1, gru2, neurons1, neurons2, neurons3, neurons4, drop):

        in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = Bert.BertLayer(n_fine_tune_layers=0, pooling="mean")(bert_inputs)

        x = tf.keras.layers.BatchNormalization()(bert_output)
        x = tf.keras.layers.SpatialDropout1D(drop)(x)

        c = MyLayers.my_conv1d(2, x, filters=256, act=MyLayers.gelu)
        m = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')(c)
        d = tf.keras.layers.Dropout(drop)(m)

        c = MyLayers.my_conv1d(3, d, filters=256, act=MyLayers.gelu)
        m = tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid')(c)
        d = tf.keras.layers.Dropout(drop)(m)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(gru1, kernel_initializer='glorot_uniform',
                                                                    return_sequences=True))(d)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(gru2, kernel_initializer='glorot_uniform',
                                          return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = AttentionWithContext()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dropout(drop)(x)
        x = MyLayers.my_dense(neurons1, x, act=MyLayers.gelu, ini='he_uniform',
                              reg=tf.keras.regularizers.l2(0.0001))
        x = tf.keras.layers.Dropout(drop)(x)
        x = MyLayers.my_dense(neurons2, x, act=MyLayers.gelu, ini='he_uniform', reg=None)
        x = tf.keras.layers.Dropout(drop)(x)
        x = MyLayers.my_dense(neurons3, x, act=MyLayers.gelu, ini='he_uniform',
                              reg=tf.keras.regularizers.l2(0.0001))
        x = tf.keras.layers.Dropout(drop)(x)
        x = MyLayers.my_dense(neurons4, x, act=MyLayers.gelu, ini='he_uniform', reg=None)
        x = tf.keras.layers.Dropout(drop)(x)
        x = tf.keras.layers.Dense(18, activation=MyLayers.custom_sigmoid, kernel_regularizer=None)(x)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=x)

        self.model = model

        return model

    def train_model(self, max_seq_length, X_train, y_train, X_dev, y_dev):

        self.max_sequence = max_seq_length

        def initialize_vars(sess):
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            K.set_session(sess)

        # model = self.genre_model(max_seq_length, 256, 256, 2048, 1024, 512, 256, 0.1)
        model = self.model

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])

        # Instantiate variables
        initialize_vars(self.sess)

        # Convert data to InputExample format
        train_examples = Bert.convert_text_to_examples(X_train, y_train)

        dev_examples = Bert.convert_text_to_examples(X_dev, y_dev)

        # Convert to features
        (train_input_ids, train_input_masks, train_segment_ids, train_labels
         ) = Bert.convert_examples_to_features(self.tokenizer, train_examples, max_seq_length=max_seq_length)

        (dev_input_ids, dev_input_masks, dev_segment_ids, test_labels
         ) = Bert.convert_examples_to_features(self.tokenizer, dev_examples, max_seq_length=max_seq_length)

        model.fit(
            [train_input_ids, train_input_masks, train_segment_ids],
            y_train,
            validation_data=([dev_input_ids, dev_input_masks, dev_segment_ids], y_dev),
            epochs=15,
            batch_size=32,
        )
        self.model = model

    def predict(self, X_test, y_test):

        # Convert data to InputExample format
        test_examples = Bert.convert_text_to_examples(X_test, y_test)

        # Convert to features
        (test_input_ids, test_input_masks, test_segment_ids, test_labels
         ) = Bert.convert_examples_to_features(self.tokenizer, test_examples, max_seq_length=self.max_sequence)

        y_score = self.model.predict([test_input_ids, test_input_masks, test_segment_ids], verbose=1)
        threshold = 0.5

        y_pred = y_score.copy()
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0

        self.y_pred = y_pred

        return y_pred

    def evaluate(self, labels_list, y_test, y_pred=None):

        if y_pred is None:
            y_pred = self.y_pred

        print("F1 global: {:.4f}".format(f1_score(y_test, y_pred, average='micro')))
        print("Jaccard score: {:.4f}".format(jaccard_score(y_test, y_pred, average='samples')))
        print("Hamming loss: {:.4f}".format(hamming_loss(y_test, y_pred)))
        print("Log loss: {:.4f}".format(log_loss(y_test, y_pred)))

        for idx, label in enumerate(labels_list):
            print(label + '\n')
            print("Confusion Matrix: \n" + str(confusion_matrix(y_test[:, idx], y_pred[:, idx])))
            print("Accuracy: {:.4f}".format(accuracy_score(y_test[:, idx], y_pred[:, idx])))
            print("F1: {:.4f}".format(f1_score(y_test[:, idx], y_pred[:, idx], pos_label=1)))
            print("Auc: {:.4f}".format(roc_auc_score(y_test[:, idx], y_pred[:, idx])))
            print("-------------------------------------------------------------------")

        return "F1 global: {:.4f}".format(f1_score(y_test, y_pred, average='micro'))
