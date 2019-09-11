import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from tqdm import tqdm_notebook
from bert.tokenization import FullTokenizer


class Bert:

    def __init__(self, sess, path, max_seq_length):

        self.sess = sess
        self.bert_path = path
        self.max_seq_length = max_seq_length

    class PaddingInputExample(object):
        """Fake example so the num input examples is a multiple of the batch size.
      When running eval/predict on the TPU, we need to pad the number of examples
      to be a multiple of the batch size, because the TPU requires a fixed batch
      size. The alternative is to drop the last batch, which is bad because it means
      the entire output data won't be generated.
      We use this class instead of `None` because treating `None` as padding
      battches could cause silent errors.
      """

    class InputExample(object):
        """A single training/test example for simple sequence classification."""

        def __init__(self, guid, text_a, text_b=None, label=None):
            """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        bert_module = hub.Module(self.bert_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )

        return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def convert_single_example(self, tokenizer, example, max_seq_length=256):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, self.PaddingInputExample):
            input_ids = [0] * max_seq_length
            input_mask = [0] * max_seq_length
            segment_ids = [0] * max_seq_length
            label = 0
            return input_ids, input_mask, segment_ids, label

        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0: (max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids, example.label

    def convert_examples_to_features(self, tokenizer, examples, max_seq_length=256):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for example in tqdm_notebook(examples, desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = self.convert_single_example(
                tokenizer, example, max_seq_length
            )
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
        return (
            np.array(input_ids),
            np.array(input_masks),
            np.array(segment_ids),
            np.array(labels).reshape(-1, 1),
        )

    def convert_text_to_examples(self, texts, labels):
        """Create InputExamples"""
        InputExamples = []
        for text, label in zip(texts, labels):
            InputExamples.append(
                self.InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
            )
        return InputExamples

    def convert_to_features(self, X, y):

        # Instantiate tokenizer
        tokenizer = self.create_tokenizer_from_hub_module()

        # Convert data to InputExample format
        examples = self.convert_text_to_examples(X, y)

        # Convert to features
        (input_ids, input_masks, segment_ids, labels
         ) = self.convert_examples_to_features(tokenizer, examples, max_seq_length=self.max_seq_length)

        return input_ids, input_masks, segment_ids, labels

    class BertLayer(tf.keras.layers.Layer):
        def __init__(
                self,
                n_fine_tune_layers=10,
                pooling="mean",
                bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
                **kwargs,
        ):
            self.n_fine_tune_layers = n_fine_tune_layers
            self.trainable = True
            self.output_size = 768
            self.pooling = pooling
            self.bert_path = bert_path
            if self.pooling not in ["first", "mean"]:
                raise NameError(
                    f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
                )

            super().__init__(**kwargs)

        def build(self, input_shape):
            self.bert = hub.Module(
                self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
            )

            # Remove unused layers
            trainable_vars = self.bert.variables
            if self.pooling == "first":
                trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
                trainable_layers = ["pooler/dense"]

            elif self.pooling == "mean":
                trainable_vars = [
                    var
                    for var in trainable_vars
                    if not "/cls/" in var.name and not "/pooler/" in var.name
                ]
                trainable_layers = []
            else:
                raise NameError(
                    f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
                )

            # Select how many layers to fine tune
            for i in range(self.n_fine_tune_layers):
                trainable_layers.append(f"encoder/layer_{str(11 - i)}")

            # Update trainable vars to contain only the specified layers
            trainable_vars = [
                var
                for var in trainable_vars
                if any([l in var.name for l in trainable_layers])
            ]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

            super().build(input_shape)

        def call(self, inputs):
            inputs = [K.cast(x, dtype="int32") for x in inputs]
            input_ids, input_mask, segment_ids = inputs
            bert_inputs = dict(
                input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
            )
            if self.pooling == "first":
                result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                    "pooled_output"
                ]
            elif self.pooling == "mean":
                result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                    "sequence_output"
                ]

            else:
                raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

            return result

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.output_size
