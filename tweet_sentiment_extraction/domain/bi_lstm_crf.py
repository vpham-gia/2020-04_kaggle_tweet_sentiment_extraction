"""Recurrent neural network models.

Classes
-------
BidirectionalLSTM
"""
import tensorflow as tf
from tensorflow.keras import regularizers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, TimeDistributed, SpatialDropout1D, Dense, Input, Bidirectional, LSTM, concatenate
from tensorflow_addons.text.crf import crf_decode
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

from tweet_sentiment_extraction.domain.tensorflow_crf import CRF

class BidirectionalLSTMCRF:
    """Builds a bidirectional LSTM to perform predictions.

    Attributes
    ----------
    LENGTH_OF_LONGEST_SENTENCE: integer
    hidden_dim: integer
        Size of the hidden layer
    word_embedding_initialization: np.array
        Embedding matrix of size (vocabulary_size, embedding_vector_length)
    model: tensorflow.keras.Model

    Methods
    -------
    fit(X, y, pad_sentences=True, **kwargs)
    predict(X_test, pad_sentences=True)
    """

    LENGTH_OF_LONGEST_SENTENCE = 40
    NUMBER_EXTRA_FEATURES = 6
    N_TARGET = 3

    def __init__(self, hidden_dim, word_embedding_initialization):
        """Initialize class."""
        self.hidden_dim = hidden_dim
        self.word_embedding_initialization = word_embedding_initialization
        self.layer_crf = CRF(self.N_TARGET)
        self.model = self._model


    @property
    def _model(self):
        """Model structure.

        Returns
        -------
        model: tensorflow.keras.Model
        """
        inputs = Input(shape=(self.LENGTH_OF_LONGEST_SENTENCE, ), name='word_indexes')

        embedding = Embedding(input_dim=self.word_embedding_initialization.shape[0],
                              output_dim=self.word_embedding_initialization.shape[1],
                              weights=[self.word_embedding_initialization],
                              trainable=True)(inputs)  # change trainable to False

        extra_features = Input(shape=(self.LENGTH_OF_LONGEST_SENTENCE, self.NUMBER_EXTRA_FEATURES),
                               name='extra_features')
        embedding_with_extra_features = concatenate([embedding, extra_features])

        bidirection_lstm = Bidirectional(LSTM(self.hidden_dim,
                                              return_sequences=True,  #kernel_regularizer=regularizers.l2(0.01)
                                              ))(embedding_with_extra_features)

        dropout = SpatialDropout1D(rate=0.4)(bidirection_lstm)
        bilstm_with_extra_features = concatenate([dropout, extra_features])

        layer_crf_dense = Dense(self.N_TARGET, activation='softmax')(bilstm_with_extra_features)
        output_crf = self.layer_crf(layer_crf_dense)
#
        model = Model(inputs=[inputs, extra_features], outputs=output_crf)
        model.compile(optimizer=Adam(lr=0.001), loss=self.layer_crf.loss, metrics=[self.layer_crf.viterbi_accuracy])
        return model

    def load_model_weights(self, model_weights_path):
        """Load previously saved weights.

        Parameters
        ----------
        model_weights_path: string
        """
        self.model.load_weights(model_weights_path)

    def fit(self, X_word_indexes, X_features, y, pad_sentences=True, **kwargs):
        """Override fit method.

        Parameters
        ----------
        X: array-like of shape (n_samples,) containing list of tokens indexes
        y: array-like of shape (n_samples,) containing list of target sequences
        pad_sentences: boolean, default True
            If True, pads sequences to LENGTH_OF_LONGEST_SENTENCE

        Returns
        -------
        self: object
        """
        if pad_sentences:
            X_word = pad_sequences(X_word_indexes, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
            X_features = pad_sequences(X_features, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
            y_to_fit = pad_sequences(y, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_word, y_to_fit = X_word_indexes, y

        return self.model.fit({'word_indexes': X_word, 'extra_features': X_features}, y_to_fit, **kwargs)

    def predict(self, X_test_word, X_test_features, pad_sentences=True):
        """Override predict method.

        Parameters
        ----------
        X_test: array-like of shape (n_samples,) containing list of tokens indexes
        pad_sentences: boolean, default True
            If True, pads sequences to LENGTH_OF_LONGEST_SENTENCE

        Returns
        -------
        unpaded_preds: array-like of shape (n_samples,)
        """
        if pad_sentences:
            X_word = pad_sequences(X_test_word, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
            X_test_features = pad_sequences(X_test_features, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_word = X_test_word

        predictions = self.model.predict({'word_indexes': X_word, 'extra_features': X_test_features})

        unpaded_preds = [pred[:len(x)] for pred, x in zip(predictions, X_test_word)]
        return unpaded_preds

if __name__ == "__main__":
    word_embedding = np.ones((20, 500))
    bilstm = BidirectionalLSTMCRF(8, word_embedding)
    bilstm.model.summary()
    # 4 exemples
    # 3 tokens à chauqe fois
    # 6 features par token
    x_train_indexed = [
        [0, 3, 16],  # les indices des mots (entre 1 et 20 car on a 20 mots dans l'embedding)
        [0, 5, 6],
        [0, 14, 16],
        [0, 13, 16]
    ]
    train_extra_features = [  # train set
        [  # tweet (list of tokens)
            [-1, 12, 0.03, 0, 0, 0],  # token (sentiment, length, compound, ...)
            [-1, 12, -0.35, 0, 1, 0],
            [-1, 12, 0.03, 0, 1, 1],
        ],
        [  # tweet (list of tokens)
            [-1, 26, 0.63, 0, 1, 1],  # token (sentiment, length, compound, ...)
            [-1, 26, 0.03, 0, 0, 0],
            [-1, 26, 0.03, 0, 0, 0],
        ],
        [  # tweet (list of tokens)
            [-1, 26, -0.63, 0, 1, 1],  # token (sentiment, length, compound, ...)
            [-1, 26, 0.03, 0, 0, 0],
            [-1, 26, -0.03, 0, 0, 0],
        ],
        [  # tweet (list of tokens)
            [-1, 26, 0.33, 0, 1, 1],  # token (sentiment, length, compound, ...)
            [-1, 26, 0.03, 0, 0, 0],
            [-1, 26, 0.33, 1, 1, 0],
        ],
    ]
    tag_to_id = {"O": 0, "B-I": 1, "I": 2}
    id_to_tag = {0: "O", 1: "B-I", 2: "I"}
    train_labels = [
        ["O", "B-I", "I"],
        ["O", "O", "O"],
        ["B-I", "I", "I"],
        ["B-I", "I", "O"],
    ]

    train_labels_one_hot = [
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],  # [0, B-I, I]
        ],
        [
            [1, 0, 0], [1, 0, 0], [1, 0, 0],  # [0, 0, 0]
        ],
        [
            [0, 1, 0], [0, 0, 1], [0, 0, 1],  # [B-I, I, I]
        ],
        [
            [0, 1, 0], [0, 0, 1], [1, 0, 0],  # [B-I, I, 0]
        ],
    ]
    model_fit = bilstm.fit(X_word_indexes=x_train_indexed, X_features=train_extra_features, y=train_labels_one_hot,
                           batch_size=2, epochs=5, validation_split=0.25, verbose=1)

    train_pred = bilstm.predict(X_test_word=x_train_indexed, X_test_features=train_extra_features)


    print([[id_to_tag[np.argmax(prob)] for prob in pred] for pred in train_pred])
    print([[np.argmax(prob) > 0 for prob in pred] for pred in train_pred])
