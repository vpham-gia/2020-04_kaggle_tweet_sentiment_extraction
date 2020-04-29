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


class BidirectionalLSTM:
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

    def __init__(self, hidden_dim, word_embedding_initialization):
        """Initialize class."""
        self.hidden_dim = hidden_dim
        self.word_embedding_initialization = word_embedding_initialization
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
                                              return_sequences=True,
                                              kernel_regularizer=regularizers.l2(0.01)))(embedding_with_extra_features)

        dropout = SpatialDropout1D(rate=0.2)(bidirection_lstm)
        # prediction = Dense(1, activation='sigmoid')(dropout)
        prediction = TimeDistributed(Dense(1, activation="sigmoid"))(dropout)

        crf_predictions, _ = crf_decode(potentials=prediction,
                                        transition_params=tf.constant(np.ones((2, 2))),
                                        sequence_length=self.LENGTH_OF_LONGEST_SENTENCE)

        model = Model(inputs=[inputs, extra_features], outputs=crf_predictions)
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
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
