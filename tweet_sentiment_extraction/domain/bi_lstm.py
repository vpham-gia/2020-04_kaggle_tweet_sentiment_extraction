"""Recurrent neural network models.

Classes
-------
BidirectionalLSTM
"""
from tensorflow.keras import regularizers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, TimeDistributed, Dropout, Dense, Input, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

    LENGTH_OF_LONGEST_SENTENCE = 35

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
        inputs = Input(shape=(self.LENGTH_OF_LONGEST_SENTENCE, ))

        embedding = Embedding(input_dim=self.word_embedding_initialization.shape[0],
                              output_dim=self.word_embedding_initialization.shape[1],
                              weights=[self.word_embedding_initialization],
                              trainable=True)(inputs)  # change trainable to False

        # TODO: concat with aditional features

        bidirection_lstm = Bidirectional(LSTM(self.hidden_dim,
                                              return_sequences=True,
                                              kernel_regularizer=regularizers.l2(0.01)))(embedding)

        dropout = Dropout(0.2)(bidirection_lstm)
        prediction = Dense(1, activation='sigmoid')(dropout)

        model = Model(inputs=inputs, outputs=prediction)
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y, pad_sentences=True, **kwargs):
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
            X_to_fit = pad_sequences(X, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
            y_to_fit = pad_sequences(y, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_to_fit, y_to_fit = X, y

        return self.model.fit(X_to_fit, y_to_fit, **kwargs)

    def predict(self, X_test, pad_sentences=True):
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
            X_to_predict = pad_sequences(X_test, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_to_predict = X_test

        predictions = self.model.predict(X_to_predict)

        unpaded_preds = [pred[:len(x)] for pred, x in zip(predictions, X_test)]
        return unpaded_preds
