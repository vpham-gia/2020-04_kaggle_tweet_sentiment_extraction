"""Bidirectional LSTM for name entity recognition-like problem."""

from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from os.path import join
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, TimeDistributed, Dropout, Dense, Input, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import re
import sys

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner, SentencePreprocessor
from tweet_sentiment_extraction.utils.decorators import timer
from tweet_sentiment_extraction.utils.metrics import jaccard_score
from tweet_sentiment_extraction import settings as stg


train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

train = train.query('sentiment != "neutral"')
validation = validation.query('sentiment != "neutral"')

train_data = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize=stg.TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(df=validation, column_name_to_tokenize=stg.TEXT_COL)

train_data = SentenceCleaner.add_tokenized_column(df=train_data, column_name_to_tokenize=stg.SELECTED_TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(df=validation_data,
                                                       column_name_to_tokenize=stg.SELECTED_TEXT_COL)


# def clean_text_with_repeated_letters(text):
#     # 'REPEATED_LETTERS': r'([A-Za-z])\1{2,}'
#     pass

# stg.PADDED_SEQUENCE_COL: lambda df: df[stg.INDEXED_TOKENS_COL].apply(lambda x: pad_sequences(x, MAX_LEN))


dictionary = Dictionary([["<OOV>", "<PAD>"]])
train_selected_dictionary = Dictionary(train_data[stg.TOKENS_SELECTED_TEXT_COL].apply(
    lambda x: [SentencePreprocessor._lower_and_add_flag_pattern(word) for word in x]))
train_dictionary = Dictionary(train_data[stg.TOKENS_TEXT_COL].apply(
    lambda x: [SentencePreprocessor._lower_and_add_flag_pattern(word) for word in x]))
train_dictionary.filter_extremes(keep_n=3000)
filtered_train_dictionary = train_dictionary.token2id
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)

sp = SentencePreprocessor(df=train_data)

train_data_preprocessed = sp.preprocess_dataset(vocabulary=dictionary.token2id)\
                            .assign(to_exclude=lambda df: df[stg.TARGET_SEQUENCE_COL].apply(lambda x: max(x) == 0))\
                            .query('not to_exclude')


@timer
def load_word_embedding(filename='glove.twitter.27B/glove.twitter.27B.200d.txt'):
    """Word embeddings from pre-trained Glove."""
    # tqdm.pandas()
    f = open(join(stg.WORD_EMBEDDING_DIR, filename))

    embedding_values = {}
    for line in f:
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_values[word] = coef
    return embedding_values


@timer
def adapt_glove_to_dictionary(pre_trained_glove_values, dictionary_token2id=dictionary.token2id):
    all_embs = np.stack(pre_trained_glove_values.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    emb_mean, emb_std

    embedding_matrix = np.random.normal(emb_mean, emb_std, (len(dictionary_token2id), 200))
    embedding_matrix[1] = 0  # Padding embedding

    out_of_vocabulary = []
    for word, i in dictionary_token2id.items():
        if i >= 2:  # 0 is the index of OOV, 1 is the index of Padding
            values = pre_trained_glove_values.get(word)
            if values is not None:
                embedding_matrix[i] = values
            else:
                out_of_vocabulary.append(word)
    print(f'out_of_vocabulary: {len(out_of_vocabulary)}')

    return embedding_matrix  # , out_of_vocabulary


pre_train_glove = load_word_embedding()
our_dict = adapt_glove_to_dictionary(pre_trained_glove_values=pre_train_glove, dictionary_token2id=dictionary.token2id)


class BidirectionalLSTM:
    """TODO."""

    LENGTH_OF_LONGEST_SENTENCE = 35

    def __init__(self, hidden_dim, word_embedding_initialization):
        """Initialize class."""
        self.hidden_dim = hidden_dim
        self.word_embedding_initialization = word_embedding_initialization
        self.model = self._model

    @property
    def _model(self):
        """Model structure."""
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
        """Override fit method."""
        if pad_sentences:
            X_to_fit = pad_sequences(X, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
            y_to_fit = pad_sequences(y, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_to_fit, y_to_fit = X, y

        return self.model.fit(X_to_fit, y_to_fit, **kwargs)

    def predict(self, X_test, pad_sentences=True):
        """Override predict method to match NER application."""
        if pad_sentences:
            X_to_predict = pad_sequences(X_test, maxlen=self.LENGTH_OF_LONGEST_SENTENCE, padding='post')
        else:
            X_to_predict = X_test

        predictions = self.model.predict(X_to_predict)

        unpaded_preds = [pred[:len(x)] for pred, x in zip(predictions, X_test)]
        return unpaded_preds


BLSTM = BidirectionalLSTM(hidden_dim=128, word_embedding_initialization=our_dict)

callbacks = ModelCheckpoint(join(stg.ML_DATA_DIR, "best_model.hdf5"), monitor='loss', verbose=1, save_best_only=True)
model_fit = BLSTM.fit(X=train_data_preprocessed[stg.INDEXED_TOKENS_COL],
                      y=train_data_preprocessed[stg.TARGET_SEQUENCE_COL],
                      batch_size=32, epochs=30, validation_split=0.2, callbacks=[callbacks], verbose=1)

train_pred = BLSTM.predict(X_test=train_data_preprocessed[stg.INDEXED_TOKENS_COL])

train_data_with_pred = train_data_preprocessed.assign(
    pred=train_pred,
    pred_in_bool=lambda df: df['pred'].apply(lambda x: [True if score > 0.5 else False for score in x]),
    tokens_pred=lambda df: df.apply(lambda row: np.array(row[stg.TOKENS_TEXT_COL])[row['pred_in_bool']], axis=1),
    sentence_pred=lambda df: df['tokens_pred'].apply(lambda x: ' '.join(x))
)

train_all_sentiments = pd.merge(left=pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL]),
                                right=train_data_with_pred[[stg.ID_COL, 'sentence_pred']],
                                on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

train_score = jaccard_score(y_true=train_all_sentiments[stg.SELECTED_TEXT_COL],
                            y_pred=train_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'train score: {train_score}')
print('--------------------------')

sp_validation = SentencePreprocessor(df=validation)
validation_data_preprocessed = sp.preprocess_dataset(vocabulary=dictionary.token2id)
validation_pred = BLSTM.predict(X_test=validation_data_preprocessed[stg.INDEXED_TOKENS_COL])

validation_data_with_pred = validation_data_preprocessed.assign(
    pred=validation_pred,
    pred_in_bool=lambda df: df['pred'].apply(lambda x: [True if score > 0.5 else False for score in x]),
    tokens_pred=lambda df: df.apply(lambda row: np.array(row[stg.TOKENS_TEXT_COL])[row['pred_in_bool']], axis=1),
    sentence_pred=lambda df: df['tokens_pred'].apply(lambda x: ' '.join(x))
)

validation_all_sentiments = pd.merge(left=pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv')).dropna(subset=[stg.TEXT_COL]),
                                     right=validation_data_with_pred[[stg.ID_COL, 'sentence_pred']],
                                     on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

validation_score = jaccard_score(y_true=validation_all_sentiments[stg.SELECTED_TEXT_COL],
                                 y_pred=validation_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'validation score: {validation_score}')
print('--------------------------')
