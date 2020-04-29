"""Bidirectional LSTM for name entity recognition-like problem."""

from gensim.models import KeyedVectors
from gensim.corpora import Dictionary
from os.path import join
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import re
import sys

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner, SentencePreprocessor
from tweet_sentiment_extraction.utils.decorators import timer
from tweet_sentiment_extraction.utils.metrics import jaccard_score
from tweet_sentiment_extraction import settings as stg
from tweet_sentiment_extraction.domain.bi_lstm import BidirectionalLSTM
from tweet_sentiment_extraction.domain.word_embedding import load_word_embedding, adapt_glove_to_dictionary


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


pre_train_glove = load_word_embedding()
our_dict = adapt_glove_to_dictionary(pre_trained_glove_values=pre_train_glove, dictionary_token2id=dictionary.token2id)


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
