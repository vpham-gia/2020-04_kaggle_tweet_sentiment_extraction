from gensim.corpora import Dictionary

import re

import sys
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner
from tweet_sentiment_extraction.utils.metrics import jaccard_score
from tweet_sentiment_extraction import settings as stg

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, TimeDistributed, Dropout, Dense, Input, Bidirectional, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

######################################################################
# Example: An LSTM for Part-of-Speech Tagging in Tensorflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######################################################################


##########################################
# Load and Clean Text same as RNN Pytorch
##########################################

regex_pattern = {
    '<link>': r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))',
    '<hash>': r'#\w*',
    '<mention>': r'@\w*',
    '<retweet>': r'^(RT|FAV)'
}


def clean_text_with_repeated_letters(text):
    # 'REPEATED_LETTERS': r'([A-Za-z])\1{2,}'
    pass


def lower_and_add_flag_pattern(text):
    text = text.lower()

    for key, value in regex_pattern.items():
        text = re.sub(value, key, text)

    return text


train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

train = train.query('sentiment != "neutral"')
validation = validation.query('sentiment != "neutral"')

train_data = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize=stg.TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(df=validation, column_name_to_tokenize=stg.TEXT_COL)

train_data = SentenceCleaner.add_tokenized_column(df=train_data, column_name_to_tokenize=stg.SELECTED_TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(
    df=validation_data, column_name_to_tokenize=stg.SELECTED_TEXT_COL)

train_preprocessed_rnn = [
    (sentence, ["1" if word in target else "0" for word in sentence])
    for sentence, target in zip(train_data['tokens_text'], train_data['tokens_selected_text'])
]

validation_preprocessed_rnn = [
    (sentence, ["1" if word in target else "0" for word in sentence])
    for sentence, target in zip(validation_data['tokens_text'], validation_data['tokens_selected_text'])
]

dictionary = Dictionary([["<OOV>", "<PAD>"]])
train_selected_dictionary = Dictionary(train_data[stg.TOKENS_SELECTED_TEXT_COL].apply(
    lambda x: [lower_and_add_flag_pattern(word) for word in x]))
train_dictionary = Dictionary(train_data[stg.TOKENS_TEXT_COL].apply(
    lambda x: [lower_and_add_flag_pattern(word) for word in x]))
train_dictionary.filter_extremes(keep_n=3000)
filtered_train_dictionary = train_dictionary.token2id
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)

##########################################
##########################################
MAX_LEN = 35


def prepare_sequence(seq, to_ix):
    """
    Prepare Sequence by adding pattern and transform to lowercase.

    Parameters:
    seq: list
    to_ix: gensim dict

    Return:
    list"""
    idxs = [to_ix.get(lower_and_add_flag_pattern(w), to_ix["<OOV>"]) for w in seq]
    return idxs


def prepare_seq_X_y(df, to_ix):
    """
    Return:
    X: list
    y: list
    """
    seq_X = []
    seq_y = []
    for i in range(len(train_preprocessed_rnn)):
        seq_X_in = prepare_sequence(train_preprocessed_rnn[i][0], dictionary.token2id)
        seq_X.append(seq_X_in)

        seq_y_in = train_preprocessed_rnn[i][1]
        seq_y.append(seq_y_in)

    return seq_X, seq_y


def load_glove():
    """Word embeddings from pre-trained Glove."""
    tqdm.pandas()
    f = open(join(stg.RAW_DATA_DIR, 'glove.840B.300d.txt'))

    embedding_values = {}
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_values[word] = coef
    return embedding_values


def fit_glove(embedding_values):

    all_embs = np.stack(embedding_values.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    emb_mean, emb_std

    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, 300))
    OFV = []
    for word, i in tqdm(token.word_index.items()):
        values = embedding_values.get(re.sub(r"[^A-Za-z]", "", word))
        if values is not None:
            embedding_matrix[i] = values
        else:
            OFV.append(word)
    print(f'OFV:{len(OFV)}')

    return embedding_matrix


seq_X, seq_y = prepare_seq_X_y(train_preprocessed_rnn, dictionary.token2id)

train_pad_seq_x = pad_sequences(seq_X, maxlen=MAX_LEN)
train_pad_seq_y = pad_sequences(seq_y, maxlen=MAX_LEN)

token = Tokenizer(num_words=54000, filters='')
token.fit_on_texts(dictionary.token2id)
vocab_size = len(token.word_index) + 1

embedding_values = load_glove()
embedding_matrix = fit_glove(embedding_values)

######################################################################
# Create  model:
######################################################################


def bidirectional_lstm_model(hidden_dim):
    print('Building LSTM model...')
    model = Sequential()

    model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=True))  # change trainable to False
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(f'LSTM Built: {model.summary()}')
    return model


model = bidirectional_lstm_model(hidden_dim=128)

X_train, X_test, y_train, y_test = train_test_split(train_pad_seq_x, train_pad_seq_y, test_size=0.2, random_state=50)
# Fit the model
callbacks = ModelCheckpoint(join(stg.ML_DATA_DIR, "best_model.hdf5"), monitor='loss', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[callbacks],
                    verbose=0)


def reformat_pred(df, predictions):
    """
    Return:
    result_pred: list
        text list of predictions
    """
    result_pred = []

    for i in range(len(df)):

        padding_value = MAX_LEN - len(df[i][1])
        pred_value = np.array(predictions[i][padding_value:].flatten(), dtype=bool)
        if not any(pred_value) == True:  # if prediction is empty
            pred_value[pred_value == False] = True
        text_value = np.array(df[i][0])

        result_tolist = text_value[pred_value].tolist()
        result_as_string = ' '.join(result_tolist)

        result_pred.append(result_as_string)
    return result_pred


train_pred = model.predict(train_pad_seq_x).round().astype(int)
result_pred = reformat_pred(train_preprocessed_rnn, train_pred)

train_score = jaccard_score(train_data[stg.SELECTED_TEXT_COL], pd.Series(result_pred))

print('--------------------------')
print(f'train score: {train_score}')
print('--------------------------')