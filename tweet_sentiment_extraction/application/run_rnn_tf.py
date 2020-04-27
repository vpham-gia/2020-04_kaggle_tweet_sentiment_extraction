"""Bidirectional LSTM for name entity recognition-like problem."""

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

toto = sp.preprocess_dataset(vocabulary=dictionary.token2id)

sys.exit()

##########################################
##########################################


class BidirectionalLSTM:
    """TODO."""

    LENGTH_OF_LONGEST_SENTENCE = 35

    def __init__(self, hidden_dim, word_embedding_initialization):
        """Initialize class."""
        self.hidden_dim = hidden_dim
        self.word_embedding_initialization = word_embedding_initialization

    @property
    def model(self):
        """Model structure."""
        pass

    def _custom_loss_to_exclude_paddings(self):
        pass

    def fit(self, XX):
        """Override fit method."""

    def predict(self, X_test):
        """Override predict method to match NER application."""



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
