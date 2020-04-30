import sys
from gensim.corpora import Dictionary
from os.path import join
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import numpy as np
import pandas as pd
import spacy
import re

from tweet_sentiment_extraction import settings as stg
from tweet_sentiment_extraction.domain.bi_lstm_crf import BidirectionalLSTMCRF
from tweet_sentiment_extraction.domain.dataset_builder import Featurizer
from tweet_sentiment_extraction.domain.word_embedding import WordEmbedding
from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentencePreprocessor
from tweet_sentiment_extraction.utils.metrics import jaccard_score

#embedding_matrix = np.load(join('data', 'word_embeddings', 'init_embedding_matrix.npy'))
#print(embedding_matrix.shape)
#logging.info(f'embedding shape: {embedding_matrix.shape}')
#BLSTM = BidirectionalLSTMCRF(hidden_dim=64, word_embedding_initialization=embedding_matrix)

analyzer = SentimentIntensityAnalyzer()

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))
logging.info('Train and validation loaded')

train_with_sentiments = (
    train.query('sentiment != "neutral"')
         .assign(**{
             'start_position': lambda df: df.apply(lambda x: x['text'].find(x['selected_text']), axis=1),
             'len_selected_text': lambda df: df['selected_text'].apply(len),
             'end_position': lambda df: df['start_position'] + df['len_selected_text']
         })
)
train_sentiments_featurized = Featurizer.encode_sentiment_column(df=train_with_sentiments)
train_sentiments_featurized = Featurizer.add_length_of_tweet_column(df=train_sentiments_featurized)

train_sentiments_featurized['extra_features'] = train_sentiments_featurized.apply(
    lambda row: [row[stg.SENTIMENT_COL], row[stg.FEAT_LENGTH_TWEET_COL]], axis=1
)

validation_with_sentiments = (
    validation.query('sentiment != "neutral"')
              .assign(**{
                  'start_position': lambda df: df.apply(lambda x: x['text'].find(x['selected_text']), axis=1),
                  'len_selected_text': lambda df: df['selected_text'].apply(len),
                  'end_position': lambda df: df['start_position'] + df['len_selected_text']
              })
)
validation_sentiments_featurized = Featurizer.encode_sentiment_column(df=validation_with_sentiments)
validation_sentiments_featurized = Featurizer.add_length_of_tweet_column(df=validation_sentiments_featurized)

validation_sentiments_featurized['extra_features'] = validation_sentiments_featurized.apply(
    lambda row: [row[stg.SENTIMENT_COL], row[stg.FEAT_LENGTH_TWEET_COL]], axis=1
)

nlp = spacy.load('en_core_web_md')

def remove_duplicates_char(text, regex=stg.REGEX_PATTERN['<duplicate_chars>']):
    replace_text = re.sub(regex, r"\1", text)
    return replace_text


train_docs = []
train_selected_docs = []
train_labels = []
train_extra_features = []

def encode_tokens_labels(doc, start_pos, end_pos):
    """for each token we attribute a one-hot label
    with 3 classes : B-I (begin of entity)
    I : for inside the entity
    O : out of entity"""
    labels_of_doc = []
    begin_of_entity = True
    for token in doc:

        if start_pos <= token.idx < end_pos:
            if begin_of_entity:
                label_of_token = [0, 1, 0]  #"B-I"
                begin_of_entity = False
            else:
                label_of_token = [0, 0, 1]  #"I"
        else:
            label_of_token = [1, 0, 0]  #"O"
        labels_of_doc.append(label_of_token)
    return labels_of_doc


for doc, selected_doc, start_pos, end_pos, feats in zip(nlp.pipe(train_sentiments_featurized[stg.TEXT_COL]),
                                                        nlp.pipe(train_sentiments_featurized[stg.SELECTED_TEXT_COL]),
                                                        train_sentiments_featurized['start_position'],
                                                        train_sentiments_featurized['end_position'],
                                                        train_sentiments_featurized['extra_features']):
    train_docs.append(doc)
    train_selected_docs.append(selected_doc)
    train_label = encode_tokens_labels(doc, start_pos, end_pos)
    # train_label = [[1, 0] if start_pos <= token.idx < end_pos else [0, 1] for token in doc]
    train_labels.append(train_label)

    token_feats = [[
        analyzer.polarity_scores(remove_duplicates_char(token.text))['compound'],
        stg.BOOLEAN_ENCODING[token.is_stop],
        stg.BOOLEAN_ENCODING[token.like_url],
        stg.BOOLEAN_ENCODING[bool(re.search(stg.REGEX_PATTERN['<duplicate_chars>'], token.text))]
    ] for token in doc]

    train_extra_features.append([feats + token_feat for token_feat in token_feats])

# idx = 0
# print(train_docs[idx])
# print(train_selected_docs[idx])
# print(train_labels[idx])
# print(train_extra_features[idx])

validation_docs = []
validation_selected_docs = []
validation_labels = []
validation_extra_features = []
for doc, selected_doc, start_pos, end_pos, feats in zip(nlp.pipe(validation_sentiments_featurized[stg.TEXT_COL]),
                                                        nlp.pipe(
                                                            validation_sentiments_featurized[stg.SELECTED_TEXT_COL]),
                                                        validation_sentiments_featurized['start_position'],
                                                        validation_sentiments_featurized['end_position'],
                                                        validation_sentiments_featurized['extra_features']):
    validation_docs.append(doc)
    validation_selected_docs.append(selected_doc)
    #validation_label = [[1, 0] if start_pos <= token.idx < end_pos else [0, 1] for token in doc]
    validation_label = encode_tokens_labels(doc, start_pos, end_pos)
    validation_labels.append(validation_label)

    token_feats = [[
        analyzer.polarity_scores(remove_duplicates_char(token.text))['compound'],
        stg.BOOLEAN_ENCODING[token.is_stop],
        stg.BOOLEAN_ENCODING[token.like_url],
        stg.BOOLEAN_ENCODING[bool(re.search(stg.REGEX_PATTERN['<duplicate_chars>'], token.text))]
    ] for token in doc]

    validation_extra_features.append([feats + token_feat for token_feat in token_feats])

logging.info('Extra features created')
# text = "Hi, I'm late. Soooory "
# list(nlp(text))
# labels_exemple = [0, 0, 1, 1, 0, 0, 0]
# labels_exemple = [0, 0, 1, 0, 0, 0, 0]
# labels_exemple = [0, 0, 0, 1, 0, 0, 0]

dictionary = Dictionary([["<OOV>", "<PAD>"]])

x_train = [[remove_duplicates_char(token.lower_) for token in doc] for doc in train_docs]
train_dictionary = Dictionary(x_train)

train_selected_dictionary = Dictionary([[remove_duplicates_char(token.lower_) for token in doc]
                                        for doc in train_selected_docs])
train_dictionary.filter_extremes(no_above=0.7, no_below=5)
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)
dictionary.save(join(stg.MODELS_DIR, 'rnn_spacy_tokens_dict'))
print(f'taille du vocabulaire : {len(dictionary)}')

x_train_indexed = [[dictionary.token2id.get(remove_duplicates_char(token.lower_), 0) for token in doc]
                   for doc in train_docs]
x_validation_indexed = [[dictionary.token2id.get(remove_duplicates_char(token.lower_), 0) for token in doc]
                        for doc in validation_docs]

embedding_matrix = WordEmbedding(dictionary_token2id=dictionary.token2id).global_embedding_matrix
np.save(join('data', 'word_embeddings', 'init_embedding_matrix.npy'), embedding_matrix)
print(embedding_matrix.shape)
logging.info(f'embedding shape: {embedding_matrix.shape}')
BLSTM = BidirectionalLSTMCRF(hidden_dim=64, word_embedding_initialization=embedding_matrix)

print('---------------------------------------------------------------------------------------------------------------')
logging.info(f'BLSTM model: {BLSTM.model.summary()}')
print('---------------------------------------------------------------------------------------------------------------')

callbacks = ModelCheckpoint(join(stg.MODELS_DIR, "rnn_spacy_tokens.hdf5"),
                            monitor='loss',
                            verbose=0,
                            save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='auto', restore_best_weights=True)

model_fit = BLSTM.fit(X_word_indexes=x_train_indexed, X_features=train_extra_features, y=train_labels,
                      batch_size=128, epochs=150, validation_split=0.2, callbacks=[callbacks, earlystop], verbose=1)

train_pred = BLSTM.predict(X_test_word=x_train_indexed, X_test_features=train_extra_features)

train_bool_labels = [[1 if score[0] == 0.0 else 0 for score in scores] for scores in train_pred]

selected_text_predicted = []
for doc, labels in zip(train_docs, train_bool_labels):
    predicted_text = ''.join([token.text_with_ws for token, label in zip(doc, labels) if label == 1])
    selected_text_predicted.append(predicted_text)

train_preds_to_df = train_sentiments_featurized.filter(items=[stg.ID_COL])\
                                               .assign(sentence_pred=selected_text_predicted)

train_all_sentiments = pd.merge(left=train,
                                right=train_preds_to_df,
                                on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

train_score = jaccard_score(y_true=train_all_sentiments[stg.SELECTED_TEXT_COL],
                            y_pred=train_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'train score: {train_score}')
logging.info(f'train score: {train_score}')
print('--------------------------')

validation_pred = BLSTM.predict(X_test_word=x_validation_indexed, X_test_features=validation_extra_features)

validation_bool_labels = [[1 if score[0] == 0.0 else 0 for score in scores] for scores in validation_pred]

selected_text_predicted = []
for doc, labels in zip(validation_docs, validation_bool_labels):
    predicted_text = ''.join([token.text_with_ws for token, label in zip(doc, labels) if label == 1])
    selected_text_predicted.append(predicted_text)

validation_preds_to_df = validation_sentiments_featurized.filter(items=[stg.ID_COL])\
                                                         .assign(sentence_pred=selected_text_predicted)

validation_all_sentiments = pd.merge(left=validation,
                                     right=validation_preds_to_df[[stg.ID_COL, 'sentence_pred']],
                                     on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

validation_score = jaccard_score(y_true=validation_all_sentiments[stg.SELECTED_TEXT_COL],
                                 y_pred=validation_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'validation score: {validation_score}')
logging.info(f'validation score: {train_score}')
print('--------------------------')
