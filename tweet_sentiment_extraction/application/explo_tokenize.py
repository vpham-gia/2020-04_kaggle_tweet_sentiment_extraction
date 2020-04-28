from os.path import join

import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary

from tweet_sentiment_extraction.domain.dataset_builder import Featurizer
from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentencePreprocessor
from tweet_sentiment_extraction import settings as stg
from tweet_sentiment_extraction.domain.bi_lstm import BidirectionalLSTM
from tweet_sentiment_extraction.utils.metrics import jaccard_score

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

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

train_docs = []
train_selected_docs = []
train_labels = []
train_extra_features = []
for doc, selected_doc, start_pos, end_pos, feats in zip(nlp.pipe(train_sentiments_featurized[stg.TEXT_COL]),
                                                        nlp.pipe(train_sentiments_featurized[stg.SELECTED_TEXT_COL]),
                                                        train_sentiments_featurized['start_position'],
                                                        train_sentiments_featurized['end_position'],
                                                        train_sentiments_featurized['extra_features']):
    train_docs.append(doc)
    train_selected_docs.append(selected_doc)
    train_label = [1 if start_pos <= token.idx < end_pos else 0 for token in doc]
    train_labels.append(train_label)
    train_extra_features.append([feats for i in range(len(list(doc)))])

validation_docs = []
validation_selected_docs = []
validation_labels = []
validation_extra_features = []
for doc, selected_doc, start_pos, end_pos, feats in zip(nlp.pipe(validation_sentiments_featurized[stg.TEXT_COL]),
                                                        nlp.pipe(validation_sentiments_featurized[stg.SELECTED_TEXT_COL]),
                                                        validation_sentiments_featurized['start_position'],
                                                        validation_sentiments_featurized['end_position'],
                                                        validation_sentiments_featurized['extra_features']):
    validation_docs.append(doc)
    validation_selected_docs.append(selected_doc)
    validation_label = [1 if start_pos <= token.idx < end_pos else 0 for token in doc]
    validation_labels.append(validation_label)
    validation_extra_features.append([feats for i in range(len(list(doc)))])

# text = "Hi, I'm late. Soooory "
# list(nlp(text))
# labels_exemple = [0, 0, 1, 1, 0, 0, 0]
# labels_exemple = [0, 0, 1, 0, 0, 0, 0]
# labels_exemple = [0, 0, 0, 1, 0, 0, 0]

dictionary = Dictionary([["<OOV>", "<PAD>"]])

x_train = [[token.lower_ for token in doc] for doc in train_docs]
train_dictionary = Dictionary(x_train)

train_selected_dictionary = Dictionary([[token.lower_ for token in doc] for doc in train_selected_docs])
train_dictionary.filter_extremes(keep_n=3000)
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)

x_train_indexed = [[dictionary.token2id.get(token.lower_, 0) for token in doc] for doc in train_docs]
x_validation_indexed = [[dictionary.token2id.get(token.lower_, 0) for token in doc] for doc in validation_docs]


def adapt_spacy_to_dictionary(pre_trained_embedding, dictionary_token2id=dictionary.token2id):
    embedding_matrix = np.zeros((len(dictionary_token2id), 300))
    out_of_vocabulary = []
    for word, i in dictionary_token2id.items():
        if i >= 2:  # 0 is the index of OOV, 1 is the index of Padding
            values = pre_trained_embedding.get_vector(word)
            if values is not None:
                embedding_matrix[i] = values
            else:
                out_of_vocabulary.append(word)
    print(f'out_of_vocabulary: {len(out_of_vocabulary)}')

    return embedding_matrix  # , out_of_vocabulary


embedding_matrix = adapt_spacy_to_dictionary(nlp.vocab, dictionary.token2id)

BLSTM = BidirectionalLSTM(hidden_dim=128, word_embedding_initialization=embedding_matrix)

callbacks = ModelCheckpoint(join(stg.ML_DATA_DIR, "best_model.hdf5"), monitor='loss', verbose=0, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

model_fit = BLSTM.fit(X_word_indexes=x_train_indexed, X_features=train_extra_features,
                      y=train_labels,
                      batch_size=32, epochs=5, validation_split=0.2, callbacks=[callbacks, earlystop], verbose=1)

train_pred = BLSTM.predict(X_test_word=x_train_indexed, X_test_features=train_extra_features)

train_bool_labels = [[1 if score[0] > 0.5 else 0 for score in scores] for scores in train_pred]

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
print('--------------------------')

validation_pred = BLSTM.predict(X_test_word=x_validation_indexed, X_test_features=validation_extra_features)

validation_bool_labels = [[1 if score[0] > 0.5 else 0 for score in scores] for scores in validation_pred]

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
print('--------------------------')
