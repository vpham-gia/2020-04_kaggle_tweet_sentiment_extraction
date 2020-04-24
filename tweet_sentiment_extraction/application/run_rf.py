"""Random Forest model."""
from os.path import join, basename
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import argparse
import pandas as pd
import spacy

from tweet_sentiment_extraction.domain.dataset_builder import DatasetCreator
from tweet_sentiment_extraction.domain.sentence_constructor import SentenceConstructor as sc
from tweet_sentiment_extraction.utils.metrics import jaccard_score

import tweet_sentiment_extraction.settings as stg

from datetime import datetime as dt
print(f'{dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")} - Start')

PARSER = argparse.ArgumentParser(description='Parser for Tweet Extractor project.')

PARSER.add_argument('--read_saved_files', '-read', help='Boolean to read existing files.',
                    type=str, choices=['y', 'n'], default='y')

ARGS = PARSER.parse_args()

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))
test = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'test.csv'))

if ARGS.read_saved_files == 'y':
    train_dataset = pd.read_csv(join(stg.ML_DATA_DIR, 'ml_train_spacy_encoding_and_sentiment.csv'))
else:
    train_dataset = DatasetCreator(df=train, bool_train_mode=True).build_dataset()

nlp = spacy.load('en_core_web_md')
stop_words = nlp.Defaults.stop_words
print(f'Shape: {train_dataset.shape}')
train_dataset_no_stopwords = (
    train_dataset.assign(is_stopword=lambda df: df[stg.WORD_COL].apply(lambda x: x in stop_words))\
                 .query('not is_stopword')
)
print(f'Shape: {train_dataset_no_stopwords.shape}')

model = RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=45)
model.fit(X=train_dataset_no_stopwords[stg.ML_FEATURES_COL], y=train_dataset_no_stopwords[stg.ML_TARGET_COL])
print(f'{dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")} - OK fit')

train_dataset_with_ml_pred = train_dataset.assign(**{
    stg.ML_PRED_COL: lambda df: model.predict(X=df[stg.ML_FEATURES_COL])
})

train_with_tokens_from_ml = sc.compute_tokens_from_ml_predictions(df=train_dataset_with_ml_pred)
train_with_sentence_pred = sc.add_sentence_pred_from_tokens_col(df=train_with_tokens_from_ml)

train_for_score_assesment = pd.merge(left=train.rename(columns={'selected_text': stg.SENTENCE_TARGET_COL}),
                                     right=train_with_sentence_pred,
                                     on=stg.ID_COL, how='left')\
    .fillna({stg.SENTENCE_PRED_COL: ' '})

train_score = jaccard_score(y_true=train_for_score_assesment[stg.SENTENCE_TARGET_COL],
                            y_pred=train_for_score_assesment[stg.SENTENCE_PRED_COL])
print('--------------------------')
print(f'Train score: {train_score}')
print('--------------------------')

if ARGS.read_saved_files == 'y':
    validation_dataset = pd.read_csv(join(stg.ML_DATA_DIR, 'ml_validation_spacy_encoding_and_sentiment.csv'))
else:
    validation_dataset = DatasetCreator(df=validation, bool_train_mode=False).build_dataset()

validation_dataset_with_ml_pred = validation_dataset.assign(**{
    stg.ML_PRED_COL: lambda df: model.predict(X=df[stg.ML_FEATURES_COL])
})

validation_with_tokens_from_ml = sc.compute_tokens_from_ml_predictions(df=validation_dataset_with_ml_pred)
validation_with_sentence_pred = sc.add_sentence_pred_from_tokens_col(df=validation_with_tokens_from_ml)

validation_for_score_assesment = pd.merge(left=validation.rename(columns={'selected_text': stg.SENTENCE_TARGET_COL}),
                                          right=validation_with_sentence_pred,
                                          on=stg.ID_COL, how='left')\
    .fillna({stg.SENTENCE_PRED_COL: ' '})

validation_score = jaccard_score(y_true=validation_for_score_assesment[stg.SENTENCE_TARGET_COL],
                                 y_pred=validation_for_score_assesment[stg.SENTENCE_PRED_COL])
print('--------------------------')
print(f'Validation score: {validation_score}')
print('--------------------------')

print(f'{dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")} - End of script {basename(__file__)}')
