"""Random Forest model."""
from os.path import join
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from tweet_sentiment_extraction.domain.dataset_builder import DatasetCreator
from tweet_sentiment_extraction.domain.sentence_constructor import SentenceConstructor as sc
from tweet_sentiment_extraction.utils.metrics import jaccard_score

import tweet_sentiment_extraction.settings as stg

from datetime import datetime as dt
print(f'{dt.now()} - Start')

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))
test = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'test.csv'))

train_dataset = DatasetCreator(df=train, bool_train_mode=True).build_dataset()

rf = RandomForestClassifier(n_estimators=100, n_jobs=3)
rf.fit(X=train_dataset[stg.ML_FEATURES_COL], y=train_dataset[stg.ML_TARGET_COL])

train_dataset_with_ml_pred = train_dataset.assign(**{
    stg.ML_PRED_COL: lambda df: rf.predict(X=df[stg.ML_FEATURES_COL])
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

validation_dataset = DatasetCreator(df=validation, bool_train_mode=False).build_dataset()

validation_dataset_with_ml_pred = validation_dataset.assign(**{
    stg.ML_PRED_COL: lambda df: rf.predict(X=df[stg.ML_FEATURES_COL])
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
