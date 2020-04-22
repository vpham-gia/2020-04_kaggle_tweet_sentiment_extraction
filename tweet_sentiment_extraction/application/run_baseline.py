"""Baseline model is the cleaned up text."""
from os.path import join

import pandas as pd

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner
from tweet_sentiment_extraction.domain.sentence_constructor import SentenceConstructor as sc
from tweet_sentiment_extraction.utils.metrics import jaccard_score

import tweet_sentiment_extraction.settings as stg

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))
test = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'test.csv'))

train_tokens = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize=stg.TEXT_COL)\
                              .dropna()\
                              .rename(columns={stg.SELECTED_TEXT_COL: stg.SENTENCE_TARGET_COL,
                                               stg.TOKENS_TEXT_COL: stg.TOKENS_PRED_COL})
train_predictions = sc.add_sentence_pred_from_tokens_col(df=train_tokens)

train_score = jaccard_score(y_true=train_predictions[stg.SENTENCE_TARGET_COL],
                            y_pred=train_predictions[stg.SENTENCE_PRED_COL])
print(f'Train score: {train_score}')

validation_tokens = SentenceCleaner.add_tokenized_column(df=validation, column_name_to_tokenize=stg.TEXT_COL)\
                                   .rename(columns={stg.SELECTED_TEXT_COL: stg.SENTENCE_TARGET_COL,
                                                    stg.TOKENS_TEXT_COL: stg.TOKENS_PRED_COL})
validation_predictions = sc.add_sentence_pred_from_tokens_col(df=validation_tokens)

validation_score = jaccard_score(y_true=validation_predictions[stg.SENTENCE_TARGET_COL],
                                 y_pred=validation_predictions[stg.SENTENCE_PRED_COL])
print(f'Validation score: {validation_score}')

test_tokens = SentenceCleaner.add_tokenized_column(df=test, column_name_to_tokenize=stg.TEXT_COL)\
                             .rename(columns={stg.TOKENS_TEXT_COL: stg.TOKENS_PRED_COL})
test_predictions = sc.add_sentence_pred_from_tokens_col(df=test_tokens)
test_predictions.rename(columns={stg.SENTENCE_PRED_COL: stg.SELECTED_TEXT_COL})\
                .filter(items=[stg.ID_COL, stg.SELECTED_TEXT_COL])\
                .to_csv(join(stg.OUTPUTS_DIR, 'baseline.csv'), index=False)
