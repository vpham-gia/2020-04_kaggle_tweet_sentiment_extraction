"""Baseline model is the cleaned up text."""
from os.path import join

import pandas as pd

from tweet_sentiment_extraction.domain.metrics import jaccard_score
from tweet_sentiment_extraction.domain.text_selector import TextSelector
from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner

import tweet_sentiment_extraction.settings as stg

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))
test = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'test.csv'))

train_tokens = SentenceCleaner.add_tokenized_column(df=train)\
                              .dropna()\
                              .rename(columns={'selected_text': 'target'})
train_predictions = TextSelector.add_pred_from_tokens_col(df=train_tokens)

train_score = jaccard_score(y_true=train_predictions['target'],
                            y_pred=train_predictions['selected_text'])
print(f'Train score: {train_score}')

validation_tokens = SentenceCleaner.add_tokenized_column(df=validation)\
                                   .dropna()\
                                   .rename(columns={'selected_text': 'target'})
validation_predictions = TextSelector.add_pred_from_tokens_col(df=validation_tokens)

validation_score = jaccard_score(y_true=validation_predictions['target'],
                                 y_pred=validation_predictions['selected_text'])
print(f'Validation score: {validation_score}')

test_tokens = SentenceCleaner.add_tokenized_column(df=test)
test_predictions = TextSelector.add_pred_from_tokens_col(df=test_tokens)
test_predictions.filter(items=[stg.ID_COL, stg.SELECTED_TEXT_COL])\
                .to_csv(join(stg.OUTPUTS_DIR, 'baseline.csv'), index=False)
