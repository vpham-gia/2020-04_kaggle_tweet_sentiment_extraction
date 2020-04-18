"""Baseline model is the cleaned up text."""
from os.path import join

from tweet_sentiment_extraction.domain.metrics import jaccard_score
from tweet_sentiment_extraction.domain.text_selector import TextSelector
from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner

import tweet_sentiment_extraction.settings as stg

train_with_tokens = (
    SentenceCleaner(filename='train.csv').add_tokenized_column()
    .rename(columns={'selected_text': 'target'})
)

train_with_predictions = TextSelector(df=train_with_tokens).add_pred_from_tokens_col()

train_score = jaccard_score(y_true=train_with_predictions['target'],
                            y_pred=train_with_predictions['selected_text'])
print(f'Train score: {train_score}')

validation_with_tokens = (
    SentenceCleaner(filename='validation.csv').add_tokenized_column()
    .rename(columns={'selected_text': 'target'})
)

validation_with_predictions = TextSelector(df=validation_with_tokens).add_pred_from_tokens_col()

validation_score = jaccard_score(y_true=validation_with_predictions['target'],
                                 y_pred=validation_with_predictions['selected_text'])
print(f'Validation score: {validation_score}')

test_with_tokens = (
    SentenceCleaner(filename='test.csv').add_tokenized_column()
)

test_with_predictions = TextSelector(df=test_with_tokens).add_pred_from_tokens_col()

test_with_predictions.filter(items=[stg.ID_COL, stg.SELECTED_TEXT_COL])\
                     .to_csv(join(stg.OUTPUTS_DIR, 'baseline.csv'), index=False)
