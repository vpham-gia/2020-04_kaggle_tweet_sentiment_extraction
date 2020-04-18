"""Create and save validation data."""

from os.path import join
from sklearn.model_selection import train_test_split

import pandas as pd

import tweet_sentiment_extraction.settings as stg

raw_train = pd.read_csv(join(stg.RAW_DATA_DIR, 'train.csv'))
raw_test = pd.read_csv(join(stg.RAW_DATA_DIR, 'test.csv'))

train, validation = train_test_split(raw_train, test_size=raw_test.shape[0])

train.to_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'), index=False)
validation.to_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'), index=False)
