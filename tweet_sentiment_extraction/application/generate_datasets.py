"""Create and save datasets."""

from os.path import join
from sklearn.model_selection import train_test_split

import pandas as pd

from tweet_sentiment_extraction.domain.dataset_builder import DatasetCreator

import tweet_sentiment_extraction.settings as stg


def generate_save_train_validation_datasets():
    """Generate and save train and validation sets."""
    raw_train = pd.read_csv(join(stg.RAW_DATA_DIR, 'train.csv'))
    raw_test = pd.read_csv(join(stg.RAW_DATA_DIR, 'test.csv'))

    train, validation = train_test_split(raw_train, test_size=raw_test.shape[0], random_state=45)

    train.to_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'), index=False)
    validation.to_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'), index=False)


def generate_save_machine_learning_datasets(suffix=''):
    """Generate and save ML-ready datasets."""
    train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
    validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

    train_dataset = DatasetCreator(df=train, bool_train_mode=True).build_dataset()
    validation_dataset = DatasetCreator(df=validation, bool_train_mode=False).build_dataset()

    train_dataset.to_csv(join(stg.ML_DATA_DIR, f'ml_train_{suffix}.csv'), index=False)
    validation_dataset.to_csv(join(stg.ML_DATA_DIR, f'ml_validation_{suffix}.csv'), index=False)


if __name__ == "__main__":
    # generate_save_train_validation_datasets()
    generate_save_machine_learning_datasets(suffix='spacy_encoding_and_sentiment')
