"""Technical cleaning of tweets dataset.

Class
-----
SentenceCleaner
SentencePreprocessor

"""
import re

import tweet_sentiment_extraction.settings as stg


class SentenceCleaner:
    """Performs technical cleaning.

    Methods
    -------
    add_tokenized_column(df)
        Add column with sentence converted in tokens.
    """

    @classmethod
    def add_tokenized_column(self, df, column_name_to_tokenize):
        """Add column with sentence converted in tokens.

        Parameters
        ----------
        df: pandas.dataFrame
        column_name_to_tokenize: string

        Returns
        -------
        df_with_tokens: pandas.DataFrame
        """
        COL = column_name_to_tokenize
        df_with_tokens = df.assign(**{f'tokens_{COL}': lambda df: df[COL].apply(lambda x: str(x).split())})
        return df_with_tokens


class SentencePreprocessor:
    """Preprocess sentences from a given dataframe.

    Attributes
    ----------
    df: pandas.DataFrame
        Dataframe with tokenized columns (for text and selected_text)

    Properties
    ----------
    preprocessed_dataset()

    """

    def __init__(self, df):
        """Initialize class."""
        self.df = df

    def preprocess_dataset(self, vocabulary):
        """Preprocess dataset to be Tensorflow-ready.

        Preprocessing includes:
        * Add target sequence column
        * Add additional features column
        * Add cleaned_original tokens features

        Parameters
        ----------
        vocabulary: dict
            Key: word, value: index

        Returns
        -------
        df_preprocessed: pandas.DataFrame

        """
        df_with_label_sequence = self._compute_target_sequence(df=self.df)
        df_with_cleaned_original_tokens = self._clean_original_tokens(df=df_with_label_sequence, vocabulary=vocabulary)
        df_with_additional_features = self._compute_additional_features(df=df_with_cleaned_original_tokens)

        df_preprocessed = df_with_additional_features
        return df_preprocessed

    def _compute_target_sequence(self, df):
        df_with_target_sequence = df.assign(**{
            stg.TARGET_SEQUENCE_COL: lambda df: df.apply(lambda row: ['1' if w in row[stg.TOKENS_SELECTED_TEXT_COL]
                                                                      else '0'
                                                                      for w in row[stg.TOKENS_TEXT_COL]], axis=1)
        })
        return df_with_target_sequence

    def _clean_original_tokens(self, df, vocabulary):
        _WORDS_IN_VOCAB = set(vocabulary.keys())

        df_with_cleaned_sequence = df.assign(**{
            stg.CLEANED_TOKENS_COL: lambda df: df[stg.TOKENS_TEXT_COL].apply(
                lambda x: [self._lower_and_add_flag_pattern(token) for token in x])
        }).assign(**{
            stg.CLEANED_TOKENS_COL: lambda df: df[stg.CLEANED_TOKENS_COL].apply(
                lambda x: [token if token in _WORDS_IN_VOCAB else '<OOV>' for token in x]),
            stg.INDEXED_TOKENS_COL: lambda df: df[stg.CLEANED_TOKENS_COL].apply(
                lambda x: [vocabulary.get(token) for token in x])
        })

        return df_with_cleaned_sequence

    @staticmethod
    def _lower_and_add_flag_pattern(text):
        text = text.lower()
        for key, value in stg.REGEX_PATTERN.items():
            text = re.sub(value, key, text)
        return text

    def _compute_additional_features(self, df):
        """TODO."""
        return df


if __name__ == '__main__':
    import pandas as pd
    from os.path import join
    train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))
    a = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize='text')
    b = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize='selected_text')
