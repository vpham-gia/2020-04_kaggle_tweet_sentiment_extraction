"""Builds dataset with target from original dataset with full sentence.

Classes
-------
Featurizer
SentencePreprocessor
"""
import numpy as np
import pandas as pd
import spacy

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner
from tweet_sentiment_extraction.utils.decorators import timer

import tweet_sentiment_extraction.settings as stg


class DatasetCreator:
    """Creates ML-ready dataset.

    Attributes
    ----------
    df: pandas.DataFrame
        Initial dataframe to be preprocessed to extract features and target (optional)
    bool_train_mode: boolean

    Methods
    -------
    build_dataset()

    Properties
    ----------
    df_features
    df_target_tokens

    """

    def __init__(self, df, bool_train_mode):
        """Initialize class."""
        self.df = df
        self.bool_train_mode = bool_train_mode

    @timer
    def build_dataset(self):
        """Build dataset.

        Returns
        -------
        dataset: pandas.DataFrame
        """
        if self.bool_train_mode:
            feats = self.df_features.dropna(subset=[stg.WORD_COL])
            target_tokens = self.df_target_tokens

            dataset = pd.merge(left=feats, right=target_tokens, on=stg.ID_COL, how='left')\
                        .assign(**{
                            stg.ML_TARGET_COL: lambda df: df.apply(
                                lambda x: x[stg.WORD_COL] in x[stg.TOKENS_SELECTED_TEXT_COL], axis=1)
                        })
        else:
            dataset = self.df_features

        return dataset

    @property
    def df_features(self):
        """Compute features.

        Returns
        -------
        df_features: pandas.DataFrame
        """
        sentences_pivoted = SentencePreprocessor.pivot_sentence_in_column(df=self.df)

        df_sentiment_column_encoded = Featurizer.encode_sentiment_column(df=sentences_pivoted)
        df_with_word_vector = Featurizer.encode_word_to_vector(df=df_sentiment_column_encoded)

        df_features = df_with_word_vector.filter(
            items=[stg.ID_COL, stg.POSITION_IN_SENTENCE_COL, stg.WORD_COL] + stg.ML_FEATURES_COL
        )

        return df_features

    @property
    def df_target_tokens(self):
        """Compute target tokens.

        Returns
        -------
        df_target: pandas.DataFrame
        """
        df_target = SentencePreprocessor.compute_target_tokens(df=self.df)
        return df_target


class Featurizer:
    """Creates features from sentiment and word_in_sentence columns.

    Methods
    -------
    encode_sentiment_column(df)
        Encode sentiment to {-1, 0, 1}.
    add_length_of_tweet_column(df)
        Add column with length of tweets (number of words).
    encode_word_to_vector(df)
        Encode word to vector using spacy models.
    """

    @classmethod
    def encode_sentiment_column(cls, df):
        """Encode sentiment to {-1, 0, 1}.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        df_with_sentiment_encoding: pandas.DataFrame
        """
        df_with_sentiment_encoding = df.assign(**{
            stg.SENTIMENT_COL: lambda df: df[stg.SENTIMENT_COL].map(stg.SENTIMENT_ENCODING)
        })
        return df_with_sentiment_encoding

    @classmethod
    def add_length_of_tweet_column(cls, df):
        """Add column with length of tweets (number of words).

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        df_with_length_of_tweet: pandas.DataFrame
        """
        df_with_length_of_tweet = df.assign(**{
            stg.FEAT_LENGTH_TWEET_COL: lambda df: df[stg.TEXT_COL].apply(lambda x: len(str(x).split()))
        })
        return df_with_length_of_tweet

    @classmethod
    def encode_word_to_vector(cls, df):
        """Encode word to vector using spacy models.

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        df_with_word_encoding: pandas.DataFrame
        """
        nlp = spacy.load('en_core_web_md')

        unique_words_found = df[stg.WORD_COL].unique()
        spacy_unique_words = list(nlp.pipe(unique_words_found, n_process=1))

        df_word_encoding = pd.DataFrame({
            stg.WORD_COL: [str(word) for word in spacy_unique_words],
            stg.WORD_ENCODING_COL: [word.vector for word in spacy_unique_words]
        })

        df_encoding_split = (
            df_word_encoding[stg.WORD_ENCODING_COL].apply(pd.Series)
                                                   .rename(columns=lambda x: f'{stg.PREFIX_SPACY_ENCODING_COL}_{x}')
        )

        df_with_word_encoding = pd.merge(left=df, right=pd.concat([df_word_encoding, df_encoding_split], axis=1),
                                         on=stg.WORD_COL, how='left')

        return df_with_word_encoding


class SentencePreprocessor:
    """Preprocesses sentences.

    Preprocessing includes:
    * Collection of target tokens by textID
    * Pivot of sentences to get one word by row (with duplication of textID x sentiment).

    Methods
    -------
    pivot_sentence_in_column(df)
        Creates one row by word of a sentence with repetition of textID and sentiment.
    """

    @classmethod
    def compute_target_tokens(cls, df):
        """Compute target tokens by textID.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        df_target: pandas.DataFrame
        """
        df_with_target_tokens = SentenceCleaner.add_tokenized_column(df=df,
                                                                     column_name_to_tokenize=stg.SELECTED_TEXT_COL)
        df_target = df_with_target_tokens.filter(items=[stg.ID_COL, stg.TOKENS_SELECTED_TEXT_COL])

        return df_target

    @classmethod
    def pivot_sentence_in_column(cls, df):
        """Create one row by word of a sentence with repetition of textID and sentiment.

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        df_pivot: pandas.DataFrame

        """
        df_tokens_columns = cls._convert_tokens_to_columns(df=df)

        COLS_TO_KEEP = [stg.ID_COL, stg.SENTIMENT_COL]
        df_with_tokens_columns = pd.merge(left=df.filter(items=COLS_TO_KEEP),
                                          right=df_tokens_columns,
                                          left_index=True, right_index=True, how='inner')

        df_pivot = df_with_tokens_columns.melt(id_vars=COLS_TO_KEEP,
                                               value_vars=[col for col in df_with_tokens_columns.columns
                                                           if col not in COLS_TO_KEEP],
                                               var_name=stg.POSITION_IN_SENTENCE_COL,
                                               value_name=stg.WORD_COL)\
            .sort_values(by=[stg.ID_COL, stg.POSITION_IN_SENTENCE_COL]).dropna()

        return df_pivot

    @classmethod
    def _convert_tokens_to_columns(cls, df):
        df_with_tokens = SentenceCleaner.add_tokenized_column(df=df, column_name_to_tokenize=stg.TEXT_COL)
        df_tokens_columns = df_with_tokens[stg.TOKENS_TEXT_COL].apply(pd.Series)

        return df_tokens_columns


if __name__ == '__main__':
    from os.path import join
    df = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))

    a = DatasetCreator(df=df.head(10), bool_train_mode=True)
    toto = a.build_dataset()

    # from datetime import datetime as dt
    # start = dt.now()
    # b = Featurizer.encode_sentiment_column(df=a)
    # end = dt.now()
    # print(f'Total sentiment encoding time: {end - start}')

    # from datetime import datetime as dt
    # start = dt.now()
    # a = SentencePreprocessor.pivot_sentence_in_column(df=df.head(10))
    # c = Featurizer.encode_word_to_vector(df=a)
    # end = dt.now()
    # print(f'Total word encoding time: {end - start}')
