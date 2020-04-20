"""Builds dataset with target from original dataset with full sentence.

Classes
-------
Featurizer
SentencePreprocessor
"""
import pandas as pd
import spacy

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner

import tweet_sentiment_extraction.settings as stg


class Featurizer:
    """Creates features from sentiment and word_in_sentence columns."""

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

        df_with_word_encoding = pd.merge(left=df, right=df_word_encoding,
                                         on=stg.WORD_COL, how='left')

        return df_with_word_encoding


class SentencePreprocessor:
    """Preprocess sentences to get one word by row (with duplication of textID x sentiment).

    Methods
    -------
    pivot_sentence_in_column(df)
        Creates one row by word of a sentence with repetition of textID and sentiment.
    """

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

    a = SentencePreprocessor.pivot_sentence_in_column(df=df)

    from datetime import datetime as dt
    start = dt.now()
    b = Featurizer.encode_sentiment_column(df=a)
    end = dt.now()
    print(f'Total sentiment encoding time: {end - start}')

    start = dt.now()
    c = Featurizer.encode_word_to_vector(df=a)
    end = dt.now()
    print(f'Total word encoding time: {end - start}')

    # nlp = spacy.load('en_core_web_md')
    # print('Loaded')
    # start = dt.now()
    # word_encoding = list(nlp.pipe(a['word_in_sentence'].unique(), n_process=3))
    # end = dt.now()
    # print(f'Total word encoding time: {end - start}')
