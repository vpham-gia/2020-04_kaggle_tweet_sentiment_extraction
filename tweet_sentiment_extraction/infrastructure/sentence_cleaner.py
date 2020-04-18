"""Technical cleaning of tweets dataset.

Class
-----
SentenceCleaner

"""
from os.path import join

import pandas as pd

import tweet_sentiment_extraction.settings as stg


class SentenceCleaner:
    """Class to perform technical cleaning.

    Attributes
    ----------
    data: pandas.DataFrame

    Methods
    -------
    add_tokenized_column()
        Add column with sentence converted in tokens.
    """

    def __init__(self, filename):
        self.data = pd.read_csv(join(stg.PROCESSED_DATA_DIR, filename))

    def add_tokenized_column(self):
        """Add column with sentence converted in tokens.

        Returns
        -------
        df_with_tokens: pandas.DataFrame
        """
        df_with_tokens = self.data.assign(**{
            stg.TOKENS_COL: lambda df: df[stg.TEXT_COL].apply(lambda x: str(x).split())}
        )

        return df_with_tokens


if __name__ == '__main__':
    sc = SentenceCleaner(filename='train.csv')

    df = sc.add_tokenized_column()
