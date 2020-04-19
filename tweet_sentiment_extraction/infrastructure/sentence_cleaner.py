"""Technical cleaning of tweets dataset.

Class
-----
SentenceCleaner

"""
import tweet_sentiment_extraction.settings as stg


class SentenceCleaner:
    """Performs technical cleaning.

    Methods
    -------
    add_tokenized_column(df)
        Add column with sentence converted in tokens.
    """

    @classmethod
    def add_tokenized_column(self, df):
        """Add column with sentence converted in tokens.

        Parameters
        ----------
        df: pandas.dataFrame

        Returns
        -------
        df_with_tokens: pandas.DataFrame
        """
        df_with_tokens = df.assign(**{stg.TOKENS_COL: lambda df: df[stg.TEXT_COL].apply(lambda x: str(x).split())})
        return df_with_tokens


if __name__ == '__main__':
    import pandas as pd
    from os.path import join
    train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))
    clean = SentenceCleaner.add_tokenized_column(df=train)
