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


if __name__ == '__main__':
    import pandas as pd
    from os.path import join
    train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv'))
    a = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize='text')
    b = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize='selected_text')
