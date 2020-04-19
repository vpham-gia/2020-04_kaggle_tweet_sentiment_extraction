"""Text selection for given tweet.

Class
-----
TextSelector

"""
import tweet_sentiment_extraction.settings as stg


class TextSelector:
    """TODO.

    Methods
    -------
    add_pred_from_tokens_col(df)
        Build selected_text column from list of tokens.
    """

    @classmethod
    def add_pred_from_tokens_col(self, df):
        """Build selected_text column from list of tokens.

        Parameters
        ----------
        df: pandas.dataFrame

        Returns
        -------
        df_with_selected_text: pandas.DataFrame
        """
        df_with_selected_text = df.assign(
            **{stg.SELECTED_TEXT_COL: lambda df: df[stg.TOKENS_COL].apply(lambda x: ' '.join(x))}
        )

        return df_with_selected_text
