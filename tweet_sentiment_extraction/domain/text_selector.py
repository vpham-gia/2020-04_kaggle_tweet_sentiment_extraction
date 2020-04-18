"""Text selection for given tweet.

Class
-----
TextSelector

"""
import tweet_sentiment_extraction.settings as stg


class TextSelector:
    """TODO."""

    def __init__(self, df):
        """Initialize class."""
        self.data = df

    def add_pred_from_tokens_col(self):
        """Build selected_text column from list of tokens.

        Returns
        -------
        df_with_selected_text: pandas.DataFrame
        """
        df_with_selected_text = self.data.assign(
            **{stg.SELECTED_TEXT_COL: lambda df: df[stg.TOKENS_COL].apply(lambda x: ' '.join(x))}
        )

        return df_with_selected_text
