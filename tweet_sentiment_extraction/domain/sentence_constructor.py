"""Text selection for given tweet.

Class
-----
SentenceConstructor

"""
import tweet_sentiment_extraction.settings as stg


class SentenceConstructor:
    """Build sentence prediction from either ML output or tokens.

    Methods
    -------
    add_sentence_pred_from_tokens_col(df)
        Build selected_text column from list of tokens.
    """

    @classmethod
    def add_sentence_pred_from_tokens_col(cls, df):
        """Build sentence_pred column from list of tokens.

        Parameters
        ----------
        df: pandas.dataFrame

        Returns
        -------
        df_with_sentence_pred: pandas.DataFrame
        """
        df_with_sentence_pred = df.assign(
            **{stg.SENTENCE_PRED_COL: lambda df: df[stg.TOKENS_PRED_COL].apply(cls._concatenate_tokens)}
        )

        return df_with_sentence_pred

    @staticmethod
    def _concatenate_tokens(tokens_list):
        """Concatenate elements from tokens list or return empty string."""
        try:
            return ' '.join(tokens_list)
        except TypeError:
            return ' '

    @classmethod
    def compute_tokens_from_ml_predictions(cls, df):
        """Post-process Machine Learning predictions to extract list of tokens for each tweet.

        Parameters
        ----------
        df: pandas.dataFrame

        Returns
        -------
        df_with_tokens_from_ml: pandas.dataFrame
        """
        df_with_tokens_from_ml = df.query(f'{stg.ML_PRED_COL}')\
                                   .groupby(by=stg.ID_COL, as_index=False)\
                                   .agg({stg.WORD_COL: list})\
                                   .rename(columns={stg.WORD_COL: stg.TOKENS_PRED_COL})
        return df_with_tokens_from_ml
