import pandas as pd
import numpy as np

THRESH = 0.5


def patch_whitespace(doc, scores):
    labels = [True if score >= THRESH else False for score in scores]

    right_propagation = np.logical_and(pd.Series(labels).shift(1).fillna(False),
                                       pd.Series([token.whitespace_ != ' ' for token in doc]).shift(1).fillna(False))

    right_propagation2 = np.logical_and(pd.Series(labels).shift(2).fillna(False),
                                        pd.Series([token.whitespace_ != ' ' for token in doc]).shift(2).fillna(False))

    left_propagation = np.logical_and(pd.Series(labels).shift(-1).fillna(False),
                                      pd.Series([token.whitespace_ != ' ' for token in doc]).fillna(False))

    labels = np.logical_or.reduce((labels, right_propagation, right_propagation2, left_propagation))
    return ''.join([token.text_with_ws for token, label in zip(doc, labels) if label])

