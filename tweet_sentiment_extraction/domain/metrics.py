"""Computes performance metrics."""

import numpy as np


def jaccard_score(y_true, y_pred):
    """Jaccard score.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    """
    try:
        assert(len(y_true) == len(y_pred))

        jaccard_vector = [_jaccard(str1=str(true), str2=str(pred)) for true, pred in zip(y_true, y_pred)]
        jaccard_score = np.mean(jaccard_vector)
        return jaccard_score
    except AssertionError:
        print('y_true and y_pred lengths differ.')


def _jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
