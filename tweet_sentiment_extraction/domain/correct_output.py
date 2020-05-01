import pandas as pd
import numpy as np

THRESH = 0.5


def patch_whitespace(tokens, scores):
    labels = [True if score >= THRESH else False for score in scores]
    patched_labels = labels

    for position in range(len(patched_labels) - 1):
        if labels[position] and tokens[position].whitespace_ == '':
            patched_labels[position + 1] = True

    for position in range(len(patched_labels) - 1, 0, -1):
        if labels[position] and tokens[position - 1].whitespace_ == '':
            patched_labels[position - 1] = True

    return ''.join([token.text_with_ws for token, label in zip(tokens, labels) if label])
