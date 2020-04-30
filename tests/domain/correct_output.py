import pytest

import numpy as np
import spacy

from tweet_sentiment_extraction.domain.correct_output import get_neighbors_until_space


nlp = spacy.load('en_core_web_md')

cas1 = ' you`re looking at!! So luckyyy.'
doc1 = nlp(cas1)
scores1 = [0, 0.1, 0.1, 0.7, 0.7, 0.3, 0.1, 0.7, 0.4]
wanted1 = 'at!! luckyyy.'
np.array(doc1)[np.array(scores1) > 0.5]

cas2 = "I'm famous!! #famous"
doc2 = nlp(cas2)
scores2 = [0, 0, 0.7, 0, 0, 0.1, 0.7]
wanted2 = 'famous!! #famous'
np.array(doc2)[np.array(scores2) > 0.5]

cas3 = "I'm famous!! #famous ##sofamous"
doc3 = nlp(cas3)
scores3 = [0, 0, 0.7, 0, 0, 0.1, 0.7, 0.1, 0.1, 0.7]
wanted3 = 'famous!! #famous ##sofamous'
np.array(doc3)[np.array(scores3) > 0.5]


def test_get_neighbors_until_space():
    corrected1 = get_neighbors_until_space(doc1)
    assert corrected1 == wanted1
    corrected2 = get_neighbors_until_space(doc2)
    assert corrected2 == wanted2
    corrected3 = get_neighbors_until_space(doc3)
    assert corrected3 == wanted3
