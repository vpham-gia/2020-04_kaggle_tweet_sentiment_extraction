import pytest

import numpy as np
import spacy

from tweet_sentiment_extraction.domain.correct_output import patch_whitespace


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


def test_patch_whitespace():
    corrected1 = patch_whitespace(doc1, scores1)
    assert corrected1 == wanted1
    corrected2 = patch_whitespace(doc2, scores2)
    assert corrected2 == wanted2
    corrected3 = patch_whitespace(doc3, scores3)
    assert corrected3 == wanted3
