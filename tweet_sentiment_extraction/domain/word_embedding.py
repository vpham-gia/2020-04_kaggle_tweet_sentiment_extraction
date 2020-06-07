"""Custom word embeddings.

Classes
-------
WordEmbedding
"""
from os.path import join

import numpy as np
import spacy

from tweet_sentiment_extraction import settings as stg
from tweet_sentiment_extraction.utils.decorators import timer


class WordEmbedding:
    """Computes word embedding matrices from GloVe and spacy.

    Attributes
    ----------
    dictionary: dict
        Token to ID dictionary
    glove_matrix: numpy.array of shape (len(dictionary), 200)
    spacy_matrix: numpy.array of shape (len(dictionary), 300)
    global_embedding_matrix: numpy.array of shape (len(dictionary), 200 + 300)
    """

    def __init__(self, dictionary_token2id):
        """Initialize class."""
        self.dictionary = dictionary_token2id
        self.glove_matrix = self._glove_matrix
        self.spacy_matrix = self._spacy_matrix
        self.global_embedding_matrix = np.concatenate((self.spacy_matrix, self.glove_matrix), axis=1)

    @property
    def _glove_matrix(self):
        pre_trained_glove = self._load_word_embedding(filename='glove.twitter.27B.200d.txt')
        glove_word_embedding_matrix = self._adapt_glove_to_dictionary(pre_trained_glove_values=pre_trained_glove)
        return glove_word_embedding_matrix

    @property
    def _spacy_matrix(self):
        nlp = spacy.load('en_core_web_md')
        spacy_word_embedding_matrix = self._adapt_spacy_to_dictionary(pre_trained_spacy_values=nlp.vocab)
        return spacy_word_embedding_matrix

    @staticmethod
    def _load_word_embedding(filename):
        file_ = open(join(stg.WORD_EMBEDDING_DIR, filename))

        embedding_values = {}
        for line in file_:
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            embedding_values[word] = coef
        return embedding_values

    def _adapt_glove_to_dictionary(self, pre_trained_glove_values):
        all_embs = np.stack(pre_trained_glove_values.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        emb_mean, emb_std

        embedding_matrix = np.random.normal(emb_mean, emb_std, (len(self.dictionary), 200))
        embedding_matrix[1] = 0  # Padding embedding

        out_of_vocabulary = []
        for word, i in self.dictionary.items():
            if i >= 2:  # 0 is the index of OOV, 1 is the index of Padding
                values = pre_trained_glove_values.get(word)
                if values is not None:
                    embedding_matrix[i] = values
                else:
                    out_of_vocabulary.append(word)
        print(f'GloVe out_of_vocabulary: {len(out_of_vocabulary)}')

        return embedding_matrix

    def _adapt_spacy_to_dictionary(self, pre_trained_spacy_values):
        embedding_matrix = np.zeros((len(self.dictionary), 300))
        out_of_vocabulary = []
        for word, i in self.dictionary.items():
            if i >= 2:  # 0 is the index of OOV, 1 is the index of Padding
                values = pre_trained_spacy_values.get_vector(word)
                if values is not None:
                    embedding_matrix[i] = values
                else:
                    out_of_vocabulary.append(word)
        print(f'Spacy out_of_vocabulary: {len(out_of_vocabulary)}')

        return embedding_matrix


if __name__ == "__main__":
    from gensim.corpora import Dictionary
    vocabulary = Dictionary([["I", "love"], ["chocolate"]])
    embedding_matrix = WordEmbedding(vocabulary.token2id)
