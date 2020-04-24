import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.corpora import Dictionary

import re

import sys
from os.path import join
import pandas as pd
import numpy as np

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentenceCleaner
from tweet_sentiment_extraction.utils.metrics import jaccard_score
from tweet_sentiment_extraction import settings as stg

######################################################################
# Example: An LSTM for Part-of-Speech Tagging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section, we will use an LSTM to get part of speech tags. We will
# not use Viterbi or Forward-Backward or anything like that, but as a
# (challenging) exercise to the reader, think about how Viterbi could be
# used after you have seen what is going on.
#
# The model is as follows: let our input sentence be
# :math:`w_1, \dots, w_M`, where :math:`w_i \in V`, our vocab. Also, let
# :math:`T` be our tag set, and :math:`y_i` the tag of word :math:`w_i`.
# Denote our prediction of the tag of word :math:`w_i` by
# :math:`\hat{y}_i`.
#
# This is a structure prediction, model, where our output is a sequence
# :math:`\hat{y}_1, \dots, \hat{y}_M`, where :math:`\hat{y}_i \in T`.
#
# To do the prediction, pass an LSTM over the sentence. Denote the hidden
# state at timestep :math:`i` as :math:`h_i`. Also, assign each tag a
# unique index (like how we had word\_to\_ix in the word embeddings
# section). Then our prediction rule for :math:`\hat{y}_i` is
#
# .. math::  \hat{y}_i = \text{argmax}_j \  (\log \text{Softmax}(Ah_i + b))_j
#
# That is, take the log softmax of the affine map of the hidden state,
# and the predicted tag is the tag that has the maximum value in this
# vector. Note this implies immediately that the dimensionality of the
# target space of :math:`A` is :math:`|T|`.
#
#
# Prepare data:

regex_pattern = {
    'URL_PATTERN': r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))',
    'HASHTAG_PATTERN': r'#\w*',
    'MENTION_PATTERN': r'@\w*',
    'RESERVED_WORDS_PATTERN': r'^(RT|FAV)'
}


def clean_text_with_repeated_letters(text):
    # 'REPEATED_LETTERS': r'([A-Za-z])\1{2,}'
    pass


def lower_and_add_flag_pattern(text):
    text = text.lower()

    for key, value in regex_pattern.items():
        text = re.sub(value, key, text)

    return text


def prepare_target(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(lower_and_add_flag_pattern(w), to_ix["<OOV>"]) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

train = train.query('sentiment != "neutral"')
validation = validation.query('sentiment != "neutral"')

train_data = SentenceCleaner.add_tokenized_column(df=train, column_name_to_tokenize=stg.TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(df=validation, column_name_to_tokenize=stg.TEXT_COL)

train_data = SentenceCleaner.add_tokenized_column(df=train_data, column_name_to_tokenize=stg.SELECTED_TEXT_COL)
validation_data = SentenceCleaner.add_tokenized_column(
    df=validation_data, column_name_to_tokenize=stg.SELECTED_TEXT_COL)

train_preprocessed_rnn = [
    (sentence, ["1" if word in target else "0" for word in sentence])
    for sentence, target in zip(train_data['tokens_text'], train_data['tokens_selected_text'])
]

validation_preprocessed_rnn = [
    (sentence, ["1" if word in target else "0" for word in sentence])
    for sentence, target in zip(validation_data['tokens_text'], validation_data['tokens_selected_text'])
]

dictionary = Dictionary([["<OOV>", "<PAD>"]])
train_selected_dictionary = Dictionary(train_data[stg.TOKENS_SELECTED_TEXT_COL].apply(
    lambda x: [lower_and_add_flag_pattern(word) for word in x]))
train_dictionary = Dictionary(train_data[stg.TOKENS_TEXT_COL].apply(
    lambda x: [lower_and_add_flag_pattern(word) for word in x]))
train_dictionary.filter_extremes(keep_n=3000)
filtered_train_dictionary = train_dictionary.token2id
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)


tag_to_ix = {"0": 0, "1": 1}


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

######################################################################
# Create the model:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Train the model:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(dictionary), len(tag_to_ix))
print(repr(model))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(train_preprocessed_rnn[0][0], dictionary.token2id)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
    print(epoch)
    for sentence, tags in train_preprocessed_rnn:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, dictionary.token2id)
        targets = prepare_target(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


with torch.no_grad():
    inputs = prepare_sequence(train_preprocessed_rnn[1][0], dictionary.token2id)
    tag_scores = model(inputs)
# See what the scores are after training

with torch.no_grad():
    tag_scores_list = []

    for sentence, _ in train_preprocessed_rnn:
        inputs = prepare_sequence(sentence, dictionary.token2id)
        tag_scores = model(inputs)
        tag_scores_list.append(tag_scores)

tag_scores_1d = [[x[1] for x in list_pred] for list_pred in tag_scores_list]
train_bool_pred = [[True if score > 0.5 else False for score in list_pred] for list_pred in tag_scores_1d]

flat = []
for e1 in range(len(train_bool_pred)):
    for e2 in range(len(train_bool_pred[e1])):
        flat.append(train_bool_pred[e1][e2])


train_model_pred = [' '.join(np.array(sentence)[pred])
                    for sentence, pred in zip(train_data['tokens_text'], train_bool_pred)]

train_score = jaccard_score(y_true=train[stg.SELECTED_TEXT_COL],
                            y_pred=train_model_pred)
print('--------------------------')
print(f'train score: {train_score}')
print('--------------------------')

with torch.no_grad():
    tag_scores_list = []

    for sentence, _ in validation_preprocessed_rnn:
        inputs = prepare_sequence(sentence, dictionary.token2id)
        tag_scores = model(inputs)
        tag_scores_list.append(tag_scores)

tag_scores_1d = [[x[1] for x in list_pred] for list_pred in tag_scores_list]
validation_bool_pred = [[True if score > 0.5 else False for score in list_pred] for list_pred in tag_scores_1d]
validation_model_pred = [' '.join(np.array(sentence)[pred])
                         for sentence, pred in zip(validation_data['tokens_text'], validation_bool_pred)]

validation_score = jaccard_score(y_true=validation_data[stg.SELECTED_TEXT_COL],
                                 y_pred=validation_model_pred)
print('--------------------------')
print(f'Validation score: {validation_score}')
print('--------------------------')

######################################################################
# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the example above, each word had an embedding, which served as the
# inputs to our sequence model. Let's augment the word embeddings with a
# representation derived from the characters of the word. We expect that
# this should help significantly, since character-level information like
# affixes have a large bearing on part-of-speech. For example, words with
# the affix *-ly* are almost always tagged as adverbs in English.
#
# To do this, let :math:`c_w` be the character-level representation of
# word :math:`w`. Let :math:`x_w` be the word embedding as before. Then
# the input to our sequence model is the concatenation of :math:`x_w` and
# :math:`c_w`. So if :math:`x_w` has dimension 5, and :math:`c_w`
# dimension 3, then our LSTM should accept an input of dimension 8.
#
# To get the character level representation, do an LSTM over the
# characters of a word, and let :math:`c_w` be the final hidden state of
# this LSTM. Hints:
#
# * There are going to be two LSTM's in your new model.
#   The original one that outputs POS tag scores, and the new one that
#   outputs a character-level representation of each word.
# * To do a sequence model over characters, you will have to embed characters.
#   The character embeddings will be the input to the character LSTM.
#

for i, sentence in enumerate(tag_scores_1d):
    if sum(np.array(sentence) > 0.1) > 0:
        print(i, sentence)