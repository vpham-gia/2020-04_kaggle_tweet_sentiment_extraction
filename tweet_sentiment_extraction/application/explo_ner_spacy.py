"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
import sys
from spacy.util import minibatch, compounding
# from pathlib import Path
import random
# import plac
from datetime import datetime as dt
from os.path import join, basename

import numpy as np
import pandas as pd
import spacy

from tweet_sentiment_extraction.utils.metrics import jaccard_score

import tweet_sentiment_extraction.settings as stg

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

train = train.assign(start_position=lambda df: df.apply(lambda x: x['text'].find(x['selected_text']), axis=1),
                     len_selected_text=lambda df: df['selected_text'].apply(len),
                     end_position=lambda df: df['start_position'] + df['len_selected_text'])

train_with_sentiment = train.query('sentiment in ["positive", "negative"]')

train_data_spacy = [
    (sentence, {'entities': [(start, end, sentiment)]})
    for sentence, start, end, sentiment in zip(train_with_sentiment['text'],
                                               train_with_sentiment['start_position'],
                                               train_with_sentiment['end_position'],
                                               train_with_sentiment['sentiment'])
]

nlp = spacy.blank("en")  # create blank Language class
print("Created blank 'en' model")

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe("ner")

# add labels
for _, annotations in train_data_spacy:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# get names of other pipes to disable them during training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes):  # only train NER
    # reset and initialize the weights randomly – but only if we're
    # training a new model
    nlp.begin_training()

    for itn in range(100):
        random.shuffle(train_data_spacy)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data_spacy, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                losses=losses,
            )
        print(f'{dt.strftime(dt.now(), "%Y-%m-%d %H:%M:%S")} Losses', losses)


def toto(list_x):
    try:
        return list_x[0]
    except IndexError:
        return ' '

sys.exit()

train_pred = [[ent.text if ent.label_ == sentiment else 'no' for ent in doc.ents if ent.label_ == sentiment]
              for doc, sentiment in zip(nlp.pipe(train['text']), train['sentiment'])]

train = train.assign(
    spacy_pred=train_pred,
    sentence_pred=lambda df: np.where(df['sentiment'] == 'neutral', df['text'], df['spacy_pred'].apply(toto))
)

train_score = jaccard_score(y_true=train['selected_text'],
                            y_pred=train['sentence_pred'])
print('--------------------------')
print(f'Train score: {train_score}')
print('--------------------------')

validation_pred = [[ent.text if ent.label_ == sentiment else 'no' for ent in doc.ents if ent.label_ == sentiment]
                   for doc, sentiment in zip(nlp.pipe(validation['text']), validation['sentiment'])]

validation = validation.assign(
    spacy_pred=validation_pred,
    sentence_pred=lambda df: np.where(df['sentiment'] == 'neutral', df['text'], df['spacy_pred'].apply(toto))
)

validation_score = jaccard_score(y_true=validation['selected_text'],
                                 y_pred=validation['sentence_pred'])
print('--------------------------')
print(f'Validation score: {validation_score}')
print('--------------------------')



sys.exit()


# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]
