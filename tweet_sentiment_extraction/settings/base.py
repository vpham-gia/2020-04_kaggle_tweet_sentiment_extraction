"""Basic settings of the project.

Contains all configurations for the projectself.
Should NOT contain any secrets.

>>> import settings
>>> settings.DATA_DIR
"""
import os
import logging

# By default the data is stored in this repository's "data/" folder.
# You can change it in your own settings file.
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

DATA_DIR = os.path.join(REPO_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ML_DATA_DIR = os.path.join(DATA_DIR, 'ml-ready')
WORD_EMBEDDING_DIR = os.path.join(DATA_DIR, 'word_embeddings')

MODELS_DIR = os.path.join(REPO_DIR, 'models')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')

TESTS_DIR = os.path.join(REPO_DIR, 'tests')
TESTS_DATA_DIR = os.path.join(TESTS_DIR, 'fixtures')


# Logging
def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Set loggings parameters.

    Parameters
    ----------
    log_filename: str
    logging_level: logging.level

    """
    with open(os.path.join(LOGS_DIR, log_filename), 'a') as file:
        file.write('\n')
        file.write('\n')

    LOGGING_FORMAT = '[%(asctime)s][%(levelname)s][%(module)s] - %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=os.path.join(LOGS_DIR, log_filename)
    )


ID_COL = 'textID'
TEXT_COL = 'text'
SENTIMENT_COL = 'sentiment'
SELECTED_TEXT_COL = 'selected_text'

TOKENS_TEXT_COL = f'tokens_{TEXT_COL}'
TOKENS_SELECTED_TEXT_COL = f'tokens_{SELECTED_TEXT_COL}'
WORD_COL = 'word_in_sentence'
POSITION_IN_SENTENCE_COL = 'word_position_in_sentence'

SENTIMENT_ENCODING = {
    'negative': -1,
    'neutral': 0,
    'positive': 1
}

BOOLEAN_ENCODING = {True: 1, False: 0}

FEAT_LENGTH_TWEET_COL = 'length_of_tweet'

WORD_ENCODING_COL = 'word_encoding'
PREFIX_SPACY_ENCODING_COL = 'spacy_enc'

ML_FEATURES_COL = [f'{PREFIX_SPACY_ENCODING_COL}_{x}' for x in range(0, 300)] + [SENTIMENT_COL]
ML_PRED_COL = 'ml_pred'
ML_TARGET_COL = 'ml_target'

TOKENS_PRED_COL = 'tokens_sentence_pred'
SENTENCE_PRED_COL = 'sentence_pred'
SENTENCE_TARGET_COL = 'sentence_target'

# Variables for Tensorflow run
TARGET_SEQUENCE_COL = 'target_sequence'
CLEANED_TOKENS_COL = 'cleaned_tokens'
INDEXED_TOKENS_COL = 'indexed_tokens'

REGEX_PATTERN = {
    '<link>': r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))',
    '<hash>': r'#\w*',
    '<mention>': r'@\w*',
    '<retweet>': r'^(RT|FAV)',
    '<duplicate_chars>': r'([A-Za-z])\1{2,}'
}
