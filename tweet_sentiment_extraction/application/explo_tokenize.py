from os.path import join

import pandas as pd
import spacy
from gensim.corpora import Dictionary

from tweet_sentiment_extraction.infrastructure.sentence_cleaner import SentencePreprocessor
from tweet_sentiment_extraction import settings as stg
from tweet_sentiment_extraction.domain.bi_lstm import BidirectionalLSTM,
from tensorflow.keras.callbacks import ModelCheckpoint

train = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL])
validation = pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv'))

train = train.query('sentiment != "neutral"')\
             .assign(**{
                 'start_position': lambda df: df.apply(lambda x: x['text'].find(x['selected_text']), axis=1),
                 'len_selected_text': lambda df: df['selected_text'].apply(len),
                 'end_position': lambda df: df['start_position'] + df['len_selected_text'],
             })

validation = validation.query('sentiment != "neutral"')\
                       .assign(start_position=lambda df: df.apply(lambda x: x['text'].find(x['selected_text']), axis=1),
                               len_selected_text=lambda df: df['selected_text'].apply(len),
                               end_position=lambda df: df['start_position'] + df['len_selected_text'])

nlp = spacy.load('en_core_web_md')

train_docs = []
train_selected_docs = []
train_labels = []
#for doc, selected_doc, start_position, end_position in zip(nlp.pipe(validation[stg.TEXT_COL]), nlp.pipe(validation[stg.SELECTED_TEXT_COL]), validation['start_position'], validation['end_position']):
for doc, selected_doc, start_position, end_position in zip(nlp.pipe(train[stg.TEXT_COL]), nlp.pipe(train[stg.SELECTED_TEXT_COL]), train['start_position'], train['end_position']):
    train_docs.append(doc)
    train_selected_docs.append(selected_doc)
    train_label = [1 if start_position <= token.idx < end_position else 0 for token in doc]
    train_labels.append(train_label)

validation_docs = []
validation_selected_docs = []
validation_labels = []
for doc, selected_doc, start_position, end_position in zip(nlp.pipe(validation[stg.TEXT_COL]), nlp.pipe(validation[stg.SELECTED_TEXT_COL]), validation['start_position'], validation['end_position']):
    validation_docs.append(doc)
    validation_selected_docs.append(selected_doc)
    validation_label = [1 if start_position <= token.idx < end_position else 0 for token in doc]
    validation_labels.append(validation_label)

idx_ = 51

print(labels[idx_])
print([t for t in docs[idx_]])
print([t for t in selected_docs[idx_]])

text = "Hi, I'm late. Soooory "
list(nlp(text))
labels_exemple = [0, 0, 1, 1, 0, 0, 0]
labels_exemple = [0, 0, 1, 0, 0, 0, 0]
labels_exemple = [0, 0, 0, 1, 0, 0, 0]

for token in nlp(text):
    if token.whitespace_ == '':
        print(token.text+token.nbor(1).text)


tok1.nbor


nlp.vocab.get_vector("I")

# generate the embedding matrix

dictionary = Dictionary([["<OOV>", "<PAD>"]])

x_train = [[token.lower_ for token in doc] for doc in docs]
train_dictionary = Dictionary(x_train)

train_selected_dictionary = Dictionary([[token.lower_ for token in doc] for doc in selected_docs])

len(train_dictionary)
len(train_selected_dictionary)
train_dictionary.filter_extremes(keep_n=3000)
dictionary.merge_with(train_selected_dictionary)
dictionary.merge_with(train_dictionary)

x_train_indexed = [[dictionary.token2id.get(token.lower_, 0) for token in doc] for doc in train_docs]
x_validation_indexed = [[dictionary.token2id.get(token.lower_, 0) for token in doc] for doc in validation_docs]


def adapt_spacy_to_dictionary(pre_trained_embedding, dictionary_token2id=dictionary.token2id):
    embedding_matrix = np.zeros((len(dictionary_token2id), 300))
    out_of_vocabulary = []
    for word, i in dictionary_token2id.items():
        if i >= 2:  # 0 is the index of OOV, 1 is the index of Padding
            values = pre_trained_embedding.get_vector(word)
            if values is not None:
                embedding_matrix[i] = values
            else:
                out_of_vocabulary.append(word)
    print(f'out_of_vocabulary: {len(out_of_vocabulary)}')

    return embedding_matrix  # , out_of_vocabulary


embedding_matrix = adapt_spacy_to_dictionary(nlp.vocab, dictionary.token2id)

BLSTM = BidirectionalLSTM(hidden_dim=128, word_embedding_initialization=embedding_matrix)

callbacks = ModelCheckpoint(join(stg.ML_DATA_DIR, "best_model.hdf5"), monitor='loss', verbose=1, save_best_only=True)
model_fit = BLSTM.fit(X=x_train_indexed,
                      y=labels,
                      batch_size=32, epochs=5, validation_split=0.2, callbacks=[callbacks], verbose=1)

train_pred = BLSTM.predict(X_test=x_train_indexed)

train_bool_labels = [[1 if score[0] > 0.5 else 0 for score in scores] for scores in train_pred]

selected_text_predicted = []
for doc, labels in zip(train_docs, train_bool_labels):
    predicted_text = ''.join([token.text_with_ws for token, label in zip(doc, labels) if label == 1])
    selected_text_predicted.append(predicted_text)



train_data_with_pred = train_data_preprocessed.assign(
    pred=train_pred,
    pred_in_bool=lambda df: df['pred'].apply(lambda x: [True if score > 0.5 else False for score in x]),
    tokens_pred=lambda df: df.apply(lambda row: np.array(row[stg.TOKENS_TEXT_COL])[row['pred_in_bool']], axis=1),
    sentence_pred=lambda df: df['tokens_pred'].apply(lambda x: ' '.join(x))
)

train_all_sentiments = pd.merge(left=pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'train.csv')).dropna(subset=[stg.TEXT_COL]),
                                right=train_data_with_pred[[stg.ID_COL, 'sentence_pred']],
                                on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

train_score = jaccard_score(y_true=train_all_sentiments[stg.SELECTED_TEXT_COL],
                            y_pred=train_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'train score: {train_score}')
print('--------------------------')

sp_validation = SentencePreprocessor(df=validation)
validation_data_preprocessed = sp.preprocess_dataset(vocabulary=dictionary.token2id)
validation_pred = BLSTM.predict(X_test=validation_data_preprocessed[stg.INDEXED_TOKENS_COL])

validation_data_with_pred = validation_data_preprocessed.assign(
    pred=validation_pred,
    pred_in_bool=lambda df: df['pred'].apply(lambda x: [True if score > 0.5 else False for score in x]),
    tokens_pred=lambda df: df.apply(lambda row: np.array(row[stg.TOKENS_TEXT_COL])[row['pred_in_bool']], axis=1),
    sentence_pred=lambda df: df['tokens_pred'].apply(lambda x: ' '.join(x))
)

validation_all_sentiments = pd.merge(left=pd.read_csv(join(stg.PROCESSED_DATA_DIR, 'validation.csv')).dropna(subset=[stg.TEXT_COL]),
                                     right=validation_data_with_pred[[stg.ID_COL, 'sentence_pred']],
                                     on=stg.ID_COL, how='left')\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'].isna(), df['text'], df['sentence_pred']))\
    .assign(sentence_pred=lambda df: np.where(df['sentence_pred'] == '', df['text'], df['sentence_pred']))

validation_score = jaccard_score(y_true=validation_all_sentiments[stg.SELECTED_TEXT_COL],
                                 y_pred=validation_all_sentiments['sentence_pred'])

print('--------------------------')
print(f'validation score: {validation_score}')
print('--------------------------')
