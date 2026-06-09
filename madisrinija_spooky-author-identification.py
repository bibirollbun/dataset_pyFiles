import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, Embedding
from tensorflow.keras.callbacks import EarlyStopping


train = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")
test = pd.read_csv("/kaggle/input/spooky-author-identification/test.zip")


display(train.head())
display(test.head())


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


le = LabelEncoder()
train['author'] = le.fit_transform(train['author'].values)


def word_count(text):
    return len(text.split())

train['word count'] = train['text'].apply(word_count)
print("Avergae Number of words in each text :",round(sum(train['word count'])/len(train['word count'])))


X , Y = train['text'], train['author']

X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size = 0.3)


tfidf = TfidfVectorizer(
    min_df=3,
    max_features=None, 
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3), 
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
    stop_words = 'english'
)

tfidf.fit(list(X_train)+list(X_valid))

X_train_modi = tfidf.transform(X_train)
X_valid_modi = tfidf.transform(X_valid)
test_modi = tfidf.transform(test)


scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_modi)
X_train_scaled = scaler.transform(X_train_modi)
X_valid_scaled = scaler.transform(X_valid_modi)


svc = SVC(probability=True)
svc.fit(X_train_modi,Y_train)


predictions = svc.predict_proba(X_valid_modi)

print ("logloss: %0.3f " % multiclass_logloss(Y_valid, predictions))


tokenizer =Tokenizer(
            num_words=10000
)
tokenizer.fit_on_texts(list(X_train)+list(X_valid))
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)

X_train_pad = pad_sequences(X_train_seq)
X_valid_pad = pad_sequences(X_valid_seq)

word_index = tokenizer.word_index


glove_dir = '/kaggle/input/glove-global-vectors-for-word-representation'

embeddings_index = {}
fname = os.path.join(glove_dir, 'glove.6B.100d.txt')  # check the file name
with open(fname, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 100
num_words = min(10000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



model = Sequential()

model.add(Embedding(
                    input_dim=num_words,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=10,
                    trainable=False
))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(
    X_train_pad,Y_train,
    validation_data = (X_valid_pad,Y_valid),
    batch_size=512,
    epochs = 30,
    verbose=1,
    callbacks=[earlystop]
)


pred = model.predict(X_valid_pad)
print ("logloss: %0.3f " % multiclass_logloss(Y_valid, pred))


classes = ['EAP','HPL','MWS']

submission = pd.DataFrame(pred,columns=classes)

submission.insert(0,'id',test['id'])

submission.to_csv('submission.csv', index=False)

