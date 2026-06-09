# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords')



import zipfile
import pandas as pd

with zipfile.ZipFile("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

df = pd.read_csv("/kaggle/working/train.tsv", sep="\t")


X = df['Phrase']
y = df['Sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)



vocab_size = 10000
max_length = 100
embedding_dim = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')



encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_train_cat = to_categorical(y_train_encoded)
y_test_cat = to_categorical(y_test_encoded)


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


model.fit(X_train_pad, y_train_cat, epochs=5, batch_size=128, validation_split=0.1)



y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

print(classification_report(y_true, y_pred, target_names=encoder.classes_.astype(str)))


df.dropna(inplace=True)



train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Phrase'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
)


import tensorflow as tf



def clean_text(text):
    text = text.lower()
    text = tf.strings.regex_replace(text, '[^a-zA-Z ]', '')
    return text.numpy().decode('utf-8')

train_texts = [clean_text(t) for t in train_texts]
test_texts = [clean_text(t) for t in test_texts]


tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

max_len = 50
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

y_train = np.array(train_labels)
y_test = np.array(test_labels)


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout



y_train_binary = (y_train == 2).astype(int)
y_test_binary = (y_test == 2).astype(int)

model_stage1 = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_stage1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_stage1.fit(X_train, y_train_binary, epochs=3, validation_split=0.1, batch_size=128)



model_stage1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_stage1.fit(X_train, y_train_binary, epochs=3, validation_split=0.1, batch_size=128)


stage1_preds = (model_stage1.predict(X_test) > 0.5).astype(int).flatten()



train_mask_stage2 = y_train != 2
test_mask_stage2 = y_test != 2

X_train_stage2 = X_train[train_mask_stage2]
y_train_stage2 = y_train[train_mask_stage2]

X_test_stage2 = X_test[test_mask_stage2]
y_test_stage2 = y_test[test_mask_stage2]

# إعادة ترميز الفئات المتبقية إلى 0,1,2,3
label_map = {0: 0, 1: 1, 3: 2, 4: 3}
inv_map = {v: k for k, v in label_map.items()}

y_train_stage2 = [label_map[v] for v in y_train_stage2]
y_test_stage2 = [label_map[v] for v in y_test_stage2]


model_stage2 = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model_stage2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_stage2.fit(X_train_stage2, np.array(y_train_stage2), epochs=3, validation_split=0.1, batch_size=128)



stage2_preds_probs = model_stage2.predict(X_test_stage2)
stage2_preds = np.argmax(stage2_preds_probs, axis=1)
stage2_preds = [inv_map[p] for p in stage2_preds]



final_preds = []
j = 0
for i in stage1_preds:
    if i == 1:
        final_preds.append(2)
    else:
        final_preds.append(stage2_preds[j])
        j += 1


from sklearn.metrics import classification_report, accuracy_score



print("Accuracy:", accuracy_score(y_test, final_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, final_preds))

