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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_df = pd.read_csv('/kaggle/input/kcvanguard-deep-learning-assignment/train-reviews-gmaps.csv')
test_df = pd.read_csv('/kaggle/input/kcvanguard-deep-learning-assignment/test-review-gmaps-new.csv')

print(train_df.head())

print(train_df['label'].value_counts())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

train_df['clean'] = train_df['reviews'].apply(clean_text)
test_df['clean'] = test_df['reviews'].apply(clean_text)

train_df['label_nums'] = train_df['label'].map({
    'Positive': 1,
    'Negative': 0
})
train_df['label_nums'] = train_df['label_nums'].astype(int)


print("Train_df: ", train_df.shape)
print("Test_df: ", test_df.shape)


MAX_WORDS = 1000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['clean'])

X = tokenizer.texts_to_sequences(train_df['clean'])
X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')
y=train_df['label_nums'].values

X_test = tokenizer.texts_to_sequences(test_df['clean'])
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1
)


val_pred = (model.predict(X_val) > 0.5).astype(int)
acc = accuracy_score(y_val, val_pred)
print('Validation Accuracy: ', acc)


test_pred = (model.predict(X_test) > 0.5).astype(int)


temp_df = pd.DataFrame({
    'id': test_df['id'],
    'label_nums': test_pred.flatten()
})
temp_df['label'] = temp_df['label_nums'].map({
    0: 'Negative',
    1: 'Positive'
})


submission = pd.DataFrame({
    'id': temp_df['id'],
    'label': temp_df['label']
})

submission.to_csv('D16ADA17_Kajol')


sample_reviews = [
    "The food was amazing and the staff were very friendly!",
    "I hated the service, it was so slow and rude.",
    "The place was dirty and the food was cold.",
    "Average restaurant, nothing special."
    
]


sample_seq = tokenizer.texts_to_sequences(sample_reviews)


sample_pad = pad_sequences(sample_seq, maxlen=MAX_LEN, padding='post', truncating='post')


sample_pred_prob = model.predict(sample_pad)
sample_pred = (sample_pred_prob > 0.5).astype(int).flatten()


sample_labels = le.inverse_transform(sample_pred)


for review, label, prob in zip(sample_reviews, sample_labels, sample_pred_prob.flatten()):
    print(f"Review: {review}\nPredicted Sentiment: {label} (Probability: {prob:.2f})\n")




