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


MAX_WORDS = 10000
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

submission.to_csv('D16ADA13_RajatDisawal')




