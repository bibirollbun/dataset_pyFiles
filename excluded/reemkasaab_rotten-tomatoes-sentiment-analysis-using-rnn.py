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


!pip install contractions


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re
import contractions
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


df = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep='\t')


df.head()


df = df[['Phrase', 'Sentiment']]


label_mapping = {
    0: 0,  # negative
    1: 0,  # negative
    2: 1,  # neutral
    3: 2,  # positive
    4: 2   # positive
}

df['Sentiment'] = df['Sentiment'].map(label_mapping)


df.info()


X = df['Phrase'].values
y = df['Sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


test_df = pd.DataFrame({'Phrase': X_test, 'Sentiment': y_test})
test_df.to_csv('test_data.csv', index=False)


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    return text


X_train = pd.Series(X_train).apply(clean_text)


vocab_size = 10000  
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)


train_sequences = tokenizer.texts_to_sequences(X_train)

max_length = max(len(seq) for seq in train_sequences)  # ناخد أطول جملة
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')


embedding_dim = 128

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(train_padded, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)



model.save("sentiment_rnn_model.h5")

