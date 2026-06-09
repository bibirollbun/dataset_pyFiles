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


!pip install protobuf==3.20.*


import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




train_df = pd.read_csv("/kaggle/input/news-classification-challenge/Train.csv")
test_df = pd.read_csv("/kaggle/input/news-classification-challenge/Test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())

# Assume text column name and label
TEXT_COL = "Headline"
LABEL_COL = "Category"



MAX_VOCAB = 20000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["Headline"])

X_train = tokenizer.texts_to_sequences(train_df["Headline"])
X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post")

X_test = tokenizer.texts_to_sequences(test_df["Headline"])
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding="post")



label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["Category"])



X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)



embedding_dim = 128

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=embedding_dim, input_length=MAX_LEN),
    Dropout(0.3),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

import numpy as np
dummy_input = np.zeros((1, MAX_LEN))
model.predict(dummy_input)

# Now show the summary
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64,
    verbose=1
)



# Predict
preds = model.predict(X_test, batch_size=64)
pred_labels = np.argmax(preds, axis=1)
pred_categories = label_encoder.inverse_transform(pred_labels)



submission = pd.DataFrame({
    "id": test_df["Id"],
    "category": pred_categories
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")
submission.head()





