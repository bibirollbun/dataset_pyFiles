import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load and label data
true = pd.read_csv("/kaggle/input/5jjud6k/dataset/True.csv")
fake = pd.read_csv("/kaggle/input/5jjud6k/dataset/Fake.csv")
true['label'] = 1
fake['label'] = 0
df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)

# 2. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# 3. Tokenize
MAX_WORDS = 5000
MAX_LEN = 300
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df['label'].values

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. RNN Model (without LSTM)
model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.build(input_shape=(None, MAX_LEN))
model.summary()

# 6. Train
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 7. Evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))

# 8. Save model
model.save("rnn_fake_news_model.h5")
print("✅ Model saved as 'rnn_fake_news_model.h5'")



# lstm_fake_news_final.py

import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load and label data
true = pd.read_csv("/kaggle/input/5jjud6k/dataset/True.csv")
fake = pd.read_csv("/kaggle/input/5jjud6k/dataset/Fake.csv")
true['label'] = 1
fake['label'] = 0
df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)

# 2. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# 3. Tokenize and pad
MAX_WORDS = 5000
MAX_LEN = 300
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df['label'].values

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build LSTM Model
model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the model before summary
model.build(input_shape=(None, MAX_LEN))
model.summary()

# 6. Train
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 7. Evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))

# 8. Save the model
model.save("lstm_fake_news_model.h5")
print("✅ Model saved as 'lstm_fake_news_model.h5'")



# bilstm_fake_news.py

import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load and label data
true = pd.read_csv("/kaggle/input/5jjud6k/dataset/True.csv")
fake = pd.read_csv("/kaggle/input/5jjud6k/dataset/Fake.csv")
true['label'] = 1
fake['label'] = 0
df = pd.concat([true, fake]).sample(frac=1).reset_index(drop=True)

# 2. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['text'] = df['title'] + " " + df['text']
df['text'] = df['text'].apply(clean_text)

# 3. Tokenize
MAX_WORDS = 5000
MAX_LEN = 300
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df['label'].values

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. BiLSTM Model
model = Sequential()
model.add(Embedding(MAX_WORDS, 128, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))

# ✅ Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.build(input_shape=(None, MAX_LEN))
model.summary()

# 6. Train
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 7. Evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))

# 8. Save the model
model.save("bilstm_fake_news_model.h5")
print("✅ Model saved as 'bilstm_fake_news_model.h5'")











