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


from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense,LSTM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.test.gpu_device_name()


train_data = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep='\t')


train_data.head(10)


train_data.shape


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)



train_data['cleaned'] = train_data['Phrase'].apply(clean_text)
texts = train_data['cleaned'].astype(str).tolist()


train_set ,val_set = train_test_split(train_data, test_size = 0.2, random_state = 42,shuffle = True)


train_texts = train_set['cleaned'].astype(str).tolist()
val_texts = val_set['cleaned'].astype(str).tolist()


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=100)


val_sequences = tokenizer.texts_to_sequences(val_texts)
val_padded = pad_sequences(val_sequences, maxlen=100)


x_train=train_padded
y_train=train_set['Sentiment']
x_test=val_padded
y_test=val_set['Sentiment']


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


import joblib
joblib.dump(tokenizer,"/kaggle/working/tokenizer.pkl")
joblib.dump(le,"/kaggle/working/LabelEncoder.pkl")


y_train.shape


embedding_dim = 256
rnn_units = 512
vocab_size = 5000
def build_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(5,activation='softmax')
    ])
 
model = build_model(vocab_size, embedding_dim, rnn_units)
 
 
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=False)



model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
 
 
EPOCHS = 10
with tf.device('/device:GPU:0'):
    history = model.fit(x_train,y_train_encoded, epochs=EPOCHS)


test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.2%}")


predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(y_test, predicted_classes))


model.save('/kaggle/working/sentiment_RNN_model.h5')


def build_model_with_LSTM(vocab_size, embedding_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(216, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(5,activation='softmax')
    ])

model_LMST = build_model_with_LSTM(vocab_size, embedding_dim)
model_LMST.compile(optimizer='adam', loss=loss, metrics=['accuracy'])


with tf.device('/device:GPU:0'):
    history_LSTM= model_LMST.fit(x_train,y_train_encoded, epochs=EPOCHS)


model_LMST.save('/kaggle/working/sentiment_LMST_model.h5')

