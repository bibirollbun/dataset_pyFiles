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



import joblib
tokenizer = joblib.load("/kaggle/input/vectorizer-and-encoder/LabelEncoder.pkl")
encoder = joblib.load("/kaggle/input/vectorizer-and-encoder/tokenizer.pkl")


from tensorflow.keras.models import load_model

RNN_model = load_model('/kaggle/input/sentiment-rnn-model/keras/default/1/sentiment_RNN_model.h5')
RNN_model.summary()


from tensorflow.keras.models import load_model
model_LMST = load_model('/kaggle/input/sentiment-rnn-model/keras/default/1/sentiment_RNN_model.h5')

model.summary()


test_data = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip", sep='\t')


test_data.head(10)


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def classify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens_list = " ".join(tokens)
    texts = [tokens_list]
    sequences = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(x)
    predicted_classe = np.argmax(predictions, axis=1)
    label_predicted = encoder.inverse_transform(predicted_classe)
    return label_predicted


for t in test_data['Phrase']:
    result=classify(t)
    print(result)

