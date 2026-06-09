import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns
import spacy as sp
import string
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


data = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
data_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')


data


data_test


data.drop(['textID'], axis = 1, inplace=True)
data_test.drop(['textID'], axis = 1, inplace=True)


data.head()


data_test.head()


data.isnull().sum(axis=0)


data_test.isnull().sum(axis=0)


data.dropna(axis=0, inplace=True)


data['sentiment'] = data['sentiment'].map({'positive': 1,
                             'negative': -1,
                             'neutral': 0},
                             na_action=None)


data_test['sentiment'] = data['sentiment'].map({'positive': 1,
                             'negative': -1,
                             'neutral': 0},
                             na_action=None)


positive = data[data['sentiment'] == 1]
negative = data[data['sentiment'] == -1]
neutral = data[data['sentiment'] == 0]
positive_test = data_test[data_test['sentiment'] == 1]
negative_test = data_test[data_test['sentiment'] == -1]
neutral_test = data_test[data_test['sentiment'] == 0]


plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(positive['text']))
plt.title('Description Positive', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()


plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(negative['text']))
plt.title('Description Negative', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()


plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('fast')

wc = WordCloud(background_color = 'orange', width = 1500, height = 1500).generate(str(neutral['text']))
plt.title('Description Neutral', fontsize = 15)

plt.imshow(wc)
plt.axis('off')
plt.show()


fig2 = px.histogram(data,x='sentiment',color='sentiment',template='plotly_dark')
fig2.show()


plt.figure(figsize=(20,6))
top_30 = data.groupby('selected_text')['selected_text'].count() \
.sort_values(ascending = False).head(30)
sns.barplot(x=top_30.index, y = top_30.values)
plt.title('Top 30 Words')
plt.show()


temp = data.describe()
temp.style.background_gradient(cmap='Purples')


data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x >= 0 else 0)


X = data.iloc[:, 0].values
X


X.shape


y = data.iloc[:, 2].values
y


X, _, y, _ = train_test_split(X, y, stratify = y)


print(X.shape, y.shape )


unique, counts = np.unique(y, return_counts=True)
unique, counts


def clean_t(t):
  t = BeautifulSoup(t, 'lxml').get_text()
  t = re.sub(r"@[A-Za-z0-9]+", ' ', t)
  t = re.sub(r"https?://[A-Za-z0-9./]+", ' ', t)
  t = re.sub(r"[^a-zA-Z.!?]", ' ', t)
  t = re.sub(r" +", ' ', t)
  return t


text = "I don't like"
text = clean_t(text)
text


import spacy
nlp = spacy.blank("en")
nlp


stop_words = sp.lang.en.STOP_WORDS
print(stop_words)


len(stop_words)


string.punctuation


def clean_t2(tt):
  tt = tt.lower()
  document = nlp(tt)

  words = []
  for token in document:
    words.append(token.text)

  words = [word for word in words if word not in stop_words and word not in string.punctuation]
  words = ' '.join([str(element) for element in words])

  return words


text2 = clean_t2(text)
text2


data_clean = [clean_t2(clean_t(t)) for t in X]


for _ in range(10):
  print(data_clean[random.randint(0, len(data_clean) - 1)])


data_labels = y
data_labels


np.unique(data_labels)


tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2**16)


tokenizer.vocab_size


print(tokenizer.subwords)


ids = tokenizer.encode('I like')
ids


data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]


for _ in range(10):
  print(data_inputs[random.randint(0, len(data_inputs) - 1)])


max_len = max([len(sentence) for sentence in data_inputs])
max_len


data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                            value = 0,
                                                            padding = 'post',
                                                            maxlen=max_len)


for _ in range(10):
  print(data_inputs[random.randint(0, len(data_inputs) - 1)])


train_inputs, test_inputs, train_labels, test_labels = train_test_split(data_inputs,
                                                                        data_labels,
                                                                        test_size=0.3,
                                                                        stratify = data_labels)


print(train_inputs.shape, train_labels.shape)


print(test_inputs.shape, test_labels.shape )


class DCNN(tf.keras.Model):

  def __init__(self,
               vocab_size,
               emb_dim=128,
               nb_filters=50,
               ffn_units=512,
               nb_classes=2,
               dropout_rate=0.1,
               training=True,
               name="dcnn"):
    super(DCNN, self).__init__(name=name)
    self.embedding = layers.Embedding(vocab_size, emb_dim)
    self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding='same', activation='relu')
    self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding='same', activation='relu')
    self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding='same', activation='relu')
    self.pool = layers.GlobalMaxPool1D()
    
#estrutura da rede neural
    self.dense_1 = layers.Dense(units = ffn_units, activation = 'relu')
    self.dropout = layers.Dropout(rate = dropout_rate)
    if nb_classes == 2:
      self.last_dense = layers.Dense(units = 1, activation = 'sigmoid')
    else:
      self.last_dense = layers.Dense(units = nb_classes, activation = 'softmax')

  def call(self, inputs, training):
    x = self.embedding(inputs)
    x_1 = self.bigram(x)
    x_1 = self.pool(x_1)
    x_2 = self.trigram(x)
    x_2 = self.pool(x_2)
    x_3 = self.fourgram(x)
    x_3 = self.pool(x_3)

    merged = tf.concat([x_1, x_2, x_3], axis = -1)
    merged = self.dense_1(merged)
    merged = self.dropout(merged, training)
    output = self.last_dense(merged)

    return output


vocab_size = tokenizer.vocab_size
vocab_size


emb_dim = 200
nb_filters = 100
ffn_units = 256
batch_size = 64
nb_classes = len(set(train_labels))
nb_classes


dropout_rate = 0.2
nb_epochs = 5  


Dcnn = DCNN(vocab_size=vocab_size, emb_dim=emb_dim, nb_filters=nb_filters,
            ffn_units=ffn_units, nb_classes=nb_classes, dropout_rate=dropout_rate)


if nb_classes == 2:
  Dcnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
  Dcnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = Dcnn.fit(train_inputs, train_labels,
                   batch_size = batch_size,
                   epochs = nb_epochs,
                   verbose = 1,
                   validation_split = 0.10)


results = Dcnn.evaluate(test_inputs, test_labels, batch_size=batch_size)
print(results)


y_pred_test = Dcnn.predict(test_inputs)


y_pred_test


y_pred_test = (y_pred_test > 0.5)
y_pred_test


test_labels


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, y_pred_test)
cm


sns.heatmap(cm, annot=True)


history.history.keys()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss progress during training and validation')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.legend(['Training loss', 'Validation loss'])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy progress during training and validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy', 'Validation accuracy'])


text = "I hate"
text = tokenizer.encode(text)
Dcnn(np.array([text]), training=False).numpy()


text = "I happy"
text = tokenizer.encode(text)
Dcnn(np.array([text]), training=False).numpy()


text = "It is complicated"
text = tokenizer.encode(text)
Dcnn(np.array([text]), training=False).numpy()


''''text = str(input('write here:   '))
text = tokenizer.encode(text)
text =  Dcnn(np.array([text]), training=False).numpy()
if text >= 0.7:
  print('positivo');
elif text >= 0.4 and text <= 0.69:
    print('neutral')
else:
  print('negativo')'''

