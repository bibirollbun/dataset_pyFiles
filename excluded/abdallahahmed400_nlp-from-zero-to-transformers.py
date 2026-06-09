import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, SimpleRNN,LSTM
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


train_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# Load First 5 rows

train_df.head()


# Sum the null values for each columns

train_df.isna().sum()


# Show the shapeof the data

train_df.shape


train_df = train_df.iloc[:15000,:]
train_df.shape


# drop classes except one and drop id because non useful columns

train_df.drop(columns=["id","identity_hate", "insult", "threat", "obscene","severe_toxic"])


# Get the length of longest sentence

max_len = train_df['comment_text'].apply(lambda x: len(str(x).split())).max()
max_len


# Make function for getting auc score

def roc_auc(y_pred, y_true):

    fpr,tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr,tpr)
    return roc_auc


# Lets discove hoe many classes at toxic columns

train_df['toxic'].value_counts(normalize=True).plot(
    kind="pie",
    autopct='%1.1f%%',
    title="Toxic Comment Distribution",
)


sns.countplot(data=train_df, x="toxic")
plt.title("Distribution of Toxic Comments")
plt.xlabel("Toxic Label")
plt.ylabel("Count")
plt.show()


xtrain, xvalid, ytrain, yvalid = train_test_split(train_df.comment_text.values, train_df.toxic.values, 
                                                  stratify=train_df.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)


tokens = text.Tokenizer(num_words=None)
max_length=1500

tokens.fit_on_texts(list(xtrain) + list(xvalid))
train_seq = tokens.texts_to_sequences(xtrain)
valid_seq = tokens.texts_to_sequences(xvalid)

# zero pad the sentenecs

train_pad = sequence.pad_sequences(train_seq, maxlen = max_len)
valid_pad = sequence.pad_sequences(valid_seq, maxlen = max_len)

word_index = tokens.word_index


train_pad


from imblearn.over_sampling import SMOTE

smote = SOMTE(random_state=42)
x_sampled, y_sampled = smote.fit_transform(train_pad, train_df['toxic'])


# Strategy is used disribute to the model corss the GPU
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    model = Sequential()
    model.add(Embedding(
        input_dim=len(word_index) + 1,
        output_dim=300,
        input_length=max_len
    ))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation="sigmoid"))
    model.build(input_shape=(None, max_len)) 
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()  # âœ… INSIDE the scope



model.fit(train_pad,ytrain, epochs=10, batch_size=64*strategy.num_replicas_in_sync)


batch_size = 64 * strategy.num_replicas_in_sync
print(f"Using batch size: {batch_size}")



embeddings_index = {}

with open("/kaggle/input/glove6b100dtxt/glove.6B.100d.txt", "r", encoding="utf-8") as f:
    for line in f:
        value = line.strip().split()
        word = value[0]
        bs = np.asarray([float(val) for val in value[1:]], dtype='float32')
        embeddings_index[word] = bs

print(f"Found {len(embeddings_index)} word vectors.")



# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Strategy is used disribute to the model corss the GPU
strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    model = Sequential()
    model.add(Embedding(
        len(word_index)+1,
        100,
        weights=[embedding_matrix],
        input_length = max_len,
        trainable=False
    ))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.2))
    model.add(Dense(1,activation = "sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(train_pad, ytrain, epochs=5, batch_size=64*strategy.num_replicas_in_sync)


score = model.predict(valid_pad)


score = model.predict(valid_pad)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



roc_auc = roc_auc(score, yvalid)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()



with strategy.scope():
    model = Sequential()
    model.add(Embedding(len(word_index)+1)
             , 100,
             weights=[embeddings_matrix],
             input_length=max_len,
             trainable=False)
    model.add(SaptialDrpout(0.3))
    model.add(GRU(300))
    model.Dense(1, activation="sigmoid")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


model.fit(train_pad,ytrain, epochs=10, batch_size=64*strategy.num_replicas_in_sync)


score_gru = model.predict(valid_pad)


print(f"accuracy: {roc_auc(score, yvalid)}")


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



roc_auc = roc_auc(score_gru, yvalid)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()



with strategy.scope():

    model = Sequential()
    model.add(Embedding(len(word_index) +1),
             100,
             input_length=max_len,
             weights=[embeddings_matrix],
             trainable=False)
    model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy", opimizer="adam", metrics=["accuracy"])

model.summary()


model.fit(train_pad,ytrain, epochs=10, batch_size=64*strategy.num_replicas_in_sync)


y_pred = model.predict(valid_pad)


print(f"accuracy: {roc_auc(y_pred, yvalid)}")


# loading libararies

import os
import tensorflow as tf
import tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.model import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tokenizers import BertWordPieceTokenizer


# LOADING THE DATA

train_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')



def encoding(text, tokenizer, chunk_size=256, max_len=512):

    """
        it's making encoding to convert the row text to sequence numerical token ids for bert model understanding the text.
    """

    tokenizer.enable_padding(max_length=max_len)
    tokenizer.enable_truncation(max_length=max_len)
    ids = []
    
    for i in range(0, len(text), chunk_size):
        text_chunk = text[i: i+chunk_size].tolist()
        encode = tokenizer.encode_batch(text_chunk)
        ids.extend([encode.ids for e in encode])

    return ids


AUTO = tf.data.experimental.AUTOTUNE

epochs = 3
batch_size = 16 * strategy.num_replicas_in_sync
max_len = 192




