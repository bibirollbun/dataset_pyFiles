import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout

from keras.layers import BatchNormalization

# from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection,metrics, pipeline

# from keras.layers import GlobalMaxPooling1d, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

# from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')




try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ',tpu.master())

except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initalize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:
    strategy = tf.distribute.get_strategy()


print('REPLICAS: ',strategy.num_replicas_in_sync)


train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


train.columns


train.drop(['severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate'],axis = 1, inplace = True)


train.columns


train.shape


train = train.loc[:12000,:]
train.shape


train['comment_text'].apply(lambda x: len(str(x).split())).max()


def roc_auc(prediction, traget):
    fpr,tpr,thresholds = metrics.roc_curve(target,predictions)
    roc_auc=metrics.auc(fpr,tpr)
    return roc_auc


xtrain,xvalid,ytrain,yvalid=train_test_split(train.comment_text.values,train.toxic.values, random_state=42,test_size=0.2,shuffle = True)


yvalid.shape


token = tf.keras.preprocessing.text.Tokenizer(num_words = None)
max_len = 1500

token.fit_on_texts(list(xtrain) +list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

xtrain_pad = tf.keras.preprocessing.sequence.pad_sequences(xtrain_seq,maxlen=max_len)

xvalid_pad = tf.keras.preprocessing.sequence.pad_sequences(xvalid_seq,maxlen=max_len)

word_index = token.word_index





type(token.fit_on_texts)


type(xtrain_seq)


type(xtrain_pad)


len(xtrain_seq[1])


xtrain_pad[1]


len(word_index)



model = Sequential([tf.keras.layers.Embedding(len(word_index)+1,300,input_shape = (1500,)),
                     tf.keras.layers.SimpleRNN(100),
                     tf.keras.layers.Dense(1,activation='sigmoid')])

model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])


model.summary()


model.fit(xtrain_pad,ytrain,epochs = 5)


score = model.predict(xvalid_pad)


score[1]


yvalid[1]


xvalid[1]


fpr, tpr, thresholds = metrics.roc_curve( yvalid,score)


roc_auc = metrics.roc_auc_score(yvalid,score)


roc_auc


metrics.auc(fpr,tpr)


plt.figure(figsize = (15,10))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(fpr,tpr,color = 'darkorange',label = f'ROC curve (area = {roc_auc:.2f}')
plt.title('The ROC Cruve')
plt.legend(loc='lower right')
plt.show()


score_dict = {}


score_dict['Simple_RNN'] = roc_auc


score_dict


embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding = 'utf-8')

for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors. '% len(embeddings_index))


embedding_matrix = np.zeros((len(word_index)+1,300))

for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


k  = np.zeros((len(word_index)+1,300))


k.shape


xtrain[1]


word_index['you']


weights = [embedding_matrix]


len(weights)


k =10
for key,i in word_index.items():
    print(i,'-',key)
    if k < 21:
        k += 1
        continue
    else:
        break


embedding_matrix[1]


vec = embeddings_index.get('you')


type([embedding_matrix])


vec.shape


model1 = tf.keras.Sequential([tf.keras.layers.Embedding(len(word_index)+1,300,weights = [embedding_matrix],input_shape=(1500,),trainable=False),
                             tf.keras.layers.LSTM(100,dropout=0.3),
                             tf.keras.layers.Dense(1,activation = 'sigmoid')])

model1.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])


model1.summary()


model1.fit(xtrain_pad,ytrain,epochs = 5)


score1 = model1.predict(xvalid_pad)


fpr,tpr,thresholds = metrics.roc_curve(yvalid,score1)


roc_score1 = metrics.roc_auc_score(yvalid,score1)


roc_score1


plt.plot(fpr,tpr)
plt.show()


score_dict['LSTM']=roc_score1


score_dict


model3 = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index)+1,300,weights = [embedding_matrix],input_shape=(1500,),trainable = False),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.GRU(300),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])
model3.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])


model3.summary()


model3.fit(xtrain_pad,ytrain,epochs = 5)


score3 = model3.predict(xvalid_pad)


roc_score3 = metrics.roc_auc_score(yvalid,score3)


roc_score3


score_dict['GRU(RNN)'] = roc_score3


score_dict




