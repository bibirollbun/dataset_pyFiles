# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential  # ✅ Corrected
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN  # ✅ Corrected
from tensorflow.keras.layers import Dense, Activation, Dropout  # ✅ Corrected
from tensorflow.keras.layers import Embedding  # ✅ Corrected
from tensorflow.keras.layers import BatchNormalization  # ✅ Correct
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  # ✅ Corrected
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer  # ✅ Correct
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import time 


import pandas as pd
dtrain = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
dval = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
dtest = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu ='local')
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy =tf.distribute.TPUStrategy(tpu)
else:
    print('TPU is not active')
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


print("REPLICAS: ", strategy.num_replicas_in_sync)
core = strategy.num_replicas_in_sync


dtrain


dtest


#Limitting the data
dtrain = dtrain.loc[:12000,:]


dtrain = dtrain[['id','comment_text','toxic']]
dtrain.head(5)


dtrain.shape


each_len = []

for k in dtrain['comment_text']:
    each_len.append(len(k.split(' ')))
    
dtrain['comment_len'] = each_len


dtrain_sorted = dtrain.groupby('comment_len')[['comment_text']].max().sort_values(by='comment_len', ascending=False).reset_index()

dtrain_sorted


#print(dtrain_sorted.iloc[0]['comment_text'])
#print(dtrain_sorted.iloc[0]['comment_len'])


from sklearn.metrics import roc_auc_score, roc_curve, auc

def roc_auc_cal(pred,y):
    a=roc_auc_score(y, pred)
    return a


#EXAMPLE
'''
# Simulated dataset
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

print(roc_auc_cal(y_scores,y_true))  ---- 0.75
'''


dtrain


#Converting text to lowercase:
import re
def convt_lower(data):
    tmp = []
    for e in data:
        tmp.append(e.lower())
    return tmp

def remove_specialchar(data):
    tmp = []
    for e in data:
        e = e.replace('\n',' ')
        tmp.append(re.sub(r'[^a-zA-Z0-9\s]', '', e))
    return tmp



comment_lc = convt_lower(dtrain['comment_text'])
comment_punc = remove_specialchar(comment_lc)
dtrain['comment_pro']=comment_punc


dtrain['comment_pro'].apply(lambda x:len(str(x).split())).max()


#first split the data:
y = dtrain['toxic']
x = dtrain['comment_pro']
x_train,x_test,y_train,y_test = train_test_split(x.values,y.values,random_state=42,test_size=0.25, shuffle=True)

maxlen = 2000
#Initialize the tokenizer
token = Tokenizer(num_words=maxlen, oov_token="<OOV>") 
token.fit_on_texts(x_train)
vocab_size=token.word_index

#Apply the tokenizer here:
x_train_seq = token.texts_to_sequences(x_train) 
x_test_seq = token.texts_to_sequences(x_test)   #Truncation just so that max len of data is 2000 words

#Truncation did not take place before. I will to it now!
for e in x_train_seq:
    if len(e)>2000:
        print(len(e))
        print('blinked!')
        break

x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen, padding='pre', truncating='pre') #post or pre
x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen, padding='pre', truncating='pre') #post or pre

#now check whether the truncating is done succesfully:
checker = True
for e in x_train_pad:
    if len(e) > 2000:
        print('blinked!')
        checker == False
if checker == True:
    print('truncating is applied sucessfully')
else:
    print('there is a problem in truncating')
    


i = 0
for k,v in vocab_size.items():
    print(k,v)
    i+=1
    if i == 10:
        break
print('the number of vocabulary',len(vocab_size))


x_train_pad


from tensorflow.keras.optimizers import Adam

loss = 'binary_crossentropy'

with strategy.scope():
    optimizer = Adam(learning_rate=1e-5,clipnorm=1.0)

    model = Sequential()
    #input layer
    model.add(Embedding(len(vocab_size)+1,300,input_length = maxlen)) #input_dim,output_dim(embedding_dim),inp_len
    model.add(BatchNormalization())
    #hidden layer where recurrency happens here
    model.add(SimpleRNN(100))
    #output layer
    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 2000))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics =['accuracy'])
model.summary()


history_basicRNN = model.fit(x_train_pad,y_train,epochs = 20, batch_size=64*strategy.num_replicas_in_sync,validation_data = (x_test_pad,y_test))


train_acc = history_basicRNN.history['accuracy']
test_acc = history_basicRNN.history['val_accuracy']
train_loss = history_basicRNN.history['loss']
test_loss = history_basicRNN.history['val_loss']

epoch = range(1,len(train_acc)+1) # From  1 to epoch number


plt.plot(epoch,train_acc ,label ='train_acc')
plt.plot(epoch,test_acc ,label ='test_acc')
plt.title('Accuracy Results')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epoch,train_loss ,label ='train_loss')
plt.plot(epoch,test_loss ,label ='test_loss')
plt.title('Loss Results')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#previous model: model.add(Embedding(vocab_size+1,300,input_length = maxlen)) #input_dim, embedding_dim or output_dim, input_len

glove = '/kaggle/input/glove840b300dtxt/glove.840B.300d.txt'


#https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
embeddings_dict = {}
with open(glove, 'r',encoding="utf-8") as f:
    for line in tqdm(f):
        values = line.split(' ')
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


#let me remember:
i = 0
for k,v in vocab_size.items():
    print(k,v)
    i+=1
    if i == 10:
        break


len(vocab_size)


embedding_vector = np.zeros((len(vocab_size)+1,300)) #creating empty array
#filling them with pretrained values:

for k,i in tqdm(vocab_size.items()):
    arr_value = embeddings_dict.get(k)
    if arr_value is not None:
        embedding_vector[i] = arr_value


embeddings_dict['the']


embedding_vector[2]


embedding_vector.shape


maxlen


from tensorflow.keras.optimizers import Adam

loss = 'binary_crossentropy'

with strategy.scope():
    optimizer = Adam(learning_rate=1e-5,clipnorm=1.0)

    model = Sequential()

    model.add(Embedding(len(vocab_size)+1,300,weights=[embedding_vector],input_length = maxlen, trainable=False)) #input_dim,output_dim(embedding_dim),inp_len
    model.add(BatchNormalization())
    model.add(LSTM(100))
    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 2000))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics =['accuracy'])
model.summary()


start = time.time()
history_lstm = model.fit(x_train_pad,y_train,epochs = 20, batch_size=64*strategy.num_replicas_in_sync,validation_data = (x_test_pad,y_test))
end = time.time()

dif = (end-start)
print(f'the execution took {dif:.3f} sec')


train_acc = history_lstm.history['accuracy']
test_acc = history_lstm.history['val_accuracy']
train_loss = history_lstm.history['loss']
test_loss = history_lstm.history['val_loss']

epoch = range(1,len(train_acc)+1) # From  1 to epoch number


plt.plot(epoch,train_acc ,label ='train_acc')
plt.plot(epoch,test_acc ,label ='test_acc')
plt.title('Accuracy Results')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epoch,train_loss ,label ='train_loss')
plt.plot(epoch,test_loss ,label ='test_loss')
plt.title('Loss Results')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




from tensorflow.keras.optimizers import Adam

loss = 'binary_crossentropy'

with strategy.scope():
    optimizer = Adam(learning_rate=1e-5,clipnorm=1.0)

    model = Sequential()

    model.add(Embedding(len(vocab_size)+1,300,weights=[embedding_vector],input_length = maxlen, trainable=False)) #input_dim,output_dim(embedding_dim),inp_len
    model.add(BatchNormalization())
    model.add(GRU(100))
    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 2000))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics =['accuracy'])
model.summary() 


start = time.time()
history_gru = model.fit(x_train_pad,y_train,epochs = 20, batch_size=64*strategy.num_replicas_in_sync,validation_data = (x_test_pad,y_test))
end = time.time()

dif = (end-start)
print(f'the execution took {dif:.3f} sec')


train_acc = history_gru.history['accuracy']
test_acc = history_gru.history['val_accuracy']
train_loss = history_gru.history['loss']
test_loss = history_gru.history['val_loss']

epoch = range(1,len(train_acc)+1) # From  1 to epoch number


plt.plot(epoch,train_acc ,label ='train_acc')
plt.plot(epoch,test_acc ,label ='test_acc')
plt.title('Accuracy Results')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epoch,train_loss ,label ='train_loss')
plt.plot(epoch,test_loss ,label ='test_loss')
plt.title('Loss Results')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




from tensorflow.keras.layers import LSTM, Bidirectional, Dense

loss = 'binary_crossentropy'

with strategy.scope():
    optimizer = Adam(learning_rate=1e-5,clipnorm=1.0)

    model = Sequential()

    model.add(Embedding(len(vocab_size)+1,300,weights=[embedding_vector],input_length = maxlen, trainable=False)) #input_dim,output_dim(embedding_dim),inp_len
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 2000))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics =['accuracy'])
model.summary() 


start = time.time()
history_bilstm = model.fit(x_train_pad,y_train,epochs = 20, batch_size=64*strategy.num_replicas_in_sync,validation_data = (x_test_pad,y_test))
end = time.time()

dif = (end-start)
print(f'the execution took {dif:.3f} sec')


train_acc = history_bilstm.history['accuracy']
test_acc = history_bilstm.history['val_accuracy']
train_loss = history_bilstm.history['loss']
test_loss = history_bilstm.history['val_loss']

epoch = range(1,len(train_acc)+1) # From  1 to epoch number


plt.plot(epoch,train_acc ,label ='train_acc')
plt.plot(epoch,test_acc ,label ='test_acc')
plt.title('Accuracy Results')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epoch,train_loss ,label ='train_loss')
plt.plot(epoch,test_loss ,label ='test_loss')
plt.title('Loss Results')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


dtrain = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
dval = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
dtest = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

dtrain = dtrain[['id','comment_text','toxic']]

comment_lc = convt_lower(dtrain['comment_text'])
comment_punc = remove_specialchar(comment_lc)
dtrain['comment_pro']=comment_punc


dtrain.shape


#first split the data:
y = dtrain['toxic']
x = dtrain['comment_pro']
x_train,x_test,y_train,y_test = train_test_split(x.values,y.values,random_state=42,test_size=0.25, shuffle=True)

maxlen = 2000 #it was 2000
#Initialize the tokenizer
token = Tokenizer(num_words=maxlen, oov_token="<OOV>") 
token.fit_on_texts(x_train)
vocab_size=token.word_index

#Apply the tokenizer here:
x_train_seq = token.texts_to_sequences(x_train) 
x_test_seq = token.texts_to_sequences(x_test)   #Truncation just so that max len of data is 2000 words

#Truncation did not take place before. I will to it now!
for e in x_train_seq:
    if len(e)>2000:
        print(len(e))
        print('blinked!')
        break

x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen, padding='pre', truncating='pre') #post or pre
x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen, padding='pre', truncating='pre') #post or pre

#now check whether the truncating is done succesfully:
checker = True
for e in x_train_pad:
    if len(e) > 2000:
        print('blinked!')
        checker == False
if checker == True:
    print('truncating is applied sucessfully')
else:
    print('there is a problem in truncating')
    


len(vocab_size.keys())


#let me remember:
print('input dim is ', len(vocab_size.keys()))
i = 0
for k,v in vocab_size.items():
    print(k,v)
    i+=1
    if i == 10:
        break


embedding_vector = np.zeros((len(vocab_size)+1,300)) #creating empty array
#filling them with pretrained values:

for k,i in tqdm(vocab_size.items()):
    arr_value = embeddings_dict.get(k)
    if arr_value is not None:
        embedding_vector[i] = arr_value


len(embedding_vector)


embedding_vector.shape


maxlen


# First things first, LSTM

from tensorflow.keras.optimizers import Adam
from keras import regularizers
kernel_regularizer = regularizers.L1L2(l1=1e-4, l2=1e-4)
bias_regularizer = regularizers.L2(1e-4)
activity_regularizer = regularizers.L2(1e-4)

loss = 'binary_crossentropy'
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    mode="min")

with strategy.scope():
    optimizer = Adam(learning_rate=1e-4,clipnorm=1.0)

    model = Sequential()

    model.add(Embedding(len(vocab_size)+1,300,weights=[embedding_vector],input_length = maxlen, trainable=False)) #input_dim,output_dim(embedding_dim),inp_len

    model.add(LSTM(128,
                   dropout=0.2,
                   kernel_regularizer=kernel_regularizer,
                  bias_regularizer = bias_regularizer,
                  activity_regularizer = activity_regularizer))
    model.add(BatchNormalization())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1,activation='sigmoid'))

    model.build(input_shape=(None, 2000))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics =['accuracy'])
model.summary()


start = time.time()
history_lstm = model.fit(x_train_pad,
                         y_train,
                         epochs = 5, 
                         batch_size=64*strategy.num_replicas_in_sync,
                         validation_data = (x_test_pad,y_test),
                         callbacks = [early_stop])
end = time.time()

dif = (end-start)
print(f'the execution took {dif:.3f} sec')


start = time.time()
history_lstm = model_bi.fit(x_train_pad,
                         y_train,
                         epochs = 20, 
                         batch_size=64*strategy.num_replicas_in_sync,
                         validation_data = (x_test_pad,y_test),
                         callbacks = [early_stop])
end = time.time()

dif = (end-start)
print(f'the execution took {dif:.3f} sec')

