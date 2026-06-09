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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
%matplotlib inline


df = pd.read_csv('/kaggle/input/lstm-7-30/spamtrain.csv',delimiter=',',encoding='latin-1')
df.head()


X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


max_words = 1000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('sigmoid')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])


model.fit(sequences_matrix,Y_train,batch_size=32,epochs=30,
          validation_split=0.2)


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


accr = model.evaluate(test_sequences_matrix,Y_test)


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# 讀取測試資料
test_df = pd.read_csv("/kaggle/input/lstm-7-30/testcsv.csv", encoding="latin1")

test_X = test_df.V1  


max_words = 1000
max_len = 30
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(test_X)
sequences = tok.texts_to_sequences(test_X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_pred = model.predict(sequences_matrix)



import pandas as pd
labels = ['ham' if p < 0.5 else 'spam' for p in test_pred.flatten()]
test_pred = test_pred.flatten()
submission = pd.DataFrame({
    "ID": range(1, len(labels) + 1),  
    'Predict': labels
})
submission.to_csv('submit.csv', index=False)

