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


import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split
import numpy as np 
import random 
import json 
  
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd


df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
# df = df.drop('Unnamed: 0', axis=1)
df.head()


import plotly.express as px
grouped = df.groupby(['label']).count().reset_index()
grouped

fig = px.bar(grouped, x='label', y='Question')
fig.show()


#convert labels   from list to  n dismision array
def list_to_cat(list_data):
    x=len(list(set(list_data)))
    vec = tf.keras.utils.to_categorical(list_data, num_classes=x) 
    return vec


def number_to_cat(index):
    arr=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    arr[index]=1.0
    return arr


x_train = df['Question']
training_labels = list_to_cat(df['label'])
len(df['label'])


df["label_cat"]=list(training_labels)
df.head()


tokenizer = Tokenizer(filters='',oov_token='<unk>') 
tokenizer.fit_on_texts(x_train) 

training_sequences = tokenizer.texts_to_sequences(x_train) 
training_pad = pad_sequences(training_sequences, padding='pre') 



def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed


epochs=15



model = tf.keras.models.Sequential([ 
    tf.keras.layers.Embedding(input_dim=1000, output_dim=1000), 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_num, dropout=0.2)),   
    tf.keras.layers.Dense(lstm_num, activation='relu'), 
   
   
  
    tf.keras.layers.Dense(lstm_num, activation='softmax') 
]) 
  
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)   
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
model.compile(optimizer=optimizer, loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy']) 
model.summary()


X_train, X_test, y_train, y_test = train_test_split(training_pad, training_labels, test_size=0.2)




hist = model.fit(
               np.array(X_train),
               np.array(y_train),
               epochs=epochs,
               batch_size=5, 
               verbose=1,
                )


import matplotlib.pyplot as plt

# Plot Utility
def plot_graphs(history, string,l):
  plt.plot(history.history[string][:l:])
 
  plt.xlabel("Epochs")
  plt.ylabel(string)

  plt.show()


y_pred=model.predict(X_test)
pred_class = np.argmax(y_pred, axis=1) 
pred_cats=[]


for i in pred_class:
    # pred_cats.append()
    pred_cats=pred_cats+ [number_to_cat(i)]
    


from sklearn.metrics import f1_score

f1_score(y_test, pred_cats, average="micro")


plot_graphs(hist, 'accuracy',epochs)
plot_graphs(hist, 'loss',epochs)


df_test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv') 
df_test.head()


x_test = df_test['Question']
tokenizer = Tokenizer(filters='',oov_token='<unk>') 
tokenizer.fit_on_texts(x_test) 

test_sequences = tokenizer.texts_to_sequences(x_test) 
test_pad = pad_sequences(test_sequences, padding='pre')


predicted_vals=model.predict(test_pad)
pred_class = np.argmax(predicted_vals, axis=1)  


id_arra=np.array(df_test['id'])
res_arr =np.array(pred_class)

df_submision= pd.DataFrame({'id': id_arra, 'label': res_arr})
df_submision.head()


df_submision.to_csv('/kaggle/working/submission2.csv',index=False)


res_arr




