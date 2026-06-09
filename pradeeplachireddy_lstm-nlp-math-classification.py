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


df_train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
df_test = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')
sample_submission = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv')


df_train.head()


df_train['label'].value_counts()


df_train.info()


df_test.info()


df_test.head()


sample_submission.head()


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional,Dense,Dropout,SpatialDropout1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,load_model


df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)


X_train,X_val,y_train,y_val = train_test_split(df_train['Question'],df_train['label'],test_size=0.2,random_state=42,stratify=df_train['label'])


X_test = df_test['Question']


tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)


train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)


vocab_size=len(tokenizer.word_index)+1
vocab_size


maxlenwords = [len(word) for word in train_sequences]
maxlen = int(np.percentile(maxlenwords,99))
maxlen


train_sequences_pad = pad_sequences(train_sequences,maxlen=maxlen,padding='post',truncating='post')
val_sequences_pad = pad_sequences(val_sequences,maxlen=maxlen,padding='post',truncating='post')
test_sequences_pad = pad_sequences(test_sequences,maxlen=maxlen,padding='post',truncating='post')


model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=128))
model.add(Bidirectional(LSTM(64,kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),dropout=0.2,recurrent_dropout=0.2)))
#model.add(LSTM(64,kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(8,activation='softmax'))


model.summary()


optimizer=Adam(learning_rate=0.001)


model.compile(optimizer=optimizer,metrics=['accuracy'],loss='sparse_categorical_crossentropy')


EarlyStopping=EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
ModelCheckpoint = ModelCheckpoint(filepath='best_model.keras',monitor='val_loss',mode='min',save_best_only=True)
                              


history = model.fit(x=train_sequences_pad,y=y_train,batch_size=64,epochs=20,callbacks=[EarlyStopping,ModelCheckpoint],validation_data=(val_sequences_pad,y_val))


model_loaded=load_model('best_model.keras')


train_loss, train_acc = model_loaded.evaluate(train_sequences_pad,y_train)


validation_loss, validation_acc = model_loaded.evaluate(val_sequences_pad,y_val)


test_predicted=model_loaded.predict(test_sequences_pad)
test_predicted


y_pred_binary=np.argmax(test_predicted,axis=1)
y_pred_binary


df_test['label'] = y_pred_binary


sample_submission['label']=df_test['label']
sample_submission.head()


df_test.info()


sample_submission.to_csv("submission.csv", index=False)

