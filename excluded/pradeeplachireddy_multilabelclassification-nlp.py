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


df_train = pd.read_csv('/kaggle/input/voai-2025-hsgs-nlp/train.csv')
df_test = pd.read_csv('/kaggle/input/voai-2025-hsgs-nlp/test.csv')
sample_sub = pd.read_csv('/kaggle/input/voai-2025-hsgs-nlp/sample_submission.csv')


df_train.head()


df_train.columns


df_train.isna().sum()


import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\s+',' ',text).strip()
    return text


df_train['clean_text'] = df_train['comment_text'].apply(clean_text)
df_test['clean_text'] = df_test['comment_text'].apply(clean_text)
df_train.head()


labels_col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df_train['labels'] = df_train[labels_col].values.tolist()
df_train.head()


df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)


# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(df_train['labels'])


y=np.array(df_train['labels'].tolist()).astype('float32')
class_counts = y.sum(axis=0)
class_counts


y.dtype


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.model_selection import train_test_split


# y = np.array(df_train['labels'].tolist())





X_train,X_val,y_train,y_val = train_test_split(df_train['clean_text'],y,test_size=0.2,random_state=42)
# stratify should not be used for multi label classification,
# it only works for single label classification


y_val.shape


tokenizer = Tokenizer(oov_token='<OOV>')


X_test = df_test['clean_text'].copy()


tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)


vocab_size = len(tokenizer.word_index)+1
vocab_size


maxlenwords = [len(word) for word in train_sequences]


max_len = int(np.percentile(maxlenwords,99))
max_len


train_sequences_pad = pad_sequences(train_sequences,maxlen = max_len,padding='post')
val_sequences_pad = pad_sequences(val_sequences,maxlen=max_len,padding='post')
test_sequences_pad = pad_sequences(test_sequences,maxlen=max_len,padding='post')


from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Bidirectional,LSTM,Embedding,Dense
from tensorflow.keras.regularizers import l2


model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=128))
model.add(Bidirectional(LSTM(128,kernel_regularizer=l2(0.01),recurrent_regularizer=l2(0.01),dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(6,activation='sigmoid'))


model.summary()


import tensorflow
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


model.compile(optimizer='adam',metrics=[tensorflow.keras.metrics.AUC(multi_label=True)],loss='binary_crossentropy')


Early_Stopping=EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)
Model_Checkpoint = ModelCheckpoint(filepath='best_model.keras',monitor='val_loss',mode='min',save_best_only=True)





history = model.fit(x=train_sequences_pad,y=y_train,batch_size=128,epochs=20,callbacks=[Early_Stopping,Model_Checkpoint],validation_data=(val_sequences_pad,y_val))


y_probs = model.predict(test_sequences_pad)
y_probs





threshold=0.5
y_pred_labels = (y_probs >=threshold).astype(int)
y_pred_labels


pred_df = pd.DataFrame(y_probs,columns = labels_col)


pred_df  = pred_df.reset_index(drop=True)


test_id_df = df_test[['id']].reset_index(drop=True)


final_df = pd.concat([test_id_df,pred_df],axis=1)
final_df


final_df.to_csv('mlsubmission.csv',index=False)

