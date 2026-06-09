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


train_csv=pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
train_csv.head()


train_csv.shape


train_csv['id']=train_csv['id'].astype('str')


for i in range(0,len(train_csv['id'])):
    
    if i/10 < 1:
        train_csv.loc[i,'id']='000' + train_csv.loc[i,'id']
    else:
        train_csv.loc[i,'id']='00' + train_csv.loc[i,'id']




input_data=[]

for dirname,dirs,_ in os.walk('/kaggle/input/fake-or-real-the-impostor-hunt/data/train/'):
    for directory in dirs:
        extracted_id=directory[8:]
        file_1_path=os.path.join(dirname,directory,'file_1.txt')
        file_2_path=os.path.join(dirname,directory,'file_2.txt')

        with open(file_1_path,'r') as file:
            file_1_text=file.read()

        with open(file_2_path,'r') as file:
            file_2_text=file.read()

        real_text_id_value=train_csv.loc[train_csv['id']==extracted_id,'real_text_id'].iloc[0]
        
        if real_text_id_value== 1 :
            input_data.append([file_1_text,'real'])
            input_data.append([file_2_text,'fake'])
        else:
            input_data.append([file_2_text,'real'])
            input_data.append([file_1_text,'fake'])
    break

print(input_data[0])


test_data=[]
test_id=[]
for dirname,dirs,_ in os.walk('/kaggle/input/fake-or-real-the-impostor-hunt/data/test/'):
    for directory in dirs:
        extracted_id=directory[8:]
        file_1_path=os.path.join(dirname,directory,'file_1.txt')
        file_2_path=os.path.join(dirname,directory,'file_2.txt')

        with open(file_1_path,'r') as file:
            file_1_text=file.read()

        with open(file_2_path,'r') as file:
            file_2_text=file.read()

        test_id.append(extracted_id)
        test_id.append(extracted_id)
        test_data.append(file_1_text)
        test_data.append(file_2_text)
    break

print(test_data[0])


X = [row[0] for row in input_data]

#make X as desired input for Tokenizer like ['ashish','vedang','om']

X_flat=[]           
for s in X:
    X_flat.append(s)




import re

def remove_symbols_regex(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

X_flat=[remove_symbols_regex(text) for text in X_flat]
test_data=[remove_symbols_regex(text) for text in test_data]

test_data[0]


#removing stopwords(common words)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

filtered_X=[]
for text in X_flat:
    filtered_words = [word for word in text.lower().split() if word not in stop_words]
    filtered_X.append(filtered_words)

filtered_test=[]
for text in test_data:
    filtered_words = [word for word in text.lower().split() if word not in stop_words]
    filtered_test.append(filtered_words)
    



filtered_X_flat=[]           
for s in filtered_X:
    filtered_X_flat.append(s)


test=[]           
for s in filtered_test:
    test.append(s)


print(test[3])
print(filtered_X_flat[3])


#creating the vocablary ( assigning a number to each unique word)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
tokenizer.fit_on_texts(filtered_X_flat)
tokenizer.fit_on_texts(test)
tokenizer.word_index



y=[row[1] for row in input_data]
y[1]


vocab_length=len(tokenizer.word_index)




tokenized_X=[]

for text in X_flat:
    tokenized_text=tokenizer.texts_to_sequences([text])       #returns a 2D list
    tokenized_X.append(tokenized_text[0])

print(tokenized_X[0])

tokenized_test=[]

for text in test:
    tokenized_text=tokenizer.texts_to_sequences([text])       #returns a 2D list
    tokenized_test.append(tokenized_text[0])

print(tokenized_test[0])



for i in range(len(y)):
    if y[i]=='fake':
        y[i]=0
    else:
        y[i]=1

y


max_len=max([len(item) for item in tokenized_X])


#applying padding at end
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences=pad_sequences(tokenized_X,max_len,padding='post')
padded_test_sequences=pad_sequences(tokenized_test,max_len,padding='post')


X_test=np.array(padded_test_sequences)


X=np.array(padded_input_sequences)
X.shape


y=np.array(y)
y.shape


#creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,GRU,Dense,BatchNormalization,Dropout,Bidirectional

model=Sequential()
model.add(Embedding(vocab_length+1,output_dim=100,input_length=max_len))  # +1 is for padding zeros


model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(100)))

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))


from tensorflow.keras.optimizers import Adam
optimizer=Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


model.build(input_shape=(None, max_len))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=3),
    ModelCheckpoint('imposter_hunt_model.h5', save_best_only=True)
]
model.fit(X, y, epochs=20, batch_size=32, callbacks=callbacks)


y_pred = model.predict(X_test)
y_pred


y_pred.shape



flatten_y_pred = y_pred.flatten()



y_pred=flatten_y_pred


my_id=[]
real_text_id=[]

def remove_leading_zeros(s):
    return s.lstrip('0') or '0'

i=0

while i < len(y_pred):
    prob_file_1=y_pred[i]
    prob_file_2=y_pred[i+1]

    my_id.append(remove_leading_zeros(test_id[i]))
    
    if prob_file_1>prob_file_2 :
        real_text_id.append(1)
    else :
        real_text_id.append(2)

    i=i+2



result=pd.DataFrame({
    'id':my_id,
    'real_text_id':real_text_id
})

result.head()



result['real_text_id'].value_counts()


result.to_csv("imposter_hunt16.csv",index=False)




