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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head()


train_df = train_df.dropna()
ids = test_df['id']
ids
test_df.drop(columns=['id'])


train_df.isnull().sum()


test_df.fillna(test_df.mean(), inplace=True)
test_df.isnull().sum()


X = train_df.drop(columns=['rainfall','id'])
y = train_df['rainfall']
test_id = test_df['id'] 
test_df = test_df.drop(columns=['id'])
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)



test_id


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)




test_df = test_df[X_train.columns]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 



import tensorflow
from tensorflow import keras
from keras.layers import Dense,Dropout
from keras import Sequential
from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)), 
    Dropout(0.3), 

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(1, activation='sigmoid') 
])



X_train.shape


model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss',    
    min_delta=0.001,       
    patience=10,           
    verbose=1,  
    mode=min,
    baseline=None,        
    restore_best_weights=True  
)
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50,validation_split=0.2, callbacks=[early_stopping])



y_pred =model.predict(test_df)
test_id = test_id.ravel()
y_pred = y_pred.ravel()
y_pred.shape



predictions = (y_pred > 0.5).astype(int)



submission = pd.DataFrame({"id": test_id, "rainfall": y_pred.flatten()})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")

