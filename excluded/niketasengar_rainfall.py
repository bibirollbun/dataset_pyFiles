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


import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential,Input
from keras.layers import Dense,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df


df.info()


df.isnull().sum()


df.describe()


plt.figure(figsize=(9,5))
corr=df.corr()
sns.heatmap(corr,annot=True,cmap="YlOrBr",linewidths=0.6,fmt=".1f",linecolor="black")
plt.show()


# dropping redundant and highly correlated column

df.drop(columns=['id','temparature','mintemp','day'],inplace=True,axis=1)



sns.countplot(x=df['rainfall'])
plt.title("Rainfall Distribution (0 = No Rain, 1 = Rain)")
plt.xlabel("Rainfall")
plt.ylabel("Count")
plt.show()


columns=df.drop(columns=['rainfall'])
for x in columns:
    sns.kdeplot(df[x],label=x,fill=True,linewidth=2)
    plt.title("KDE Plot")
    plt.legend()
    plt.show()


# log transformation for converting skewed data 
exclude_col=df[['rainfall']]
for x in df.columns:
    if x not in exclude_col:
        df[x]=np.log1p(df[x])


sns.scatterplot(x=df['maxtemp'], y=df['humidity'], hue=df['rainfall'])


X=df.drop(columns=['rainfall'])
y=df['rainfall']


scaler=StandardScaler()
X_Scaled=scaler.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X_Scaled,y,random_state=42,test_size=.2)


model=Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32,activation='relu'),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()


model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['AUC'])


early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)


history=model.fit(X_train,y_train,
                 validation_data=(X_test,y_test),
                 epochs=80,
                 batch_size=32,
                 callbacks=[early_stop], 
                 verbose=1)


test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test.head()


test.isnull().sum()


df_test=test.drop(columns=['id','temparature','mintemp','day'],axis=1)


exclude_col=df[['rainfall']]
for x in df_test.columns:
    if x not in exclude_col:
        df[x]=np.log1p(df[x])


df_test_Scaled=scaler.transform(df_test)


test_pred=model.predict(df_test_Scaled).flatten()
if np.isnan(test_pred).sum()>0:
    print("Nan value found")
    test_pred=np.nan_to_num(test_pred)


submission=pd.DataFrame({'id': test['id'],'rainfall':test_pred})
submission.to_csv("submission.csv",index=False)
print("successfully saved")

