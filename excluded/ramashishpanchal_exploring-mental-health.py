# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data=pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
X_test=pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")
test_id=X_test.id
data.shape


data.head()


data.isnull().sum()


data.info()


data.Depression.value_counts()


fig,axis=plt.subplots(2,2,figsize=(12,8))

sns.kdeplot(x='CGPA',data=data,ax=axis[0,0])

sns.barplot(y='CGPA',x='Depression',data=data,ax=axis[0,1])

sns.barplot(y='CGPA',x='Have you ever had suicidal thoughts ?',data=data,ax=axis[1,0])

sns.barplot(y="CGPA",x='Financial Stress',data=data,ax=axis[1,1])


## cleaning some data based on observation

#feature engineering 
data['Academic Pressure']=data['Academic Pressure'].fillna(0.0)
data['Work Pressure']=data['Work Pressure'].fillna(0.0)
data['Study Satisfaction']=data['Study Satisfaction'].fillna(0.0)
data['Job Satisfaction']=data['Job Satisfaction'].fillna(0.0)

data['Pressure']=data['Academic Pressure']+data['Work Pressure']
data['Satisfaction']=data['Study Satisfaction']+data['Job Satisfaction']

data.head()


X_test['Academic Pressure']=X_test['Academic Pressure'].fillna(0.0)
X_test['Work Pressure']=X_test['Work Pressure'].fillna(0.0)
X_test['Study Satisfaction']=X_test['Study Satisfaction'].fillna(0.0)
X_test['Job Satisfaction']=X_test['Job Satisfaction'].fillna(0.0)

X_test['Pressure']=X_test['Academic Pressure']+X_test['Work Pressure']
X_test['Satisfaction']=X_test['Study Satisfaction']+X_test['Job Satisfaction']

X_test.head()


##removing unwanted columns
data=data.drop(columns=['CGPA','id','Name','Academic Pressure','Work Pressure','Study Satisfaction','Job Satisfaction'])
X_test=X_test.drop(columns=['CGPA','id','Name','Academic Pressure','Work Pressure','Study Satisfaction','Job Satisfaction'])
data.head()


data.shape


data.info()


data.Profession.value_counts()


#trying to filling Profession column
mask1=data['Profession'].isna()
mask2=data['Working Professional or Student']=='Student'
data.loc[mask1 & mask2,'Profession']='Student'
data.head()


data.info()


data.dropna(inplace=True)


data.info()



X_train=data.drop(columns=['Depression'])
y_train=data['Depression']





#encoding all categorical columns
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
encoder={}
categorical_columns=data.select_dtypes(include='object')
for col in categorical_columns:
    encoder[col]=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[col]=encoder[col].fit_transform(X_train[[col]])
    X_test[col]=encoder[col].transform(X_test[[col]])

X_test.head()




from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled=scaler.transform(X_test)
X_test=pd.DataFrame(X_test_scaled,columns=X_test.columns)

X_train.head()



import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=14))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(X_train,y_train,epochs=20,batch_size=64,validation_split=0.2)


y_pred=model.predict(X_test).ravel()

for i in range(len(y_pred)):
    if y_pred[i] < 0.5 :
        y_pred[i]=0
    else :
        y_pred[i]=1

y_pred=pd.Series(y_pred,name='Depression')

y_pred


result_dict={
    'id':test_id,
    'Depression':y_pred
}

result=pd.DataFrame(result_dict)
result.to_csv('mental_health_submission.csv',index=False)




