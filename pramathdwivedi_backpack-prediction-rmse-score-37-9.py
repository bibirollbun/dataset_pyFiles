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


sample_submission= pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
training_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


sample_submission


train


training_extra


test


df1=pd.merge(test,sample_submission,on='id',how='inner')


df=pd.concat([train,df1,training_extra],axis=0)


df.head()


print(df.shape)
print(df.columns)


df.info()



df.describe()


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Price'],bins=30,kde=True)
plt.show()


df.isnull().sum()


((df.isnull().sum())/df.shape[0])*100


for i in df.columns:
    if df[i].dtype == object:
        print(df[i].unique())
        print('mode is:',df[i].mode()[0])


for i in df.columns:
    if df[i].dtype == object:
        df[i].fillna(df[i].mode()[0],inplace=True)
    


df.isnull().sum()


df=df.dropna()


df.isnull().sum()


df.shape


df.duplicated().sum()


df=df.drop_duplicates()


df.duplicated().sum()


plt.figure(figsize=(10,5))
sns.boxplot(x=df['Price'])
plt.show()


for i in df.columns:
    if df[i].dtype==object:
        print(i)


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder()


df['Laptop Compartment']=le.fit_transform(df['Laptop Compartment'])
df['Waterproof']=le.fit_transform(df['Waterproof'])


df=pd.get_dummies(df,columns=['Material','Size','Style','Color','Brand'],dtype=int)


df.head()


df.info()


df.head()


x=df.drop(columns='Price')


y=df['Price']


x


y


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


ss=StandardScaler()


x_train[['Compartments','Weight Capacity (kg)']]=ss.fit_transform(x_train[['Compartments','Weight Capacity (kg)']])
x_test[['Compartments','Weight Capacity (kg)']]=ss.transform(x_test[['Compartments','Weight Capacity (kg)']])


x_train=x_train.drop(columns='id')



x_test1=x_test.drop(columns='id')


x_train.shape[1]


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
def mape_metric(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / y_true)) * 100


x_test=x_test.head(200000)


x_test.shape


model=Sequential()


model.add(Dense(128,input_dim=x_train.shape[1],activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))

model.add(Dense(1))


optimizer = Adam(learning_rate=0.001)



model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError()])


history = model.fit(x_train, y_train, epochs=15, batch_size=1000, validation_data=(x_test1, y_test), verbose=1)


from sklearn.metrics import mean_absolute_error
y_pred=model.predict(x_test1)
def regression_accuracy(y_test, y_pred):
    return 100 - (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100

accuracy = regression_accuracy(y_test, y_pred)
print("Regression Accuracy (%):", accuracy)



from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


import pandas as pd
y_pred = model.predict(x_test1).flatten()


y_pred = y_pred[:200000]


submission = pd.DataFrame({'id': x_test['id'][:200000], 'target': y_pred})


submission.to_csv("submission.csv", index=False)

print("Submission file saved successfully!")








