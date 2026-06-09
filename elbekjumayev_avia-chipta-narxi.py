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


df=pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv', index_col=0)


df.head()


df.airline.value_counts()


print(df['airline'].unique())



df['airline'] = df['airline'].map({'Vistara': 0, 'SpiceJet': 1,'Indigo':2,'Air_India':3,'GO_FIRST':4,'AirAsia':5})


print(df['departure_time'].unique())


df['departure_time'] = df['departure_time'].map({'Early_Morning': 0, 'Evening': 1,'Morning':2,'Afternoon':3,'Night':4,'Late_Night':5})


print(df['stops'].unique())


df['stops'] = df['stops'].map({'zero': 0, 'one': 1,'two_or_more':2})


print(df['arrival_time'].unique())


df['arrival_time'] = df['arrival_time'].map({'Early_Morning': 0, 'Evening': 1,'Morning':2,'Afternoon':3,'Night':4,'Late_Night':5})


print(df['destination_city'].unique())


df['destination_city'] = df['destination_city'].map({'Mumbai': 0, 'Kolkata': 1,'Delhi':2,'Hyderabad':3,'Chennai':4,'Bangalore':5})


print(df['class'].unique())


df['class'] = df['class'].map({'Economy': 0, 'Business': 1})


df.head()


df=df.drop('flight',axis=1)


df=df.drop('source_city',axis=1)


df.head()


df.corrwith(df['price']).sort_values(ascending=False)


from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler()
df2=pd.DataFrame(min_max_scaler.fit_transform(df),columns=df.columns)


df2.head()


import sklearn
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df2, test_size=0.2,random_state=42)


from sklearn import linear_model
from sklearn.linear_model import LinearRegression

MLR_model=linear_model.LinearRegression()
x_train=np.asanyarray(train_set[['airline','departure_time','stops','arrival_time','destination_city','class','duration','days_left']])
y_train=np.asanyarray(train_set[['price']])
                      


MLR_model.fit(x_train, y_train)


x_test=np.asanyarray(test_set[['airline','departure_time','stops','arrival_time','destination_city','class','duration','days_left']])
y_test=np.asanyarray(test_set[['price']])
y_natija=MLR_model.predict(x_test)


MAE=mean_absolute_error(y_test,y_natija)
RMSE=np.sqrt(mean_squared_error(y_test,y_natija))
print(MAE)
print(RMSE)


MLR_model=linear_model.LinearRegression()
x_train=np.asanyarray(train_set[['class','duration','airline']])
y_train=np.asanyarray(train_set[['price']])


MLR_model.fit(x_train, y_train)


x_test=np.asanyarray(test_set[['class','duration','airline']])
y_test=np.asanyarray(test_set[['price']])
y_natija=MLR_model.predict(x_test)


MAE=mean_absolute_error(y_test,y_natija)
RMSE=np.sqrt(mean_squared_error(y_test,y_natija))
print(MAE)
print(RMSE)


df.head(5)


from sklearn.preprocessing import LabelEncoder


encoder=LabelEncoder()


df['airline']=encoder.fit_transform(df['airline'].values)
df['flight']=encoder.fit_transform(df['flight'].values)
df['source_city']=encoder.fit_transform(df['source_city'].values)
df['departure_time']=encoder.fit_transform(df['departure_time'].values)
df['stops']=encoder.fit_transform(df['stops'].values)
df['arrival_time']=encoder.fit_transform(df['arrival_time'].values)
df['destination_city']=encoder.fit_transform(df['destination_city'].values)
df['class']=encoder.fit_transform(df['class'].values)


df.head(5)


df.corrwith(df['price']).sort_values(ascending=False)


import sklearn
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2,random_state=42)


from sklearn import linear_model
from sklearn.linear_model import LinearRegression

MLR_model=linear_model.LinearRegression()
x_train=np.asanyarray(train_set[['airline','stops','arrival_time','class','duration','days_left','flight']])
y_train=np.asanyarray(train_set[['price']])

MLR_model.fit(x_train, y_train)


x_test=np.asanyarray(test_set[['airline','stops','arrival_time','class','duration','days_left','flight']])
y_test=np.asanyarray(test_set[['price']])
y_natija=MLR_model.predict(x_test)


MAE=mean_absolute_error(y_test,y_natija)
RMSE=np.sqrt(mean_squared_error(y_test,y_natija))

print(MAE)
print(RMSE)


