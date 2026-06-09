import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


train_data=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train_data.info()


train_data.isnull().sum()


#filling in null values with median values
train_data['num_sold']=train_data['num_sold'].fillna(train_data['num_sold'].median())


train_data.isnull().sum()


train_data['date']=pd.to_datetime(train_data['date'])
test_data['date']=pd.to_datetime(test_data['date'])
train_data['year']=train_data['date'].dt.year
train_data['month']=train_data['date'].dt.month
train_data['day']=train_data['date'].dt.day
test_data['year']=test_data['date'].dt.year
test_data['month']=test_data['date'].dt.month
test_data['day']=test_data['date'].dt.day


categorical_columns=[feature for feature in train_data.columns if train_data[feature].dtype=='O']
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_train = encoder.fit_transform(train_data[categorical_columns])
one_hot_df_train= pd.DataFrame(one_hot_encoded_train, columns=encoder.get_feature_names_out(categorical_columns))
train_data_encoded=pd.concat([train_data, one_hot_df_train],axis=1)
train_data_encoded=train_data_encoded.drop(categorical_columns, axis=1)

#encoding test data
one_hot_encoded_test=encoder.transform(test_data[categorical_columns])
one_hot_df_test=pd.DataFrame(one_hot_encoded_test, columns=encoder.get_feature_names_out(categorical_columns))
test_data_encoded=pd.concat([test_data,one_hot_df_test],axis=1)



y=train_data_encoded['num_sold']
train_data_encoded=train_data_encoded.drop(['id', 'date','num_sold'], axis=1)


X_train, X_test,y_train,y_test=train_test_split(train_data_encoded, y, test_size=0.2, random_state=42) 
#using linear regression
model=LinearRegression()
model.fit(X_train,y_train)



y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("Accuracy:", model.score(X_train,y_train))


test_data_encoded=test_data_encoded.drop(categorical_columns,axis=1)
test_data_encoded=test_data_encoded.drop(['id','date'],axis=1)
y_predicted=model.predict(test_data_encoded)


sample_submission['num_sold']=y_predicted


sample_submission


sample_submission.to_csv("submission.csv",index=False)

