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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv") #The training dataset
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv") #The testing dataset
test_id = test['id'] #An array containing the id numbers of 'test' dataset, will be used later~


train.head()


train.isnull().sum()


def clean(data):
    data = data.drop('id',axis = 1)
    #Numerical comlumns
    data['Time_spent_Alone'] = data['Time_spent_Alone'].fillna(data['Time_spent_Alone'].median())
    data['Social_event_attendance'] = data['Social_event_attendance'].fillna(data['Social_event_attendance'].median())
    data['Going_outside'] = data['Going_outside'].fillna(data['Going_outside'].median())
    data['Friends_circle_size'] = data['Friends_circle_size'].fillna(data['Friends_circle_size'].median())
    data['Post_frequency'] = data['Post_frequency'].fillna(data['Post_frequency'].median())

    #Categorical columns
    data['Stage_fear'] = data['Stage_fear'].fillna('No')
    data['Drained_after_socializing'] = data['Drained_after_socializing'].fillna('No')
    return data


train = clean(train)
test = clean(test)
train.isnull().sum()


def manual_encoding(data):
    #To encode Stage_fear
    data_list_yes = []
    data_list_no = []
    for i in data['Stage_fear']:
        if i == 'Yes':
            data_list_yes.append(1)
            data_list_no.append(0)
        elif i == 'No':
            data_list_yes.append(0)
            data_list_no.append(1)
    data = data.drop('Stage_fear',axis = 1)
    data_list_yes = np.array(data_list_yes)
    data_list_no = np.array(data_list_no)
    data['Stage_fear_Yes'] = data_list_yes
    data['Stage_fear_No'] = data_list_no

    #To encode Drained_after_socializing
    data_list_yes = []
    data_list_no = []
    for i in data['Drained_after_socializing']:
        if i == 'Yes':
            data_list_yes.append(1)
            data_list_no.append(0)
        elif i == 'No':
            data_list_yes.append(0)
            data_list_no.append(1)
    data = data.drop('Drained_after_socializing',axis = 1)
    data_list_yes = np.array(data_list_yes)
    data_list_no = np.array(data_list_no)
    data['Drained_after_socializing_Yes'] = data_list_yes
    data['Drained_after_socializing_No'] = data_list_no
    return data


train = manual_encoding(train)
test = manual_encoding(test)


test.head()


#Creating the features dataset 'X'
X = train.drop('Personality',axis = 1)


#Creating the target value containing dataset 'y'
y = []
n = len(train['Personality'])
for value in train['Personality']:
    if value == 'Extrovert':
        y.append(1)
    elif value == 'Introvert':
        y.append(0)


#The models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#The pipeline (which is like a shell encasing the scaler and model )
from sklearn.pipeline import Pipeline

#The scalers
from sklearn.preprocessing import StandardScaler,QuantileTransformer,MinMaxScaler

#The function which splits our train dataset, one for training and other for testing
#, as a method of checking which pipeline is better
from sklearn.model_selection import train_test_split

#accuracy and precision score functions
from sklearn.metrics import accuracy_score,precision_score


#Creating all the pipelines
pipe1 = Pipeline([
    ('scaler',StandardScaler()),
    ('model',LogisticRegression(random_state = 42))
])
pipe2 = Pipeline([
    ('scaler',QuantileTransformer(random_state = 42)),
    ('model',LogisticRegression(random_state = 42))
])
pipe3 = Pipeline([
    ('scaler',MinMaxScaler()),
    ('model',LogisticRegression(random_state = 42))
])
pipe4 = Pipeline([
    ('scaler',StandardScaler()),
    ('model',KNeighborsClassifier(n_neighbors = 7))
])
pipe5 = Pipeline([
    ('scaler',QuantileTransformer(random_state = 42)),
    ('model',KNeighborsClassifier(n_neighbors = 7))
])
pipe6 = Pipeline([
    ('scaler',MinMaxScaler()),
    ('model',KNeighborsClassifier(n_neighbors = 7))
])
pipe7 = Pipeline([
    ('scaler',StandardScaler()),
    ('model',RandomForestClassifier(random_state = 42,max_depth = 5))
])
pipe8 = Pipeline([
    ('scaler',QuantileTransformer(random_state = 42)),
    ('model',RandomForestClassifier(random_state = 42,max_depth = 5))
])
pipe9 = Pipeline([
    ('scaler',MinMaxScaler()),
    ('model',RandomForestClassifier(random_state = 42,max_depth = 5))
])


pipes = [pipe1,pipe2,pipe3,pipe4,pipe5,pipe6,pipe7,pipe8,pipe9]
names = ['SS,LR','QT,LR','MMS,LR','SS,KNN','QT,KNN','MMS,KNN','SS,RFC','QT,RFC','MMS,RFC']


X_train,X_test,y_train,y_test = train_test_split(X,y)


for name,pipe in zip(names,pipes):
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    print(f'{name},accuracy:{round(acc,3)},precision:{round(prec,3)}')



#QT and LR
model1 = Pipeline([
    ('scale',QuantileTransformer(random_state = 42)),
    ('model',LogisticRegression(random_state = 42,n_jobs = -1))
])
model1.fit(X,y)

#SS and KNN
model2 = Pipeline([
    ('scale',StandardScaler()),
    ('model',KNeighborsClassifier(n_neighbors = 7,n_jobs = -1))
])
model2.fit(X,y)


#prediction variables which hold the predicted values to submit
predictions1 = model1.predict(test)
predictions2 = model1.predict(test)


reverse_encoded_predictions1 = []
n = len(predictions1)
for value in predictions1:
    if value == 1:
        reverse_encoded_predictions1.append('Extrovert')
    elif value == 0:
        reverse_encoded_predictions1.append('Introvert')

reverse_encoded_predictions2 = []
n = len(predictions2)
for value in predictions2:
    if value == 1:
        reverse_encoded_predictions2.append('Extrovert')
    elif value == 0:
        reverse_encoded_predictions2.append('Introvert')


#Creating the DataFrames
submission1 = pd.DataFrame({'id':test_id,'Personality':reverse_encoded_predictions1})
submission2 = pd.DataFrame({'id':test_id,'Personality':reverse_encoded_predictions2})


#AND FILES ARE DONE!!
submission1.to_csv('submission1.csv',index = False)
submission2.to_csv('submission2.csv',index = False)

