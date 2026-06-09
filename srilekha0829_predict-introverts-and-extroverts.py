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
df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df.head()


df.isnull().sum()


df.columns


df['Personality'].value_counts()


df.shape


df.info()


import pandas as pd
train = pd.read_csv ('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


test.head()


train.columns


train.info()
train['Personality'].value_counts()


train.isnull().sum()


test.isnull().sum()


train.describe()


train.dtypes


binary_cols = ['Stage_fear', 'Drained_after_socializing']

for col in binary_cols:
    train[col] = train[col].map({'Yes':1, 'No':0})
    test[col] = test[col].map({'Yes':1, 'No':0})


train.dtypes
test.dtypes


num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']
# fill the missing valuses with median
# train 

for col in num_cols:
    train[col]= train[col].fillna(train[col].median())

# test

for col in num_cols:
    test[col]= test[col].fillna(train[col].median())


train.isnull().sum()


test.isnull().sum()


train['Personality'].value_counts()


# frist read the CSV file
import pandas as pd
train = pd.read_csv ('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

# convert datatype
binary_cols = ['Stage_fear', 'Drained_after_socializing']

for col in binary_cols:
    train[col] = train[col].map({'Yes':1, 'No':0})
    test[col] = test[col].map({'Yes':1, 'No':0})
    
# fill the missing valuses with median
num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']

# train 

for col in num_cols:
    train[col]= train[col].fillna(train[col].median())

# test

for col in num_cols:
    test[col]= test[col].fillna(train[col].median())

# split train and test data set

from sklearn.model_selection import train_test_split
X = train.drop(columns=['id','Personality'])
Y = train['Personality']
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.2, random_state=42, stratify = Y)

# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test.drop(columns=['id']))

# train Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, Y_train)

# evaluate the model
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(X_val_scaled)

print(confusion_matrix(Y_val, Y_pred))
print(classification_report(Y_val, Y_pred))


features = ['Time_spent_Alone', 'Stage_fear','Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']


# Train Logistic Regression on full training data

from sklearn.linear_model import LogisticRegression
X_full = train[features]
Y_full = train['Personality']

model = LogisticRegression(max_iter = 1000)
model.fit(X_full, Y_full)
test_predictions = model.predict(test[features])

submission['Personality'] = test_predictions
submission.to_csv('final_submission.csv',index=False)


pd.read_csv('final_submission.csv').head()




