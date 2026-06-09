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


# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load dataset

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Store necessary columns
ids = test['id']

# Drop unnecessary columns
train.drop(columns='id',inplace=True,axis=1)
test.drop(columns='id',inplace=True,axis=1)


# Feature Enginnering

def feature_engineering(data):
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3])
    data['BMI'] = data['Weight']/(data['Height']/100)**2
    data['HR_duration'] = data['Heart_Rate']*data['Duration']
    data['Temp_duration'] = data['Body_Temp']*data['Duration']
    data['Calories_Burned'] = np.where(data['Sex'] == 'male',(-55.0969 + (0.6309 * data['Heart_Rate']) + (0.1988 * data['Weight']) + (0.2017 * data['Age'])) / 4.184 * data['Duration'],(-20.4022 + (0.4472 * data['Heart_Rate']) - (0.1263 * data['Weight']) + (0.074 * data['Age'])) / 4.184 * data['Duration'])
    return data

train = feature_engineering(train)
test = feature_engineering(test)


# Convert categorical to Numeric data

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


# Split dataset into Train and Test

X = train.drop(columns='Calories')
y = train['Calories']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# Feature scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test = scaler.transform(test)


# Define base regressors (Parameters set after finetuning)

rf = RandomForestRegressor(n_estimators=200,max_depth=20,min_samples_split=5, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200,max_depth=7,learning_rate=0.05, random_state=42)


# Ensemble: Voting Regressor

ensemble = VotingRegressor(estimators=[
    ('rf', rf),
    ('gb', gb)
])


# Training

ensemble.fit(X_train_scaled, y_train)


# Prediction

y_pred = ensemble.predict(X_test_scaled)


# Defining RMSLE

def rmsle_score(y,preds):
    y = np.maximum(0, y)
    preds = np.maximum(0, preds)
    return np.sqrt(np.mean((np.log1p(preds) - np.log1p(y)) ** 2))


# Evaluate

rmsle = rmsle_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Prediction on test data

y_test_pred = ensemble.predict(test)

submission['id'] = ids
submission['Calories'] = y_test_pred
submission.to_csv("submission.csv",index=False)

