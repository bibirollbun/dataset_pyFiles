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
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


#Import libraries

import pandas as pd

#Loading the dataset
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


#Check info and preview
print(train.info())
print(train.head())



from sklearn.preprocessing import LabelEncoder

#Encode the target variable
le = LabelEncoder()
train['Target_encoded'] = le.fit_transform(train['Target'])

#Feature engineering
#1.Total credits earned
train['Total_credits'] = train['Curricular units 1st sem (credited)'] + train['Curricular units 2nd sem (credited)']

#2.Average grade of both semesters
train['Avg_grade'] = (train['Curricular units 1st sem (grade)'] + train['Curricular units 2nd sem (grade)']) / 2

#Confirm changes
print(train[['Target', 'Target_encoded', 'Total_credits', 'Avg_grade']].head())


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Split the data
X = train.drop(columns=['id', 'Target', 'Target_encoded'])
y = train['Target_encoded']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#Build the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

#Train the model
pipeline.fit(X_train, y_train)

#Make predictions and evaluate
y_pred = pipeline.predict(X_valid)
print(classification_report(y_valid, y_pred))


#Define X_test by dropping the 'id'
X_test = test.drop(columns=['id'])

#Check the first few rows of X_test to see its structure
print(X_test.head())

#Check if 'id' exists in the test set and remove iif present
if 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])

#Apply feature engineering steps
X_test['Total_credits'] = X_test['Curricular units 1st sem (credited)'] + X_test['Curricular units 2nd sem (credited)']
X_test['Avg_grade'] = (X_test['Curricular units 1st sem (grade)'] + X_test['Curricular units 2nd sem (grade)']) / 2

#Verify the changes
print(X_test[['Total_credits', 'Avg_grade']].head())

#Make predictions
test_predictions = pipeline.predict(X_test)


#Prepare the submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Target': le.inverse_transform(test_predictions)
})

#Save the submission
submission.to_csv('submission.csv', index=False)

#Check if file is saved
print('Submission file created: submission.csv')

