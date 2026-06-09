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


train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


train.head(10)


train.isnull().sum()


train.info()


test.isnull().sum()





categorical_col = train.select_dtypes(include=['object']).columns
numerical_col = train.select_dtypes(exclude=['object']).columns


categorical_col, numerical_col


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')


col_to_impute = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress']
train[col_to_impute] = imputer.fit_transform(train[col_to_impute])


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder


for col in categorical_col:
    encoder = LabelEncoder()
    train[col] = encoder.fit_transform(train[col])    


train.isnull().sum()


test.isnull().sum()


test[col_to_impute] = imputer.fit_transform(test[col_to_impute])


for col in categorical_col:
    encoder = LabelEncoder()
    test[col] = encoder.fit_transform(test[col]) 


test.isnull().sum()


X = train.drop(columns=['Depression'])
y = train['Depression']


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score


accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.4f}') 


final_prediction = model.predict(test)


submission = pd.DataFrame({
    'id': test['id'],
    'Depression': final_prediction 
})


submission.to_csv('submission.csv', index=False)


submission

