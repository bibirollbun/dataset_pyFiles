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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# 2. Fill missing values
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(test[col].mean())

cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

for col in cat_cols:
    train[col] = train[col].map({'Yes': 1, 'No': 0})
    test[col] = test[col].map({'Yes': 1, 'No': 0})

train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})


X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)

# 4. Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred))

# 7. Predict on test set
test_predictions = model.predict(X_test)

# 8. Prepare submission
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': ['Extrovert' if p==1 else 'Introvert' for p in test_predictions]
})
submission.to_csv('my_submission.csv', index=False)
print("Submission file 'my_submission.csv' is ready!")




