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


import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

train.head()


# Copy of test ids for submission
test_ids = test["id"]


# Basic info
print(train.info())


print(train.isnull().sum() / len(train) * 100)


# Separate numeric and categorical features

num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('id')
cat_cols = train.select_dtypes(include=['object']).columns.drop('Personality')

# Fill numeric columns with median
for col in num_cols:
    median_val = train[col].median()
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

# Fill categorical columns with 'Unknown' to preserve missing signal
for col in cat_cols:
    train[col].fillna('Unknown', inplace=True)
    test[col].fillna('Unknown', inplace=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split for Local Validation

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Model

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


# Validate
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


# Train on Full Data
# ======================
model.fit(X, y)

# ======================
# Predict on Test
# ======================
test_pred = model.predict(test.drop('id', axis=1))

# Map predictions back to labels
pred_labels = np.where(test_pred == 0, 'Introvert', 'Extrovert')

# ======================
# Save Submission
# ======================
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': pred_labels
})
submission.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv")

