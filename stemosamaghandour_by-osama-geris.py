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


sample=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')



print(sample.columns)
print(train_df.columns)
print(test_df.columns)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Encode target variable
label_encoder = LabelEncoder()
train_df['Personality'] = label_encoder.fit_transform(train_df['Personality'])  # Extrovert = 1, Introvert = 0

# Separate features and target
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
X_test = test_df.drop(['id'], axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align train/test columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Build pipeline: Imputation + Classifier
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Validate
val_preds = pipeline.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print("âœ… Validation Accuracy:", val_acc)

# Predict on test
test_preds = pipeline.predict(X_test)
test_preds_labels = label_encoder.inverse_transform(test_preds)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_preds_labels
})
submission.to_csv('sample_submission.csv', index=False)
print("ğŸ“� sample_submission.csv saved!")


