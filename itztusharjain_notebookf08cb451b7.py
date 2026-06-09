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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load datasets
train_path = "/kaggle/input/thapar-kaggle-hack-v-01/X_train.csv"  # Update with actual path
test_path = "/kaggle/input/thapar-kaggle-hack-v-01/X_test.csv"  # Update with actual path

X_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)

# Preserve 'id' column from X_test for final submission
if 'id' in X_test.columns:
    test_ids = X_test['id']
    X_test.drop(columns=['id'], inplace=True)
else:
    raise ValueError("ID column not found in X_test")

# Separate target variable from training data
if 'target' in X_train.columns:
    y_train = X_train.pop('target')  # Remove target column

# Identify categorical features
categorical_features = ['feature_10', 'feature_11', 'feature_12']

# Encode categorical features using Label Encoding
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])

    # Handle unseen labels in X_test
    X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Fill missing values (just in case)
X_test.fillna(X_train.median(), inplace=True)

# Save processed data
X_train.to_csv("X_train_processed.csv", index=False)
X_test.to_csv("X_test_processed.csv", index=False)

# Ensure submission format: ID and target predictions
submission = pd.DataFrame({"id": test_ids, "target": [0] * len(test_ids)})  # Replace [0] with model predictions later
submission.to_csv("submission.csv", index=False)

print("Preprocessing completed! Submission file created.")


