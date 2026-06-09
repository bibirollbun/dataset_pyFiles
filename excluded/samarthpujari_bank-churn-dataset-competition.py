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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


train_data = pd.read_csv('/kaggle/input/classify-binaries-with-bank-churn-dataset/train.csv')
train_data.head()


test_data = pd.read_csv('/kaggle/input/classify-binaries-with-bank-churn-dataset/test.csv')
test_data.head()


sample_submission = pd.read_csv('/kaggle/input/classify-binaries-with-bank-churn-dataset/sample_submission.csv')
sample_submission.head()


# Preprocessing
# Separate features and target
X = train_data.drop(columns=['Exited', 'id'])
y = train_data['Exited']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])


# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


# Align test set to match train set columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model training
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1)  # Use parallel processing
model.fit(X_train, y_train)


# Validation
val_preds = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC Score: {roc_auc}")


# Prediction on test set
test_preds = model.predict_proba(X_test)[:, 1]


# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Exited': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv.")

