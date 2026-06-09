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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
submission_data = pd.read_csv(submission_path)



# Drop 'id' since it is not a feature
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']  # Store test IDs for submission
test_data.drop(columns=['id'], inplace=True)

# Handle missing values (fill with median)
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Separate features and target
X = train_data.drop(columns=['rainfall'])  # Features
y = train_data['rainfall']                 # Target

# Split into train-validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)



model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = model.predict_proba(X_val)[:, 1]

# Evaluate using ROC-AUC score
auc_score = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC-AUC Score: {auc_score:.4f}")



# Predict probabilities for test data
test_predictions = model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'rainfall': test_predictions})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")


