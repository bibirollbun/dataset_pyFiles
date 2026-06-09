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
from sklearn.metrics import roc_auc_score



train = pd.read_csv('/kaggle/input/churn-challenge-ai/train.csv')
test = pd.read_csv('/kaggle/input/churn-challenge-ai/test.csv')
sample_submission = pd.read_csv('/kaggle/input/churn-challenge-ai/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# Separate target
y = train["Exited"]

# Drop id and target to get features
X = train.drop(columns=["id", "Exited"])
X_test = test.drop(columns=["id"])

# Combine train and test to apply consistent encoding
X_all = pd.concat([X, X_test], axis=0)

# Convert categorical columns to dummy/one-hot encoding
X_all_encoded = pd.get_dummies(X_all)

# Split back to X and X_test
X = X_all_encoded.iloc[:len(X), :]
X_test = X_all_encoded.iloc[len(X):, :]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict_proba(X_val)[:, 1]
val_score = roc_auc_score(y_val, val_preds)
print(f"Validation ROC AUC Score: {val_score:.4f}")



model.fit(X, y)
test_preds = model.predict_proba(X_test)[:, 1]



submission = sample_submission.copy()
submission["Exited"] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head()


