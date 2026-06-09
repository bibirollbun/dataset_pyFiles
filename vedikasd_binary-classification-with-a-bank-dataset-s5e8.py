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
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train.head()


train.info()


train.isnull().sum()


# Keep test ids safe for submission
test_ids = test["id"]


# Drop id (for training only)
train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


# Separate features & target
X = train.drop("y", axis=1)
y = train["y"]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Identify categorical columns
onehot_cols = ["marital", "default", "housing", "loan", "education"]
label_cols = ["job", "contact", "month", "poutcome"]
numeric_cols = [col for col in X.columns if col not in onehot_cols + label_cols]

# Apply Label Encoding for label_cols
for col in label_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])  # same mapping for test

# One-hot encode onehot_cols
X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
test = pd.get_dummies(test, columns=onehot_cols, drop_first=True)


# Align train and test (important!)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# Define XGBoost model
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# Fit model
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50, early_stopping_rounds=30)


# Evaluate
y_pred = xgb.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))


# Predictions on test data
test_preds = xgb.predict(test)


# Create submission file
submission = pd.DataFrame({"id": test_ids, "y": test_preds})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")




