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


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



y = train["y"]
X = train.drop(columns=["id","y"])
X_test = test.drop(columns=["id"])



for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    le.fit(pd.concat([X[col], X_test[col]], axis=0).astype(str))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)



xgb = XGBClassifier(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)
xgb.fit(X, y)
print("✅ Model training completed")



test_pred = xgb.predict(X_test)



submission = pd.DataFrame({
    "id": test["id"],
    "y": test_pred
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv is ready! Go to Output → Download → Upload to Kaggle competition")
submission.head()





