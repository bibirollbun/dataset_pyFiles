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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")



print("Training:", train.shape)
print("Test:", test.shape)
print("\nTrain Head:")
print(train.head())



X = train.drop(columns=["id", "y"])
y = train["y"]
X_test = test.drop(columns=["id"])
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
print("After encoding:")
print("X shape:", X.shape)
print("X_test shape:", X_test.shape)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1         
)
model.fit(X_train, y_train)
val_accuracy = model.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")


model.fit(X, y)
predictions = model.predict_proba(X_test)[:, 1]
submission = sample_submission.copy()
submission["y"] = predictions
submission.to_csv("submission.csv", index=False)





