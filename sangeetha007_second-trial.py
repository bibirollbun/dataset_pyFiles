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


train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train.head()


print(train.isnull())


print(train.isna().any())
#the train dataset is clean


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print(test.head())
print(test.isna().any())


from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print(train.columns)
print(test.columns)


print(train.describe())


cat_cols = ["gender", "ethnicity", "education_level",
            "income_level", "smoking_status", "employment_status"]

# Convert to category dtype before splitting
for col in cat_cols:
    train[col] = train[col].astype("category")

enable_categorical=True
tree_method="approx"


train_dropped = train.drop(columns=["gender", "ethnicity", "education_level",
            "income_level", "smoking_status", "employment_status"])
X = train_dropped.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]

#X_test = test.drop("diagnosed_diabetes", axis=1)
#y_test = test["diagnosed_diabetes"]


print(train_dropped.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True,
    tree_method="approx",
)

model.fit(X_train, y_train)


# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)


test_dropped = test.drop(columns=["gender", "ethnicity", "education_level",
            "income_level", "smoking_status", "employment_status"])
# for col in cat_cols:
#     test_dropped[col] = test_dropped[col].astype("category")
test_predictions = model.predict(test_dropped)
print(test_predictions)


output = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_predictions
})

output.to_csv("predictions.csv", index=False)

