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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data.head()


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_data.head()


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train_data.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())

categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

for column in categorical_cols:
    train[column] = train[column].fillna("Unknown")
    test[column] = test[column].fillna("Unknown")


label_encoders = {}
for column in categorical_cols:
    lEnc = LabelEncoder()
    train[column] = lEnc.fit_transform(train[column])
    test[column] = lEnc.transform(test[column])
    label_encoders[column] = lEnc


A = train.drop(["id", "Price"], axis=1)
B = train["Price"]
A_test = test.drop(["id"], axis=1)
A_train, A_val, B_train, B_val = train_test_split(A, B, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(A_train, B_train)


val_preds = model.predict(A_val)
rmse = mean_squared_error(B_val, val_preds, squared=False)
print("Validation Root Mean Squared Error:", rmse)


test_preds = model.predict(A_test)
submission["Predicted Price"] = test_preds
submission.to_csv("baseline_submission.csv", index=False)

