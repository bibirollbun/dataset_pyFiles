# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.simplefilter(action='ignore', category=Warning)        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/mlx-2-0-classification/train.csv")
test = pd.read_csv("/kaggle/input/mlx-2-0-classification/test.csv")


df.head()


df.tail()


test.head()


test.tail()


df.info()


df.sample()


df.describe()


df.columns


df.shape


df.isna().sum()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Data Heatmap")
plt.show()


numerical_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
categorical_cols = list(df.select_dtypes(include=["object"]).columns)
categorical_cols.remove("target")


df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

test[numerical_cols] = test[numerical_cols].fillna(df[numerical_cols].mean())
test[categorical_cols] = test[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


df.isna().sum()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    if col in test.columns:
        test[col] = le.transform(test[col])


X = df.drop(columns=["target", "id"])
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test.drop(columns=["id"]))


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


test_predictions = model.predict(X_test)
test["target"] = test_predictions
submission = test[["id", "target"]]


submission.head()


submission.tail()


submission.info()


submission.sample()


submission.describe()


submission.columns


submission.shape


submission.isna().sum()


submission.isna()


submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")

