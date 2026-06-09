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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn


train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')


train.head()


print(train.info(), '\n\n')
train.describe()


train.isna().sum()


train.info()


train["Age"] = train["Age"] / 365.25
test["Age"] = test["Age"] / 365.25


train['Age'].describe()


train = train[(train["Age"] >= 0) & (train["Age"] <= 100)]
test = test[(test["Age"] >= 0) & (test["Age"] <= 100)]


test_ids = test['id'].copy()


y = train["Status"]
X_train_raw = train.drop(columns=["id", "Status"])
X_test_raw = test.drop(columns=["id"])


num_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns
ojb_cols = X_train_raw.select_dtypes(include=["object"]).columns


num_imputer = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train_raw[num_cols]), columns=num_cols, index=X_train_raw.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test_raw[num_cols]), columns=num_cols, index=X_test_raw.index)


cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train_raw[ojb_cols]), columns=ojb_cols, index=X_train_raw.index)
X_test_cat = pd.DataFrame(cat_imputer.transform(X_test_raw[ojb_cols]), columns=ojb_cols, index=X_test_raw.index)


encoders = {}
for col in ojb_cols:
    le = LabelEncoder()
    X_train_cat[col] = le.fit_transform(X_train_cat[col])
    X_test_cat[col] = le.transform(X_test_cat[col])
    encoders[col] = le


X_train_combined = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_combined = pd.concat([X_test_num, X_test_cat], axis=1)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_combined), columns=X_train_combined.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_combined), columns=X_test_combined.columns)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42)



model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}")
print(classification_report(y_val, y_pred))


test_predictions = model.predict(X_test_scaled)


submission = pd.DataFrame({"ID": test_ids, "Status": test_predictions})
submission.to_csv("/submission.csv", index=False)

submission.head()


