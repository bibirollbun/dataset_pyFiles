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
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train_data.shape, test_data.shape, submission.shape


train_data.isnull().sum()


train_data.dtypes


train_data["Episode_Length_minutes"] = train_data["Episode_Length_minutes"].astype(float)
test_data["Episode_Length_minutes"] = test_data["Episode_Length_minutes"].astype(float)


train_data["Episode_Title_int"] = train_data["Episode_Title"].str.strip("Episode").astype(int)
train_data.drop(["Episode_Title", "id"], axis=1, inplace=True)

test_data["Episode_Title_int"] = test_data["Episode_Title"].str.strip("Episode").astype(int)
test_data.drop(["Episode_Title", "id"], axis=1, inplace=True)


train_data


for n in train_data.columns:
    if train_data[n].isnull().sum():
        print(n)
        train_data[n + "_na"] = train_data[n].isnull().sum()
        train_data[n].fillna(train_data[n].median(), inplace=True)


for n in test_data.columns:
    if test_data[n].isnull().sum():
        print(n)
        test_data[n + "_na"] = test_data[n].isnull().sum()
        test_data[n].fillna(train_data[n].median(), inplace=True)


train_data.head()


train_data.dtypes


for n,c in train_data.items():
    if train_data[n].dtype.name == 'object':
        train_data[n] = train_data[n].astype('category').cat.as_ordered()

for n,c in test_data.items():
    if train_data[n].dtype.name == "category":
        test_data[n] = pd.Categorical(c, categories=train_data[n].cat.categories, ordered=True)


for n in train_data.columns:
    if train_data[n].dtype.name == "category":
        train_data[n] = train_data[n].cat.codes + 1

for n in test_data.columns:
    if test_data[n].dtype.name == "category":
        test_data[n] = test_data[n].cat.codes + 1


train_data.head()


test_data.head()


from sklearn.metrics import mean_squared_error
def print_score(estimator, X_train, y_train, X_val, y_val):
    y_pred = estimator.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    y_pred = estimator.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    train_r2 = estimator.score(X_train, y_train)
    val_r2 = estimator.score(X_val, y_val)
    print(train_rmse, val_rmse, train_r2, val_r2)


# from sklearn.model_selection import train_test_split
# train_, val_ = train_test_split(train_data, test_size=0.1, random_state=42)


# y_train = train_["Listening_Time_minutes"]
# X_train = train_.drop(["Listening_Time_minutes"], axis=1)

# y_val = val_["Listening_Time_minutes"]
# X_val = val_.drop(["Listening_Time_minutes"], axis=1)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=0.5, n_jobs=-1)
# rf.fit(X_train, y_train)


# print_score(rf, X_train, y_train, X_val, y_val)


y = train_data["Listening_Time_minutes"]
X = train_data.drop(["Listening_Time_minutes"], axis=1)


from sklearn.model_selection import KFold


# clf = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=0.5, n_jobs=-1)


from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100)


cv = KFold(n_splits=8, random_state=42, shuffle=True)

for (train, test), i in zip(cv.split(X, y), range(8)):
    print(f"Fold {i}")
    print(X.iloc[train].shape, X.iloc[test].shape)
    gb.fit(X.iloc[train], y.iloc[train])
    print_score(gb, X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test])


test_data["Number_of_Ads_na"] = 0


ypred = gb.predict(test_data)


train_data["Listening_Time_minutes"].mean(), np.mean(ypred)


submission["Listening_Time_minutes"] = ypred


submission


submission.to_csv('submission.csv', index=False)




