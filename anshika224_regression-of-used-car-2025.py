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


train = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


train.info()


train.head()


train.describe(include="all")


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])


X = train.drop(["price"], axis=1)
y = train["price"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "early_stopping_rounds": 100
}

lgbModel = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[valid_data])


import matplotlib.pyplot as plt

feature_importances = lgbModel.feature_importance(importance_type="gain")
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.show()


importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})

# 重要度が高い順にソートし、上位9個を取得
top_features = importance_df.sort_values(by="Importance", ascending=False).head(9)
top_features = top_features["Feature"].tolist()


from sklearn.ensemble import RandomForestRegressor

X = train[top_features]
y = train["price"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "early_stopping_rounds": 100
}

lgb_model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[valid_data])


rf_pred = rf_model.predict(X_valid)
lgb_pred = lgb_model.predict(X_valid)

ensemble_pred = (rf_pred + lgb_pred) / 2

rmse = mean_squared_error(y_valid, ensemble_pred, squared=False)
print(f"Ensemble RMSE: {rmse}")


categorical_cols = test.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])


rf_pred = rf_model.predict(test[top_features])
lgb_pred = lgb_model.predict(test[top_features])

ensemble_pred = (rf_pred + lgb_pred) / 2

data = {"id": test["id"], "price": ensemble_pred}
submission = pd.DataFrame(data)
submission.to_csv("submission.csv", index=False)




