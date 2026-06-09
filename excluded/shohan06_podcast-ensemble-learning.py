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


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")


# Load train and test data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


df_train.head(10)


df_test.head(5)


df_train.shape, df_test.shape


# Unique values
train_uniques = {
    col: df_train[col].nunique() for col in df_train.columns
}

test_uniques = {
    col: df_test[col].nunique() for col in df_test.columns
}

display(train_uniques)
display(test_uniques)


train = df_train.copy()
test = df_test.copy()


# Remove id from train data
train.drop("id", axis=1, inplace=True)
train.head()


# Fix missing "Number_of_Ads"
train["Number_of_Ads"] = train["Number_of_Ads"].fillna(train["Number_of_Ads"].median())

test["Number_of_Ads"] = test["Number_of_Ads"].fillna(test["Number_of_Ads"].median())

# Check if filled 
train["Number_of_Ads"].isnull().sum(), test["Number_of_Ads"].isnull().sum()


# Filter categorical and numeric columns
cat_cols = [col for col in train.columns if train[col].dtype in ["object", "category"]]
num_cols = [col for col in train.columns if train[col].dtype in ["float64"]]

# Target feature
target = "Listening_Time_minutes"

print(f"Categorical features: {cat_cols}")
print(f"Numeric features: {num_cols}")


# Target encoding categorical features using KFold
def target_encoding(train, test, col, target, n_split=5):
    out_vals = pd.Series(index=train.index, dtype=float)
    test_encoded = pd.Series(index=test.index, dtype=float)
    kf = KFold(n_splits=n_split, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]

        means = X_train.groupby(col)[target].mean()
        out_vals.iloc[val_idx] = X_val[col].map(means)

    test_encoded = test[col].map(train.groupby(col)[target].mean())
    new_col = f"{col}_TE"
    train[new_col] = out_vals.fillna(train[target].mean())
    test[new_col] = test_encoded.fillna(train[target].mean())


# Apply encoding to the categorical features
for col in cat_cols:
    target_encoding(train, test, col, target)


# check new features
train.columns


# Emsemble learning
features = ["Podcast_Name_TE", "Episode_Title_TE", "Genre_TE", "Publication_Day_TE", 
            "Episode_Sentiment_TE", "Publication_Time_TE", "Guest_Popularity_percentage",
            "Host_Popularity_percentage", "Episode_Length_minutes", "Number_of_Ads",
           ]

X = train[features]
y = train["Listening_Time_minutes"]
X_test = test[features]

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# test_preds = np.zeros(len(test))
# scores = []

# base models
lgb_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=50,
    subsample=0.5,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=-1
)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.5,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=0
)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0
)

# stacked ensemble model
stacked_model = StackingRegressor(
    estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model)],
    final_estimator=Ridge(),
    passthrough=False,
    n_jobs=-1
)

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     stacked_model.fit(X_train, y_train)

#     y_pred = stacked_model.predict(X_val)
#     rmse = mean_squared_error(y_val, y_pred, squared=False)
#     scores.append(rmse)

#     test_preds += stacked_model.predict(X_test) / kf.n_splits

# print(f"\nStacked RMSE: {np.mean(scores)}")

# train/valildation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"\nValidation RMSE: {rmse}")

# Predict on the test set
test_preds = stacked_model.predict(X_test)


# submission
submission["Listening_Time_minutes"] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head(10)




