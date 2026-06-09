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


# DRW Crypto Market Prediction - Starter Notebook

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os
from lightgbm import LGBMRegressor, early_stopping

# --- Load Data (Parquet) ---
train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

# Ensure file paths exist
assert os.path.exists(train_path), "Train parquet file not found."
assert os.path.exists(test_path), "Test parquet file not found."
assert os.path.exists(sample_sub_path), "Sample submission file not found."

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)
sample_submission = pd.read_csv(sample_sub_path)



column_names = train.columns.tolist()
print(column_names)


test.head(10)


train.head(10)


# --- Features and Target ---
target = train["label"]
features = train.drop(columns=["label"])
test_features = test#.drop(columns=["timestamp_id"])
# --- Features and Target ---

# --- Clean Data: Replace inf/-inf with NaN, then fill or drop ---
features.replace([np.inf, -np.inf], np.nan, inplace=True)
test_features.replace([np.inf, -np.inf], np.nan, inplace=True)


common_columns = features.columns.intersection(test_features.columns)
features = features[common_columns]
test_features = test_features[common_columns]


#imputer = SimpleImputer(strategy="median")
imputer = SimpleImputer(strategy="constant", fill_value=0)
features_imputed = imputer.fit_transform(features)
test_imputed = imputer.transform(test_features)


# --- Feature Selection (optional) ---
selector = SelectKBest(score_func=f_regression, k=50)
features_selected = selector.fit_transform(features_imputed, target)
test_selected = selector.transform(test_imputed)

# --- Model Training ---
tscv = TimeSeriesSplit(n_splits=5)
preds = np.zeros(len(test))





preds = np.zeros(len(test_selected))

for fold, (train_idx, val_idx) in enumerate(tscv.split(features_selected)):
    X_train, X_val = features_selected[train_idx], features_selected[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
    )

    preds += model.predict(test_selected) / tscv.get_n_splits()


sample_submission.head(10)


# --- Create Submission ---
sample_submission["prediction"] = preds
sample_submission.to_csv("submission.csv", index=False)


print("Submission file saved as submission.csv")

