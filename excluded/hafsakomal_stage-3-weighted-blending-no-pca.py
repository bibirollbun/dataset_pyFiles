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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb 


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


# Define features
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy']

X = train[features]
y = train["BeatsPerMinute"]
X_test = test[features]


# Split for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# LightGBM
lgb_model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, random_state=42)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)


# XGBoost
xgb_model = XGBRegressor(
    n_estimators=2000, learning_rate=0.05, max_depth=6, random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    early_stopping_rounds=100,
    verbose=200
)


# CatBoost
cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.05, depth=6, random_state=42, verbose=200
)
cat_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100
)


preds_lgb = lgb_model.predict(X_valid)
preds_xgb = xgb_model.predict(X_valid)
preds_cat = cat_model.predict(X_valid)

rmse_lgb = mean_squared_error(y_valid, preds_lgb, squared=False)
rmse_xgb = mean_squared_error(y_valid, preds_xgb, squared=False)
rmse_cat = mean_squared_error(y_valid, preds_cat, squared=False)

print("Validation RMSEs:")
print(f"LightGBM: {rmse_lgb:.5f}")
print(f"XGBoost: {rmse_xgb:.5f}")
print(f"CatBoost: {rmse_cat:.5f}")


# Inverse RMSE weights
w_lgb = 1 / rmse_lgb
w_xgb = 1 / rmse_xgb
w_cat = 1 / rmse_cat
total = w_lgb + w_xgb + w_cat


# Blended validation
blended_valid = (preds_lgb*w_lgb + preds_xgb*w_xgb + preds_cat*w_cat) / total
rmse_blend = mean_squared_error(y_valid, blended_valid, squared=False)
print(f"Blended Validation RMSE: {rmse_blend:.5f}")



preds_lgb_test = lgb_model.predict(X_test)
preds_xgb_test = xgb_model.predict(X_test)
preds_cat_test = cat_model.predict(X_test)

blended_test = (preds_lgb_test*w_lgb + preds_xgb_test*w_xgb + preds_cat_test*w_cat) / total



submission = sample_submission.copy()
submission["BeatsPerMinute"] = blended_test
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission_stage3.csv")




