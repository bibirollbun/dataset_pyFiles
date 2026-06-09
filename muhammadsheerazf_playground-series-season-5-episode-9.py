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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb



# ------------ Config ------------
TRAIN_FILE = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_FILE = "/kaggle/input/playground-series-s5e9/test.csv"
SUB_FILE = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

ID_COL = "id"
TARGET = "BeatsPerMinute"
N_SPLITS = 5
SEED = 42



# ------------ Load data ------------
print("Loading data...")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
print("train shape:", train.shape, "test shape:", test.shape)



# ------------ Features ------------
features = [col for col in train.columns if col not in [ID_COL, TARGET]]

X = train[features]
y = train[TARGET]
X_test = test[features]



# ------------ CV Setup ------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))
preds_xgb = np.zeros(len(test))



# ------------ LightGBM + XGBoost Training ------------
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    params_lgb = {
        "objective": "regression",
        "metric": "rmse",
        "random_state": SEED,
        "learning_rate": 0.05,
        "num_leaves": 31,
    }
    model_lgb = lgb.train(
    params_lgb, lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

    
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    preds_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / N_SPLITS

    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    params_xgb = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": SEED,
        "eta": 0.05,
        "max_depth": 6,
    }
    model_xgb = xgb.train(
        params_xgb, dtrain,
        evals=[(dtrain, "train"), (dval, "valid")],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    oof_xgb[val_idx] = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration))
    preds_xgb += model_xgb.predict(dtest, iteration_range=(0, model_xgb.best_iteration)) / N_SPLITS



# ------------ Stacking with Ridge Regression ------------
print("\nStacking predictions...")
stack_train = np.vstack([oof_lgb, oof_xgb]).T
stack_test = np.vstack([preds_lgb, preds_xgb]).T

ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(stack_train, y)
final_preds = ridge.predict(stack_test)



# ------------ Evaluate OOF ------------
rmse_lgb = mean_squared_error(y, oof_lgb, squared=False)
rmse_xgb = mean_squared_error(y, oof_xgb, squared=False)
rmse_stack = mean_squared_error(y, ridge.predict(stack_train), squared=False)

print(f"CV RMSE LightGBM: {rmse_lgb:.5f}")
print(f"CV RMSE XGBoost: {rmse_xgb:.5f}")
print(f"CV RMSE Stacked: {rmse_stack:.5f}")



# ------------ Save Submission ------------
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: final_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


