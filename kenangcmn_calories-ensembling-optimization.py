# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head()


test_df.head()


print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


nums = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]


def add_feature_cross_terms(df, features):
    df_new = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df_new[f"{f1}_x_{f2}"] = df_new[f1] * df_new[f2]
    return df_new
    
train_df = add_feature_cross_terms(train_df, nums)
test_df = add_feature_cross_terms(test_df, nums)


train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0}).astype('category')
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0}).astype('category')


X = train_df.drop(columns=["id", "Calories"])
y = np.log1p(train_df["Calories"])
X_test = test_df.drop(columns=["id"])


%%time

FOLDS = 50
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof_cb = np.zeros(len(train_df))
pred_cb = np.zeros(len(test_df))


for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"CatBoost Fold {fold+1}")
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=100,
        verbose=0,
        random_state=42,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU",
        cat_features=[X.columns.get_loc("Sex")]
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cb[val_idx] = model_cb.predict(X_val)
    pred_cb += model_cb.predict(X_test)

pred_cb /= FOLDS


%%time

FOLDS = 50
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train_df))
pred_xgb = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"XGBoost Fold {fold+1}")
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model_xgb = XGBRegressor(
        max_depth=10,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        tree_method="gpu_hist" if os.environ.get("CUDA_VISIBLE_DEVICES") else "hist"
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    pred_xgb += model_xgb.predict(X_test)

pred_xgb /= FOLDS


def optimize_ensemble_weights(oof_cb, oof_xgb, train_df):
    rmsle_scores = []

    for i in np.arange(0.0, 1.0, 0.01):
        w1 = i
        w2 = 1 - i
        ensemble_preds = w1 * np.expm1(oof_cb) + w2 * np.expm1(oof_xgb)
        rmsle = np.sqrt(mean_squared_log_error(train_df["Calories"], ensemble_preds))
        
        rmsle_scores.append({
            'w1': w1,
            'w2': w2,
            'rmsle': rmsle
        })

    results_df = pd.DataFrame(rmsle_scores)
    best_row = results_df.loc[results_df['rmsle'].idxmin()]

    final_preds = best_row['w1'] * np.expm1(pred_cb) + best_row['w2']  * np.expm1(pred_xgb) 
    final_preds = np.clip(final_preds, 1, 314)

    return best_row, results_df, final_preds

best_row, results_df, final_preds = optimize_ensemble_weights(oof_cb, oof_xgb, train_df)

print(f"Best RMSLE is {best_row['rmsle']} with w1 = {best_row['w1']} and w2 = {best_row['w2']} !")


sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub["Calories"] = final_preds
sub.to_csv("submission.csv", index=False)


sub.head()

