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


org=pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")


df=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
dt=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


df.info()


org.info()


df=df.drop(columns=['id'])


ids = dt['id']


dt=dt.drop(columns=['id'])


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation   # ✅ import callbacks

# ---------- USER CONFIG ----------
TARGET_COL = "accident_risk"      # change if your target column has a different name
ID_COL = None                     # set to column name if you have an ID column to exclude, or None
N_SPLITS = 5
RANDOM_STATE = 42
N_BINS = 10                       # number of bins to create for stratification (adjust if necessary)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100
# ---------------------------------

def train_lgbm_stratified_cv(df,
                             target_col=TARGET_COL,
                             id_col=ID_COL,
                             n_splits=N_SPLITS,
                             random_state=RANDOM_STATE,
                             n_bins=N_BINS):

    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in dataframe")

    # features = all columns except target and id (if provided)
    exclude = {target_col}
    if id_col:
        exclude.add(id_col)
    features = [c for c in df.columns if c not in exclude]

    X = df[features]
    y = df[target_col].values

    # Handle categorical/object features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    # Create stratification bins from continuous target
    try:
        y_binned = pd.qcut(y, q=n_bins, duplicates="drop", labels=False)
    except Exception:
        y_binned = pd.cut(y, bins=n_bins, labels=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmse_scores = []
    oof_preds = np.zeros(len(df))

    fold = 0
    for train_idx, val_idx in skf.split(X, y_binned):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",   # LightGBM hist tree
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "verbosity": -1,
            "seed": random_state,
            "device": "gpu",          # ✅ use GPU
            "force_col_wise": True,   # ✅ better stability on GPU
        }

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS),
                log_evaluation(VERBOSE_EVAL)
            ]
        )

        val_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
        oof_preds[val_idx] = val_pred
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        rmse_scores.append(rmse)
        print(f"Fold {fold} RMSE: {rmse:.6f}  (best_iter={bst.best_iteration})")

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"\nCV RMSE scores: {['{:.6f}'.format(s) for s in rmse_scores]}")
    print(f"Mean CV RMSE: {mean_rmse:.6f}")
    print(f"Std  CV RMSE: {std_rmse:.6f}")

    return {
        "rmse_scores": rmse_scores,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "oof_preds": oof_preds,
        "features": features,
        "cat_cols": cat_cols
    }






# -------------------------
# Example usage:
# -------------------------
results = train_lgbm_stratified_cv(df)
print("Final Mean CV RMSE:", results['mean_rmse'])


def train_full_model_and_predict(df, dt, target_col=TARGET_COL, id_col=ID_COL, params=None, num_boost_round=2000):
    df = df.copy()
    dt = dt.copy()

    exclude = {target_col}
    if id_col:
        exclude.add(id_col)
    features = [c for c in df.columns if c not in exclude]

    X = df[features]
    y = df[target_col]

    # categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        if c in dt.columns:
            dt[c] = dt[c].astype("category")

    dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "device": "gpu",
            "force_col_wise": True,
        }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round
    )

    preds = model.predict(dt[features])
    return preds



test_preds = train_full_model_and_predict(df, dt)
print(test_preds[:10])


sub = pd.DataFrame({
    'id': ids,                  # ids should be your test set ID column
    'accident_risk': test_preds # predictions from the model
})



submy = pd.read_csv("/kaggle/input/mysubs/SUBMISSION (35).csv")


sub.to_csv("submission_lbg.csv", index=False)
sub.head()



import pandas as pd


weight_sub = 0.2
weight_submy = 0.8


sub = sub.sort_values('id').reset_index(drop=True)
submy = submy.sort_values('id').reset_index(drop=True)

supersub = sub.copy()
supersub['accident_risk'] = weight_sub * sub['accident_risk'] + weight_submy * submy['accident_risk']

supersub.to_csv('supersub.csv', index=False)



supersub.head()


# results = train_xgb_stratified_cv(df)
# print("Final Mean CV RMSE:", results['mean_rmse'])


# import xgboost as xgb

# def train_full_model_and_predict(df, dt, target_col=TARGET_COL, id_col=ID_COL, params=None, num_boost_round=2000):
#     df = df.copy()
#     dt = dt.copy()

#     # Exclude target & ID columns
#     exclude = {target_col}
#     if id_col:
#         exclude.add(id_col)
#     features = [c for c in df.columns if c not in exclude]

#     X = df[features]
#     y = df[target_col]

#     # categorical columns → cast to category
#     cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
#     for c in cat_cols:
#         X[c] = X[c].astype("category")
#         if c in dt.columns:
#             dt[c] = dt[c].astype("category")

#     dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
#     dtest = xgb.DMatrix(dt[features], enable_categorical=True)

#     if params is None:
#         params = {
#             "objective": "reg:squarederror",
#             "eval_metric": "rmse",
#             "tree_method": "hist",   # histogram tree
#             "device": "cuda",        # ✅ GPU
#             "learning_rate": 0.05,
#             "max_depth": 8,
#             "subsample": 0.8,
#             "colsample_bytree": 0.8,
#             "seed": RANDOM_STATE,
#         }

#     model = xgb.train(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round
#     )

#     # ✅ Use full model if no early stopping was applied
#     preds = model.predict(dtest)
#     return preds



# test_preds = train_full_model_and_predict(df, dt)



# sub = pd.DataFrame({
#     "id": ids,   # adjust if your test set ID column has a different name
#     "accident_risk": test_preds
# })



# sub.to_csv("submission_xgb.csv", index=False)
# sub.head()


