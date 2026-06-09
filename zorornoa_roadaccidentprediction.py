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


!pip install xgboost optuna


import warnings, time, os, pickle, math
warnings.filterwarnings("ignore")
t0 = time.time()

import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import optuna
import xgboost as xgb


TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"      
TEST_PATH = "/kaggle/input/playground-series-s5e10/test.csv"
TARGET = "accident_risk"
SAMPLE_FOR_OPTUNA = 100_000   
N_TRIALS = 100                
CV_FOLDS = 3                 
RANDOM_STATE = 42
N_JOBS = 4


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train.csv")
y = train[TARGET]
X = train.drop(columns=[TARGET])


def feature_engineer(df):
    df = df.copy()
    if "speed_limit" in df.columns and "curvature" in df.columns:
        df["speed_curv_interaction"] = df["speed_limit"].fillna(0) * df["curvature"].fillna(0)
    if "num_lanes" in df.columns:
        df["lanes_sq"] = df["num_lanes"].fillna(0).astype(float) ** 2
    if "lighting" in df.columns:
        df["lighting"] = df["lighting"].astype(str)
        df["is_night"] = df["lighting"].str.lower().isin(["night","dark","dusk","dawn"]).astype(int)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)
    return df


X_fe = feature_engineer(X)
test_fe = feature_engineer(test.copy())


common_cols = [c for c in X_fe.columns if c in test_fe.columns]
X_fe = X_fe[common_cols]
test_fe = test_fe[common_cols]


numeric_cols = X_fe.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in common_cols if c not in numeric_cols]
low_card_cats = [c for c in categorical_cols if X_fe[c].nunique(dropna=False) <= 10]
high_card_cats = [c for c in categorical_cols if c not in low_card_cats]


num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")


preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("impute", num_imputer), ("scale", StandardScaler())]), numeric_cols),
    ("lowcat", Pipeline([("impute", cat_imputer), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))]), low_card_cats),
    ("highcat", Pipeline([("impute", cat_imputer), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), high_card_cats)
], remainder="drop")


preprocessor.fit(X_fe, y)
X_pre = preprocessor.transform(X_fe)
test_pre = preprocessor.transform(test_fe)


def get_feature_names(preproc, numeric_cols, low_card_cats, high_card_cats):
    names = []
    names += numeric_cols
    if low_card_cats:
        ohe = preproc.named_transformers_["lowcat"].named_steps["onehot"]
        for i, col in enumerate(low_card_cats):
            cats = ohe.categories_[i]
            names += [f"{col}__{str(cat)}" for cat in cats]
    names += high_card_cats
    return names


feature_names = get_feature_names(preprocessor, numeric_cols, low_card_cats, high_card_cats)
X_df = pd.DataFrame(X_pre, columns=feature_names)
test_df = pd.DataFrame(test_pre, columns=feature_names)


n_sample = min(SAMPLE_FOR_OPTUNA, X_df.shape[0])
if n_sample < X_df.shape[0]:
    X_sample = X_df.sample(n_sample, random_state=RANDOM_STATE)
    y_sample = y.loc[X_sample.index]
else:
    X_sample = X_df.copy()
    y_sample = y.copy()

print("Sample size for Optuna CV:", X_sample.shape[0])


def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": N_JOBS,
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 300, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
    }
    try:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        rmses = []
        for tr_idx, val_idx in cv.split(X_sample):
            X_tr, X_val = X_sample.iloc[tr_idx], X_sample.iloc[val_idx]
            y_tr, y_val = y_sample.iloc[tr_idx], y_sample.iloc[val_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            bst = xgb.train(param, dtrain, num_boost_round=2000,
                            evals=[(dvalid, "valid")],
                            early_stopping_rounds=50, verbose_eval=False)

            # robust predict: prefer iteration_range when best_iteration exists
            best_it = getattr(bst, "best_iteration", None)
            if best_it is not None:
                # iteration_range end is exclusive, so pass best_it+1
                pred = bst.predict(dvalid, iteration_range=(0, best_it + 1))
            else:
                pred = bst.predict(dvalid)

            rmse = mean_squared_error(y_val, pred, squared=False)
            rmses.append(rmse)
        return float(np.mean(rmses))
    except Exception as e:
        # Log and re-raise so Optuna marks the trial as failed rather than returning None
        print(f"Trial error: {e}")
        raise


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best trial RMSE:", study.best_value)
print("Best params:", study.best_params)


# After finding study.best_params
best_params = study.best_params.copy()
best_params.update({
    "verbosity": 0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": N_JOBS
})

# Train/val split for early stopping
X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(X_df, y, test_size=0.1, random_state=RANDOM_STATE)
dtrain_full = xgb.DMatrix(X_tr_full, label=y_tr_full)
dval_full = xgb.DMatrix(X_val_full, label=y_val_full)

bst_final = xgb.train(best_params, dtrain_full, num_boost_round=5000,
                      evals=[(dval_full, "valid")], early_stopping_rounds=100, verbose_eval=100)

# Predict on test
dtest = xgb.DMatrix(test_df)
best_it = getattr(bst_final, "best_iteration", None)
if best_it is not None:
    test_pred = bst_final.predict(dtest, iteration_range=(0, best_it + 1))
else:
    test_pred = bst_final.predict(dtest)


sid = test["id"] if "id" in test.columns else np.arange(len(test))
submission = pd.DataFrame({"id": sid, TARGET: test_pred})
submission.to_csv("submission.csv", index=False)




