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


import os, gc, random, warnings, itertools, typing as tp
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")


SEED = 42
N_FOLDS = 5
DATA_DIR = "/kaggle/input/playground-series-s5e7"

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
seed_everything()


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

TARGET = "Personality"
IDCOL  = "id"

X = train.drop(columns=[TARGET])
y = train[TARGET].map({"Extrovert":1, "Introvert":0})
test_ids = test[IDCOL]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("cat_cols:", len(cat_cols), "num_cols:", len(num_cols))


import itertools
import numpy as np

for col in X.columns:
    X[f"{col}_null"] = X[col].isnull().astype(int)
    test[f"{col}_null"] = test[col].isnull().astype(int)

num_top = num_cols[:10]
for a, b in itertools.combinations(num_top, 2):
    X[f"{a}_{b}_prod"] = X[a] * X[b]
    test[f"{a}_{b}_prod"] = test[a] * test[b]

print("追加後 shape:", X.shape, test.shape)


from sklearn.preprocessing import LabelEncoder

lbl_dict: dict[str, LabelEncoder] = {}
for col in cat_cols:
    le = LabelEncoder()

    le.fit(pd.concat([X[col], test[col]], axis=0))

    X[col]    = le.transform(X[col])
    test[col] = le.transform(test[col])
    lbl_dict[col] = le


def cat_objective(trial):
    params = {
        "iterations": 1500,
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.07, log=True),
        "l2_leaf_reg": trial.suggest_float("l2", 1.0, 10.0, log=True),
        "random_seed": SEED,
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "verbose": False,
    }
    skf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for tr_idx, va_idx in skf.split(X, y):
        model = cb.CatBoostClassifier(**params, cat_features=cat_idx)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
                  use_best_model=True)
        preds = model.predict(X.iloc[va_idx])
        accs.append(accuracy_score(y.iloc[va_idx], preds))
    return np.mean(accs)

def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "seed": SEED,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("leaves", 16, 64),
        "min_data_in_leaf": trial.suggest_int("min_data", 10, 100),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
    }

    skf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    accs = []

    for tr_idx, va_idx in skf.split(X, y):
        lgb_tr = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx],
                             categorical_feature=cat_cols, free_raw_data=False)
        lgb_va = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx],
                             categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=3000,
            valid_sets=[lgb_va],
            callbacks=[

                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ],
        )

        preds = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration) > 0.5
        accs.append(accuracy_score(y.iloc[va_idx], preds))

    return np.mean(accs)

def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("depth", 3, 8),
        "subsample": trial.suggest_float("sub", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.6, 1.0),
        "lambda": trial.suggest_float("l2", 1e-3, 10.0, log=True),
        "alpha":  trial.suggest_float("l1", 1e-3, 10.0, log=True),
        "seed": SEED,
        "verbosity": 0,
    }
    skf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, va_idx in skf.split(X, y):
        dtr = xgb.DMatrix(X.iloc[tr_idx], y.iloc[tr_idx])
        dva = xgb.DMatrix(X.iloc[va_idx], y.iloc[va_idx])
        model = xgb.train(params, dtr, num_boost_round=3000,
                          evals=[(dva, "val")],
                          early_stopping_rounds=200, verbose_eval=False)
        preds = model.predict(dva) > 0.5
        accs.append(accuracy_score(y.iloc[va_idx], preds))
    return np.mean(accs)

print("Optuna tuning (各モデル 26 trials)")
for name, obj in [("Cat", cat_objective), ("LGBM", lgb_objective), ("XGB", xgb_objective)]:
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=26, show_progress_bar=False)
    globals()[f"{name}_BEST"] = study.best_params
    print(f"{name} best {study.best_value:.4f} →", study.best_params)


Cat_BEST.update(dict(iterations=2000, loss_function="Logloss", eval_metric="Accuracy",
                     random_seed=SEED, verbose=False))

LGBM_BEST.update(dict(objective="binary", metric="binary_error", seed=SEED, verbosity=-1))

XGB_BEST.update(dict(objective="binary:logistic", eval_metric="logloss", seed=SEED, verbosity=0))


Cat_BEST_FIXED = Cat_BEST.copy()
Cat_BEST_FIXED["border_count"] = 255
Cat_BEST_FIXED["learning_rate"] = Cat_BEST_FIXED.pop("lr")
Cat_BEST_FIXED["l2_leaf_reg"]   = Cat_BEST_FIXED.pop("l2")
Cat_BEST_FIXED.update(
    dict(iterations=2000,
         loss_function="Logloss",
         eval_metric="Accuracy",
         random_seed=SEED,
         verbose=False)
)

LGBM_BEST_FIXED = LGBM_BEST.copy()
LGBM_BEST_FIXED["learning_rate"]    = LGBM_BEST_FIXED.pop("lr")
LGBM_BEST_FIXED["num_leaves"]       = LGBM_BEST_FIXED.pop("leaves")
LGBM_BEST_FIXED["min_data_in_leaf"] = LGBM_BEST_FIXED.pop("min_data")
LGBM_BEST_FIXED["feature_fraction"] = LGBM_BEST_FIXED.pop("ff")
LGBM_BEST_FIXED.update(
    dict(objective="binary",
         metric="binary_error",
         seed=SEED,
         verbosity=-1)
)

XGB_BEST_FIXED = XGB_BEST.copy()
XGB_BEST_FIXED["eta"]        = XGB_BEST_FIXED.pop("lr")
XGB_BEST_FIXED["max_depth"]  = XGB_BEST_FIXED.pop("depth")
XGB_BEST_FIXED["lambda"]     = XGB_BEST_FIXED.pop("l2")
XGB_BEST_FIXED["alpha"]      = XGB_BEST_FIXED.pop("l1")
XGB_BEST_FIXED.update(
    dict(objective="binary:logistic",
         eval_metric="logloss",
         seed=SEED,
         verbosity=0)
)



oof_preds  = np.zeros((len(X), 3))
test_preds = np.zeros((len(test), 3))

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")
    
    m_cat = cb.CatBoostClassifier(**Cat_BEST_FIXED, cat_features=cat_idx)
    m_cat.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof_preds[va_idx, 0] = m_cat.predict_proba(X.iloc[va_idx])[:, 1]
    test_preds[:, 0]    += m_cat.predict_proba(test)[:, 1] / N_FOLDS
    
    tr_ds = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx],
                        categorical_feature=cat_cols, free_raw_data=False)
    va_ds = lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx],
                        categorical_feature=cat_cols, free_raw_data=False)
    m_lgb = lgb.train(
        LGBM_BEST_FIXED, tr_ds, num_boost_round=3000,
        valid_sets=[va_ds],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    oof_preds[va_idx, 1] = m_lgb.predict(X.iloc[va_idx], num_iteration=m_lgb.best_iteration)
    test_preds[:, 1]    += m_lgb.predict(test, num_iteration=m_lgb.best_iteration) / N_FOLDS

    dtr = xgb.DMatrix(X.iloc[tr_idx], y.iloc[tr_idx])
    dva = xgb.DMatrix(X.iloc[va_idx], y.iloc[va_idx])
    dvs = xgb.DMatrix(test)
    m_xgb = xgb.train(
        XGB_BEST_FIXED, dtr, num_boost_round=4000,
        evals=[(dva, "val")],
        early_stopping_rounds=300, verbose_eval=False
    )
    oof_preds[va_idx, 2] = m_xgb.predict(dva)
    test_preds[:, 2]    += m_xgb.predict(dvs) / N_FOLDS
    
    del m_cat, m_lgb, m_xgb, tr_ds, va_ds, dtr, dva, dvs
    gc.collect()

print("Base model OOF Accuracy:",
      accuracy_score(y, (oof_preds.mean(1) > 0.5).astype(int)).round(4))



stacker = LogisticRegression(max_iter=1000, random_state=SEED)
stacker.fit(oof_preds, y)
meta_oof = stacker.predict_proba(oof_preds)[:,1]
META_ACC = accuracy_score(y, (meta_oof>0.5).astype(int))
print("Stacked OOF Accuracy:", META_ACC.round(4))

meta_test = stacker.predict_proba(test_preds)[:,1]



import optuna
from sklearn.metrics import accuracy_score

def weight_objective(trial):
    w1 = trial.suggest_float("w1", 0.2, 0.6)
    w2 = trial.suggest_float("w2", 0.2, 0.6)
    w3 = 1.0 - w1 - w2
    if w3 < 0:
        raise optuna.exceptions.TrialPruned()
    blend = oof_preds @ np.array([w1, w2, w3])
    return accuracy_score(y, (blend > 0.5).astype(int))

study_w = optuna.create_study(direction="maximize")
study_w.optimize(weight_objective, n_trials=78, show_progress_bar=False)
w1, w2 = study_w.best_params["w1"], study_w.best_params["w2"]
w3 = 1 - w1 - w2
print("best weights:", round(w1,3), round(w2,3), round(w3,3))

oof_blend = oof_preds @ np.array([w1, w2, w3])



best_thr, best_acc = 0.5, 0
for th in np.arange(0.47, 0.51, 0.0001):
    acc = accuracy_score(y, (oof_blend > th).astype(int))
    if acc > best_acc:
        best_thr, best_acc = th, acc
print("best_th:", round(best_thr,4), "OOF_acc:", round(best_acc,5))



test_blend = test_preds @ np.array([w1, w2, w3])
submission = pd.DataFrame({
    IDCOL: test_ids,
    TARGET: np.where(test_blend > best_thr, "Extrovert", "Introvert")
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved :", submission.shape)


