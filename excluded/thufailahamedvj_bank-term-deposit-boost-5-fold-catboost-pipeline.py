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


!pip install -q catboost optuna category_encoders scikit-learn


import pandas as pd, numpy as np, optuna, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder

warnings.filterwarnings("ignore")
SEED = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub   = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


def preprocess(df):
    df = df.copy()
    # cyclical
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
              "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    df["month_num"] = df["month"].map(months)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    # interactions
    df["bal_dur"] = df["balance"] / (df["duration"] + 1)
    df["pd_prev"] = df["pdays"] * (df["previous"] + 1)

    # clipping
    for col in ["balance","duration","campaign"]:
        lo, hi = df[col].quantile([.01, .99])
        df[col] = df[col].clip(lo, hi)
    return df

train = preprocess(train)
test  = preprocess(test)


cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
num_cols = [c for c in train.columns if c not in cat_cols + ["id","y"]]


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 800, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "cat_features": cat_cols,
        "random_seed": SEED,
        "task_type": "GPU",
        "verbose": False
    }
    
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = []
    
    for tr_idx, va_idx in cv.split(train, train["y"]):
        model = CatBoostClassifier(**params)
        model.fit(train.iloc[tr_idx][cat_cols + num_cols], train.iloc[tr_idx]["y"])
        
        preds = model.predict_proba(train.iloc[va_idx][cat_cols + num_cols])[:, 1]
        auc = roc_auc_score(train.iloc[va_idx]["y"], preds)
        aucs.append(auc)
    
    return np.mean(aucs)

# Run Optuna study
import optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Use best parameters from study
best_params = study.best_params
best_params["random_seed"] = SEED
best_params["cat_features"] = cat_cols
best_params["task_type"] = "GPU"
best_params["verbose"] = False

# Train final model
best = CatBoostClassifier(**best_params)
best.fit(train[cat_cols + num_cols], train["y"])



probs = best.predict_proba(test[cat_cols + num_cols])[:, 1]
sub["y"] = probs
sub.to_csv("submission.csv", index=False)

