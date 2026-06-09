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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# TARGET = "diagnosed_diabetes"
test_id = test["id"]



# def feature_engineering(df):
#     df = df.copy()

#     df["age_bmi"] = df["age"] * df["bmi"]
#     df["age_waist"] = df["age"] * df["waist_to_hip_ratio"]
#     df["bmi_waist"] = df["bmi"] * df["waist_to_hip_ratio"]

#     df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1)
#     df["ldl_hdl_ratio"]  = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)

#     df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)

#     df["sedentary"] = (df["physical_activity_minutes_per_week"] < 150).astype(int)

#     return df



# train = feature_engineering(train)
# test  = feature_engineering(test)



TARGET = "diagnosed_diabetes"
test_ids = test["id"]
X = train.drop(columns=[TARGET])
y = train[TARGET]
test_X = test.copy()

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
cat_features = [X.columns.get_loc(col) for col in cat_cols]



cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 5000,          # FAST enough
    "learning_rate": 0.04,
    "depth": 9,
    "l2_leaf_reg": 10,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.2,
    "min_data_in_leaf": 50,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": 0,
    "task_type": "GPU"           # GPU optional
}



SEEDS = [42, 52, 62]
N_SPLITS = 5

test_preds = np.zeros(len(test))

for seed in SEEDS:
    print(f"\nTraining seed: {seed}")
    
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    seed_test_preds = np.zeros(len(test))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            **cat_params,
            random_seed=seed
        )

        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            use_best_model=True
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(auc)

        seed_test_preds += model.predict_proba(test)[:, 1] / N_SPLITS

    print(f"Seed {seed} mean AUC:", np.mean(fold_scores))
    test_preds += seed_test_preds / len(SEEDS)



submission = pd.DataFrame({
    "id": test_id,
    TARGET: test_preds
})

submission.to_csv("catboost_strategy1_seed_avg.csv", index=False)
submission.head()


