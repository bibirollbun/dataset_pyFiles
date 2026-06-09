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


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from lightgbm.callback import early_stopping



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm.callback import early_stopping

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"
test_ids = test_df["id"]

# =========================
# FEATURE SELECTION (PROVEN)
# =========================
cols_drop = [
    "id",
    "alcohol_consumption_per_week",
    "sleep_hours_per_day",
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

X = train_df.drop(cols_drop + [TARGET], axis=1)
y = train_df[TARGET]
X_test = test_df.drop(cols_drop, axis=1)

# =========================
# CLASS IMBALANCE
# =========================
pos = y.sum()
neg = len(y) - pos
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)

# =========================
# OOF TRAINING
# =========================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=42 + fold,
        n_jobs=-1
    )

    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[val], y.iloc[val])],
        eval_metric="auc",
        callbacks=[early_stopping(200, verbose=False)]
    )

    oof[val] = model.predict_proba(X.iloc[val])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# =========================
# OOF SCORE
# =========================
print("\nOOF AUC:", roc_auc_score(y, oof))

# =========================
# SUBMISSION
# =========================
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission saved")


