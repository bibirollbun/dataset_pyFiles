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
from lightgbm import LGBMClassifier

DATA_PATH = "/kaggle/input/mercor-cheating-detection/"

train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

FEATURES = [c for c in train.columns if c.startswith("feature_")]

# -----------------------------
# Build training data
# -----------------------------

# 1. Labeled data
labeled = train[train["is_cheating"].notna()].copy()

# 2. High-confidence clean negatives
hc_clean = train[(train["high_conf_clean"] == 1) & (train["is_cheating"].isna())].copy()
hc_clean["is_cheating"] = 0

# 3. Combine
train_full = pd.concat([labeled, hc_clean], axis=0)

X = train_full[FEATURES].fillna(-1)
y = train_full["is_cheating"]

X_test = test[FEATURES].fillna(-1)

# -----------------------------
# Model (Recall-heavy)
# -----------------------------
model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    class_weight={0: 1, 1: 12},  # VERY IMPORTANT
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Predict
# -----------------------------
test_probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": test_probs
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created")
submission.head()


