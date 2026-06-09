# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Package Import and Data Searching:
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Data Initialisation and Shape Determination:
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(train.shape)
print(test.shape)


# Training Data EDA:
train.head()


# Training Data Coloumns:
train.columns


# Splitting Data:
X = train.drop('diagnosed_diabetes', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = train['diagnosed_diabetes']
test_encoded = pd.get_dummies(test, drop_first=True)
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)


# Training Data:
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Data Type Check:
X.dtypes


from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import numpy as np

# ------------------
# CONFIG
# ------------------
seeds = [42, 2023]
n_splits = 3

lgb_test_preds = np.zeros(len(test_encoded))
xgb_test_preds = np.zeros(len(test_encoded))

# ------------------
# TRAINING LOOP
# ------------------
for seed in seeds:
    print(f"\nSeed {seed}")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    lgb_seed_preds = np.zeros(len(test_encoded))
    xgb_seed_preds = np.zeros(len(test_encoded))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # ------------------
        # LightGBM
        # ------------------
        lgb_model = LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.05,
            num_leaves=64,
            min_child_samples=50,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=seed,
            n_jobs=4,
            verbosity=-1
        )

        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )

        lgb_seed_preds += lgb_model.predict_proba(
            test_encoded,
            num_iteration=lgb_model.best_iteration_
        )[:, 1]

        # ------------------
        # XGBoost (FAST MODE)
        # ------------------
        xgb_model = XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=4,
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            verbosity=0
        )

        xgb_model.fit(X_tr, y_tr)

        xgb_seed_preds += xgb_model.predict_proba(test_encoded)[:, 1]

    # average folds
    lgb_seed_preds /= n_splits
    xgb_seed_preds /= n_splits

    # accumulate seeds
    lgb_test_preds += lgb_seed_preds
    xgb_test_preds += xgb_seed_preds

# average seeds
lgb_test_preds /= len(seeds)
xgb_test_preds /= len(seeds)


# Area Under Training Data ROC Curve:
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# example with ensemble on validation set
lgb_val = lgb_model.predict_proba(X_valid)[:, 1]
xgb_val = xgb_model.predict_proba(X_valid)[:, 1]

ensemble_val = 0.5 * lgb_val + 0.5 * xgb_val
auc = roc_auc_score(y_valid, ensemble_val)
print("ROC-AUC:", auc)
fpr, tpr, _ = roc_curve(y_valid, ensemble_val)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Prediction Probability Distribution Graph:
lgb_test = lgb_model.predict_proba(test_encoded)[:, 1]
xgb_test = xgb_model.predict_proba(test_encoded)[:, 1]

ensemble_test = 0.3 * lgb_test_preds + 0.7 * xgb_test_preds

plt.hist(ensemble_test, bins=50)
plt.title("Prediction Probability Distribution")
plt.show()


# Checking Ensembled Probability
ensemble_test[:10]


# Submission CSV Update:
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission['diagnosed_diabetes'] = ensemble_test
submission.to_csv('submission.csv', index=False)
submission.head()

