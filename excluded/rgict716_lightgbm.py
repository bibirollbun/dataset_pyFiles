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


# ================== Imports ==================
import os
import numpy as np
import pandas as pd
import optuna
import wandb
wandb.login(key="06579f8bdd9dda339e9e91dcd130fd8f44c6d5ae")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ================== Load Data ==================
train1 = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test1 = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")
train2 = pd.read_csv("/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv")

# ================== Prepare IDs ==================
test1_ids = test1["id"]

# ================== Basic Cleaning ==================
train1.columns = train1.columns.str.replace(' ', '_')
test1.columns = test1.columns.str.replace(' ', '_')
train2.columns = train2.columns.str.replace(' ', '_')

missing_cols = set(train1.columns) - set(train2.columns) - {'id', 'smoking', 'is_train'}
for col in missing_cols:
    train2[col] = np.nan

train2 = train2[train1.drop(['id', 'smoking'], axis=1).columns.tolist() + ['smoking']]
train1["is_train"] = 1
test1["is_train"] = 0
train2["is_train"] = 1
test1["smoking"] = -1

# ================== Concatenate Datasets ==================
full_df = pd.concat([train1, train2, test1], axis=0, ignore_index=True)

# ================== Feature Engineering ==================
full_df['dental_caries_sq'] = full_df['dental_caries'] ** 2
full_df['weight(kg)_sq'] = full_df['weight(kg)'] ** 2
full_df['weightxheight'] = full_df['weight(kg)'] * full_df['height(cm)']
full_df['ALT_sq'] = full_df['ALT'] ** 2
full_df['hg_height'] = full_df['hemoglobin'] * full_df['height(cm)']
full_df['Gtp_sq'] = full_df['Gtp'] ** 2

full_df.drop(['hearing(left)', 'hearing(right)', 'Urine_protein'], axis=1, inplace=True)

# ================== Re-Split Data ==================
train_df = full_df[full_df["is_train"] == 1].drop(["id", "is_train"], axis=1)
test_df = full_df[full_df["is_train"] == 0].drop(["id", "is_train", "smoking"], axis=1)

X = train_df.drop("smoking", axis=1)
y = train_df["smoking"]

# ================== Train Final Model with Best Params ==================
final_params = {
    'n_estimators': 10000,
    'learning_rate': 0.07144789619704596,
    'max_depth': 6,
    'subsample': 0.9283108972367735,
    'min_child_samples': 73,
    'colsample_bytree': 0.8704589331404177,
    'random_state': 42
    # GPU params removed for Kaggle
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))
auc_scores = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**final_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(100), log_evaluation(0)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test_df)[:, 1] / kf.n_splits

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    auc_scores.append(fold_auc)
    print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

print(f"\nMean AUC: {np.mean(auc_scores):.5f}")

# ================== Save Submission ==================
submission = pd.DataFrame({
    "id": test1_ids,
    "smoking": test_preds
})
submission.to_csv("smoking_prediction_submission_lightgbm_cpu.csv", index=False)
print("Submission file saved successfully!")


