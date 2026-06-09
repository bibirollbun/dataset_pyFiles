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


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

# Model evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Encoding & preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train


original


# Concat
train_combined = pd.concat([train, original], ignore_index=True)

# Checking
train_combined


train_combined.info()


missing = train_combined.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_df = pd.DataFrame({
    'Missing Count': missing,
})
print(missing_df)



personality_le = LabelEncoder()
train_combined["target"] = personality_le.fit_transform(train_combined["Personality"])

cat_cols = ["Stage_fear", "Drained_after_socializing"]
for col in cat_cols:
    train_combined[col] = LabelEncoder().fit_transform(train_combined[col].astype(str))




X = train_combined.drop(columns=["id", "Personality", "target"])
y = train_combined["target"]

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
scores = []

best_xgb_params = {
    "n_estimators": 148,
    "max_depth": 10,
    "learning_rate": 0.10679585520157442,
    "subsample": 0.8749220840926396,
    "colsample_bytree": 0.7714642846061028,
    "gamma": 2.8856530062239614,
    "reg_alpha": 0.8503419801725314,
    "reg_lambda": 2.8082541634976272,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "mlogloss"
}

best_lgb_params = {
    "n_estimators": 195,
    "max_depth": 7,
    "learning_rate": 0.25597221473758025,
    "subsample": 0.6392495522418413,
    "colsample_bytree": 0.6483305935794208,
    "num_leaves": 68,
    "reg_alpha": 4.326613265620098,
    "reg_lambda": 1.3707558728103124,
    "random_state": 42,
    "n_jobs": -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    proba_xgb = xgb_model.predict_proba(X_val)

    # LightGBM
    lgb_model = LGBMClassifier(**best_lgb_params, verbosity=-1)
    lgb_model.fit(X_train, y_train)
    proba_lgb = lgb_model.predict_proba(X_val)

    # Soft Voting
    avg_proba = (proba_xgb + proba_lgb) / 2
    final_preds = np.argmax(avg_proba, axis=1)

    acc = accuracy_score(y_val, final_preds)
    scores.append(acc)
    print(f"Fold {fold+1} Weighted Soft Voting Accuracy: {acc:.4f}")

print(f"\nAverage Weighted Soft Voting CV Accuracy: {np.mean(scores):.4f}")



for col in ["Stage_fear", "Drained_after_socializing"]:
    test[col] = test[col].astype(str)  # NaN'leri de string yap
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])




X_full = train_combined.drop(columns=["id", "Personality", "target"])
y_full = train_combined["target"]
X_test = test.drop(columns=["id"])


xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_full, y_full)
proba_xgb = xgb_model.predict_proba(X_test)

lgb_model = LGBMClassifier(**best_lgb_params, verbosity=-1)
lgb_model.fit(X_full, y_full)
proba_lgb = lgb_model.predict_proba(X_test)


avg_proba = (proba_xgb + proba_lgb) / 2
final_preds = np.argmax(avg_proba, axis=1)


submission_preds = personality_le.inverse_transform(final_preds)
submission["Personality"] = submission_preds
submission.to_csv("submission.csv", index=False)

submission_check = pd.read_csv("submission.csv")

print(" First 5 rows:")
print(submission_check.head())

print("\n Last 5 rows:")
print(submission_check.tail())



