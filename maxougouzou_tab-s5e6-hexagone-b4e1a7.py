# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

import category_encoders as ce

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



df_train.drop(columns=["id"], inplace=True)
path_fp_1 = "/kaggle/input/db-s5e6/FertilizerPrediction1. csv"
df_train_extra_1 = pd. read_csv(path_fp_1)
path_fp_2 = "/kaggle/input/db-s5e6/FertilizerPrediction2. csv"
df_ train_extra_2 = pd. read_csv(path_fp_2)
df_train = pd.concat([df_train, df_train_extra_1, df_train_extra_2]). reset_index(drop=True)


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Lecture des donnÃ©es
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Colonnes
CAT_COLS = ["Soil Type", "Crop Type"]
NUM_COLS = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

FEATURES = NUM_COLS + CAT_COLS
LABEL = "Fertilizer Name"

# Imputation des colonnes numÃ©riques
imputer = SimpleImputer(strategy="median")
for col in NUM_COLS:
    df_train[col] = imputer.fit_transform(df_train[[col]])
    df_test[col] = imputer.transform(df_test[[col]])

# Encodage des colonnes catÃ©gorielles
for col in CAT_COLS:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Encodage de la target
label_enc = LabelEncoder()
y = label_enc.fit_transform(df_train[LABEL])

X = df_train[FEATURES]
X_test = df_test[FEATURES]

# Fonction MAP@3
def fast_mapk(y_true, y_pred, k=3):
    score = 0
    for actual, pred in zip(y_true, y_pred):
        if actual in pred:
            score += 1 / (pred.tolist().index(actual) + 1)
    return score / len(y_true)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_enc.classes_),
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=2,
        reg_lambda=2,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"  # ou "gpu_hist" si GPU dispo
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        verbose=False
    )

    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test) / skf.n_splits

    top3_val = np.argsort(oof_preds[val_idx], axis=1)[:, -3:][:, ::-1]
    score = fast_mapk(y_val, top3_val)
    fold_scores.append(score)
    print(f"ğŸ“Š Fold {fold + 1} MAP@3 score: {score:.5f}")

# Score global
top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
map3_score = fast_mapk(y, top3_oof)

print("\nğŸ“ˆ RÃ©sumÃ© des scores par fold:")
for i, score in enumerate(fold_scores):
    print(f" - Fold {i+1}: {score:.5f}")
print(f"\nğŸ“Š OOF MAP@3 score (optimisÃ©): {map3_score:.5f}")

# GÃ©nÃ©ration du fichier de soumission
top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
label_to_class = dict(zip(range(len(label_enc.classes_)), label_enc.classes_))
top3_test_labels = np.vectorize(label_to_class.get)(top3_test)

sub[LABEL] = [' '.join(row) for row in top3_test_labels]
sub.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")

