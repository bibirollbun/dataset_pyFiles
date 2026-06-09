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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
# load train and test data 
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
# Combine train and test for feature engineering
train['is_train'] = 1
test['is_train'] = 0
test['y'] = -1  # Dummy target for uniform processing
data = pd.concat([train, test], axis=0)

# ---------------- Feature Engineering ---------------- #
def feature_engineering(df):
    df['balance_to_duration'] = df['balance'] / (df['duration'] + 1)
    df['duration_per_day'] = df['duration'] / (df['day'] + 1)
    df['campaign_per_previous'] = df['campaign'] / (df['previous'] + 1)
    df['pdays_missing'] = (df['pdays'] == -1).astype(int)
    df['contact_month'] = df['contact'] + "_" + df['month']
    return df

data = feature_engineering(data)

# ---------------- Label Encoding ---------------- #
cat_cols = data.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split back to train and test
train = data[data['is_train'] == 1].drop(columns=['is_train'])
test = data[data['is_train'] == 0].drop(columns=['is_train', 'y'])
X = train.drop(columns=['y'])
y = train['y']
X_test = test.copy()

# ---------------- Tri-Ensemble ---------------- #
N_SPLITS = 5
SEED = 42
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds_lgb = np.zeros(X.shape[0])
oof_preds_xgb = np.zeros(X.shape[0])
oof_preds_cat = np.zeros(X.shape[0])
test_preds_lgb = np.zeros(X_test.shape[0])
test_preds_xgb = np.zeros(X_test.shape[0])
test_preds_cat = np.zeros(X_test.shape[0])

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt", n_estimators=2000, learning_rate=0.06,
        num_leaves=100, max_depth=10, min_child_samples=9, subsample=0.8,
        colsample_bytree=0.5, reg_alpha=0.79, reg_lambda=3.0, max_bin=4523,
        random_state=SEED, verbosity=-1, n_jobs=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    oof_preds_lgb[valid_idx] = lgb_model.predict_proba(X_valid)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", use_label_encoder=False,
        n_estimators=2000, learning_rate=0.06, max_depth=10, subsample=0.8,
        colsample_bytree=0.5, reg_alpha=0.8, reg_lambda=3.0, tree_method="hist",
        random_state=SEED, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, early_stopping_rounds=100,
                  eval_set=[(X_valid, y_valid)], verbose=100)
    oof_preds_xgb[valid_idx] = xgb_model.predict_proba(X_valid)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=2000, learning_rate=0.06, depth=10, eval_metric="AUC",
        random_seed=SEED, verbose=100, early_stopping_rounds=100, task_type="CPU"
    )
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    oof_preds_cat[valid_idx] = cat_model.predict_proba(X_valid)[:, 1]
    test_preds_cat += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS

# ---------------- Final Ensemble ---------------- #
oof_ensemble = (oof_preds_lgb + oof_preds_xgb + oof_preds_cat) / 3
test_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3
final_score = roc_auc_score(y, oof_ensemble)
submission['y'] = test_ensemble
submission_path = "/mnt/data/tri_ensemble_submission.csv"
submission.to_csv(submission_path, index=False)

final_score, submission_path


