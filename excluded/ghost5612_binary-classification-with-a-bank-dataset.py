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
import gc
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')



ID_COL = 'id'
TARGET = 'class' if 'class' in train.columns else 'target' if 'target' in train.columns else 'y'
SEED = 42
FOLDS = 5


test_ids = test[ID_COL]
train.drop(columns=[ID_COL], inplace=True)
test.drop(columns=[ID_COL], inplace=True)


initial_features = train.select_dtypes(include=['number']).columns.tolist()
if TARGET in initial_features:
    initial_features.remove(TARGET)


def add_basic_features(df, feature_cols):
    df = df.copy()
    df["row_sum"] = df[feature_cols].sum(axis=1)
    df["row_mean"] = df[feature_cols].mean(axis=1)
    df["row_min"] = df[feature_cols].min(axis=1)
    df["row_max"] = df[feature_cols].max(axis=1)
    df["row_std"] = df[feature_cols].std(axis=1)
    return df

train = add_basic_features(train, initial_features)
test = add_basic_features(test, initial_features)

# Final feature list
features = train.select_dtypes(include=['number']).columns.tolist()
if TARGET in features:
    features.remove(TARGET)


oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
preds_lgb = np.zeros(len(test))
preds_xgb = np.zeros(len(test))
preds_cat = np.zeros(len(test))
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


import gc
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

oof_lgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(train[features], train[TARGET])):
    print(f"Fold {fold + 1}")

    X_train, y_train = train.iloc[train_idx][features], train.iloc[train_idx][TARGET]
    X_val, y_val = train.iloc[val_idx][features], train.iloc[val_idx][TARGET]

    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.01,
        n_estimators=1000,
        random_state=SEED + fold,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(100),
            log_evaluation(100)
        ]
    )

    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_preds = model.predict_proba(test[features])[:, 1]
    preds_lgb += fold_preds / FOLDS

    del model, X_train, y_train, X_val, y_val, fold_preds
    gc.collect()

    break

print(f"LGB OOF AUC: {roc_auc_score(train[TARGET], oof_lgb):.5f}")



submission = pd.DataFrame({
    "target": preds_lgb
})
submission.to_csv("submission.csv", index=False)
submission.head()



sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission = sample_submission.copy()
submission['y'] = preds_lgb  # Make sure preds_lgb is the right length & order
submission.to_csv('submission.csv', index=False)
submission.head()




