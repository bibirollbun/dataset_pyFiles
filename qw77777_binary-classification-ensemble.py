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


!pip install /kaggle/input/whl-packages/imbalanced_learn-0.10.0-py3-none-any.whl 


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

X = train.drop(['y', 'id'], axis=1)
y = train['y']

X_test = test.drop("id", axis=1)

categorical_vars = X.select_dtypes(include='object').columns
numerical_vars = X.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer(transformers=[
    ('one_hot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_vars),
    ('num', MinMaxScaler(), numerical_vars)
])

rf_params = {'n_estimators': 300, 'max_depth': 23, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
lgb_params = {'n_estimators': 500, 'max_depth': 23, 'learning_rate': 0.05, 'random_state': 42}
xgb_params = {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05, 'use_label_encoder': False, 'eval_metric':'logloss', 'random_state':42}

n_folds = 2
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_rf = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_rf = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rf_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('balance', RandomUnderSampler(random_state=42)),
        ('model', RandomForestClassifier(**rf_params))
    ])
    rf_pipeline.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_pipeline.predict_proba(X_val)[:,1]
    test_rf += rf_pipeline.predict_proba(X_test)[:,1] / n_folds

    lgb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('balance', RandomUnderSampler(random_state=42)),
        ('model', lgb.LGBMClassifier(**lgb_params))
    ])
    lgb_pipeline.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_pipeline.predict_proba(X_val)[:,1]
    test_lgb += lgb_pipeline.predict_proba(X_test)[:,1] / n_folds

    xgb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('balance', RandomUnderSampler(random_state=42)),
        ('model', xgb.XGBClassifier(**xgb_params))
    ])
    xgb_pipeline.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_pipeline.predict_proba(X_val)[:,1]
    test_xgb += xgb_pipeline.predict_proba(X_test)[:,1] / n_folds

X_meta_train = np.vstack([oof_rf, oof_lgb, oof_xgb]).T
X_meta_test  = np.vstack([test_rf, test_lgb, test_xgb]).T

meta_model = LogisticRegression()
meta_model.fit(X_meta_train, y)
final_proba = meta_model.predict_proba(X_meta_test)[:,1]

roc_auc = roc_auc_score(y, meta_model.predict_proba(X_meta_train)[:,1])
print("OOF ROC AUC of meta-model:", roc_auc)


submission = pd.DataFrame({
    "id": test["id"],
    "y": final_proba
})
submission.to_csv("submission.csv", index=False)


s = pd.read_csv('/kaggle/working/submission.csv')
s.head()

