# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

warnings.filterwarnings("ignore")



train_path = "/kaggle/input/playground-series-s5e11/train.csv"
test_path = "/kaggle/input/playground-series-s5e11/test.csv"
sample_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("âœ… Data Loaded Successfully!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()



cat_features = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

print("âœ… Label Encoding Completed!")



X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']
X_test = test.drop('id', axis=1)

print(f"Feature Matrix: {X.shape}")
print(f"Test Matrix: {X_test.shape}")



def objective(trial):
    params = {
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric': 'auc',
        'tree_method': 'hist', 
        'device' : "cuda",
        'predictor': 'gpu_predictor',  
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=500
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("âœ… Optuna Optimization Completed!")
print(f"Best AUC: {study.best_value:.4f}")
print("Best Parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")



best_params = study.best_params
best_params.update({
    'n_estimators': 5000,            
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',       
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'n_jobs': -1,
    'use_label_encoder': False
})

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
best_model = None
best_fold_auc = 0.0

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), start=1):
    print(f"\nğŸ“˜ Training fold {fold} ...")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=200,
        early_stopping_rounds=100
    )
    
    preds_valid = model.predict_proba(X_valid)[:, 1]
    oof_preds[valid_idx] = preds_valid
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits
    
    auc = roc_auc_score(y_valid, preds_valid)
    print(f"Fold {fold} AUC: {auc:.4f} (Best iteration: {model.best_iteration})")
    
    # âœ… Save best model (highest AUC)
    if auc > best_fold_auc:
        best_fold_auc = auc
        best_model = model

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nğŸ�¯ Overall CV AUC: {overall_auc:.4f}")



sample['loan_paid_back'] = test_preds
sample.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as submission.csv")
sample.head()


