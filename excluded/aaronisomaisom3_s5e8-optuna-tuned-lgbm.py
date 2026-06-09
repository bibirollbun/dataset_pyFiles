# Install Optuna
!pip install optuna --quiet
!pip install optuna-integration[lightgbm] --quiet


# Author: Aaron Isom
# Kaggle Playground-Series-S5e8 - Binary Classification with a Bank Dataset

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback

import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import optuna

warnings.filterwarnings('ignore')
tune = False # Toggle for Optuna tuning and final submission
run_cv = True # Toggle for final model validation using best parameters


# Optuna Tuning
def objective(trial):
  
    params = {
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 32, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_bin': trial.suggest_int('max_bin', 64, 1023)
    }
      
    #scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
    model = lgb.LGBMClassifier(**params, objective='binary', metric='auc', is_unbalance=True, random_state=42, verbosity=-1)
    # scale_pos_weight=scale_pos_weight) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=";")
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

original['y'] = original['y'].map({'yes': 1, 'no': 0})

train = pd.concat([train, original], axis=0, ignore_index=True)

# Features for training - drop id and target
X = train.drop(['id', 'y'], axis=1)
y = train['y']

# Features for test set drop id
X_test = test.drop(['id'], axis=1)

# Encode object and category columns to ensure unique values are mapped
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    le.fit(list(X[col].astype(str)) + list(X_test[col].astype(str)))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# Cross-validated LGBMClassifier
if tune:
    # Optuna Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, timeout=5400, show_progress_bar=True)
    best_params = study.best_trial.params
    print('Best Parameters:', best_params)
    print('Best Trial:', study.best_trial)

else:
    best_params = {"n_estimators": 10000, "learning_rate": 0.05, "min_child_samples": 10,
                   "subsample": 0.8, "colsample_bytree": 0.7, "num_leaves": 64, "max_depth": 9,
                  "max_bin": 4500, "reg_alpha": 0.8, "reg_lambda": 1.5}

clf = lgb.LGBMClassifier(**best_params, objective='binary', metric='auc', is_unbalance=True, random_state=42, verbosity=-1)
    
clf.fit(X, y)

if run_cv:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    
    print("ROC AUC scores (CV):", cv_scores)
    print("Mean ROC AUC:", np.mean(cv_scores))

preds = clf.predict_proba(X_test)[:, 1] 


# Final submission
submission['y'] = preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

