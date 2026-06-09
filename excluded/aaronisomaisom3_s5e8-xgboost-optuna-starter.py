!pip install optuna --quiet


# Author: Aaron Isom
# Kaggle Playground-Series-S5e8 - Binary Classification with a Bank Dataset
import pandas as pd
import numpy as np
import optuna
import warnings

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')
tune = False # Toggle for Optuna tuning and Final Submission


# Optuna objective for XGBoost
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000, step=100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, log=True)
    }
    
    model = XGBClassifier(**params, objective='binary:logistic', eval_metric='auc', random_state=42, device='cuda', n_jobs=-1,
                          enable_categorical=True, tree_method='hist')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    return cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=";")
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

original['y'] = original['y'].replace({'yes': 1, 'no': 0})

train = pd.concat([train, original], axis=0, ignore_index=True)

# Features for training (drop id and target)
X = train.drop(['id', 'y'], axis=1)
y = train['y']

# Features for test set (drop only id)
X_test = test.drop(['id'], axis=1)

# Find object columns
cat_cols = X.select_dtypes(include='object').columns

# Encode object and category columns to ensure unique values are mapped
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    le.fit(list(X[col].astype(str)) + list(X_test[col].astype(str)))
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


if tune:
    # Optuna Study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=5400, show_progress_bar=True)
    best_params = study.best_trial.params
    print('Best Parameters:', best_params)
    print('Best Trial:', study.best_trial)

else:
    best_params = {'max_depth': 8, 'learning_rate': 0.013438247465442936, 'subsample': 0.8008903067253942, 'colsample_bytree': 0.5816817925051649, 'n_estimators': 6500, 
        'reg_alpha': 0.026068275170423927, 'reg_lambda': 0.0013608054178647067}
    
# Final XGBoost model
final_model = XGBClassifier(**best_params, objective='binary:logistic', eval_metric='auc', random_state=42, device='cuda',  n_jobs=-1, 
                          enable_categorical=True, tree_method='hist')

final_model.fit(X, y)


# Final submission
preds = final_model.predict_proba(X_test)[:, 1]
submission['y'] = preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

