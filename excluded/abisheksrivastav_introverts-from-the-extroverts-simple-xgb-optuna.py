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


# Personality Classification - XGBoost Only with StratifiedKFold (GPU + Optuna)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Encode target
target_map = {'Introvert': 0, 'Extrovert': 1}
train['Personality'] = train['Personality'].map(target_map)

# Combine train & test for consistent preprocessing
train['is_test'] = 0
test['is_test'] = 1
test['Personality'] = np.nan
data = pd.concat([train, test], axis=0)

# Fill missing values (mode)
data.fillna(data.mode().iloc[0], inplace=True)

# Convert all numericals to categorical (via binning)
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.difference(['id', 'Personality', 'is_test'])
for col in numeric_cols:
    data[col] = pd.qcut(data[col], q=10, duplicates='drop').astype(str)

# Label encode all features
features = data.columns.difference(['id', 'Personality', 'is_test'])
le = LabelEncoder()
for col in features:
    data[col] = le.fit_transform(data[col].astype(str))

# Split back
train = data[data['is_test'] == 0].drop(columns='is_test')
test = data[data['is_test'] == 1].drop(columns=['is_test', 'Personality'])

X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns='id')

###############################
# OPTUNA TUNING FOR XGBOOST  #
###############################

def objective_xgb(trial):
    params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "lambda": trial.suggest_loguniform('lambda', 1e-3, 10),
        "alpha": trial.suggest_loguniform('alpha', 1e-3, 10),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1),
        "subsample": trial.suggest_float('subsample', 0.5, 1),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3),
        "max_depth": trial.suggest_int('max_depth', 3, 10),
        "n_estimators": 1000,
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "valid")],
                          early_stopping_rounds=50,
                          verbose_eval=False)
        preds = model.predict(dval) > 0.5
        scores.append(accuracy_score(y.iloc[val_idx], preds))
    return np.mean(scores)

# Run Optuna
print("Running Optuna for XGBoost...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=20)
params = study_xgb.best_params
params.update({"verbosity": 0, "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "gpu_hist", "predictor": "gpu_predictor"})

#############################
# Stratified KFold Training #
#############################

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=[(dval, "valid")],
                      early_stopping_rounds=10, verbose_eval=False)

    oof_preds[val_idx] = model.predict(dval) > 0.5
    test_preds += model.predict(dtest) / skf.n_splits

# Evaluate
cv_acc = accuracy_score(y, oof_preds)
print(f"Cross-Validation Accuracy: {cv_acc:.4f}")

# Create submission
final_preds = (test_preds > 0.5).astype(int)
submission["Personality"] = ["Extrovert" if p == 1 else "Introvert" for p in final_preds]
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


