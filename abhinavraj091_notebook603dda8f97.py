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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna

# Load datasets
train = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv")
test = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv")

# Feature and target separation
if 'id' in train.columns:
    X = train.drop(columns=["target", "id"])
else:
    X = train.drop(columns=["target"])
y = train["target"]
X_test = test.drop(columns=["id"])
test_ids = test["id"]

# Feature Engineering
X['feature_sum'] = X.sum(axis=1)
X['feature_mean'] = X.mean(axis=1)
X['feature_std'] = X.std(axis=1)
X_test['feature_sum'] = X_test.sum(axis=1)
X_test['feature_mean'] = X_test.mean(axis=1)
X_test['feature_std'] = X_test.std(axis=1)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(X_test)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Hyperparameter Tuning Function
def tune_model(trial, model_type):
    if model_type == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        model = XGBRegressor(**params, random_state=42)
    elif model_type == 'cat':
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        }
        model = CatBoostRegressor(**params, verbose=0, random_state=42)
    elif model_type == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        }
        model = LGBMRegressor(**params, random_state=42)
    model.fit(X_scaled[:1000], y.iloc[:1000])  # Use a subset for faster tuning
    preds = model.predict(X_scaled[1000:1200])
    return r2_score(y.iloc[1000:1200], preds)

# Optimize each model
xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(lambda trial: tune_model(trial, 'xgb'), n_trials=50)
xgb_best_params = xgb_study.best_params

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(lambda trial: tune_model(trial, 'cat'), n_trials=50)
cat_best_params = cat_study.best_params

lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(lambda trial: tune_model(trial, 'lgbm'), n_trials=50)
lgbm_best_params = lgbm_study.best_params

# Create models with tuned parameters
xgb_model = XGBRegressor(**xgb_best_params, random_state=42)
cat_model = CatBoostRegressor(**cat_best_params, random_state=42)
lgbm_model = LGBMRegressor(**lgbm_best_params, random_state=42)
gbr_model = GradientBoostingRegressor(random_state=42)
etr_model = ExtraTreesRegressor(random_state=42)

# Stacking Regressor
meta_model = Ridge(alpha=1.0, random_state=42)
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgbm', lgbm_model),
        ('gbr', gbr_model),
        ('etr', etr_model)
    ],
    final_estimator=meta_model
)

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_preds = np.zeros(len(X_test))

for train_idx, valid_idx in kf.split(X_scaled):
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    stacking_model.fit(X_train, y_train)
    final_preds += stacking_model.predict(X_test_scaled) / kf.n_splits

# Prepare the submission
test['target'] = final_preds
submission = test[['id', 'target']]
submission.to_csv("submission_optimized.csv", index=False)
print("Optimized submission file saved as 'submission_optimized.csv'")


