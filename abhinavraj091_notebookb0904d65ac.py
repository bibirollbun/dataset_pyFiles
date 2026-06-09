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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load Data
test = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/test.csv")
train = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/train.csv")
submissions = pd.read_csv("/kaggle/input/thapar-kaggle-hack-v02/sample_submission.csv")

# Preprocessing: Handle categorical features (example if any exist)
# Assuming categorical features are present (modify as per actual data)
# categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
# numerical_cols = [col for col in train.columns if col not in categorical_cols + ['id', 'target']]

# Preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#     ])

# For simplicity, if all features are numerical:
train.drop("id", axis=1, inplace=True)
X = train.drop("target", axis=1)
y = train["target"]

# Split into original train and validation sets
X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_orig, y_train_orig)

# Hyperparameter Tuning with Cross-Validation
def tune_model(model, params, X_train, y_train, X_val, y_val):
    search = RandomizedSearchCV(
        model, params, n_iter=50, cv=StratifiedKFold(3), 
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    val_pred = best_model.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, val_pred)}")
    return best_model

# RandomForest Parameters
rf_params = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ['balanced', None]
}

# XGBoost Parameters
xgb_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

# LightGBM Parameters
lgb_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Tune Models
print("Tuning RandomForest...")
best_rf = tune_model(
    RandomForestClassifier(random_state=42),
    rf_params, X_train, y_train, X_val, y_val
)

print("\nTuning XGBoost...")
best_xgb = tune_model(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    xgb_params, X_train, y_train, X_val, y_val
)

print("\nTuning LightGBM...")
best_lgb = tune_model(
    lgb.LGBMClassifier(random_state=42),
    lgb_params, X_train, y_train, X_val, y_val
)

# Evaluate Best Model on Validation Set
models = {'RandomForest': best_rf, 'XGBoost': best_xgb, 'LightGBM': best_lgb}
best_model_name = max(models, key=lambda k: accuracy_score(y_val, models[k].predict(X_val)))
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Retrain on Full Training Data (Original + SMOTE)
X_full = pd.concat([X_train_orig, X_val])
y_full = pd.concat([y_train_orig, y_val])
X_full_res, y_full_res = smote.fit_resample(X_full, y_full)

best_model.fit(X_full_res, y_full_res)

# Generate Predictions
x_test = test.drop("id", axis=1)
test_pred = best_model.predict(x_test)

# Save Submission
submissions['target'] = test_pred
submissions.to_csv('improved_submission.csv', index=False)

