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
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool

# Suppress warnings
warnings.filterwarnings('ignore')

# ====================================================
# 1. Configuration (SPEED OPTIMIZED)
# ====================================================
TARGET = 'diagnosed_diabetes'
N_FOLDS = 5   # Kept at 5 for speed
SEED = 42

print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

train = train.drop('id', axis=1, errors='ignore')
test_ids = test['id'] 
test = test.drop('id', axis=1, errors='ignore')

# ====================================================
# 2. Preprocessing & Feature Engineering
# ====================================================
def feature_engineering(df):
    df = df.copy()
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    existing_cols = [c for c in cols_with_zeros if c in df.columns]
    
    for col in existing_cols:
        df[col] = df[col].replace(0, np.nan)
        df[f'{col}_was_missing'] = df[col].isna().astype(int)

    if 'Insulin' in df.columns and 'Glucose' in df.columns:
        df['Insulin_Glucose_Ratio'] = df['Insulin'] * df['Glucose'] / 405.0
        
    if 'Age' in df.columns and 'Pregnancies' in df.columns:
        df['Pregnancies_per_Age'] = df['Pregnancies'] / (df['Age'] + 1)
        
    return df

train = feature_engineering(train)
test = feature_engineering(test)

cat_cols = [col for col in train.columns if train[col].dtype == 'object']
num_cols = [col for col in train.columns if col not in cat_cols and col != TARGET]

imputer = SimpleImputer(strategy='median')
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.transform(test[num_cols])

for col in cat_cols:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train.drop(TARGET, axis=1)
y = train[TARGET]

# ====================================================
# 3. Model Definition (FAST PARAMETERS)
# ====================================================

# LightGBM: Fast learning rate, reduced estimators
lgbm_params = {
    'n_estimators': 1000,           
    'learning_rate': 0.05,          
    'max_depth': 8,
    'num_leaves': 31,              
    'subsample': 0.8,              
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'verbose': -1
}

# XGBoost: Hist tree method for CPU speed
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,          
    'max_depth': 6,                 
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1.0,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'tree_method': 'hist' 
}

# CatBoost: Reduced depth for massive speed gain
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,          
    'depth': 6,                    
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': SEED,
    'early_stopping_rounds': 50
}

# ====================================================
# 4. Training Loop (Generating OOF Predictions)
# ====================================================
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# These arrays will store the predictions from each model to be used in the blend
oof_preds_lgbm = np.zeros(len(X))
test_preds_lgbm = np.zeros(len(test))

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(test))

oof_preds_cat = np.zeros(len(X))
test_preds_cat = np.zeros(len(test))

print(f"Starting Training on {N_FOLDS} folds...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- LightGBM ---
    model_lgbm = LGBMClassifier(**lgbm_params)
    model_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_preds_lgbm[val_idx] = model_lgbm.predict_proba(X_val)[:, 1]
    test_preds_lgbm += model_lgbm.predict_proba(test)[:, 1] / N_FOLDS
    
    # --- XGBoost ---
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50
    )
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(test)[:, 1] / N_FOLDS

    # --- CatBoost ---
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_preds_cat += model_cat.predict_proba(test)[:, 1] / N_FOLDS

    print(f"Fold {fold+1}/{N_FOLDS} complete.")

# ====================================================
# 5. Automated Blending (Stacking)
# ====================================================
print("\nPerforming Logistic Regression Blend...")

# 1. Stack the OOF predictions (Train set for Blender)
X_blend = np.column_stack((oof_preds_lgbm, oof_preds_xgb, oof_preds_cat))

# 2. Stack the Test predictions (Test set for Blender)
X_test_blend = np.column_stack((test_preds_lgbm, test_preds_xgb, test_preds_cat))

# 3. Train Meta-Model (Logistic Regression)
# We use positive=True to force positive weights (ensemble logic)
meta_model = LogisticRegression(random_state=SEED, solver='lbfgs') 
meta_model.fit(X_blend, y)

# 4. Get Coefficients (Weights)
weights = meta_model.coef_[0]
intercept = meta_model.intercept_[0]

print(f"Blend Weights -> LGBM: {weights[0]:.2f}, XGB: {weights[1]:.2f}, CAT: {weights[2]:.2f}")

# 5. Predict on full dataset to check score
blend_oof_preds = meta_model.predict_proba(X_blend)[:, 1]
blend_score = roc_auc_score(y, blend_oof_preds)

print(f"LightGBM AUC: {roc_auc_score(y, oof_preds_lgbm):.5f}")
print(f"XGBoost AUC:  {roc_auc_score(y, oof_preds_xgb):.5f}")
print(f"CatBoost AUC: {roc_auc_score(y, oof_preds_cat):.5f}")
print(f"---------------------------------------")
print(f"BLEND AUC:    {blend_score:.5f}")

# ====================================================
# 6. Submission
# ====================================================

# Use the meta model to predict on the test stack
final_predictions = meta_model.predict_proba(X_test_blend)[:, 1]

submission = pd.DataFrame({'id': test_ids, TARGET: final_predictions})
submission.to_csv('submissions.csv', index=False)
print("Submission saved successfully.")

