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
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# Separate target
y_train = df_train['diagnosed_diabetes'].values
X_train = df_train.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = df_test.drop(['id'], axis=1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# Identify categorical columns
categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]


print(f"\nCategorical columns: {categorical_cols}")
print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"\nNumerical columns: {numerical_cols}")

target = df_train['diagnosed_diabetes']
X_raw = df_train.drop(['diagnosed_diabetes', 'id'], axis=1, errors='ignore')
X_test_raw = df_test.drop(['id'], axis=1, errors='ignore')

# --- A) Create Label Encoded Version (For XGBoost & LightGBM) ---
print("Creating Label Encoded dataset for XGB/LGBM...")
X_le = X_raw.copy()
X_test_le = X_test_raw.copy()

for col in X_le.columns:
    if col in categorical_cols or X_le[col].dtype == 'object':
        le = LabelEncoder()
        # Handle NaNs before encoding to prevent crash
        X_le[col] = X_le[col].fillna("MISSING").astype(str)
        X_test_le[col] = X_test_le[col].fillna("MISSING").astype(str)
        
        # Fit on combined to cover all categories
        full_data = pd.concat([X_le[col], X_test_le[col]])
        le.fit(full_data)
        X_le[col] = le.transform(X_le[col])
        X_test_le[col] = le.transform(X_test_le[col])

# --- B) Create Native Categorical Version (For CatBoost) ---
print("Creating Native Categorical dataset for CatBoost...")
X_cat = X_raw.copy()
X_test_cat = X_test_raw.copy()

# CatBoost likes NaNs in categories to be filled with a string
for col in categorical_cols:
    if col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("Missing").astype(str)


xgb_params ={
    'n_estimators': 10000, # We control this via early stopping
    'early_stopping_rounds': 50,
    'booster': 'gbtree',
    'tree_method': 'hist',     # Fast training
    'eval_metric': 'logloss',
    'learning_rate': 0.010586281318793418, 
    'max_depth': 5, 
    'subsample': 0.9419910623833896, 
    'colsample_bytree': 0.5244058847875112, 
    'min_child_weight': 7, 
    'reg_alpha': 0.00015151084454479046, 
    'reg_lambda': 2.161158791085214e-08, 
    'gamma': 2.240078485583776e-07}


lgb_params = {
    'n_estimators': 10000,
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.04151567000333162, 
    'num_leaves': 93, 'max_depth': 3, 
    'min_child_samples': 97, 
    'subsample': 0.8336810469662667, 
    'colsample_bytree': 0.5021699121748862, 
    'reg_alpha': 0.015640727219830758, 
    'reg_lambda': 1.374990603296636e-06
}



cat_params = {'iterations': 5000,
    'eval_metric': 'AUC',
    'verbose': 0,
    'task_type': 'CPU', # Change to 'GPU' if available
    'cat_features': [c for c in categorical_cols if c in X_cat.columns],
 'learning_rate': 0.08141363864155182, 
 'depth': 4, 
 'l2_leaf_reg': 2.721242066354407, 
 'random_strength': 0.3197413721687479, 
 'subsample': 0.8585190651619243}


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF (Out of Fold) Predictions for CV scoring
oof_xgb = np.zeros(len(X_le))
oof_lgb = np.zeros(len(X_le))
oof_cat = np.zeros(len(X_cat))

# Test Predictions
pred_xgb = np.zeros(len(X_test_le))
pred_lgb = np.zeros(len(X_test_le))
pred_cat = np.zeros(len(X_test_cat))

print(f"\n{'='*20} Starting Cross-Validation {'='*20}")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_le, target)):
    
    # --- XGBoost & LightGBM (Use Label Encoded Data) ---
    X_tr_le, X_val_le = X_le.iloc[train_idx], X_le.iloc[val_idx]
    y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_le)[:, 1]
    pred_xgb += model_xgb.predict_proba(X_test_le)[:, 1] / 5
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val_le)[:, 1]
    pred_lgb += model_lgb.predict_proba(X_test_le)[:, 1] / 5
    
    # --- CatBoost (Use Native Data) ---
    X_tr_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
    pred_cat += model_cat.predict_proba(X_test_cat)[:, 1] / 5
    
    print(f"Fold {fold+1} done.")





# 1. Single Model Scores
auc_xgb = roc_auc_score(target, oof_xgb)
auc_lgb = roc_auc_score(target, oof_lgb)
auc_cat = roc_auc_score(target, oof_cat)

# 2. Probability Blend (Average)
oof_blend_prob = (oof_xgb + oof_lgb + oof_cat) / 3
auc_blend_prob = roc_auc_score(target, oof_blend_prob)

# 3. Rank Blend (Average of Ranks) - FIXED
# Use pct=True for both OOF and test, and divide by 3 for both
oof_blend_rank = (pd.Series(oof_xgb).rank(pct=True) + 
                  pd.Series(oof_lgb).rank(pct=True) + 
                  pd.Series(oof_cat).rank(pct=True)) / 3
auc_blend_rank = roc_auc_score(target, oof_blend_rank)

from scipy.optimize import minimize

#  4. OPTIMIZED WEIGHTED BLEND (The Scientific Method) ---
print(f"\n{'='*20} FINDING OPTIMAL BLEND WEIGHTS {'='*20}")

# Define the function we want to minimize (Negative AUC)
def minimize_auc(weights):
    # Normalize weights so they sum to 1.0
    # (We use absolute values to prevent negative weights, just in case)
    w = np.abs(weights)
    w = w / w.sum()
    
    # Create the blended OOF prediction
    final_oof = (w[0] * oof_xgb + 
                 w[1] * oof_lgb + 
                 w[2] * oof_cat)
    
    # Return negative AUC (because minimize() tries to make numbers smaller)
    return -roc_auc_score(target, final_oof)

# Starting guess: Equal weights [0.33, 0.33, 0.33]
initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1), (0, 1), (0, 1)] # Weights must be between 0 and 1

# Run the mathematical optimization
result = minimize(minimize_auc, initial_weights, bounds=bounds, method='SLSQP')

# Extract the best weights
opt_weights = np.abs(result.x) / np.abs(result.x).sum()
best_auc_optimized = -result.fun

print(f"Optimal Weights Found:")
print(f"  XGBoost:  {opt_weights[0]:.4f}")
print(f"  LightGBM: {opt_weights[1]:.4f}")
print(f"  CatBoost: {opt_weights[2]:.4f}")
print(f"Optimized Blend AUC: {best_auc_optimized:.6f}")

# Calculate the final Optimized Test Predictions
pred_optimized = (opt_weights[0] * pred_xgb + 
                  opt_weights[1] * pred_lgb + 
                  opt_weights[2] * pred_cat)

# --- 5. COMPARISON & SELECTION ---

# 1. Single Model Scores
auc_xgb = roc_auc_score(target, oof_xgb)
auc_lgb = roc_auc_score(target, oof_lgb)
auc_cat = roc_auc_score(target, oof_cat)

# 2. Probability Blend (Simple Average)
oof_blend_prob = (oof_xgb + oof_lgb + oof_cat) / 3
auc_blend_prob = roc_auc_score(target, oof_blend_prob)

# 3. Rank Blend
oof_blend_rank = (pd.Series(oof_xgb).rank(pct=True) + 
                  pd.Series(oof_lgb).rank(pct=True) + 
                  pd.Series(oof_cat).rank(pct=True)) / 3
auc_blend_rank = roc_auc_score(target, oof_blend_rank)

# Store everything in the results dictionary
results = {
    'XGBoost': {'score': auc_xgb, 'preds': pred_xgb},
    'LightGBM': {'score': auc_lgb, 'preds': pred_lgb},
    'CatBoost': {'score': auc_cat, 'preds': pred_cat},
    'Prob_Blend': {'score': auc_blend_prob, 'preds': (pred_xgb + pred_lgb + pred_cat) / 3},
    'Rank_Blend': {'score': auc_blend_rank, 'preds': (pd.Series(pred_xgb).rank(pct=True) + 
                                                      pd.Series(pred_lgb).rank(pct=True) + 
                                                      pd.Series(pred_cat).rank(pct=True)) / 3},
    'Optimized_Blend': {'score': best_auc_optimized, 'preds': pred_optimized}
}

# Print Summary
print(f"\n{'='*10} FINAL SCOREBOARD {'='*10}")
print(f"XGBoost:         {auc_xgb:.6f}")
print(f"LightGBM:        {auc_lgb:.6f}")
print(f"CatBoost:        {auc_cat:.6f}")
print(f"Simple Average:  {auc_blend_prob:.6f}")
print(f"Rank Blend:      {auc_blend_rank:.6f}")
print(f"Optimized Blend: {best_auc_optimized:.6f}  <-- EXPECTED WINNER")
print(f"{'='*40}")

# --- 6. SUBMISSION ---
# Automatically pick the best one
best_method = max(results, key=lambda x: results[x]['score'])
best_score = results[best_method]['score']
final_predictions = results[best_method]['preds']

print(f"\n✅ Best Strategy: {best_method} with CV: {best_score:.6f}")
print("Generating submission file...")

submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': np.clip(final_predictions, 0.001, 0.999) # Clip is good practice
})

submission.to_csv('submission.csv', index=False)
print("Saved to 'submission.csv'")



