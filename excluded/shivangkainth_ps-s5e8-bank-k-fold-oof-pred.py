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


# ===================================================================
# This is Stacked Ensemble Pipeline
#
# This script implements a full K-fold oof approach:
# 1. Hyperparameter Tuning: Uses Optuna to find the best parameters
#    for each base model via K-Fold Cross-Validation.
# 2. Stacking: Generates Out-of-Fold (OOF) predictions using the
#    best parameters to train a meta-model.
# 3. Final Training: Retrains the base models on the ENTIRE
#    training dataset and uses the meta-model to predict on the
#    test set for the final submission.
# ===================================================================


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import os
import gc
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Optional These are optional and used for parameter tunning if Not found then manually set params used .
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not found.")

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not found.")



# -------------------------------
# 1) Load and Prepare Data
# -------------------------------
TRAIN_PATH = '/kaggle/input/ps-s5e8/train.csv'
TEST_PATH = '/kaggle/input/ps-s5e8/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/ps-s5e8/sample_submission.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

TARGET = "y"
if TARGET not in train_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train.csv")

y = train_df[TARGET].astype(int)
X_full = train_df.drop(columns=[TARGET])
X_test_full = test_df.copy()

# Align columns - crucial for consistency
train_cols = X_full.columns
test_cols = X_test_full.columns
shared_cols = list(set(train_cols) & set(test_cols))
X_full = X_full[shared_cols]
X_test_full = X_test_full[shared_cols]

id_col = next((col for col in ["id", "ID", "Id"] if col in test_df.columns), None)
sub_target_col = sample_sub.columns[1] if id_col else sample_sub.columns[-1]



print("Step 2: Preprocessing data...")
cat_cols = [c for c in X_full.columns if X_full[c].dtype == "object"]
num_cols = [c for c in X_full.columns if c not in cat_cols]

for c in num_cols:
    med = X_full[c].median()
    X_full[c] = X_full[c].fillna(med)
    X_test_full[c] = X_test_full[c].fillna(med)

for c in cat_cols:
    mode = X_full[c].mode()[0]
    X_full[c] = X_full[c].astype("category").cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
    X_test_full[c] = X_test_full[c].astype("category").cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

def kfold_target_encode(train_series, target, test_series, n_splits=5, smoothing=20):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train_series), dtype=float)
    test_encoded = np.zeros(len(test_series), dtype=float)
    global_mean = target.mean()

    for tr_idx, va_idx in skf.split(train_series, target):
        tr_vals, va_vals = train_series.iloc[tr_idx], train_series.iloc[va_idx]
        tr_target = target.iloc[tr_idx]
        
        counts = tr_vals.value_counts()
        means = tr_vals.groupby(tr_vals).apply(lambda s: tr_target.loc[s.index].mean())
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
        
        oof[va_idx] = va_vals.map(smooth).fillna(global_mean).values
        test_encoded += test_series.map(smooth).fillna(global_mean).values / n_splits
        
    return oof, test_encoded

X_enc = X_full[num_cols].copy()
X_test_enc = X_test_full[num_cols].copy()

for c in cat_cols:
    tr_enc, te_enc = kfold_target_encode(X_full[c], y, X_test_full[c])
    X_enc[f"TE_{c}"] = tr_enc
    X_test_enc[f"TE_{c}"] = te_enc



print("\nPART 1: HYPERPARAMETER TUNING WITH OPTUNA")
NFOLDS_TUNE = 3 
N_TRIALS = 20 

def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': 1000,
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
    }
    skf = StratifiedKFold(n_splits=NFOLDS_TUNE, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X_enc, y):
        Xtr, Xva = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        model = xgb.XGBClassifier(**params, early_stopping_rounds=100, random_state=42)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        preds = model.predict_proba(Xva)[:, 1]
        scores.append(roc_auc_score(yva, preds))
    return np.mean(scores)

def lgb_objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    skf = StratifiedKFold(n_splits=NFOLDS_TUNE, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(X_enc, y):
        Xtr, Xva = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict_proba(Xva)[:, 1]
        scores.append(roc_auc_score(yva, preds))
    return np.mean(scores)

if HAS_OPTUNA:
    print(f"Tuning XGBoost ({N_TRIALS} trials)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS)
    xgb_best_params = study_xgb.best_params
    print("Best XGBoost Params:", xgb_best_params)

    print(f"\nTuning LightGBM ({N_TRIALS} trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lgb_objective, n_trials=N_TRIALS)
    lgb_best_params = study_lgb.best_params
    print("Best LightGBM Params:", lgb_best_params)
    
else:
    # This params are gonna used ifparameters if Optuna is not available
    xgb_best_params = {'lambda': 1.5, 'alpha': 1.5, 'colsample_bytree': 0.7, 'subsample': 0.8, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 1, 'gamma': 0.1}
    lgb_best_params = {'learning_rate': 0.02, 'num_leaves': 64, 'max_depth': 7, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

if HAS_CATBOOST:
    cat_best_params = {'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3.0, 'loss_function': 'Logloss', 'eval_metric': 'AUC', 'task_type': 'GPU'}



print("\nPART 2: GENERATING OUR OOF PREDICTIONS FOR STACKING")
NFOLDS_STACK = 5
skf = StratifiedKFold(n_splits=NFOLDS_STACK, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_enc))
oof_lgb = np.zeros(len(X_enc))
oof_cat = np.zeros(len(X_enc)) if HAS_CATBOOST else None

best_iter_xgb, best_iter_lgb, best_iter_cat = [], [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y), 1):
    print(f"===== Stacking Fold {fold}/{NFOLDS_STACK} =====")
    Xtr_enc, Xva_enc = X_enc.iloc[tr_idx], X_enc.iloc[va_idx]
    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

    xgb_model = xgb.XGBClassifier(**xgb_best_params, n_estimators=2000, early_stopping_rounds=200, random_state=42)
    xgb_model.fit(Xtr_enc, ytr, eval_set=[(Xva_enc, yva)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict_proba(Xva_enc)[:, 1]
    best_iter_xgb.append(xgb_model.best_iteration)
    print(f"  XGB Fold AUC: {roc_auc_score(yva, oof_xgb[va_idx]):.6f}")

    lgb_model = lgb.LGBMClassifier(**lgb_best_params, n_estimators=2000, random_state=42)
    lgb_model.fit(Xtr_enc, ytr, eval_set=[(Xva_enc, yva)], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_lgb[va_idx] = lgb_model.predict_proba(Xva_enc)[:, 1]
    # FIX: Use .best_iteration_ for the scikit-learn API
    best_iter_lgb.append(lgb_model.best_iteration_)
    print(f"  LGB Fold AUC: {roc_auc_score(yva, oof_lgb[va_idx]):.6f}")
    
    if HAS_CATBOOST:
        cat_model = CatBoostClassifier(**cat_best_params, iterations=3000, od_type="Iter", od_wait=300, random_seed=42, verbose=0)
        cat_model.fit(X_full.iloc[tr_idx], ytr, eval_set=(X_full.iloc[va_idx], yva), cat_features=cat_cols)
        oof_cat[va_idx] = cat_model.predict_proba(X_full.iloc[va_idx])[:, 1]
        best_iter_cat.append(cat_model.get_best_iteration())
        print(f"  CAT Fold AUC: {roc_auc_score(yva, oof_cat[va_idx]):.6f}")
        
    gc.collect()


print("\nPART 3: TRAINING META-MODEL ON OOF PREDICTIONS")
if HAS_CATBOOST:
    oof_stack = np.vstack([oof_xgb, oof_lgb, oof_cat]).T
else:
    oof_stack = np.vstack([oof_xgb, oof_lgb]).T

meta_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear", random_state=42))
])
meta_model.fit(oof_stack, y)
meta_oof_auc = roc_auc_score(y, meta_model.predict_proba(oof_stack)[:, 1])
print(f"Meta-Model OOF AUC: {meta_oof_auc:.6f}")


print("\nPART 4: TRAINING FINAL MODELS ON FULL DATASET")

n_est_xgb = int(np.mean(best_iter_xgb) * 1.1)
n_est_lgb = int(np.mean(best_iter_lgb) * 1.1)
print(f"Final rounds -> XGB: {n_est_xgb}, LGB: {n_est_lgb}")

print("Training final XGBoost model...")
final_xgb = xgb.XGBClassifier(**xgb_best_params, n_estimators=n_est_xgb, random_state=42)
final_xgb.fit(X_enc, y, verbose=False)
test_pred_xgb = final_xgb.predict_proba(X_test_enc)[:, 1]

print("Training final LightGBM model...")
final_lgb = lgb.LGBMClassifier(**lgb_best_params, n_estimators=n_est_lgb, random_state=42)
final_lgb.fit(X_enc, y)
test_pred_lgb = final_lgb.predict_proba(X_test_enc)[:, 1]

if HAS_CATBOOST:
    n_est_cat = int(np.mean(best_iter_cat) * 1.1)
    print(f"Final rounds -> CAT: {n_est_cat}")
    print("Training final CatBoost model...")
    final_cat = CatBoostClassifier(**cat_best_params, iterations=n_est_cat, random_seed=42, verbose=0)
    final_cat.fit(X_full, y, cat_features=cat_cols)
    test_pred_cat = final_cat.predict_proba(X_test_full)[:, 1]


print("\nPART 5: CREATING FINAL SUBMISSION")

if HAS_CATBOOST:
    test_stack_full = np.vstack([test_pred_xgb, test_pred_lgb, test_pred_cat]).T
else:
    test_stack_full = np.vstack([test_pred_xgb, test_pred_lgb]).T

# Use the meta-model to make the final prediction
# meta model is our master model which uses the oof predictions made by other models and learns from that then predicts y
final_predictions = meta_model.predict_proba(test_stack_full)[:, 1]

submission = pd.DataFrame()
if id_col:
    submission[id_col] = test_df[id_col]
submission[sub_target_col] = final_predictions
submission.to_csv("submission_final.csv", index=False)

print("\nsubmission_final.csv created successfully!")
print("Pipeline finished.")


