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


# =============================================================================
# 0. SETUP & LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep Optuna's output clean
print("Setup Complete: All libraries and Optuna are ready.")


# =============================================================================
# 1. DATA LOADING & PREPARATION
# =============================================================================
print("\n--- Loading and Preparing Data ---")
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

original_features = [col for col in df_train.columns if col not in ['id', 'BeatsPerMinute']]
X_raw = df_train[original_features]
y = df_train['BeatsPerMinute']
X_test_raw = df_test[original_features]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=original_features)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=original_features)
test_ids = df_test['id']


# #############################################################################
# PART A: THE SEARCH FOR THE BEST GENERALIST (LightGBM)
# #############################################################################
print("\n--- PART A: Starting Optuna Search for the Generalist Model ---")

def objective_generalist(trial):
    """
    This objective function runs the full pipeline with a default Specialist/Arbiter
    to find the Generalist parameters that give the best *final system score*.
    """
    params = {
        'objective': 'regression', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    generalist_model = lgb.LGBMRegressor(**params)
    oof_preds_stage1 = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
        generalist_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds_stage1[val_idx] = generalist_model.predict(X_val_fold)

    errors = (y - oof_preds_stage1) ** 2
    is_hard_sample = errors >= np.percentile(errors, 90)
    X_spec = X.copy()
    X_spec['generalist_prediction'] = oof_preds_stage1
    scaler_spec = StandardScaler()
    X_spec[['generalist_prediction']] = scaler_spec.fit_transform(X_spec[['generalist_prediction']])
    
    sample_weights = np.where(is_hard_sample, 10.0, 1.0)
    specialist_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1) # Default Specialist
    oof_preds_stage2 = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_spec, y)):
        X_train_fold, y_train_fold = X_spec.iloc[train_idx], y.iloc[train_idx]
        specialist_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights[train_idx])
        oof_preds_stage2[val_idx] = specialist_model.predict(X_spec.iloc[val_idx])
    
    meta_train = pd.DataFrame({'generalist_preds': oof_preds_stage1, 'specialist_preds': oof_preds_stage2})
    meta_scaler = StandardScaler()
    meta_train_scaled = meta_scaler.fit_transform(meta_train)
    arbiter = Ridge(random_state=42) # Default Arbiter
    arbiter.fit(meta_train_scaled, y)
    
    final_system_oof_preds = arbiter.predict(meta_train_scaled)
    return np.sqrt(mean_squared_error(y, final_system_oof_preds))

study_generalist = optuna.create_study(direction='minimize')
study_generalist.optimize(objective_generalist, n_trials=50) # Increase for better results
best_generalist_params = study_generalist.best_params
print("\n--- Generalist Search Complete ---")
print("Best Generalist (LGBM) Hyperparameters:", best_generalist_params)


# #############################################################################
# PART B: THE SEARCH FOR THE BEST SPECIALIST (LightGBM)
# #############################################################################
print("\n\n--- PART B: Preparing to Tune the Specialist Model ---")
print("--- Generating OOF preds with the BEST Generalist ---")
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
generalist_model_opt = lgb.LGBMRegressor(**best_generalist_params, n_estimators=2000, random_state=42, verbose=-1)
oof_preds_stage1_opt = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    generalist_model_opt.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_stage1_opt[val_idx] = generalist_model_opt.predict(X_val_fold)

errors_opt = (y - oof_preds_stage1_opt) ** 2
is_hard_sample_opt = errors_opt >= np.percentile(errors_opt, 90)
X_spec_opt = X.copy()
X_spec_opt['generalist_prediction'] = oof_preds_stage1_opt
scaler_spec_opt = StandardScaler()
X_spec_opt[['generalist_prediction']] = scaler_spec_opt.fit_transform(X_spec_opt[['generalist_prediction']])
sample_weights_opt = np.where(is_hard_sample_opt, 10.0, 1.0)

def objective_specialist(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 10, 80),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
    }
    specialist_model = lgb.LGBMRegressor(**params)
    oof_preds_stage2 = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_spec_opt, y)):
        X_train_fold, y_train_fold = X_spec_opt.iloc[train_idx], y.iloc[train_idx]
        specialist_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_opt[train_idx])
        oof_preds_stage2[val_idx] = specialist_model.predict(X_spec_opt.iloc[val_idx])
    
    meta_train = pd.DataFrame({'generalist_preds': oof_preds_stage1_opt, 'specialist_preds': oof_preds_stage2})
    meta_scaler = StandardScaler()
    meta_train_scaled = meta_scaler.fit_transform(meta_train)
    arbiter = Ridge(random_state=42) # Default Arbiter
    arbiter.fit(meta_train_scaled, y)
    
    final_system_oof_preds = arbiter.predict(meta_train_scaled)
    return np.sqrt(mean_squared_error(y, final_system_oof_preds))

print("\n--- Starting Optuna Search for Specialist (LGBM) ---")
study_specialist = optuna.create_study(direction='minimize')
study_specialist.optimize(objective_specialist, n_trials=30) # Increase for better results
best_specialist_params = study_specialist.best_params
print("\n--- Specialist Search Complete ---")
print("Best Specialist (LGBM) Hyperparameters:", best_specialist_params)


# #############################################################################
# PART C: THE FINAL RUN AND SUBMISSION
# #############################################################################
print("\n\n--- PART C: Rerunning pipeline with fully optimized components ---")

# STAGE 1: THE GENERALIST MODEL (with best_params)
print("\n--- STAGE 1: Training the FINAL Generalist Model ---")
generalist_model = lgb.LGBMRegressor(**best_generalist_params, n_estimators=2000, random_state=42, verbose=-1)
oof_preds_stage1 = np.zeros(len(X))
test_preds_stage1 = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    generalist_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_preds_stage1[val_idx] = generalist_model.predict(X_val_fold)
    test_preds_stage1 += generalist_model.predict(X_test) / N_SPLITS

# STAGES 2 & 3: DIAGNOSTIC & SPECIALIST FEATURE CREATION
print("\n--- STAGES 2 & 3: Creating FINAL Specialist Features ---")
errors = (y - oof_preds_stage1) ** 2
is_hard_sample = errors >= np.percentile(errors, 90)
X_spec = X.copy()
X_test_spec = X_test.copy()
X_spec['generalist_prediction'] = oof_preds_stage1
X_test_spec['generalist_prediction'] = test_preds_stage1
# Clustering is not needed for features, so we simplify this part
scaler_spec_final = StandardScaler()
X_spec[['generalist_prediction']] = scaler_spec_final.fit_transform(X_spec[['generalist_prediction']])
X_test_spec[['generalist_prediction']] = scaler_spec_final.transform(X_test_spec[['generalist_prediction']])

# STAGE 4: THE SPECIALIST MODEL (with best_params)
print("\n--- STAGE 4: Training the FINAL Specialist Model ---")
sample_weights = np.where(is_hard_sample, 10.0, 1.0)
specialist_model = lgb.LGBMRegressor(**best_specialist_params, n_estimators=2000, random_state=42, verbose=-1)
oof_preds_stage2 = np.zeros(len(X))
test_preds_stage2 = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_spec, y)):
    X_train_fold, y_train_fold = X_spec.iloc[train_idx], y.iloc[train_idx]
    specialist_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights[train_idx])
    oof_preds_stage2[val_idx] = specialist_model.predict(X_spec.iloc[val_idx])
    test_preds_stage2 += specialist_model.predict(X_test_spec) / N_SPLITS

# STAGE 5: THE ARBITER
print("\n--- STAGE 5: Training the FINAL Arbiter ---")
meta_train = pd.DataFrame({'generalist_preds': oof_preds_stage1, 'specialist_preds': oof_preds_stage2})
meta_test = pd.DataFrame({'generalist_preds': test_preds_stage1, 'specialist_preds': test_preds_stage2})
meta_scaler = StandardScaler()
meta_train_scaled = meta_scaler.fit_transform(meta_train)
meta_test_scaled = meta_scaler.transform(meta_test)
arbiter = Ridge(random_state=42)
arbiter.fit(meta_train_scaled, y)

# STAGE 6: FINAL PREDICTIONS
print("\n--- STAGE 6: Generating Final 'In-Cage' Predictions ---")
final_incage_predictions = arbiter.predict(meta_test_scaled)


# STAGE 8: FINAL SUBMISSION
print("\n--- STAGE 8: Creating Final Submission File ---")
submission_df = pd.DataFrame({'id': test_ids, 'BeatsPerMinute': final_incage_predictions})
submission_df.to_csv('submission_fully_tuned_lgbm_architecture.csv', index=False)
print("Submission file 'submission_fully_tuned_lgbm_architecture.csv' created successfully!")
print("Final submission head:")
print(submission_df.head())

