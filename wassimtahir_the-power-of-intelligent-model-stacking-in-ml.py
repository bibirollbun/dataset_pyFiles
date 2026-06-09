import numpy as np
import pandas as pd
import warnings
import gc

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import optuna
from optuna.samplers import TPESampler
from scipy.optimize import minimize

SEED = 42
np.random.seed(SEED)

class Config:
    N_SPLITS = 5
    N_TRIALS = 30  
    RANDOM_STATE = 42

config = Config()

print("="*80)
print("ğŸš€ SIMPLIFIED PIPELINE - QUALITY OVER QUANTITY")
print("="*80)


print("\nğŸ“‚ Loading data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

test_ids = test_df['id'].copy()


print("\nğŸ”§ Creating SMART features (only the useful ones)...")

def create_smart_features(df):
    df = df.copy()
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_le'] = le.fit_transform(df[col].astype(str))
    df = df.drop(cat_cols, axis=1)
    
    if 'speed_limit' in df.columns and 'curvature' in df.columns:
        df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
        df['danger_score'] = (df['speed_limit'] / 100) * (df['curvature'] ** 2)
    if 'num_reported_accidents' in df.columns and 'num_lanes' in df.columns:
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    if 'weather_le' in df.columns and 'lighting_le' in df.columns:
        df['env_risk'] = df['weather_le'] * df['lighting_le']
    if 'num_reported_accidents' in df.columns:
        df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    if 'curvature' in df.columns and 'speed_limit' in df.columns:
        df['road_complexity'] = df['curvature'] * (df['speed_limit'] / 100)
    if 'time_of_day_le' in df.columns and 'school_season_le' in df.columns:
        df['rush_hour_risk'] = df['time_of_day_le'] * df['school_season_le']
    
    return df

train_processed = create_smart_features(train_df)
test_processed = create_smart_features(test_df)

TARGET = 'accident_risk'
feature_cols = [col for col in train_processed.columns if col != TARGET]

common_cols = list(set(feature_cols) & set(test_processed.columns))
common_cols.sort()

X = train_processed[common_cols].values
y = train_processed[TARGET].values
X_test = test_processed[common_cols].values

print(f"âœ… Features created: {len(common_cols)}")
print(f"X shape: {X.shape}")


print("\nğŸ�¯ Optimizing hyperparameters...")

kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=SEED)
train_idx, val_idx = next(kf.split(X))
X_train_opt, X_val_opt = X[train_idx], X[val_idx]
y_train_opt, y_val_opt = y[train_idx], y[val_idx]

def objective_xgb(trial):
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'random_state': SEED,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_opt, y_train_opt, 
              eval_set=[(X_val_opt, y_val_opt)],
              verbose=False)
    
    preds = model.predict(X_val_opt)
    return np.sqrt(mean_squared_error(y_val_opt, preds))

def objective_lgb(trial):
    params = {
        'device': 'gpu',
        'num_leaves': trial.suggest_int('num_leaves', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'random_state': SEED,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    preds = model.predict(X_val_opt)
    return np.sqrt(mean_squared_error(y_val_opt, preds))

def objective_cat(trial):
    params = {
        'task_type': 'GPU',
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': SEED,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=(X_val_opt, y_val_opt),
              early_stopping_rounds=100,
              verbose=False)
    
    preds = model.predict(X_val_opt)
    return np.sqrt(mean_squared_error(y_val_opt, preds))


print("\nâš¡ Optimizing XGBoost...")
study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=config.N_TRIALS, show_progress_bar=True)
best_xgb = study_xgb.best_params
best_xgb.update({'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED, 'verbosity': 0})
print(f"âœ… XGBoost Best RMSE: {study_xgb.best_value:.6f}")

print("\nâš¡ Optimizing LightGBM...")
study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=config.N_TRIALS, show_progress_bar=True)
best_lgb = study_lgb.best_params
best_lgb.update({'device': 'gpu', 'random_state': SEED, 'verbose': -1})
print(f"âœ… LightGBM Best RMSE: {study_lgb.best_value:.6f}")

print("\nâš¡ Optimizing CatBoost...")
study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_cat.optimize(objective_cat, n_trials=config.N_TRIALS, show_progress_bar=True)
best_cat = study_cat.best_params
best_cat.update({'task_type': 'GPU', 'random_seed': SEED, 'verbose': False})
print(f"âœ… CatBoost Best RMSE: {study_cat.best_value:.6f}")

gc.collect()



print("\n" + "="*80)
print("ğŸŒ² Training models with CV...")
print("="*80)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

scores = {'xgb': [], 'lgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # XGBoost
    print("Training XGBoost...")
    model_xgb = xgb.XGBRegressor(**best_xgb)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / config.N_SPLITS
    scores['xgb'].append(np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx])))
    print(f"  XGBoost RMSE: {scores['xgb'][-1]:.6f}")
    del model_xgb
    gc.collect()
    
    # LightGBM
    print("Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(**best_lgb)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / config.N_SPLITS
    scores['lgb'].append(np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx])))
    print(f"  LightGBM RMSE: {scores['lgb'][-1]:.6f}")
    del model_lgb
    gc.collect()
    
    # CatBoost
    print("Training CatBoost...")
    model_cat = CatBoostRegressor(**best_cat)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=100, verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / config.N_SPLITS
    scores['cat'].append(np.sqrt(mean_squared_error(y_val, oof_cat[val_idx])))
    print(f"  CatBoost RMSE: {scores['cat'][-1]:.6f}")
    del model_cat
    gc.collect()

print("\n" + "="*80)
print("ğŸ“Š CV RESULTS")
print("="*80)
for name, score_list in scores.items():
    print(f"{name.upper():10s}: {np.mean(score_list):.6f} Â± {np.std(score_list):.6f}")


print("\n" + "="*80)
print("âš–ï¸�  Finding Optimal Ensemble Weights")
print("="*80)


oof_stack = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_stack = np.column_stack([test_xgb, test_lgb, test_cat])

def optimize_weights(preds, target):
    def objective(w):
        ensemble = np.average(preds, axis=1, weights=w)
        return np.sqrt(mean_squared_error(target, ensemble))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(preds.shape[1])]
    initial = np.ones(preds.shape[1]) / preds.shape[1]
    
    result = minimize(objective, initial, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x if result.success else initial

weights = optimize_weights(oof_stack, y)
print("\nğŸ“Š Optimal Weights:")
print(f"  XGBoost:  {weights[0]:.4f}")
print(f"  LightGBM: {weights[1]:.4f}")
print(f"  CatBoost: {weights[2]:.4f}")

ensemble_oof = np.average(oof_stack, axis=1, weights=weights)
ensemble_test = np.average(test_stack, axis=1, weights=weights)

ensemble_test = np.clip(ensemble_test, 0, 1)

ensemble_cv = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"\nğŸ�† Ensemble CV RMSE: {ensemble_cv:.6f}")


print("\n" + "="*80)
print("ğŸ�¯ FINAL COMPARISON")
print("="*80)

xgb_cv = np.sqrt(mean_squared_error(y, oof_xgb))
lgb_cv = np.sqrt(mean_squared_error(y, oof_lgb))
cat_cv = np.sqrt(mean_squared_error(y, oof_cat))

print(f"XGBoost alone:  {xgb_cv:.6f}")
print(f"LightGBM alone: {lgb_cv:.6f}")
print(f"CatBoost alone: {cat_cv:.6f}")
print(f"Ensemble:       {ensemble_cv:.6f}")

best_single = min(xgb_cv, lgb_cv, cat_cv)
improvement = best_single - ensemble_cv

if improvement > 0:
    print(f"\nâœ… Ensemble improves by: {improvement:.6f}")
    final_predictions = ensemble_test
    method = "Weighted Ensemble"
else:
    print(f"\nâš ï¸�  Single model is better! Using best single model.")
    if xgb_cv == best_single:
        final_predictions = test_xgb
        method = "XGBoost alone"
    elif lgb_cv == best_single:
        final_predictions = test_lgb
        method = "LightGBM alone"
    else:
        final_predictions = test_cat
        method = "CatBoost alone"


import pandas as pd

print("\n" + "="*80)
print("ğŸ“� Creating Submission")
print("="*80)

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_predictions
})

other_submission = pd.read_csv('/kaggle/input/s5e10-nn-stacking-baseline/test_nn_ensemble.csv')




submission['accident_risk'] = (submission['accident_risk'] * 0.1 + other_submission['accident_risk']*0.9)

submission['accident_risk'] = submission['accident_risk'].clip(0, 1)


print(f"\nğŸ“Š Submission Statistics:")
print(submission['accident_risk'].describe())


submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission saved to 'submission.csv'")

