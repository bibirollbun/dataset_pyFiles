import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

# ==========================
# Data Loading
# ==========================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
ss = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

train = train.drop(columns=['id'])
test_ids = test['id']
test = test.drop(columns=['id'])

# ==========================
# Feature Engineering
# ==========================
def feature_engineering(df):
    df = df.copy()
    # Interaction features
    df['Loudness_Energy'] = df['AudioLoudness'] * df['Energy']
    df['Duration_per_Mood'] = df['TrackDurationMs'] / (df['MoodScore'] + 1e-6)
    df['Rhythm_Mood'] = df['RhythmScore'] * df['MoodScore']
    df['Vocal_Instrumental'] = df['VocalContent'] * (df['InstrumentalScore'] + 1e-6)
    df['LogDuration'] = np.log1p(df['TrackDurationMs'])

    # Polynomial / Ratio features
    df['Energy2'] = df['Energy']**2
    df['Loudness2'] = df['AudioLoudness']**2
    df['Mood_to_Acoustic'] = df['MoodScore'] / (df['AcousticQuality'] + 1e-6)
    df['Energy_to_Vocal'] = df['Energy'] / (df['VocalContent'] + 1e-6)
    df['Rhythm_per_Duration'] = df['RhythmScore'] / (df['TrackDurationMs'] + 1e-6)
    df['Duration_Mood_Ratio'] = df['TrackDurationMs'] / (df['MoodScore'] + 1e-6)
    df['Energy_Acoustic'] = df['Energy'] * df['AcousticQuality']
    df['Mood_Energy'] = df['MoodScore'] * df['Energy']
    df['Vocal_Rhythm'] = df['VocalContent'] * df['RhythmScore']
    df['Acoustic_Instrumental'] = df['AcousticQuality'] * df['InstrumentalScore']
    df['Loudness_per_Duration'] = df['AudioLoudness'] / (df['TrackDurationMs'] + 1e-6)
    df['Duration_per_Instrumental'] = df['TrackDurationMs'] / (df['InstrumentalScore'] + 1e-6)
    return df

X = feature_engineering(train.drop(columns=['BeatsPerMinute']))
y = train['BeatsPerMinute']
X_test = feature_engineering(test)

# ==========================
# Optuna Optimization Functions
# ==========================
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'seed': 42
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval])
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmses.append(mean_squared_error(y_val, preds, squared=False))
    return np.mean(rmses)

def objective_cat(trial):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        rmses.append(mean_squared_error(y_val, preds, squared=False))
    return np.mean(rmses)

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'seed': 42
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**params, n_estimators=1000)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],  verbose=False)
        preds = model.predict(X_val)
        rmses.append(mean_squared_error(y_val, preds, squared=False))
    return np.mean(rmses)

# ==========================
# Run Optuna (10 trials each)
# ==========================
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=5)

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=5)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=5)

best_lgbm_params = study_lgb.best_trial.params
best_cat_params = study_cat.best_trial.params
best_xgb_params = study_xgb.best_trial.params

best_lgbm_params.update({'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'seed': 42})
best_cat_params.update({'iterations': 2000, 'loss_function': 'RMSE', 'random_seed': 42, 'verbose': False})
best_xgb_params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42, 'n_estimators': 2000})

# ==========================
# K-Fold Training with Tuned Params
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)


lgbm_oof = np.zeros(len(X))
cat_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))


lgbm_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

lgbm_fold_rmse, cat_fold_rmse, xgb_fold_rmse = [], [], []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    
    # LightGBM
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_valid, y_valid, reference=dtrain)
    model_lgbm = lgb.train(best_lgbm_params, dtrain, num_boost_round=2000, valid_sets=[dval])
    pred_val_lgb = model_lgbm.predict(X_valid, num_iteration=model_lgbm.best_iteration)
    lgbm_oof[valid_idx] = pred_val_lgb
    lgbm_preds += model_lgbm.predict(X_test, num_iteration=model_lgbm.best_iteration) / kf.n_splits
    lgbm_fold_rmse.append(mean_squared_error(y_valid, pred_val_lgb, squared=False))
    
    
    # CatBoost
    model_cat = CatBoostRegressor(**best_cat_params)
    model_cat.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    pred_val_cat = model_cat.predict(X_valid)
    cat_oof[valid_idx] = pred_val_cat
    cat_preds += model_cat.predict(X_test) / kf.n_splits
    cat_fold_rmse.append(mean_squared_error(y_valid, pred_val_cat, squared=False))
    
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(**best_xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    pred_val_xgb = model_xgb.predict(X_valid)
    xgb_oof[valid_idx] = pred_val_xgb
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits
    xgb_fold_rmse.append(mean_squared_error(y_valid, pred_val_xgb, squared=False))
# ==========================
# CV Scores & Weights (lower RMSE -> higher weight)
# ==========================
lgbm_cv = mean_squared_error(y, lgbm_oof, squared=False)
cat_cv = mean_squared_error(y, cat_oof, squared=False)
xgb_cv = mean_squared_error(y, xgb_oof, squared=False)


print("\nPer-fold RMSE:")
print(f"LGBM folds: {[round(v, 4) for v in lgbm_fold_rmse]} | mean={np.mean(lgbm_fold_rmse):.4f} ± {np.std(lgbm_fold_rmse):.4f}")
print(f"CAT folds: {[round(v, 4) for v in cat_fold_rmse]} | mean={np.mean(cat_fold_rmse):.4f} ± {np.std(cat_fold_rmse):.4f}")
print(f"XGB folds: {[round(v, 4) for v in xgb_fold_rmse]} | mean={np.mean(xgb_fold_rmse):.4f} ± {np.std(xgb_fold_rmse):.4f}")


print("\nOOF CV RMSE:")
print(f"LGBM OOF RMSE: {lgbm_cv:.4f}")
print(f"CAT OOF RMSE: {cat_cv:.4f}")
print(f"XGB OOF RMSE: {xgb_cv:.4f}")

# inverse-RMSE weighting (so lower RMSE => larger weight). Optional power>1 to emphasize differences
inv = np.array([1.0/max(lgbm_cv, 1e-6), 1.0/max(cat_cv, 1e-6), 1.0/max(xgb_cv, 1e-6)])
# sharpen weights a bit
power = 1.5
weights = inv ** power
weights = weights / weights.sum()


w_lgbm, w_cat, w_xgb = weights.tolist()
print(f"\nEnsemble Weights (based on OOF RMSE): LGBM={w_lgbm:.4f}, CAT={w_cat:.4f}, XGB={w_xgb:.4f}")


# ==========================
# Weighted Averaging Ensemble
# ==========================
weighted_oof = w_lgbm * lgbm_oof + w_cat * cat_oof + w_xgb * xgb_oof
weighted_cv = mean_squared_error(y, weighted_oof, squared=False)
print(f"Weighted Ensemble OOF RMSE: {weighted_cv:.4f}")
weighted_test_preds = w_lgbm * lgbm_preds + w_cat * cat_preds + w_xgb * xgb_preds

sub_weighted = ss.copy()
sub_weighted['BeatsPerMinute'] = weighted_test_preds
sub_weighted.to_csv("submission_weighted.csv", index=False)
final = ss.copy()
final['BeatsPerMinute'] = weighted_test_preds
final.to_csv("submission.csv", index=False)
print("Saved submission_weighted.csv")





