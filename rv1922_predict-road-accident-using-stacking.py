pip install "optuna-integration[xgboost]"


import pandas as pd
import numpy as np
import time 

import warnings 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from category_encoders import TargetEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import lightgbm as lgb
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.info()


if train.duplicated().any():
    train = train.drop_duplicates()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target_col = 'accident_risk'


for c in bool_cols:
    if c in train.columns or test.columns:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)


encoder = TargetEncoder(cols=cat_cols)
train[cat_cols] = encoder.fit_transform(train[cat_cols], train[target_col])
test[cat_cols] = encoder.transform(test[cat_cols])


def feature_engineering(df):
    df = df.copy()

    if 'num_reported_accidents' in df.columns:
        df['accidents_log'] = np.log1p(df['num_reported_accidents'])
        df['accidents_sqrt'] = np.sqrt(df['num_reported_accidents'])

    if 'curvature' in df.columns:
        df['curvature_squared'] = df['curvature'] ** 2
        df['curv_log'] = np.log1p(df['curvature'])
        df['curv_sqrt'] = np.sqrt(df['curvature'])
        df['curv_inv'] = 1.0 / (df['curvature'] + 1e-5)

    if 'speed_limit' in df.columns:
        df['speed_sq'] = df['speed_limit'] ** 2
        df['speed_log'] = np.log1p(df['speed_limit'])
        df['inv_speed'] = 1.0 / (df['speed_limit'] + 1.0)

    if 'num_lanes' in df.columns:
        df['lanes_log'] = np.log1p(df['num_lanes'])
        df['lanes_inv'] = 1.0 / (df['num_lanes'] + 1.0)

    # --------------------
    # Step 4: Intermediate Features - Interactions, ratios, densities
    # --------------------
    if all(col in df.columns for col in ['speed_limit', 'curvature']):
        df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
        df['curv_speed'] = df['curvature'] * df['speed_limit']
        df['danger_score'] = (df['speed_limit'] / 100.0) * (df['curvature'] ** 2)
        df['road_complexity'] = df['curvature'] * (df['speed_limit'] / 100.0)
        df['risk_index'] = (df['curv_speed'] * df.get('accidents_per_lane', 1.0)) / (df['speed_limit'] + 1.0)
        df['stability_score'] = (df.get('num_lanes', 1.0) / (1.0 + df['curvature'])) * df['speed_limit']

    if all(col in df.columns for col in ['num_reported_accidents', 'num_lanes']):
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1.0)
        df['accident_density'] = df['accidents_per_lane'] * df.get('speed_x_curvature', 1.0)

    if 'num_lanes' in df.columns:
        if 'curvature' in df.columns:
            df['curv_per_lane'] = df['curvature'] / (df['num_lanes'] + 1.0)
        if 'speed_limit' in df.columns:
            df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1.0)
        df['risk_density'] = df.get('curv_speed', 1.0) / (df['num_lanes'] + 1.0)

    if 'weather' in df.columns:
        if 'curvature' in df.columns:
            df['weather_curv_interact'] = df['weather'] * df['curvature']
        if 'speed_limit' in df.columns:
            df['weather_speed_interact'] = df['weather'] * df['speed_limit']
        if 'lighting' in df.columns:
            df['env_risk'] = df['weather'] * df['lighting']

    if 'time_of_day' in df.columns:
        if 'school_season' in df.columns:
            df['rush_hour_risk'] = df['time_of_day'] * df['school_season']
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24.0)
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24.0)

    if 'holiday' in df.columns and 'time_of_day' in df.columns:
        df['holiday_rush'] = df['holiday'] * df.get('rush_hour_risk', df['time_of_day'])

    # --------------------
    # Step 5: Advanced Features - Polynomials, power transforms, binnings, flags
    # --------------------
    if all(col in df.columns for col in ['curvature', 'speed_limit']):
        df['poly_mix1'] = np.sqrt(np.maximum(df['curvature'] * df['speed_limit'], 0.0))
        df['poly_mix2'] = (df.get('num_reported_accidents', 0.0) ** 0.3) * df['speed_limit']
        df['poly_high'] = df['speed_limit'] ** 2 * df['curvature'] ** 1.5

    if 'num_lanes' in df.columns:
        df['tight_lane'] = (df['num_lanes'] <= 2).astype(int)

    if 'curvature' in df.columns:
        df['sharp_curve'] = (df['curvature'] > 0.6).astype(int)

    if 'speed_limit' in df.columns:
        df['high_speed_zone'] = (df['speed_limit'] > 80).astype(int)

    if all(col in df for col in ['sharp_curve', 'high_speed_zone']):
        df['critical_zone'] = (df['sharp_curve'] & df['high_speed_zone']).astype(int)

    if 'curvature' in df.columns:
        df['curv_bin'] = pd.cut(df['curvature'], bins=5, labels=False)

    if 'speed_limit' in df.columns:
        df['speed_bin'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 70, 90, np.inf], labels=False)

    if 'env_risk' in df and 'curvature' in df.columns:
        df['total_env_exposure'] = df['env_risk'] * df['curvature']

    # --------------------
    # Step 6: Clean up - Handle inf/nan
    # --------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    return df


train = feature_engineering(train)
test = feature_engineering(test)


train.head()


train.info()


X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']
X_test = test.drop(columns=['id'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


random_state = 42

# Define objective for Optuna
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'verbosity': 0,
        'n_estimators': 5000,  # large number with early stopping
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 0.1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.5, 1.5),
        'random_state': random_state
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    rmse_scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=150,
            verbose=False
        )
        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# Create Optuna study with pruning
#study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state), pruner=optuna.pruners.MedianPruner())
#study.optimize(objective, n_trials=60, timeout=3600)  # adjust n_trials and timeout as needed


#print("Best RMSE:", study.best_value)
#print("Best params:", study.best_params)


random_state = 42
n_splits = 3

# Stratified regression bins (for fold stratification)
def create_bins(y, n_bins=10):
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

# Define the objective for Optuna (stacked model)
def objective(trial):
    # XGBoost params
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'verbosity': 0,
        'n_estimators': 5000,
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 15),
        'learning_rate': trial.suggest_loguniform('xgb_learning_rate', 0.005, 0.05),
        'subsample': trial.suggest_uniform('xgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_uniform('xgb_colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_loguniform('xgb_gamma', 0.001, 0.1),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('xgb_reg_alpha', 0.001, 0.1),
        'reg_lambda': trial.suggest_loguniform('xgb_reg_lambda', 0.5, 1.5),
        'random_state': random_state
    }

    # LightGBM params
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbosity': -1,
        'num_iterations': 5000,
        'max_depth': trial.suggest_int('lgb_max_depth', 6, 15),
        'learning_rate': trial.suggest_loguniform('lgb_learning_rate', 0.005, 0.05),
        'subsample': trial.suggest_uniform('lgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_uniform('lgb_colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('lgb_min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_loguniform('lgb_reg_alpha', 0.001, 0.1),
        'reg_lambda': trial.suggest_loguniform('lgb_reg_lambda', 0.5, 1.5),
        'min_split_gain': trial.suggest_loguniform('lgb_min_split_gain', 0.001, 0.1),
        'random_state': random_state
    }

    # Stratified folds for regression
    bins = create_bins(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmse_scores = []

    for train_idx, valid_idx in kf.split(X, bins):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # XGBoost model
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=150,
            verbose=False
        )
        xgb_pred = xgb_model.predict(X_valid)

        # LightGBM model
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse'
        )
        lgb_pred = lgb_model.predict(X_valid)

        # Stack predictions via linear regression
        stack_input = np.column_stack([xgb_pred, lgb_pred])
        stack_model = LinearRegression()
        stack_model.fit(stack_input, y_valid)
        stack_pred = stack_model.predict(stack_input)

        rmse = mean_squared_error(y_valid, stack_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# Run Optuna study with TPE sampler and pruning
"""study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=random_state),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(objective, n_trials=60, timeout=3600)"""


#print("Best RMSE:", study.best_value)
#print("Best parameters:", study.best_params)


random_state = 42
n_splits = 10

# Stratified bins for regression
def create_bins(y, n_bins=10):
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

bins = create_bins(y)

# Optimized params (kept from Optuna, but reduce n_estimators for efficiency)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'verbosity': 0,
    'n_estimators': 2000,  # Reduced from 5000 based on avg best iter ~1252
    'max_depth': 13,
    'learning_rate': 0.01155307726815254,
    'subsample': 0.7810228559035284,
    'colsample_bytree': 0.7156507413411545,
    'gamma': 0.01811266513974936,
    'min_child_weight': 6,
    'reg_alpha': 0.0052798792801754275,
    'reg_lambda': 0.5124782891193855,
    'random_state': random_state,
    'early_stopping_rounds': 150
}

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'verbosity': -1,
    'n_estimators': 1000,  # Reduced from 5000 based on avg best iter ~634
    'max_depth': 9,
    'learning_rate': 0.04912840558965971,
    'subsample': 0.8771424068781188,
    'colsample_bytree': 0.8078938276001452,
    'min_child_weight': 1,
    'reg_alpha': 0.07476866922455089,
    'reg_lambda': 0.9612236653475524,
    'min_split_gain': 0.0011048829258412144,
    'random_state': random_state
}

# Arrays for base OOF and test predictions
xgb_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, bins), 1):
    print(f"\n========== Fold {fold} ==========")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # XGBoost
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],  # Add train for monitoring
        verbose=100
    )
    best_iter = xgb_model.get_booster().best_iteration
    print(f"XGBoost best_iteration (Fold {fold}): {best_iter}")
    xgb_oof[valid_idx] = xgb_model.predict(X_valid)
    xgb_test += xgb_model.predict(X_test) / n_splits

    # Print train/valid RMSE at best iter (for overfitting check)
    train_rmse = mean_squared_error(y_train, xgb_model.predict(X_train), squared=False)
    valid_rmse = mean_squared_error(y_valid, xgb_oof[valid_idx], squared=False)
    print(f"XGBoost Train RMSE (Fold {fold}): {train_rmse:.5f}, Valid RMSE: {valid_rmse:.5f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],  # Add train for monitoring
        eval_metric='rmse' # Log every 100 iterations
    )
    best_iter = lgb_model.best_iteration_
    print(f"LightGBM best_iteration (Fold {fold}): {best_iter}")
    lgb_oof[valid_idx] = lgb_model.predict(X_valid)
    lgb_test += lgb_model.predict(X_test) / n_splits

    # Print train/valid RMSE
    train_rmse = mean_squared_error(y_train, lgb_model.predict(X_train), squared=False)
    valid_rmse = mean_squared_error(y_valid, lgb_oof[valid_idx], squared=False)
    print(f"LightGBM Train RMSE (Fold {fold}): {train_rmse:.5f}, Valid RMSE: {valid_rmse:.5f}")

# Stacking: Collect base OOF, train Ridge on full OOF
stack_input_oof = np.column_stack([xgb_oof, lgb_oof])
stack_input_test = np.column_stack([xgb_test, lgb_test])

stack_model = Ridge(alpha=1.0)  # Add regularization for stability
stack_model.fit(stack_input_oof, y)

# OOF predictions (note: slightly biased, but better than per-fold)
stack_oof = stack_model.predict(stack_input_oof)
oof_rmse = mean_squared_error(y, stack_oof, squared=False)
print("\nOOF RMSE:", oof_rmse)


stack_test = stack_model.predict(stack_input_test)
predictions = pd.Series(stack_test, name="accident_risk")
submission = pd.DataFrame({'id': submission.id, 'accident_risk': predictions})
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved. Preview:")
print(submission.head())

