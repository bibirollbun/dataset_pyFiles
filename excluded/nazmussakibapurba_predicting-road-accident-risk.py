# ====================================================
# Kaggle Playground S5E10: Optuna Hyperparameter Optimization
# Goal: Beat 0.05537 with systematic tuning
# ====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings("ignore")

# ============================
# 1. Load Data
# ============================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("âœ… Data Loaded Successfully")
print("Train shape:", train_df.shape)

# ============================
# 2. Advanced Feature Engineering
# ============================
def create_advanced_features(df):
    df_copy = df.copy()
    
    df_copy['speed_curvature'] = df_copy['speed_limit'] * df_copy['curvature']
    df_copy['lanes_speed'] = df_copy['num_lanes'] * df_copy['speed_limit']
    df_copy['curvature_lanes'] = df_copy['curvature'] * df_copy['num_lanes']
    
    weather_risk = {'clear': 0, 'rainy': 1, 'foggy': 2}
    lighting_risk = {'daylight': 0, 'dim': 1, 'night': 2}
    time_risk = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    road_risk = {'urban': 1, 'rural': 1.5, 'highway': 2}
    
    df_copy['weather_risk'] = df_copy['weather'].map(weather_risk)
    df_copy['lighting_risk'] = df_copy['lighting'].map(lighting_risk)
    df_copy['time_risk'] = df_copy['time_of_day'].map(time_risk)
    df_copy['road_type_encoded'] = df_copy['road_type'].map(road_risk)
    
    df_copy['environment_risk'] = (df_copy['weather_risk'] + 
                                   df_copy['lighting_risk'] + 
                                   df_copy['time_risk'])
    
    df_copy['complexity_score'] = (df_copy['curvature'] * df_copy['speed_limit'] * 
                                   df_copy['num_lanes']) / 100
    
    lighting_scores = {'daylight': 0, 'dim': 2, 'night': 3}
    weather_scores = {'clear': 0, 'rainy': 2, 'foggy': 3}
    df_copy['visibility_risk'] = (df_copy['lighting'].map(lighting_scores) + 
                                  df_copy['weather'].map(weather_scores))
    
    time_scores = {'morning': 1, 'afternoon': 1.2, 'evening': 1.5, 'night': 2}
    df_copy['time_amplifier'] = df_copy['time_of_day'].map(time_scores)
    
    df_copy['composite_risk'] = (df_copy['complexity_score'] * 
                                 df_copy['visibility_risk'] * 
                                 df_copy['time_amplifier'] * 
                                 df_copy['road_type_encoded']) / 10
    
    df_copy['peak_hour'] = ((df_copy['time_of_day'].isin(['morning', 'evening'])) & 
                            (df_copy['holiday'] == False)).astype(int)
    
    df_copy['high_risk_combo'] = (
        (df_copy['weather'].isin(['foggy', 'rainy'])) &
        (df_copy['lighting'].isin(['dim', 'night'])) &
        (df_copy['curvature'] > 0.5)
    ).astype(int)
    
    df_copy['speed_to_lanes_ratio'] = df_copy['speed_limit'] / (df_copy['num_lanes'] + 1)
    df_copy['accident_tendency'] = df_copy['num_reported_accidents'] * df_copy['environment_risk']
    
    return df_copy

train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# ============================
# 3. Prepare Features
# ============================
features_to_use = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
    'weather', 'road_signs_present', 'public_road', 'time_of_day', 
    'holiday', 'school_season', 'num_reported_accidents',
    'speed_curvature', 'lanes_speed', 'curvature_lanes', 'weather_risk', 
    'lighting_risk', 'time_risk', 'environment_risk', 'complexity_score', 
    'visibility_risk', 'time_amplifier', 'road_type_encoded', 'composite_risk', 
    'peak_hour', 'high_risk_combo', 'speed_to_lanes_ratio', 'accident_tendency'
]

X = train_df[features_to_use]
y = train_df['accident_risk']
X_test = test_df[features_to_use]

categorical_features = [
    'road_type', 'lighting', 'weather', 'road_signs_present', 
    'public_road', 'time_of_day', 'holiday', 'school_season'
]

for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"âœ… Features prepared: {len(features_to_use)} total")

# ============================
# 4. Prepare data for all models
# ============================
X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()

def prepare_xgboost_features(X_train, X_val, X_test):
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()
    
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                       'road_signs_present', 'public_road', 'holiday', 'school_season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_xgb[col] = le.fit_transform(X_train_xgb[col].astype(str))
        X_val_xgb[col] = le.transform(X_val_xgb[col].astype(str))
        X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    
    return X_train_xgb, X_val_xgb, X_test_xgb

X_train_xgb, X_val_xgb, X_test_xgb = prepare_xgboost_features(X_train, X_val, X_test)
X_train_lgb, X_val_lgb, X_test_lgb = prepare_xgboost_features(X_train, X_val, X_test)

# ============================
# 5. Optuna Optimization for CatBoost
# ============================
print("\n" + "="*60)
print("OPTUNA: OPTIMIZING CATBOOST HYPERPARAMETERS")
print("="*60 + "\n")

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
    }
    
    model = CatBoostRegressor(
        **params,
        cat_features=categorical_features,
        verbose=False,
        random_state=42
    )
    
    model.fit(X_train_cat, y_train)
    preds = model.predict(X_val_cat)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study_cat = optuna.create_study(direction='minimize', pruner=MedianPruner())
study_cat.optimize(objective_catboost, n_trials=30, show_progress_bar=True)

best_cat_params = study_cat.best_params
best_cat_rmse = study_cat.best_value
print(f"\nâœ… Best CatBoost RMSE: {best_cat_rmse:.6f}")
print(f"Best CatBoost params: {best_cat_params}\n")

# ============================
# 6. Optuna Optimization for XGBoost
# ============================
print("="*60)
print("OPTUNA: OPTIMIZING XGBOOST HYPERPARAMETERS")
print("="*60 + "\n")

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
    }
    
    model = XGBRegressor(**params, objective="reg:squarederror", verbosity=0, random_state=42)
    model.fit(X_train_xgb, y_train)
    preds = model.predict(X_val_xgb)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study_xgb = optuna.create_study(direction='minimize', pruner=MedianPruner())
study_xgb.optimize(objective_xgboost, n_trials=30, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
best_xgb_rmse = study_xgb.best_value
print(f"\nâœ… Best XGBoost RMSE: {best_xgb_rmse:.6f}")
print(f"Best XGBoost params: {best_xgb_params}\n")

# ============================
# 7. Optuna Optimization for LightGBM
# ============================
print("="*60)
print("OPTUNA: OPTIMIZING LIGHTGBM HYPERPARAMETERS")
print("="*60 + "\n")

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
    }
    
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
    model.fit(X_train_lgb, y_train)
    preds = model.predict(X_val_lgb)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study_lgb = optuna.create_study(direction='minimize', pruner=MedianPruner())
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)

best_lgb_params = study_lgb.best_params
best_lgb_rmse = study_lgb.best_value
print(f"\nâœ… Best LightGBM RMSE: {best_lgb_rmse:.6f}")
print(f"Best LightGBM params: {best_lgb_params}\n")

# ============================
# 8. Train Final Models with Best Params
# ============================
print("="*60)
print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
print("="*60 + "\n")

cat_model = CatBoostRegressor(
    **best_cat_params,
    cat_features=categorical_features,
    verbose=False,
    random_state=42
)
cat_model.fit(X_train_cat, y_train)
cat_test_pred = cat_model.predict(X_test_cat)

xgb_model = XGBRegressor(**best_xgb_params, objective="reg:squarederror", verbosity=0, random_state=42)
xgb_model.fit(X_train_xgb, y_train)
xgb_test_pred = xgb_model.predict(X_test_xgb)

lgb_model = lgb.LGBMRegressor(**best_lgb_params, verbose=-1, random_state=42)
lgb_model.fit(X_train_lgb, y_train)
lgb_test_pred = lgb_model.predict(X_test_lgb)

print("âœ… All models trained with optimized hyperparameters\n")

# ============================
# 9. Simple Average Ensemble
# ============================
final_pred = (cat_test_pred + xgb_test_pred + lgb_test_pred) / 3
final_pred = np.clip(final_pred, 0, 1)

# ============================
# 10. Create Submission
# ============================
submission_df = pd.DataFrame({
    'id': test_df['id'].values,
    'accident_risk': final_pred
})

submission_df.to_csv('/kaggle/working/lastsubmission3.csv', index=False)

print("="*60)
print("âœ… SUBMISSION READY!")
print("="*60)
print(f"Prediction range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")
print(f"Mean prediction: {final_pred.mean():.6f}")
print(f"\nBest individual RMSEs found:")
print(f"  CatBoost: {best_cat_rmse:.6f}")
print(f"  XGBoost: {best_xgb_rmse:.6f}")
print(f"  LightGBM: {best_lgb_rmse:.6f}")
print(f"\nðŸŽ¯ Expected to beat 0.05537!")


# ====================================================
# Kaggle Playground S5E10: Fast Optuna Optimization
# XGBoost + LightGBM only (no CatBoost - too slow)
# ====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings("ignore")

# ============================
# 1. Load Data
# ============================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("âœ… Data Loaded Successfully")
print("Train shape:", train_df.shape)

# ============================
# 2. Advanced Feature Engineering
# ============================
def create_advanced_features(df):
    df_copy = df.copy()
    
    df_copy['speed_curvature'] = df_copy['speed_limit'] * df_copy['curvature']
    df_copy['lanes_speed'] = df_copy['num_lanes'] * df_copy['speed_limit']
    df_copy['curvature_lanes'] = df_copy['curvature'] * df_copy['num_lanes']
    
    weather_risk = {'clear': 0, 'rainy': 1, 'foggy': 2}
    lighting_risk = {'daylight': 0, 'dim': 1, 'night': 2}
    time_risk = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    road_risk = {'urban': 1, 'rural': 1.5, 'highway': 2}
    
    df_copy['weather_risk'] = df_copy['weather'].map(weather_risk)
    df_copy['lighting_risk'] = df_copy['lighting'].map(lighting_risk)
    df_copy['time_risk'] = df_copy['time_of_day'].map(time_risk)
    df_copy['road_type_encoded'] = df_copy['road_type'].map(road_risk)
    
    df_copy['environment_risk'] = (df_copy['weather_risk'] + 
                                   df_copy['lighting_risk'] + 
                                   df_copy['time_risk'])
    
    df_copy['complexity_score'] = (df_copy['curvature'] * df_copy['speed_limit'] * 
                                   df_copy['num_lanes']) / 100
    
    lighting_scores = {'daylight': 0, 'dim': 2, 'night': 3}
    weather_scores = {'clear': 0, 'rainy': 2, 'foggy': 3}
    df_copy['visibility_risk'] = (df_copy['lighting'].map(lighting_scores) + 
                                  df_copy['weather'].map(weather_scores))
    
    time_scores = {'morning': 1, 'afternoon': 1.2, 'evening': 1.5, 'night': 2}
    df_copy['time_amplifier'] = df_copy['time_of_day'].map(time_scores)
    
    df_copy['composite_risk'] = (df_copy['complexity_score'] * 
                                 df_copy['visibility_risk'] * 
                                 df_copy['time_amplifier'] * 
                                 df_copy['road_type_encoded']) / 10
    
    df_copy['peak_hour'] = ((df_copy['time_of_day'].isin(['morning', 'evening'])) & 
                            (df_copy['holiday'] == False)).astype(int)
    
    df_copy['high_risk_combo'] = (
        (df_copy['weather'].isin(['foggy', 'rainy'])) &
        (df_copy['lighting'].isin(['dim', 'night'])) &
        (df_copy['curvature'] > 0.5)
    ).astype(int)
    
    df_copy['speed_to_lanes_ratio'] = df_copy['speed_limit'] / (df_copy['num_lanes'] + 1)
    df_copy['accident_tendency'] = df_copy['num_reported_accidents'] * df_copy['environment_risk']
    
    return df_copy

train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# ============================
# 3. Prepare Features
# ============================
features_to_use = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
    'weather', 'road_signs_present', 'public_road', 'time_of_day', 
    'holiday', 'school_season', 'num_reported_accidents',
    'speed_curvature', 'lanes_speed', 'curvature_lanes', 'weather_risk', 
    'lighting_risk', 'time_risk', 'environment_risk', 'complexity_score', 
    'visibility_risk', 'time_amplifier', 'road_type_encoded', 'composite_risk', 
    'peak_hour', 'high_risk_combo', 'speed_to_lanes_ratio', 'accident_tendency'
]

X = train_df[features_to_use]
y = train_df['accident_risk']
X_test = test_df[features_to_use]

for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"âœ… Features prepared: {len(features_to_use)} total")

# ============================
# 4. Prepare data for XGB and LGB
# ============================
def prepare_xgboost_features(X_train, X_val, X_test):
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()
    
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                       'road_signs_present', 'public_road', 'holiday', 'school_season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_xgb[col] = le.fit_transform(X_train_xgb[col].astype(str))
        X_val_xgb[col] = le.transform(X_val_xgb[col].astype(str))
        X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    
    return X_train_xgb, X_val_xgb, X_test_xgb

X_train_xgb, X_val_xgb, X_test_xgb = prepare_xgboost_features(X_train, X_val, X_test)
X_train_lgb, X_val_lgb, X_test_lgb = prepare_xgboost_features(X_train, X_val, X_test)

# ============================
# 5. Optuna Optimization for XGBoost (50 trials - faster)
# ============================
print("\n" + "="*60)
print("OPTUNA: OPTIMIZING XGBOOST HYPERPARAMETERS (50 trials)")
print("="*60 + "\n")

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
    }
    
    model = XGBRegressor(**params, objective="reg:squarederror", verbosity=0, random_state=42)
    model.fit(X_train_xgb, y_train)
    preds = model.predict(X_val_xgb)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study_xgb = optuna.create_study(direction='minimize', pruner=MedianPruner())
study_xgb.optimize(objective_xgboost, n_trials=100, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
best_xgb_rmse = study_xgb.best_value
print(f"\nâœ… Best XGBoost RMSE: {best_xgb_rmse:.6f}")
print(f"Best XGBoost params: {best_xgb_params}\n")

# ============================
# 6. Optuna Optimization for LightGBM (50 trials - faster)
# ============================
print("="*60)
print("OPTUNA: OPTIMIZING LIGHTGBM HYPERPARAMETERS (50 trials)")
print("="*60 + "\n")

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
    }
    
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
    model.fit(X_train_lgb, y_train)
    preds = model.predict(X_val_lgb)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

study_lgb = optuna.create_study(direction='minimize', pruner=MedianPruner())
study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True)

best_lgb_params = study_lgb.best_params
best_lgb_rmse = study_lgb.best_value
print(f"\nâœ… Best LightGBM RMSE: {best_lgb_rmse:.6f}")
print(f"Best LightGBM params: {best_lgb_params}\n")

# ============================
# 7. Train Final Models
# ============================
print("="*60)
print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
print("="*60 + "\n")

xgb_model = XGBRegressor(**best_xgb_params, objective="reg:squarederror", verbosity=0, random_state=42)
xgb_model.fit(X_train_xgb, y_train)
xgb_test_pred = xgb_model.predict(X_test_xgb)

lgb_model = lgb.LGBMRegressor(**best_lgb_params, verbose=-1, random_state=42)
lgb_model.fit(X_train_lgb, y_train)
lgb_test_pred = lgb_model.predict(X_test_lgb)

print("âœ… Both models trained with optimized hyperparameters\n")

# ============================
# 8. Ensemble with Best Weights
# ============================
# Use inverse RMSE weighting
weight_xgb = 1 / best_xgb_rmse
weight_lgb = 1 / best_lgb_rmse
total_weight = weight_xgb + weight_lgb

weight_xgb /= total_weight
weight_lgb /= total_weight

final_pred = weight_xgb * xgb_test_pred + weight_lgb * lgb_test_pred
final_pred = np.clip(final_pred, 0, 1)

# ============================
# 9. Create Submission
# ============================
submission_df = pd.DataFrame({
    'id': test_df['id'].values,
    'accident_risk': final_pred
})

submission_df.to_csv('/kaggle/working/finalsubmission100.csv', index=False)

print("="*60)
print("âœ… SUBMISSION READY!")
print("="*60)
print(f"Prediction range: [{final_pred.min():.6f}, {final_pred.max():.6f}]")
print(f"Mean prediction: {final_pred.mean():.6f}")
print(f"\nBest individual RMSEs found:")
print(f"  XGBoost: {best_xgb_rmse:.6f}")
print(f"  LightGBM: {best_lgb_rmse:.6f}")
print(f"\nEnsemble Weights:")
print(f"  XGBoost: {weight_xgb:.4f}")
print(f"  LightGBM: {weight_lgb:.4f}")
print(f"\nExpected Ensemble RMSE: ~{min(best_xgb_rmse, best_lgb_rmse):.6f}")
print(f"ðŸŽ¯ Expected to beat 0.05537!")

