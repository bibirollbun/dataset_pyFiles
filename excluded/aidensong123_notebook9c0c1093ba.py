import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
import warnings
import gc
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
import os
input_path = '/kaggle/input/'
comp_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
data_path = os.path.join(input_path, comp_folders[0]) if comp_folders else input_path

TRAIN_PATH = os.path.join(data_path, 'train.csv')
TEST_PATH = os.path.join(data_path, 'test.csv')
SUBMISSION_PATH = 'submission.csv'

SEED = 42
N_FOLDS = 7
N_TRIALS = 30  # Optuna trials per model
N_OOF_MODELS = 6  # Number of diverse OOF models

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_bayes_optimal_score(df):
    """Bayesian optimal score with clipping"""
    import scipy.stats
    
    def f(X):
        return (
            0.3 * X["curvature"] + 
            0.2 * (X["lighting"] == "night").astype(int) + 
            0.1 * (X["weather"] != "clear").astype(int) + 
            0.2 * (X["speed_limit"] >= 60).astype(int) + 
            0.1 * (X["num_reported_accidents"] > 2).astype(int)
        )
    
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    
    df['bayes_optimal'] = clip_f(df)
    return df

def engineer_features(df):
    """Essential high-impact features"""
    df = df.copy()
    
    # Bayes optimal
    df = add_bayes_optimal_score(df)
    
    # Physics-based
    df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    df['danger_score'] = (df['speed_limit'] / 100) * (df['curvature'] ** 2)
    df['kinetic_risk'] = (df['speed_limit'] ** 2) * df['curvature'] / 1000
    
    # Ratios
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
    
    # Polynomials
    df['curvature_2'] = df['curvature'] ** 2
    df['curvature_3'] = df['curvature'] ** 3
    df['speed_2'] = df['speed_limit'] ** 2
    df['accidents_2'] = df['num_reported_accidents'] ** 2
    
    # Transforms
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['sqrt_speed'] = np.sqrt(df['speed_limit'])
    df['sqrt_curve'] = np.sqrt(df['curvature'])
    
    # Boolean flags
    df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
    df['sharp_curve'] = (df['curvature'] > 0.5).astype(int)
    df['high_accidents'] = (df['num_reported_accidents'] > 2).astype(int)
    
    # Time features
    time_map = {'morning': 8, 'afternoon': 14, 'evening': 19, 'night': 1}
    df['hour'] = df['time_of_day'].map(time_map)
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Weather interactions
    df['bad_weather'] = (~df['weather'].isin(['clear'])).astype(int)
    df['poor_lighting'] = (df['lighting'] == 'poor').astype(int)
    df['danger_combo'] = df['high_speed'] * df['sharp_curve'] * df['bad_weather']
    
    return df

def encode_features(train_df, test_df):
    """Label encode categorical features"""
    cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day',
                'road_signs_present', 'public_road', 'holiday', 'school_season']
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    
    return train_df, test_df

# ============================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================

def optimize_xgboost(X_train, y_train, X_val, y_val):
    """Optimize XGBoost with Optuna"""
    
    def objective(trial):
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'random_state': SEED,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({'tree_method': 'hist', 'device': 'cuda', 'random_state': SEED, 'verbosity': 0})
    
    return best_params, study.best_value

def optimize_lightgbm(X_train, y_train, X_val, y_val):
    """Optimize LightGBM with Optuna"""
    
    def objective(trial):
        params = {
            'device': 'gpu',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100.0, log=True),
            'random_state': SEED,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({'device': 'gpu', 'random_state': SEED, 'verbose': -1})
    
    return best_params, study.best_value

def optimize_catboost(X_train, y_train, X_val, y_val):
    """Optimize CatBoost with Optuna"""
    
    def objective(trial):
        params = {
            'task_type': 'GPU',
            'depth': trial.suggest_int('depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'iterations': trial.suggest_int('iterations', 1000, 3000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 128, 255),
            'random_seed': SEED,
            'verbose': False
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=100, verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({'task_type': 'GPU', 'random_seed': SEED, 'verbose': False})
    
    return best_params, study.best_value

# ============================================================
# GENERATE DIVERSE OOF META-FEATURES
# ============================================================

def generate_optimized_oof(X, y, X_test, best_params_dict):
    """Generate OOF predictions using optimized hyperparameters"""
    
    print("\nğŸ”® Generating Optimized OOF Meta-Features...")
    
    all_oofs = []
    all_preds = []
    names = []
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # XGBoost with optimized params
    print(f"\n  [1/6] XGBoost (optimized)...", end='')
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = xgb.XGBRegressor(**best_params_dict['xgb'])
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_test) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f" RMSE: {rmse:.6f}")
    all_oofs.append(oof)
    all_preds.append(pred)
    names.append('xgb_opt')
    
    # LightGBM with optimized params
    print(f"  [2/6] LightGBM (optimized)...", end='')
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**best_params_dict['lgb'])
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_test) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f" RMSE: {rmse:.6f}")
    all_oofs.append(oof)
    all_preds.append(pred)
    names.append('lgb_opt')
    
    # CatBoost with optimized params
    print(f"  [3/6] CatBoost (optimized)...", end='')
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = CatBoostRegressor(**best_params_dict['cat'])
        model.fit(X_trn, y_trn, eval_set=(X_val, y_val),
                  early_stopping_rounds=100, verbose=False)
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_test) / N_FOLDS
    
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f" RMSE: {rmse:.6f}")
    all_oofs.append(oof)
    all_preds.append(pred)
    names.append('cat_opt')
    
    # Add 3 more diverse models with different seeds
    for i, seed in enumerate([123, 456, 789]):
        print(f"  [{i+4}/6] XGBoost (seed={seed})...", end='')
        oof = np.zeros(len(X))
        pred = np.zeros(len(X_test))
        
        params = best_params_dict['xgb'].copy()
        params['random_state'] = seed
        
        for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
            X_trn, X_val = X[trn_idx], X[val_idx]
            y_trn, y_val = y[trn_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
            oof[val_idx] = model.predict(X_val)
            pred += model.predict(X_test) / N_FOLDS
        
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f" RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        names.append(f'xgb_{seed}')
    
    return all_oofs, all_preds, names

# ============================================================
# NEURAL NETWORK STACKER
# ============================================================

def build_nn(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(128)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_nn_stacker(X_meta, y, X_meta_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_meta))
    preds = np.zeros(len(X_meta_test))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_meta)
    X_test_scaled = scaler.transform(X_meta_test)
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        print(f"  Fold {fold}/5", end='\r')
        X_trn, X_val = X_scaled[trn_idx], X_scaled[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = build_nn(X_trn.shape[1])
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        
        model.fit(X_trn, y_trn, validation_data=(X_val, y_val),
                  epochs=500, batch_size=512, callbacks=[early_stop], verbose=0)
        
        oof[val_idx] = model.predict(X_val, verbose=0).flatten()
        preds += model.predict(X_test_scaled, verbose=0).flatten() / 5
    
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"\nNeural Network RMSE: {rmse:.6f}")
    return oof, preds

# ============================================================
# OPTIMIZED BLENDING
# ============================================================

def optimize_blend_weights(oof_predictions, y_true):
    """Find optimal blend weights"""
    
    def blend_rmse(weights):
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        blended = sum(w * pred for w, pred in zip(weights, oof_predictions))
        return np.sqrt(mean_squared_error(y_true, blended))
    
    x0 = [1.0] * len(oof_predictions)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - len(oof_predictions)})
    bounds = [(0, None)] * len(oof_predictions)
    
    result = minimize(blend_rmse, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / np.sum(result.x)
    optimal_rmse = blend_rmse(result.x)
    
    return optimal_weights, optimal_rmse

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    import time
    start_time = time.time()
    
    print("="*80)
    print("ğŸš€ ULTIMATE PIPELINE: Optuna Optimization + OOF Meta-Features")
    print("="*80)
    
    print("\n[1/8] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test_ids = test['id'].values
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    if 'accident_risk' in test.columns:
        test = test.drop('accident_risk', axis=1)
    
    print("\n[2/8] Feature engineering...")
    train = engineer_features(train)
    test = engineer_features(test)
    
    print("\n[3/8] Encoding features...")
    train, test = encode_features(train, test)
    
    X = train.drop(['id', 'accident_risk'], axis=1).values
    y = train['accident_risk'].values
    X_test = test.drop(['id'], axis=1).values
    
    print(f"\nâœ¨ Total features: {X.shape[1]}")
    
    print("\n" + "="*80)
    print("[4/8] OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Split for optimization
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(kf.split(X))
    X_train_opt, X_val_opt = X[train_idx], X[val_idx]
    y_train_opt, y_val_opt = y[train_idx], y[val_idx]
    
    print("\nâš¡ Optimizing XGBoost...")
    best_xgb, best_xgb_rmse = optimize_xgboost(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
    print(f"âœ… XGBoost Best RMSE: {best_xgb_rmse:.6f}")
    gc.collect()
    
    print("\nâš¡ Optimizing LightGBM...")
    best_lgb, best_lgb_rmse = optimize_lightgbm(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
    print(f"âœ… LightGBM Best RMSE: {best_lgb_rmse:.6f}")
    gc.collect()
    
    print("\nâš¡ Optimizing CatBoost...")
    best_cat, best_cat_rmse = optimize_catboost(X_train_opt, y_train_opt, X_val_opt, y_val_opt)
    print(f"âœ… CatBoost Best RMSE: {best_cat_rmse:.6f}")
    gc.collect()
    
    best_params_dict = {
        'xgb': best_xgb,
        'lgb': best_lgb,
        'cat': best_cat
    }
    
    print("\n" + "="*80)
    print("[5/8] GENERATING OOF META-FEATURES WITH OPTIMIZED PARAMS")
    print("="*80)
    
    oof_meta_list, pred_meta_list, meta_names = generate_optimized_oof(
        X, y, X_test, best_params_dict
    )
    
    # Add OOF features to original data
    print("\nğŸ“Š Adding OOF meta-features to dataset...")
    X_with_oof = X.copy()
    X_test_with_oof = X_test.copy()
    
    for i, (oof, pred, name) in enumerate(zip(oof_meta_list, pred_meta_list, meta_names)):
        X_with_oof = np.column_stack([X_with_oof, oof])
        X_test_with_oof = np.column_stack([X_test_with_oof, pred])
    
    # Add aggregate statistics
    oof_array = np.column_stack(oof_meta_list)
    pred_array = np.column_stack(pred_meta_list)
    
    X_with_oof = np.column_stack([
        X_with_oof,
        oof_array.mean(axis=1),
        oof_array.std(axis=1),
        oof_array.max(axis=1),
        oof_array.min(axis=1)
    ])
    
    X_test_with_oof = np.column_stack([
        X_test_with_oof,
        pred_array.mean(axis=1),
        pred_array.std(axis=1),
        pred_array.max(axis=1),
        pred_array.min(axis=1)
    ])
    
    print(f"âœ¨ Total features with OOF: {X_with_oof.shape[1]}")
    
    print("\n" + "="*80)
    print("[6/8] TRAINING FINAL MODELS WITH OOF FEATURES")
    print("="*80)
    
    # Train final models with OOF features
    print("\nğŸŒ³ Training final XGBoost...")
    oof_xgb_final = np.zeros(len(X_with_oof))
    pred_xgb_final = np.zeros(len(X_test_with_oof))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_with_oof), 1):
        print(f"  Fold {fold}/{N_FOLDS}", end='\r')
        X_trn, X_val = X_with_oof[trn_idx], X_with_oof[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = xgb.XGBRegressor(**best_xgb)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb_final[val_idx] = model.predict(X_val)
        pred_xgb_final += model.predict(X_test_with_oof) / N_FOLDS
    
    rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb_final))
    print(f"\n  XGBoost (with OOF) RMSE: {rmse_xgb:.6f}")
    
    print("\nğŸŒ³ Training final LightGBM...")
    oof_lgb_final = np.zeros(len(X_with_oof))
    pred_lgb_final = np.zeros(len(X_test_with_oof))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_with_oof), 1):
        print(f"  Fold {fold}/{N_FOLDS}", end='\r')
        X_trn, X_val = X_with_oof[trn_idx], X_with_oof[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**best_lgb)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb_final[val_idx] = model.predict(X_val)
        pred_lgb_final += model.predict(X_test_with_oof) / N_FOLDS
    
    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb_final))
    print(f"\n  LightGBM (with OOF) RMSE: {rmse_lgb:.6f}")
    
    print("\n" + "="*80)
    print("[7/8] NEURAL NETWORK STACKER")
    print("="*80)
    
    # Stack all predictions
    all_oofs = oof_meta_list + [oof_xgb_final, oof_lgb_final]
    all_preds = pred_meta_list + [pred_xgb_final, pred_lgb_final]
    all_names = meta_names + ['xgb_final', 'lgb_final']
    
    X_meta = np.column_stack(all_oofs)
    X_meta_test = np.column_stack(all_preds)
    
    print("\nğŸ§  Training NN stacker...")
    oof_nn, pred_nn = train_nn_stacker(X_meta, y, X_meta_test)
    
    all_oofs.append(oof_nn)
    all_preds.append(pred_nn)
    all_names.append('nn_stacker')
    
    print("\n" + "="*80)
    print("[8/8] FINAL ENSEMBLE OPTIMIZATION")
    print("="*80)
    
    print("\nğŸ”� Optimizing weights...")
    final_weights, final_rmse = optimize_blend_weights(all_oofs, y)
    
    print(f"\nğŸ“Š Model Weights:")
    for name, weight in zip(all_names, final_weights):
        if weight > 0.01:  # Only show significant weights
            print(f"  {name}: {weight:.4f}")
    
    print(f"\nğŸ�¯ Final Ensemble RMSE: {final_rmse:.6f}")
    
    # Generate final predictions
    final_predictions = np.zeros(len(X_test))
    for weight, pred in zip(final_weights, all_preds):
        final_predictions += weight * pred
    
    # Ensure predictions are within valid range (if competition specifies constraints)
    final_predictions = np.clip(final_predictions, 0, None)  # Assuming accident_risk >= 0
    
    # Create submission file
    print("\nğŸ“� Creating submission file...")
    submission = pd.DataFrame({
        'id': test_ids,
        'accident_risk': final_predictions
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"âœ… Submission file saved to {SUBMISSION_PATH}")
    
    # Print execution time
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"\nâ�° Total execution time: {execution_time:.2f} minutes")
    
    return submission

# ============================================================
# EXECUTE PIPELINE
# ============================================================

if __name__ == "__main__":
    submission = main()

