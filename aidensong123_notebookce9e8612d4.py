import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import scipy.stats
import warnings
import gc
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
input_path = '/kaggle/input/'
comp_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
data_path = os.path.join(input_path, comp_folders[0]) if comp_folders else input_path

TRAIN_PATH = os.path.join(data_path, 'train.csv')
TEST_PATH = os.path.join(data_path, 'test.csv')
ORIG_PATH = os.path.join(input_path, 'simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
SUBMISSION_PATH = 'submission.csv'

SEED = 42
N_FOLDS = 5
SEEDS = [42, 128, 256, 510, 1024]

# ============================================================
# CRITICAL: BAYES OPTIMAL FEATURE
# ============================================================
def add_bayes_optimal(df):
    """Bayesian optimal score - CRITICAL for performance"""
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
    df['bayes_simple'] = f(df)  # Also add the simple version
    return df

# ============================================================
# NEW: ORIGINAL DATASET AGGREGATIONS (POWERFUL!)
# ============================================================
def add_original_features(train, test, orig):
    """Add aggregated features from original dataset"""
    print("  Adding original dataset aggregations...")
    
    base_cols = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
                 'weather', 'road_signs_present', 'public_road', 'time_of_day', 
                 'holiday', 'school_season', 'num_reported_accidents']
    
    target = 'accident_risk'
    
    # ensure orig has the expected target column
    if target not in orig.columns:
        raise ValueError("orig dataset missing 'accident_risk' column")
    
    for col in base_cols:
        if col not in orig.columns:
            # skip if orig doesn't have this column
            continue
        
        # Align dtypes between orig and train/test for reliable grouping/merging
        # If column is object-like in train/test, keep strings; otherwise leave numerics as-is.
        # (We don't force everything to string; we only cast object-like ones.)
        # Use copies to avoid mutating original DataFrames outside scope.
        orig_col = orig[col].copy()
        
        # Compute group stats from orig
        agg = orig.groupby(col)[target].agg(['mean', 'std', 'count']).reset_index()
        agg.columns = [col, f'orig_{col}_mean', f'orig_{col}_std', f'orig_{col}_count']
        
        # Merge; if types mismatch this will produce NaNs â€” we fill afterwards
        train = train.merge(agg, on=col, how='left')
        test = test.merge(agg, on=col, how='left')
        
        # Fill missing with sensible defaults
        global_mean = orig[target].mean()
        train[f'orig_{col}_mean'].fillna(global_mean, inplace=True)
        test[f'orig_{col}_mean'].fillna(global_mean, inplace=True)
        train[f'orig_{col}_std'].fillna(0, inplace=True)
        test[f'orig_{col}_std'].fillna(0, inplace=True)
        train[f'orig_{col}_count'].fillna(0, inplace=True)
        test[f'orig_{col}_count'].fillna(0, inplace=True)
    
    return train, test

# ============================================================
# NEW: SAFE TARGET ENCODING
# ============================================================
def add_target_encoding(train, test, y, cat_cols, n_folds=5):
    """Safe target encoding using cross-validation to prevent leakage.
       This implementation avoids assigning floats into categorical columns by
       mapping on a non-categorical view and writing results into fresh float columns.
    """
    print("  Adding target encoding...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    # Ensure y is aligned with train (train must have the target column)
    if y.name not in train.columns:
        # If train doesn't have the target column, set it from y (most callers pass full train)
        train[y.name] = y.values
    
    for col in cat_cols:
        new_col = f'{col}_te'
        # Create the new column as float (so assigning floats into it is safe)
        train[new_col] = 0.0
        
        global_mean = y.mean()
        
        for trn_idx, val_idx in kf.split(train):
            # compute mean on training partition (use train.iloc[...] which is safe)
            te_map = train.iloc[trn_idx].groupby(col)[y.name].mean()
            # Map on a non-categorical view, convert to float
            val_series = train.iloc[val_idx][col].astype(object).map(te_map).fillna(global_mean).astype(float)
            # Assign into the new float column using index alignment (safe)
            train.loc[train.index[val_idx], new_col] = val_series.values
        
        # For test, map using full-train mapping; ensure mapping uses non-categorical view and produce float series
        te_map_full = train.groupby(col)[y.name].mean()
        test[new_col] = test[col].astype(object).map(te_map_full).fillna(global_mean).astype(float)
    
    return train, test

# ============================================================
# FREQUENCY ENCODING + BINNING
# ============================================================
def create_frequency_features(train_df, test_df, cols, num_cols):
    """Add frequency and binning features"""
    train, test = train_df.copy(), test_df.copy()
    
    for col in cols:
        # Frequency encoding
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())
        
        # Binning for numeric columns
        if col in num_cols:
            for q in [5, 10, 20]:  # Added 20 bins
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(
                        train[col], q=q, labels=False, retbins=True, duplicates="drop"
                    )
                    test[f"{col}_bin{q}"] = pd.cut(
                        test[col], bins=bins, labels=False, include_lowest=True
                    )
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0
    
    return train, test

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
def engineer_features(train, test, orig, y):
    """Comprehensive feature engineering with original dataset"""
    
    # Bayes optimal (CRITICAL)
    train = add_bayes_optimal(train)
    test = add_bayes_optimal(test)
    if orig is not None:
        orig = add_bayes_optimal(orig)
    
    # NEW: Add original dataset aggregations
    if orig is not None:
        train, test = add_original_features(train, test, orig)
    
    # Identify feature types
    num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
    cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                'road_signs_present', 'public_road', 'holiday', 'school_season']
    
    # Frequency encoding + binning
    all_cols = num_cols + cat_cols
    train, test = create_frequency_features(train, test, all_cols, num_cols)
    
    # NEW: Target encoding (safe with CV)
    train, test = add_target_encoding(train, test, y, cat_cols, n_folds=N_FOLDS)
    
    # Physics-based interactions
    for df in [train, test]:
        df['speed_x_curve'] = df['speed_limit'] * df['curvature']
        df['danger_score'] = (df['speed_limit'] / 100) * (df['curvature'] ** 2)
        df['kinetic_risk'] = (df['speed_limit'] ** 2) * df['curvature'] / 1000
        
        # Ratios
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
        df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
        df['accidents_x_speed'] = df['num_reported_accidents'] * df['speed_limit']
        df['accidents_x_curve'] = df['num_reported_accidents'] * df['curvature']
        
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
        
        # NEW: More complex flags
        df['extreme_curve'] = (df['curvature'] > 0.7).astype(int)
        df['very_high_speed'] = (df['speed_limit'] >= 80).astype(int)
        
        # Categorical combinations
        df['road_weather'] = df['road_type'].astype(str) + '_' + df['weather'].astype(str)
        df['road_light'] = df['road_type'].astype(str) + '_' + df['lighting'].astype(str)
        df['weather_light'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
        df['time_weather'] = df['time_of_day'].astype(str) + '_' + df['weather'].astype(str)
        
        # Time features
        time_map = {'morning': 8, 'afternoon': 14, 'evening': 19, 'night': 1}
        df['hour'] = df['time_of_day'].map(time_map)
        df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Complex interactions
        df['bad_weather'] = (~df['weather'].isin(['clear'])).astype(int)
        df['poor_lighting'] = (df['lighting'] == 'poor').astype(int)
        df['danger_combo'] = df['high_speed'] * df['sharp_curve'] * df['bad_weather']
        
        # NEW: Interaction with original features
        if 'orig_curvature_mean' in df.columns:
            df['curve_vs_orig'] = df['curvature'] - df['orig_curvature_mean']
        if 'orig_speed_limit_mean' in df.columns:
            df['speed_vs_orig'] = df['speed_limit'] - df['orig_speed_limit_mean']
    
    # Convert categoricals
    cat_cols_extended = cat_cols + ['road_weather', 'road_light', 'weather_light', 'time_weather']
    for col in cat_cols_extended:
        if col in train.columns:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
    
    return train, test

# ============================================================
# OPTIMIZED MODEL CONFIGS
# ============================================================
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 9,
    'learning_rate': 0.007,  # Slightly lower for better convergence
    'n_estimators': 10000,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 2.0,
    'gamma': 0.01,
    'enable_categorical': True,
    'random_state': SEED,
    'verbosity': 0,
}

LGB_PARAMS = {
    'device': 'gpu',
    'num_leaves': 127,
    'max_depth': 10,
    'learning_rate': 0.007,
    'n_estimators': 10000,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'verbose': -1,
    'force_col_wise': True
}

CAT_PARAMS = {
    'task_type': 'GPU',
    'depth': 9,
    'learning_rate': 0.007,
    'iterations': 10000,
    'l2_leaf_reg': 2.0,
    'border_count': 254,
    'random_seed': SEED,
    'verbose': False
}

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_single_model(X, y, X_test, model_type, seed, n_folds=N_FOLDS, cat_cols=None):
    """Train a single model with given seed"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        X_trn, X_val = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        # Ensure categorical columns are properly typed
        if cat_cols is not None:
            for c in cat_cols:
                if c in X_trn.columns:
                    if not pd.api.types.is_categorical_dtype(X_trn[c]):
                        X_trn[c] = X_trn[c].astype('category')
                    if not pd.api.types.is_categorical_dtype(X_val[c]):
                        X_val[c] = X_val[c].astype('category')
        
        if model_type == 'xgb':
            params = XGB_PARAMS.copy()
            params['random_state'] = seed
            
            X_trn_xgb = X_trn.copy()
            X_val_xgb = X_val.copy()
            
            # Convert any remaining object columns to category for XGBoost
            for col in X_trn_xgb.columns:
                if X_trn_xgb[col].dtype == 'object':
                    X_trn_xgb[col] = X_trn_xgb[col].astype('category')
                    X_val_xgb[col] = X_val_xgb[col].astype('category')
            
            model = xgb.XGBRegressor(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
            model.fit(X_trn_xgb, y_trn, eval_set=[(X_val_xgb, y_val)], verbose=False)
        
        elif model_type == 'lgb':
            params = LGB_PARAMS.copy()
            params['random_state'] = seed
            model = lgb.LGBMRegressor(**params)
            lgb_cat = [c for c in (cat_cols or []) if c in X_trn.columns]
            model.fit(
                X_trn, y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(400, verbose=False)],
                categorical_feature=lgb_cat if lgb_cat else 'auto'
            )
        
        elif model_type == 'cat':
            params = CAT_PARAMS.copy()
            params['random_seed'] = seed
            model = CatBoostRegressor(**params)
            cat_in_fold = [c for c in (cat_cols or []) if c in X_trn.columns]
            model.fit(
                X_trn, y_trn,
                eval_set=(X_val, y_val),
                cat_features=cat_in_fold if cat_in_fold else None,
                verbose=False
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_test) / n_folds
        
        del model
        gc.collect()
    
    return oof, pred

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("ğŸš€ ENHANCED PIPELINE v2: Advanced Features + Original Dataset")
    print("=" * 80)
    
    print("\n[1/5] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Load original dataset
    try:
        orig = pd.read_csv(ORIG_PATH)
        print(f"âœ“ Original dataset loaded: {orig.shape}")
    except Exception:
        print("âš  Original dataset not found or unreadable, skipping orig features")
        orig = None
    
    test_ids = test['id'].values
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    # Prepare target
    y = train['accident_risk'].copy()
    
    print("\n[2/5] Feature engineering with original dataset...")
    train, test = engineer_features(train, test, orig, y)
    
    # Drop duplicates
    train = train.drop_duplicates()
    y = train['accident_risk']
    
    X = train.drop(['id', 'accident_risk'], axis=1)
    X_test = test.drop(['id'], axis=1)
    
    cat_cols = X.select_dtypes(include=['category']).columns.tolist()
    
    print(f"âœ¨ Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"   Categorical: {len(cat_cols)}")
    
    print("\n[3/5] Training ensemble (15 models)...")
    all_oofs = []
    all_preds = []
    model_names = []
    
    # XGBoost with 5 seeds
    for seed in SEEDS:
        print(f"  ğŸŒ² XGBoost (seed={seed})...", end=' ')
        oof, pred = train_single_model(X, y, X_test, 'xgb', seed, cat_cols=cat_cols)
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        model_names.append(f'xgb_{seed}')
    
    # LightGBM with 5 seeds
    for seed in SEEDS:
        print(f"  ğŸŒ³ LightGBM (seed={seed})...", end=' ')
        oof, pred = train_single_model(X, y, X_test, 'lgb', seed, cat_cols=cat_cols)
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        model_names.append(f'lgb_{seed}')
    
    # CatBoost with 5 seeds
    for seed in SEEDS:
        print(f"  ğŸ�± CatBoost (seed={seed})...", end=' ')
        oof, pred = train_single_model(X, y, X_test, 'cat', seed, cat_cols=cat_cols)
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        model_names.append(f'cat_{seed}')
    
    print("\n[4/5] Advanced meta-ensemble...")
    
    # Create meta-features
    X_meta = np.column_stack(all_oofs)
    X_meta_test = np.column_stack(all_preds)
    
    # Add rich statistics
    # Use axis=1 percentiles safely with np.nanpercentile
    q1 = np.nanpercentile(X_meta, 25, axis=1)
    q3 = np.nanpercentile(X_meta, 75, axis=1)
    test_q1 = np.nanpercentile(X_meta_test, 25, axis=1)
    test_q3 = np.nanpercentile(X_meta_test, 75, axis=1)
    
    X_meta = np.column_stack([
        X_meta,
        X_meta.mean(axis=1),
        X_meta.std(axis=1),
        X_meta.max(axis=1),
        X_meta.min(axis=1),
        X_meta.max(axis=1) - X_meta.min(axis=1),  # Range
        q1,  # Q1
        q3,  # Q3
    ])
    
    X_meta_test = np.column_stack([
        X_meta_test,
        X_meta_test.mean(axis=1),
        X_meta_test.std(axis=1),
        X_meta_test.max(axis=1),
        X_meta_test.min(axis=1),
        X_meta_test.max(axis=1) - X_meta_test.min(axis=1),
        test_q1,
        test_q3,
    ])
    
    # Train Ridge meta-model
    alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    meta_model = RidgeCV(alphas=alphas, scoring='neg_root_mean_squared_error', cv=5)
    meta_model.fit(X_meta, y)
    
    meta_oof = meta_model.predict(X_meta)
    meta_pred = meta_model.predict(X_meta_test)
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
    
    print(f"  Ridge meta RMSE: {meta_rmse:.6f} (alpha={meta_model.alpha_})")
    
    # Simple average
    avg_oof = np.mean(all_oofs, axis=0)
    avg_pred = np.mean(all_preds, axis=0)
    avg_rmse = np.sqrt(mean_squared_error(y, avg_oof))
    
    print(f"  Simple avg RMSE: {avg_rmse:.6f}")
    
    # Weighted average (give more weight to better models)
    model_rmses = [np.sqrt(mean_squared_error(y, oof)) for oof in all_oofs]
    model_weights = 1 / (np.array(model_rmses) ** 2)
    model_weights /= model_weights.sum()
    
    weighted_oof = sum(w * oof for w, oof in zip(model_weights, all_oofs))
    weighted_pred = sum(w * pred for w, pred in zip(model_weights, all_preds))
    weighted_rmse = np.sqrt(mean_squared_error(y, weighted_oof))
    
    print(f"  Weighted avg RMSE: {weighted_rmse:.6f}")
    
    print("\n[5/5] Creating final submission...")
    
    # Choose best approach
    best_rmse = min(meta_rmse, avg_rmse, weighted_rmse)
    
    if best_rmse == meta_rmse:
        print("  âœ“ Using Ridge meta-model")
        final_pred = meta_pred
        final_rmse = meta_rmse
    elif best_rmse == weighted_rmse:
        print("  âœ“ Using weighted average")
        final_pred = weighted_pred
        final_rmse = weighted_rmse
    else:
        print("  âœ“ Using simple average")
        final_pred = avg_pred
        final_rmse = avg_rmse
    
    # Clip predictions
    final_pred = np.clip(final_pred, 0, 1)
    
    # Save submission
    submission = pd.DataFrame({'id': test_ids, 'accident_risk': final_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "=" * 80)
    print("ğŸ“Š FINAL RESULTS")
    print("=" * 80)
    print(f"   CV RMSE:          {final_rmse:.6f}")
    print(f"   Your baseline:    0.05552")
    print(f"   Improvement:      {0.05552 - final_rmse:.6f}")
    print(f"   Runtime:          {elapsed:.1f} minutes")
    print(f"   Models:           {len(model_names)}")
    
    print("\nğŸ”‘ NEW FEATURES:")
    print("   âœ“ Original dataset aggregations (mean, std, count)")
    print("   âœ“ Safe target encoding with CV")
    print("   âœ“ 20-bin quantile features")
    print("   âœ“ Interactions with original features")
    print("   âœ“ 15 diverse models (5 XGB + 5 LGB + 5 Cat)")
    print("   âœ“ Advanced meta-ensemble (Ridge + weighted avg)")
    
    print("\nâœ… Submission saved:", SUBMISSION_PATH)
    print("=" * 80)
    
    # Top 5 models
    print("\nğŸ“ˆ Top 5 Individual Models:")
    model_perf = sorted(zip(model_names, model_rmses), key=lambda x: x[1])
    for name, rmse in model_perf[:5]:
        print(f"   {name:15s}: {rmse:.6f}")

if __name__ == "__main__":
    main()


