import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
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
comp_folders = []
if os.path.exists(input_path):
    comp_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
data_path = os.path.join(input_path, comp_folders[0]) if comp_folders else input_path

TRAIN_PATH = os.path.join(data_path, 'train.csv')
TEST_PATH = os.path.join(data_path, 'test.csv')
SUBMISSION_PATH = 'submission.csv'
SEED = 42
N_FOLDS = 5  # ä¿�æŒ� 5 æŠ˜ä»¥å¹³è¡¡é€Ÿåº¦å’Œç¨³å®šæ€§
N_SEEDS = 3  # ä¿�æŒ� 3 ä¸ªç§�å­�

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
    return df

# ============================================================
# FREQUENCY ENCODING + BINNING (Static)
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
            for q in [5, 10]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(
                        train[col], q=q, labels=False, retbins=True, duplicates="drop"
                    )
                    test[f"{col}_bin{q}"] = pd.cut(
                        test[col], bins=bins, labels=False, include_lowest=True
                    )
                except:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0
    
    return train, test

# ============================================================
# STATIC FEATURE ENGINEERING (Pre-CV)
# ============================================================
def engineer_static_features(train, test):
    """Comprehensive static feature engineering (runs before CV)"""
    
    # Bayes optimal (CRITICAL)
    train = add_bayes_optimal(train)
    test = add_bayes_optimal(test)
    
    # Identify feature types
    num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
    cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                'road_signs_present', 'public_road', 'holiday', 'school_season']
    
    # Frequency encoding + binning
    all_cols = num_cols + cat_cols
    train, test = create_frequency_features(train, test, all_cols, num_cols)
    
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
        
        # Categorical combinations
        df['road_weather'] = df['road_type'].astype(str) + '_' + df['weather'].astype(str)
        df['road_light'] = df['road_type'].astype(str) + '_' + df['lighting'].astype(str)
        df['weather_light'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
        
        # Time features
        time_map = {'morning': 8, 'afternoon': 14, 'evening': 19, 'night': 1}
        df['hour'] = df['time_of_day'].map(time_map).fillna(12) # å¡«å……æœªçŸ¥
        df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Complex interactions
        df['bad_weather'] = (~df['weather'].isin(['clear'])).astype(int)
        df['poor_lighting'] = (df['lighting'] == 'poor').astype(int)
        df['danger_combo'] = df['high_speed'] * df['sharp_curve'] * df['bad_weather']
    
    # Define which columns are categorical
    cat_cols_extended = cat_cols + ['road_weather', 'road_light', 'weather_light']
    
    return train, test, cat_cols_extended

# ============================================================
# --- NEW --- DYNAMIC FEATURE ENGINEERING (In-CV)
# ============================================================
def create_dynamic_features(X_trn, y_trn, X_val, X_test, cat_cols):
    """
    Creates Target Encoding and Groupby features inside the CV loop
    to prevent leakage.
    """
    
    # 1. Target Encoding (TE) with Smoothing
    global_mean = y_trn.mean()
    m = 30 # Smoothing factor
    
    for col in cat_cols:
        if col not in X_trn.columns: continue
        
        # Compute mapping on training fold
        mapping = y_trn.groupby(X_trn[col]).agg(['mean', 'count'])
        smooth_map = (mapping['mean'] * mapping['count'] + global_mean * m) / (mapping['count'] + m)
        
        # Apply to all splits
        X_trn[f'{col}_te'] = X_trn[col].map(smooth_map)
        X_val[f'{col}_te'] = X_val[col].map(mapping['mean']) # Use raw mean for val/test
        X_test[f'{col}_te'] = X_test[col].map(mapping['mean'])
        
        # Fill NaNs (e.g., categories in val/test not seen in trn)
        X_trn[f'{col}_te'].fillna(global_mean, inplace=True)
        X_val[f'{col}_te'].fillna(global_mean, inplace=True)
        X_test[f'{col}_te'].fillna(global_mean, inplace=True)

    # 2. Groupby Aggregation (Agg) Features
    # We aggregate key numeric features, *including* the golden feature
    num_to_agg = ['speed_limit', 'curvature', 'num_reported_accidents', 'bayes_optimal', 'kinetic_risk']
    aggs = ['mean', 'std', 'max', 'min']
    
    for col in cat_cols:
        if col not in X_trn.columns: continue
        
        for num in num_to_agg:
            if num not in X_trn.columns: continue
            
            feat_name_prefix = f'{col}_agg_{num}_'
            mapping = X_trn.groupby(col)[num].agg(aggs).add_prefix(feat_name_prefix)
            
            # Merge onto all splits
            X_trn = X_trn.merge(mapping, on=col, how='left')
            X_val = X_val.merge(mapping, on=col, how='left')
            X_test = X_test.merge(mapping, on=col, how='left')

    return X_trn, X_val, X_test

# ============================================================
# PRE-TUNED MODEL CONFIGS
# ============================================================
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 9,
    'learning_rate': 0.008,
    'n_estimators': 8000,
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
    'early_stopping_rounds': 300
}

LGB_PARAMS = {
    'device': 'gpu',
    'num_leaves': 127,
    'max_depth': 10,
    'learning_rate': 0.008,
    'n_estimators': 8000,
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
    'learning_rate': 0.008,
    'iterations': 8000,
    'l2_leaf_reg': 2.0,
    'border_count': 254,
    'random_seed': SEED,
    'verbose': False,
    'early_stopping_rounds': 300
}

# ============================================================
# --- MODIFIED --- TRAINING FUNCTIONS
# ============================================================
def train_single_model(X, y, X_test, model_type, seed, n_folds=N_FOLDS, cat_cols=None):
    """
    Train a single model.
    Dynamic FE (TE, Aggs) is now created INSIDE this loop.
    """
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    # è¿‡æ»¤ä»¥ç¡®ä¿�å�ªä½¿ç”¨ X ä¸­å®�é™…å­˜åœ¨çš„åˆ—
    base_cat_cols = [c for c in cat_cols if c in X.columns]
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        # print(f"--- Fold {fold+1}/{n_folds} ---")
        X_trn, X_val = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        X_tst_fold = X_test.copy()
        
        # åˆ›å»ºåŠ¨æ€�ç‰¹å¾�
        X_trn, X_val, X_tst_fold = create_dynamic_features(
            X_trn, y_trn, X_val, X_tst_fold, base_cat_cols 
        )
        
        # --- BUG ä¿®å¤�: è°ƒæ�¢æ“�ä½œé¡ºåº� ---
        
        # 1. (å�Ÿç¬¬2æ­¥) å…ˆå¯¹é½�åˆ—å¹¶å¡«å…… NaNs
        #    è¿™å¿…é¡»åœ¨è½¬æ�¢ä¸º 'category' ä¹‹å‰�å®Œæˆ�
        #    ä½¿ç”¨ .reindex() æ›´å®‰å…¨ï¼Œå®ƒå�¯ä»¥å¤„ç�†ä¸¢å¤±çš„åˆ—
        X_val = X_val.reindex(columns=X_trn.columns).fillna(0)
        X_tst_fold = X_tst_fold.reindex(columns=X_trn.columns).fillna(0)
        
        # 2. (å�Ÿç¬¬1æ­¥) ç�°åœ¨å†�å°†ç±»åˆ«åˆ—è½¬æ�¢ä¸º 'category'
        #    è¿™å¯¹äº� XGB/LGB/CAT æ­£ç¡®å¤„ç�†è‡³å…³é‡�è¦�
        for c in base_cat_cols:
            X_trn[c] = X_trn[c].astype('category')
            X_val[c] = X_val[c].astype('category')
            X_tst_fold[c] = X_tst_fold[c].astype('category')
        
        # --- ä¿®å¤�ç»“æ�Ÿ ---

        
        if model_type == 'xgb':
            params = XGB_PARAMS.copy()
            params['random_state'] = seed
            model = xgb.XGBRegressor(**{k: v for k, v in params.items() if k != 'early_stopping_rounds'})
            model.fit(
                X_trn, y_trn,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        elif model_type == 'lgb':
            params = LGB_PARAMS.copy()
            params['random_state'] = seed
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_trn, y_trn,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(300, verbose=False)],
                categorical_feature=base_cat_cols 
            )
        
        elif model_type == 'cat':
            params = CAT_PARAMS.copy()
            params['random_seed'] = seed
            model = CatBoostRegressor(**params)
            model.fit(
                X_trn, y_trn,
                eval_set=(X_val, y_val),
                cat_features=base_cat_cols, 
                verbose=False
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_tst_fold) / n_folds
        
        del model, X_trn, X_val, y_trn, y_val, X_tst_fold
        gc.collect()
    
    return oof, pred

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import time
    start_time = time.time()
    
    print("=" * 80)
    print("ğŸš€ HYBRID PIPELINE: Golden Features + Dynamic FE")
    print("=" * 80)
    
    print("\n[1/5] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test_ids = test['id'].values
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    print("\n[2/5] Static feature engineering...")
    # 'static_cat_cols' å­˜å‚¨äº†æ‰€æœ‰å�Ÿå§‹å’Œç»„å�ˆçš„ç±»åˆ«åˆ—å��
    train, test, static_cat_cols = engineer_static_features(train, test)
    
    # Drop duplicates and prepare data
    train = train.drop_duplicates()
    
    X = train.drop(['id', 'accident_risk'], axis=1)
    y = train['accident_risk']
    X_test = test.drop(['id'], axis=1)
    
    # ç¡®ä¿� X_test å’Œ X çš„åˆ—é¡ºåº�ä¸€è‡´
    X_test = X_test[X.columns]
    
    print(f"âœ¨ Static Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    print("\n[3/5] Training diverse models (with Dynamic FE)...")
    all_oofs = []
    all_preds = []
    model_names = []
    
    # Train XGBoost with multiple seeds
    seeds = [42, 128, 256]
    for seed in seeds:
        print(f"\n  ğŸŒ² XGBoost (seed={seed})...", end=' ')
        oof, pred = train_single_model(X, y, X_test, 'xgb', seed, cat_cols=static_cat_cols)
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        model_names.append(f'xgb_{seed}')
    
    # Train LightGBM with multiple seeds
    for seed in seeds:
        print(f"  ğŸŒ³ LightGBM (seed={seed})...", end=' ')
        oof, pred = train_single_model(X, y, X_test, 'lgb', seed, cat_cols=static_cat_cols)
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"RMSE: {rmse:.6f}")
        all_oofs.append(oof)
        all_preds.append(pred)
        model_names.append(f'lgb_{seed}')
    
    # Train CatBoost (1 seed - it's slower)
    print(f"  ğŸ�± CatBoost (seed=42)...", end=' ')
    oof, pred = train_single_model(X, y, X_test, 'cat', 42, cat_cols=static_cat_cols)
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"RMSE: {rmse:.6f}")
    all_oofs.append(oof)
    all_preds.append(pred)
    model_names.append('cat_42')
    
    print("\n[4/5] Meta-model stacking with Ridge...")
    
    # Create meta-features
    X_meta = np.column_stack(all_oofs)
    X_meta_test = np.column_stack(all_preds)
    
    # Add statistics
    X_meta = np.column_stack([
        X_meta,
        X_meta.mean(axis=1),
        X_meta.std(axis=1), # <-- ä¿®å¤�äº† 'X_mefa' æ‹¼å†™é”™è¯¯
        X_meta.max(axis=1),
        X_meta.min(axis=1)
    ])
    
    X_meta_test = np.column_stack([
        X_meta_test,
        X_meta_test.mean(axis=1),
        X_meta_test.std(axis=1),
        X_meta_test.max(axis=1),
        X_meta_test.min(axis=1)
    ])
    
    # Train Ridge meta-model
    alphas = [0.001, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    meta_model = RidgeCV(alphas=alphas, scoring='neg_root_squared_error', cv=5)
    meta_model.fit(X_meta, y)
    
    meta_oof = meta_model.predict(X_meta)
    meta_pred = meta_model.predict(X_meta_test)
    
    meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
    print(f"  Ridge meta-model RMSE: {meta_rmse:.6f}")
    print(f"  Best alpha: {meta_model.alpha_}")
    
    # Also try simple average
    avg_oof = np.mean(all_oofs, axis=0)
    avg_pred = np.mean(all_preds, axis=0)
    avg_rmse = np.sqrt(mean_squared_error(y, avg_oof))
    print(f"  Simple average RMSE: {avg_rmse:.6f}")
    
    print("\n[5/5] Creating final submission...")
    
    # Choose best approach
    if meta_rmse < avg_rmse:
        print("  âœ“ Using Ridge meta-model")
        final_pred = meta_pred
        final_rmse = meta_rmse
    else:
        print("  âœ“ Using simple average")
        final_pred = avg_pred
        final_rmse = avg_rmse
    
    # Clip predictions
    final_pred = np.clip(final_pred, 0, 1)
    
    # Save submission
    submission = pd.DataFrame({
        'id': test_ids,
        'accident_risk': final_pred
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "=" * 80)
    print("ğŸ“Š FINAL RESULTS")
    print("=" * 80)
    print(f"   CV RMSE:          {final_rmse:.6f}")
    print(f"   Runtime:          {elapsed:.1f} minutes")
    print(f"   Prediction range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    
    print("\nğŸ”‘ KEY FEATURES:")
    print("   âœ“ Bayes optimal feature (CRITICAL)")
    print("   âœ“ --- NEW: Target Encoding (Dynamic) ---")
    print("   âœ“ --- NEW: Groupby Aggregations (Dynamic) ---")
    print("   âœ“ Physics-based features")
    print("   âœ“ Multiple model types + seeds")
    print(f"   âœ“ Ridge meta-model stacking")
    
    print("\nâœ… Submission saved:", SUBMISSION_PATH)
    print("=" * 80)
    
    # Model performance breakdown
    print("\nğŸ“ˆ Individual Model Performance:")
    for name, oof in zip(model_names, all_oofs):
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"   {name:12s}: {rmse:.6f}")

if __name__ == "__main__":
    main()

