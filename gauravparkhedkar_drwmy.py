import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sys
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ===== Configuration =====
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Core original features (keep these as baseline)
    CORE_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                'X25', 'X532', 'X520', 'X329', 'X383', 'X752', 'X287', 'X298', 'X759', 'X302',
                'X55', 'X56', 'X52', 'X303', 'X51', 'X598', 'X385', 'X603', 'X674',
                'X415', 'X345', 'X174', 'X178', 'X168', 'X612', 
                'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    
    # Feature selection parameters
    FEATURE_SELECTION_SAMPLE_SIZE = 750000  # Use more recent data
    TARGET_FEATURES = 120  # Optimal balance between performance and speed

# ===== Model Parameters =====
# Optimized XGBoost parameters for faster training
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "cpu",
    "colsample_bylevel": 0.5,
    "colsample_bynode": 0.4,
    "colsample_bytree": 0.7,
    "gamma": 1.5,
    "learning_rate": 0.025,  # Slightly higher for faster training
    "max_depth": 18,         # Reduced depth
    "max_leaves": 10,        # Fewer leaves
    "min_child_weight": 15,
    "n_estimators": 1200,    # Fewer estimators but higher LR
    "subsample": 0.08,
    "reg_alpha": 35.0,
    "reg_lambda": 65.0,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}


def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe


# ===== Data Loading and Processing =====
def create_time_decay_weights(n: int, decay: float = 0.92) -> np.ndarray:
    """Create time decay weights emphasizing recent data"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

# ===== Feature Engineering =====
def feature_engineering(df):
    """Enhanced feature engineering with all the proven features"""
    # Original interaction features (proven to work)
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # === MICROSTRUCTURE FEATURES (Proven effective) ===
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median for robustness
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df


# ===== Smart Feature Selection =====
def smart_feature_selection(df, label_col, sample_size=500000, top_k=150):
    """
    Efficient feature selection using recent data samples
    Uses multiple methods and focuses on recent crypto patterns
    """
    print(f"Starting smart feature selection with {len(df)} samples...")
    
    # Use the most recent data for feature selection (crypto patterns change)
    recent_sample_size = min(sample_size, len(df))
    recent_df = df.tail(recent_sample_size).copy()
    print(f"Using {len(recent_df)} recent samples for feature selection")
    
    # Get all feature columns (excluding label)
    feature_cols = [col for col in recent_df.columns if col != label_col]
    print(f"Total features before selection: {len(feature_cols)}")
    
    X_sample = recent_df[feature_cols]
    y_sample = recent_df[label_col]
    
    # Remove features with zero variance or too many missing values
    print("Removing low-variance and high-missing features...")
    valid_features = []
    for col in feature_cols:
        if X_sample[col].var() > 1e-8 and X_sample[col].isna().sum() / len(X_sample) < 0.95:
            valid_features.append(col)
    
    X_sample = X_sample[valid_features]
    print(f"Features after variance/missing filter: {len(valid_features)}")
    
    # Method 1: Correlation with target (fast)
    print("Computing correlations...")
    correlations = {}
    for col in valid_features:
        try:
            corr = abs(pearsonr(X_sample[col], y_sample)[0])
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            continue
    
    # Method 2: Mutual Information (sample for speed)
    print("Computing mutual information...")
    mi_sample_size = min(100000, len(X_sample))
    sample_idx = np.random.choice(len(X_sample), mi_sample_size, replace=False)
    X_mi = X_sample.iloc[sample_idx]
    y_mi = y_sample.iloc[sample_idx]
    
    try:
        mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
        mi_dict = dict(zip(X_mi.columns, mi_scores))
    except:
        mi_dict = {}
    
    # Method 3: L1 regularization feature importance (fast)
    print("Computing L1 regularization scores...")
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_mi)
        lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
        lasso.fit(X_scaled, y_mi)
        l1_scores = abs(lasso.coef_)
        l1_dict = dict(zip(X_mi.columns, l1_scores))
    except:
        l1_dict = {}
    
    # Method 4: Tree-based importance (sample for speed)
    print("Computing tree-based importance...")
    try:
        rf_sample_size = min(50000, len(X_sample))
        rf_idx = np.random.choice(len(X_sample), rf_sample_size, replace=False)
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_sample.iloc[rf_idx], y_sample.iloc[rf_idx])
        tree_scores = rf.feature_importances_
        tree_dict = dict(zip(X_sample.columns, tree_scores))
    except:
        tree_dict = {}
    
    # Combine scores with weights
    print("Combining feature scores...")
    combined_scores = {}
    for col in valid_features:
        score = 0
        score += correlations.get(col, 0) * 0.3  # Correlation weight
        score += mi_dict.get(col, 0) * 0.25      # MI weight
        score += l1_dict.get(col, 0) * 0.25      # L1 weight  
        score += tree_dict.get(col, 0) * 0.2     # Tree weight
        combined_scores[col] = score
    
    # Select top features
    selected_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    final_features = [feat[0] for feat in selected_features[:top_k]]
    
    print(f"Selected {len(final_features)} features")
    print("Top 10 selected features:")
    for i, (feat, score) in enumerate(selected_features[:10]):
        print(f"  {i+1:2d}. {feat:30s} - Score: {score:.4f}")
    
    return final_features


# ===== Model Training with Recent Data Focus =====
def get_model_slices(n_samples: int):
    """Define data slices focusing on recent crypto patterns"""
    return [
        {"name": "recent_95pct", "cutoff": int(0.05 * n_samples)},  # Most recent 95%
        {"name": "recent_90pct", "cutoff": int(0.10 * n_samples)},  # Most recent 90%
        {"name": "recent_85pct", "cutoff": int(0.15 * n_samples)},  # Most recent 85%
        {"name": "recent_80pct", "cutoff": int(0.20 * n_samples)},  # Most recent 80%
print("Utility functions and feature engineering logic defined.")


def load_and_process_data():
    """Load, engineer features, and select best features efficiently"""
    print("Loading data...")
    
    # Load only necessary columns initially for memory efficiency
    initial_cols = Config.CORE_FEATURES + [Config.LABEL_COLUMN]
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=initial_cols)
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.CORE_FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Initial data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Apply feature engineering
    print("Engineering features...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Remove original base features (keep engineered ones)
    to_remove = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    train_df = train_df.drop(columns=to_remove)
    test_df = test_df.drop(columns=to_remove)
    
    print(f"After feature engineering - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Smart feature selection on recent data
    print("Performing smart feature selection...")
    selected_features = smart_feature_selection(
        train_df, 
        Config.LABEL_COLUMN, 
        sample_size=Config.FEATURE_SELECTION_SAMPLE_SIZE,
        top_k=Config.TARGET_FEATURES
    )
    
    # Keep only selected features
    train_df = train_df[selected_features + [Config.LABEL_COLUMN]]
    test_df = test_df[selected_features]
    
    # Memory optimization
    print("Optimizing memory usage...")
    train_df = reduce_mem_usage(train_df, "train")
    test_df = reduce_mem_usage(test_df, "test")
    
    # Update config with selected features
    Config.FEATURES = selected_features
    
    print(f"Final data - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Selected features: {len(Config.FEATURES)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


def train_xgb_model(X_train, y_train, X_valid, y_valid, X_test, sample_weights=None):
    """Train optimized XGBoost model"""
    model = XGBRegressor(**XGB_PARAMS)
    
    # Fit with early stopping for efficiency
    model.fit(
        X_train, y_train, 
        sample_weight=sample_weights,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    
    return valid_pred, test_pred, model



def train_and_evaluate(train_df, test_df):
    """Train models with focus on recent data patterns"""
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    
    # Initialize predictions
    oof_preds = {s["name"]: np.zeros(n_samples) for s in model_slices}
    test_preds = {s["name"]: np.zeros(len(test_df)) for s in model_slices}
    feature_importance = {s["name"]: np.zeros(len(Config.FEATURES)) for s in model_slices}
    
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]
        X_test = test_df[Config.FEATURES]
        
        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            
            # Use recent data slice
            recent_df = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            
            if len(rel_idx) == 0:
                continue
                
            X_train = recent_df.iloc[rel_idx][Config.FEATURES]
            y_train = recent_df.iloc[rel_idx][Config.LABEL_COLUMN]
            
            # Create time decay weights for recent emphasis
            sample_weights = create_time_decay_weights(len(recent_df))[rel_idx]
            
            print(f"  Training {slice_name}: {len(X_train)} samples")
            
            try:
                valid_pred, test_pred, model = train_xgb_model(
                    X_train, y_train, X_valid, y_valid, X_test, sample_weights
                )
                
                # Store predictions for validation samples in this slice
                mask = valid_idx >= cutoff
                if mask.any():
                    oof_preds[slice_name][valid_idx[mask]] = valid_pred[mask]
                
                # For samples outside the slice, use the most comprehensive slice
                if cutoff > 0 and (~mask).any():
                    oof_preds[slice_name][valid_idx[~mask]] = \
                        oof_preds["recent_95pct"][valid_idx[~mask]]
                
                test_preds[slice_name] += test_pred
                feature_importance[slice_name] += model.feature_importances_
                
                # Compute validation score
                valid_corr = pearsonr(y_valid, valid_pred)[0]
                print(f"    {slice_name} validation correlation: {valid_corr:.4f}")
                
            except Exception as e:
                print(f"    Error in {slice_name}: {str(e)}")
                continue
    
    # Average test predictions across folds
    for slice_name in test_preds:
        test_preds[slice_name] /= Config.N_FOLDS
        feature_importance[slice_name] /= Config.N_FOLDS
    
    return oof_preds, test_preds, feature_importance






# ===== Ensemble and Submission =====
def create_smart_ensemble(train_df, oof_preds, test_preds):
    """Create weighted ensemble based on recent performance"""
    print("\nEvaluating slice performance...")
    
    slice_scores = {}
    ensemble_weights = {}
    
    for slice_name in oof_preds:
        # Evaluate on recent data (more relevant for crypto)
        recent_idx = int(0.8 * len(train_df))  # Last 20% for evaluation
        recent_true = train_df.iloc[recent_idx:][Config.LABEL_COLUMN]
        recent_pred = oof_preds[slice_name][recent_idx:]
        
        # Remove zeros (unvalidated samples)
        valid_mask = recent_pred != 0
        if valid_mask.sum() > 0:
            score = pearsonr(recent_true[valid_mask], recent_pred[valid_mask])[0]
            slice_scores[slice_name] = score
            print(f"  {slice_name}: {score:.4f} (recent data correlation)")
        else:
            slice_scores[slice_name] = 0
    
    # Compute ensemble weights (higher weight for better recent performance)
    total_score = sum(max(0, score) for score in slice_scores.values())
    if total_score > 0:
        ensemble_weights = {k: max(0, v) / total_score for k, v in slice_scores.items()}
    else:
        # Equal weights if all scores are poor
        ensemble_weights = {k: 1.0 / len(slice_scores) for k in slice_scores}
    
    print("\nEnsemble weights:")
    for slice_name, weight in ensemble_weights.items():
        print(f"  {slice_name}: {weight:.3f}")
    
    # Create weighted ensemble
    ensemble_test = np.zeros(len(test_preds[list(test_preds.keys())[0]]))
    for slice_name, weight in ensemble_weights.items():
        ensemble_test += weight * test_preds[slice_name]
    
    return ensemble_test, slice_scores, ensemble_weights

def create_submission(train_df, oof_preds, test_preds, submission_df):
    """Create optimized submission"""
    
    # Create smart ensemble
    ensemble_pred, slice_scores, weights = create_smart_ensemble(train_df, oof_preds, test_preds)
    
    # Evaluate ensemble performance
    best_slice = max(slice_scores.items(), key=lambda x: x[1])
    print(f"\nBest individual slice: {best_slice[0]} ({best_slice[1]:.4f})")
    
    # Create submission
    submission = submission_df.copy()
    submission["prediction"] = ensemble_pred
    submission.to_csv("submission.csv", index=False)
    
    print(f"\nSubmission created with ensemble prediction")
    print(f"Ensemble uses {len([w for w in weights.values() if w > 0.01])} slices")
    
    return ensemble_pred


# ===== Main Execution =====
if __name__ == "__main__":
    print("=== Enhanced Crypto Prediction with Smart Feature Selection ===\n")
    
    # Load and process data
    print("Step 1: Loading and processing data...")
    train_df, test_df, submission_df = load_and_process_data()
    
    # Train models
    print("\nStep 2: Training models on recent data slices...")
    oof_preds, test_preds, feature_importance = train_and_evaluate(train_df, test_df)
    
    # Create submission
    print("\nStep 3: Creating optimized submission...")
    final_pred = create_submission(train_df, oof_preds, test_preds, submission_df)
    
    # Print feature importance
    print("\nTop 15 most important features:")
    avg_importance = np.mean(list(feature_importance.values()), axis=0)
    feature_importance_pairs = list(zip(Config.FEATURES, avg_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feat, imp) in enumerate(feature_importance_pairs[:15]):
        print(f"  {i+1:2d}. {feat:35s} - Importance: {imp:.4f}")
    
    print("\n=== Processing Complete! ===")
    print("Files created:")
    print("- submission.csv (optimized ensemble)")
    print(f"- Used {len(Config.FEATURES)} carefully selected features")
    print(f"- Focused on recent crypto market patterns")

