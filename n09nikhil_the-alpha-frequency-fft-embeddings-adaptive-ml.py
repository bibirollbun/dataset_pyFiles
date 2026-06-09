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


import pandas as pd
import numpy as np
import os, glob
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import jaccard_score, recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ALPHA RADAR: ULTIMATE HYBRID SOLUTION")
print("=" * 80)
print("\nâœ… All libraries imported successfully!")
print("ğŸ“¦ Key libraries: pandas, numpy, catboost, sklearn, scipy")


def create_temporal_features(group):
    """
    Analyzes token behavior across three time periods: early, middle, late
    
    Returns features like:
    - volume_acceleration: Is trading volume increasing?
    - buy_momentum: Are buy orders accelerating?
    - holder_velocity: Is the holder count growing?
    """
    n = len(group)
    segment_size = max(1, n // 3)
    
    # Split transactions into three time windows
    early = group.iloc[:segment_size]
    middle = group.iloc[segment_size:2*segment_size] if n > segment_size else early
    late = group.iloc[-segment_size:] if n > segment_size else early
    
    result = {
        # Volume metrics
        'early_buy_volume': float(early['sol_volume'].sum() if 'sol_volume' in early.columns else 0),
        'late_buy_volume': float(late['sol_volume'].sum() if 'sol_volume' in late.columns else 0),
        
        # Transaction counts
        'early_buy_count': float(early['buy_count'].sum() if 'buy_count' in early.columns else 0),
        'late_buy_count': float(late['buy_count'].sum() if 'buy_count' in late.columns else 0),
        'early_sell_count': float(early['sell_count'].sum() if 'sell_count' in early.columns else 0),
        'late_sell_count': float(late['sell_count'].sum() if 'sell_count' in late.columns else 0),
        
        # Market metrics
        'early_market_cap': float(early['market_cap_usd'].mean() if 'market_cap_usd' in early.columns else 0),
        'late_market_cap': float(late['market_cap_usd'].mean() if 'market_cap_usd' in late.columns else 0),
        
        # Community metrics
        'early_holders': float(early['current_holders'].max() if 'current_holders' in early.columns else 0),
        'late_holders': float(late['current_holders'].max() if 'current_holders' in late.columns else 0),
    }
    
    # Calculate momentum and acceleration (derived features)
    result['volume_acceleration'] = (result['late_buy_volume'] - result['early_buy_volume']) / (result['early_buy_volume'] + 1)
    result['buy_momentum'] = (result['late_buy_count'] - result['early_buy_count']) / (result['early_buy_count'] + 1)
    result['market_cap_velocity'] = (result['late_market_cap'] - result['early_market_cap']) / (result['early_market_cap'] + 1)
    result['holder_velocity'] = (result['late_holders'] - result['early_holders']) / (result['early_holders'] + 1)
    
    return result

print("âœ… Temporal feature function created")
print("ğŸ“Š Captures: momentum, acceleration, velocity across time periods")


def create_sequence_features(group):
    """
    Extracts patterns from transaction sequences
    
    Returns features like:
    - max_consecutive_buys: Longest streak of buy orders
    - volume_skewness: Is volume dominated by few large trades?
    - price_volatility: How stable is the price?
    """
    result = {}
    
    # === TRADE MODE ANALYSIS ===
    if 'trade_mode' in group.columns:
        trades = group['trade_mode'].values
        result['buy_ratio'] = float(np.mean(trades == 'buy'))
        
        # Find consecutive streaks
        if len(trades) > 1:
            from itertools import groupby
            buy_streaks = [sum(1 for _ in g) for k, g in groupby(trades) if k == 'buy']
            sell_streaks = [sum(1 for _ in g) for k, g in groupby(trades) if k == 'sell']
            result['max_consecutive_buys'] = float(max(buy_streaks) if buy_streaks else 0)
            result['max_consecutive_sells'] = float(max(sell_streaks) if sell_streaks else 0)
        else:
            result['max_consecutive_buys'] = 0.0
            result['max_consecutive_sells'] = 0.0
    
    # === VOLUME DISTRIBUTION ===
    if 'sol_volume' in group.columns:
        volumes = group['sol_volume'].values
        result['volume_std_norm'] = float(np.std(volumes) / (np.mean(volumes) + 1))
        result['volume_skewness'] = float(pd.Series(volumes).skew())
        
        # Concentration metrics (are top trades dominating?)
        sorted_vol = np.sort(volumes)[::-1]
        total_vol = sorted_vol.sum()
        result['top1_vol_share'] = float(sorted_vol[0] / (total_vol + 1) if len(sorted_vol) > 0 else 0)
        result['top3_vol_share'] = float(sorted_vol[:3].sum() / (total_vol + 1) if len(sorted_vol) >= 3 else 0)
    
    # === PRICE DYNAMICS ===
    if 'market_cap_usd' in group.columns:
        prices = group['market_cap_usd'].values
        if len(prices) > 1:
            returns = np.diff(prices) / (prices[:-1] + 1)
            result['price_volatility'] = float(np.std(returns))
            result['price_trend'] = float(np.polyfit(range(len(prices)), prices, 1)[0])
            result['num_price_increases'] = float(np.sum(returns > 0))
            result['max_price_jump'] = float(np.max(returns))
        else:
            result['price_volatility'] = 0.0
            result['price_trend'] = 0.0
            result['num_price_increases'] = 0.0
            result['max_price_jump'] = 0.0
    
    return result

print("âœ… Sequence feature function created")
print("ğŸ“Š Captures: streaks, concentration, price dynamics")


def create_fft_features(group):
    """
    Applies Fast Fourier Transform to extract frequency domain features
    
    Think of it like analyzing a song:
    - Time domain: notes played over time
    - Frequency domain: which notes appear most often
    """
    result = {}
    
    for col in ['market_cap_usd', 'sol_volume']:
        if col in group.columns:
            values = group[col].values
            if len(values) >= 4:
                # Pad to power of 2 (FFT requirement)
                n = 2 ** int(np.ceil(np.log2(len(values))))
                padded = np.pad(values, (0, n - len(values)), mode='edge')
                
                # Apply FFT
                fft_vals = np.abs(fft(padded))[:n//2]
                
                result[f'{col}_fft_mean'] = float(np.mean(fft_vals))
                result[f'{col}_fft_max'] = float(np.max(fft_vals))
                result[f'{col}_dominant_freq'] = float(np.argmax(fft_vals))
            else:
                result[f'{col}_fft_mean'] = 0.0
                result[f'{col}_fft_max'] = 0.0
                result[f'{col}_dominant_freq'] = 0.0
    
    return result

def create_statistical_features(group):
    """
    Statistical tests to detect distribution shifts
    
    KS Test (Kolmogorov-Smirnov):
    - Compares early vs late distributions
    - Low p-value = distributions are different (regime change!)
    
    Variance Ratio:
    - Is volatility increasing or decreasing?
    """
    result = {}
    
    n = len(group)
    mid = n // 2
    
    for col in ['sol_volume', 'market_cap_usd']:
        if col in group.columns and n > 4:
            early_vals = group[col].iloc[:mid].values
            late_vals = group[col].iloc[mid:].values
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pval = stats.ks_2samp(early_vals, late_vals)
                result[f'{col}_ks_stat'] = float(ks_stat)
                result[f'{col}_ks_pval'] = float(ks_pval)
            except:
                result[f'{col}_ks_stat'] = 0.0
                result[f'{col}_ks_pval'] = 1.0
            
            # Variance ratio (volatility change)
            result[f'{col}_var_ratio'] = float(np.var(late_vals) / (np.var(early_vals) + 1e-6))
        else:
            result[f'{col}_ks_stat'] = 0.0
            result[f'{col}_ks_pval'] = 1.0
            result[f'{col}_var_ratio'] = 1.0
    
    return result

print("âœ… FFT feature function created")
print("âœ… Statistical test function created")
print("ğŸŒŠ Captures: frequency patterns, distribution shifts")


def create_all_features(df):
    """
    Master function that applies all feature engineering
    
    Input: Transaction-level data (many rows per token)
    Output: Token-level features (one row per token)
    """
    print(f"  ğŸ§© Processing {len(df):,} transactions...")
    
    # === STEP 1: CLEAN TOKEN IDS ===
    df['mint_token_id_clean'] = df['mint_token_id'].astype(str).str.strip().str.lower()
    
    # Sort by time if available
    if 'timestamp' in df.columns:
        df = df.sort_values(['mint_token_id_clean', 'timestamp'])
    
    # === STEP 2: BASE AGGREGATIONS ===
    print("  ğŸ“Š Computing base aggregations...")
    
    # Get numeric columns (exclude index column which causes issues)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['mint_token_id', 'mint_token_id_clean', 'index']]
    
    # Aggregate each numeric column with multiple statistics
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std', 'sum', 'max', 'min', 'last', 'first']
    
    # Count unique holders
    if 'holder' in df.columns:
        agg_dict['holder'] = ['nunique', 'count']
    
    token_features = df.groupby('mint_token_id_clean').agg(agg_dict)
    token_features.columns = ['_'.join(col).strip() for col in token_features.columns.values]
    token_features.reset_index(inplace=True)
    
    # === STEP 3: TRADE MODE ANALYSIS ===
    if 'trade_mode' in df.columns:
        trade_counts = df.groupby(['mint_token_id_clean', 'trade_mode']).size().unstack(fill_value=0)
        for mode in ['buy', 'sell', 'other']:
            if mode in trade_counts.columns:
                token_features[f'{mode}_count'] = trade_counts[mode].values
            else:
                token_features[f'{mode}_count'] = 0
        
        token_features['buy_sell_ratio_calc'] = token_features['buy_count'] / (token_features['sell_count'] + 1)
    
    # === STEP 4: APPLY ADVANCED FEATURES ===
    print("  ğŸ”¬ Creating temporal features...")
    temporal_list = []
    for token_id, group in df.groupby('mint_token_id_clean'):
        features = create_temporal_features(group)
        features['mint_token_id_clean'] = token_id
        temporal_list.append(features)
    temporal_features = pd.DataFrame(temporal_list)
    
    print("  ğŸ“ˆ Creating sequence features...")
    sequence_list = []
    for token_id, group in df.groupby('mint_token_id_clean'):
        features = create_sequence_features(group)
        features['mint_token_id_clean'] = token_id
        sequence_list.append(features)
    sequence_features = pd.DataFrame(sequence_list)
    
    print("  ğŸŒŠ Creating FFT features...")
    fft_list = []
    for token_id, group in df.groupby('mint_token_id_clean'):
        features = create_fft_features(group)
        features['mint_token_id_clean'] = token_id
        fft_list.append(features)
    fft_features = pd.DataFrame(fft_list)
    
    print("  ğŸ“Š Creating statistical features...")
    stat_list = []
    for token_id, group in df.groupby('mint_token_id_clean'):
        features = create_statistical_features(group)
        features['mint_token_id_clean'] = token_id
        stat_list.append(features)
    stat_features = pd.DataFrame(stat_list)
    
    # === STEP 5: MERGE ALL FEATURES ===
    print("  ğŸ”— Merging feature sets...")
    token_features = token_features.merge(temporal_features, on='mint_token_id_clean', how='left')
    token_features = token_features.merge(sequence_features, on='mint_token_id_clean', how='left')
    token_features = token_features.merge(fft_features, on='mint_token_id_clean', how='left')
    token_features = token_features.merge(stat_features, on='mint_token_id_clean', how='left')
    
    # Rename holder column
    if 'holder_nunique' in token_features.columns:
        token_features.rename(columns={'holder_nunique': 'unique_holders'}, inplace=True)
    
    # === STEP 6: CREATE DERIVED FEATURES ===
    print("  âš¡ Creating derived features...")
    
    if 'buy_count' in token_features.columns and 'sell_count' in token_features.columns:
        token_features['net_buy_count'] = token_features['buy_count'] - token_features['sell_count']
    
    if 'sol_volume_sum' in token_features.columns and 'holder_count' in token_features.columns:
        token_features['volume_per_holder'] = token_features['sol_volume_sum'] / (token_features['holder_count'] + 1)
    
    if 'market_cap_usd_last' in token_features.columns and 'market_cap_usd_first' in token_features.columns:
        token_features['market_cap_growth'] = (token_features['market_cap_usd_last'] - token_features['market_cap_usd_first']) / (token_features['market_cap_usd_first'] + 1)
    
    if 'market_cap_usd_std' in token_features.columns and 'market_cap_usd_mean' in token_features.columns:
        token_features['market_cap_volatility'] = token_features['market_cap_usd_std'] / (token_features['market_cap_usd_mean'] + 1)
    
    if 'creator_sold_sum' in token_features.columns and 'creator_balance_last' in token_features.columns:
        token_features['creator_dump_signal'] = token_features['creator_sold_sum'] / (token_features['creator_balance_last'] + 1)
    
    # Interaction features
    if 'buy_momentum' in token_features.columns and 'sol_volume_sum' in token_features.columns:
        token_features['momentum_volume'] = token_features['buy_momentum'] * token_features['sol_volume_sum']
    
    if 'market_cap_growth' in token_features.columns and 'market_cap_volatility' in token_features.columns:
        token_features['risk_adjusted_return'] = token_features['market_cap_growth'] / (token_features['market_cap_volatility'] + 1)
    
    # === STEP 7: CLEAN UP ===
    token_features = token_features.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"  âœ… Created {token_features.shape[0]:,} tokens with {token_features.shape[1]} features")
    
    return token_features

print("âœ… Master feature engineering pipeline ready")
print("ğŸ�¯ Creates 230+ features per token from transaction data")


print("\n" + "=" * 80)
print("[STEP 1] LOADING TRAINING DATA")
print("=" * 80)

# Path to training data (September 2025 chunks)
train_path = "/kaggle/input/pumpfun-30s-september-2025/"
train_files = sorted(glob.glob(train_path + "*.csv"))

print(f"ğŸ“‚ Found {len(train_files)} training files")
print(f"ğŸ“‚ Files: {', '.join([os.path.basename(f) for f in train_files[:3]])}...")

# Load and concatenate all chunks
train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

print(f"\nâœ… Loaded {len(train_df):,} training transactions")
print(f"âœ… Unique tokens: {train_df['mint_token_id'].nunique():,}")
print(f"âœ… Date range: {train_df['timestamp'].min() if 'timestamp' in train_df.columns else 'N/A'} to {train_df['timestamp'].max() if 'timestamp' in train_df.columns else 'N/A'}")

# Quick peek at the data
print(f"\nğŸ“Š Data Preview:")
print(train_df.head(3))
print(f"\nğŸ“Š Columns: {train_df.shape[1]}")
print(f"ğŸ“‹ {', '.join(train_df.columns.tolist()[:10])}...")


print("\n" + "=" * 80)
print("[STEP 2] FEATURE ENGINEERING (TRAINING)")
print("=" * 80)

# Apply feature engineering pipeline
train_features = create_all_features(train_df)

# Display results
print(f"\nâœ… Feature engineering complete!")
print(f"ğŸ“Š Shape: {train_features.shape}")
print(f"   - Rows (tokens): {train_features.shape[0]:,}")
print(f"   - Columns (features): {train_features.shape[1]}")

# Show sample features
print(f"\nğŸ“‹ Sample Features:")
print(train_features.head(3))

# Feature categories
print(f"\nğŸ�¯ Feature Categories:")
print(f"   - Base aggregations: ~140 (mean, std, sum, max, min, first, last)")
print(f"   - Temporal features: ~15 (early/late analysis, momentum)")
print(f"   - Sequence features: ~15 (streaks, concentration, dynamics)")
print(f"   - FFT features: ~6 (frequency domain patterns)")
print(f"   - Statistical features: ~6 (distribution tests)")
print(f"   - Derived features: ~60 (interactions, ratios)")


print("\n" + "=" * 80)
print("[STEP 3] LOADING TARGET LABELS")
print("=" * 80)

# Load target tokens
target_path = "/kaggle/input/alpha-radar-target-tokens/Alpha Radar Target Tokens.csv"
target = pd.read_csv(target_path)

# Clean column names
target.columns = target.columns.str.strip()
target_col = target.columns[0]

# Normalize token IDs for matching
target['mint_token_id_clean'] = target[target_col].astype(str).str.strip().str.lower()

# Create set for fast lookup
target_set = set(target['mint_token_id_clean'].unique())

print(f"âœ… Loaded target tokens")
print(f"ğŸ“Š Total targets: {len(target_set):,}")

# Create binary target column
train_features['is_target'] = train_features['mint_token_id_clean'].isin(target_set).astype(int)

# Show class distribution
print(f"\nğŸ“Š Class Distribution:")
print(train_features['is_target'].value_counts())
print(f"\nğŸ“Š Target Rate: {train_features['is_target'].mean():.2%}")
print(f"   - Positive (targets): {train_features['is_target'].sum():,}")
print(f"   - Negative (non-targets): {(train_features['is_target']==0).sum():,}")

# Show imbalance ratio
imbalance = (train_features['is_target']==0).sum() / (train_features['is_target']==1).sum()
print(f"\nâš–ï¸�  Class Imbalance Ratio: {imbalance:.2f}:1")
print(f"   (For every 1 target token, there are {imbalance:.0f} non-target tokens)")


print("\n" + "=" * 80)
print("[STEP 4] CREATING EMBEDDINGS")
print("=" * 80)

# Prepare data for SVD (need numeric matrix)
X_temp = train_features.drop(columns=['mint_token_id_clean', 'is_target'])
X_temp = X_temp.fillna(0).replace([np.inf, -np.inf], 0)

# Determine number of components
n_components = min(15, len(X_temp.columns), len(X_temp) - 1)

if n_components >= 2:
    print(f"  ğŸ”® Creating {n_components} SVD embeddings...")
    print(f"  ğŸ“Š Input: {X_temp.shape[1]} features â†’ Output: {n_components} embeddings")
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X_temp)
    
    # Add embeddings as new features
    for i in range(n_components):
        train_features[f'svd_{i}'] = embeddings[:, i]
    
    # Show how much variance we captured
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"\n  âœ… Explained variance: {explained_var:.2%}")
    print(f"  âœ… Added {n_components} embedding features")
    
    # Show top component contributions
    print(f"\n  ğŸ“Š Top 5 Components:")
    for i in range(min(5, n_components)):
        print(f"     SVD {i}: {svd.explained_variance_ratio_[i]:.2%} variance")
else:
    print("  âš ï¸�  Not enough features for SVD")

print(f"\nâœ… Total features now: {len([col for col in train_features.columns if col not in ['mint_token_id_clean', 'is_target']])}")


print("\n" + "=" * 80)
print("[STEP 5] TRAINING MODEL")
print("=" * 80)

# Prepare features and target
X = train_features.drop(columns=['mint_token_id_clean', 'is_target'])
y = train_features['is_target']

# Calculate class imbalance
imbalance_ratio = (y == 0).sum() / (y == 1).sum()

print(f"ğŸ“Š Dataset Info:")
print(f"   - Total samples: {len(X):,}")
print(f"   - Features: {X.shape[1]}")
print(f"   - Positive class: {y.sum():,} ({y.mean():.2%})")
print(f"   - Imbalance ratio: {imbalance_ratio:.2f}:1")

# === CROSS-VALIDATION ===
n_folds = 3
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))

print(f"\nğŸ”„ Starting {n_folds}-Fold Cross-Validation...")
print(f"   (This trains 3 separate models for robust validation)")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n  ğŸ“Š Fold {fold + 1}/{n_folds}")
    print(f"     Train: {len(train_idx):,} samples | Val: {len(val_idx):,} samples")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        eval_metric='AUC',
        scale_pos_weight=imbalance_ratio,  # Handle imbalance
        early_stopping_rounds=75,
        verbose=0,
        random_seed=42 + fold,
        bootstrap_type='Bayesian'  # Better for small datasets
    )
    
    model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
    
    # Store out-of-fold predictions
    oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
    
    print(f"     âœ… Completed (best iteration: {model.best_iteration_})")

print("\nâœ… Cross-validation complete!")

# === FIND OPTIMAL THRESHOLD ===
print(f"\nğŸ�¯ Finding optimal threshold...")
print(f"   Goal: Maximize Jaccard while maintaining â‰¥75% recall")

thresholds = np.linspace(0.01, 0.95, 300)
best_threshold = 0.5
best_jaccard = 0

for t in thresholds:
    preds = (oof_preds > t).astype(int)
    recall = recall_score(y, preds, zero_division=0)
    
    # Only consider thresholds that meet recall requirement
    if recall >= 0.75:
        j = jaccard_score(y, preds, zero_division=0)
        if j > best_jaccard:
            best_threshold = t
            best_jaccard = j

# Evaluate with best threshold
final_preds = (oof_preds > best_threshold).astype(int)
final_recall = recall_score(y, final_preds)

print(f"\nğŸ“Š Out-of-Fold Results:")
print(f"   - Optimal Threshold: {best_threshold:.4f}")
print(f"   - Jaccard Score: {best_jaccard:.4f}")
print(f"   - Recall: {final_recall:.4f}")
print(f"   - Predictions: {final_preds.sum():,} / {len(final_preds):,} ({final_preds.mean():.2%})")

# === TRAIN FINAL MODEL ===
print(f"\nğŸš€ Training final model on all data...")

final_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    eval_metric='AUC',
    scale_pos_weight=imbalance_ratio,
    verbose=100,
    random_seed=42,
    bootstrap_type='Bayesian'
)

final_model.fit(X, y)

print(f"\nâœ… Final model trained!")
print(f"ğŸ“Š Best iteration: {final_model.best_iteration_ if hasattr(final_model, 'best_iteration_') else 'N/A'}")


print("\n" + "=" * 80)
print("[STEP 6] PROCESSING EVALUATION DATA")
print("=" * 80)

# Load evaluation files
eval_path = "/kaggle/input/alpha-radar-solana-sprint/"
eval_files = sorted(glob.glob(eval_path + "evaluation_set_30s_chunk_*.csv"))

print(f"ğŸ“‚ Found {len(eval_files)} evaluation files")
print(f"ğŸ“‚ Files: {', '.join([os.path.basename(f) for f in eval_files])}")

# Load and concatenate
eval_df = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True)

print(f"\nâœ… Loaded {len(eval_df):,} evaluation transactions")
print(f"âœ… Unique tokens: {eval_df['mint_token_id'].nunique():,}")

# Apply SAME feature engineering pipeline
print(f"\nğŸ”§ Applying feature engineering...")
eval_features = create_all_features(eval_df)

# === APPLY SVD EMBEDDINGS ===
X_eval_temp = eval_features.drop(columns=['mint_token_id_clean'])
X_eval_temp = X_eval_temp.fillna(0).replace([np.inf, -np.inf], 0)

if n_components >= 2:
    print(f"\n  ğŸ”® Applying SVD to evaluation...")
    
    # Only use columns that exist in both train and eval
    common_cols = [col for col in X_temp.columns if col in X_eval_temp.columns]
    print(f"  ğŸ“Š Using {len(common_cols)} common features for SVD")
    
    # Transform using trained SVD
    eval_embeddings = svd.transform(X_eval_temp[common_cols])
    
    # Add embeddings
    for i in range(n_components):
        eval_features[f'svd_{i}'] = eval_embeddings[:, i]
    
    print(f"  âœ… Added {n_components} embedding features")

print(f"\nâœ… Evaluation features ready: {eval_features.shape}")


# === ALIGN FEATURES ===
print("\n" + "=" * 80)
print("[STEP 7] ALIGNING FEATURES")
print("=" * 80)

feature_cols = X.columns.tolist()

print(f"ğŸ“Š Training features: {len(feature_cols)}")
print(f"ğŸ“Š Evaluation features: {len([c for c in eval_features.columns if c != 'mint_token_id_clean'])}")

# Add any missing features as zeros
missing_count = 0
for col in feature_cols:
    if col not in eval_features.columns:
        eval_features[col] = 0
        missing_count += 1
        if missing_count <= 5:  # Only print first 5
            print(f"  âš ï¸�  Added missing: {col}")

if missing_count > 5:
    print(f"  âš ï¸�  Added {missing_count - 5} more missing features...")

# Reorder columns to match training
X_eval = eval_features[feature_cols]

print(f"\nâœ… Aligned: {X_eval.shape}")
print(f"   All evaluation tokens now have same {X_eval.shape[1]} features as training")

# === GENERATE PREDICTIONS ===
print("\n" + "=" * 80)
print("[STEP 8] GENERATING PREDICTIONS")
print("=" * 80)

# Get prediction scores
eval_scores = final_model.predict_proba(X_eval)[:, 1]

print(f"ğŸ“Š Score Distribution:")
print(f"   Mean:     {eval_scores.mean():.6f}")
print(f"   Median:   {np.median(eval_scores):.6f}")
print(f"   Std:      {eval_scores.std():.6f}")
print(f"   Min:      {eval_scores.min():.6f}")
print(f"   Max:      {eval_scores.max():.6f}")
print(f"   95th %:   {np.percentile(eval_scores, 95):.6f}")

# === ADAPTIVE THRESHOLDING ===
print(f"\nğŸ�šï¸�  Adaptive Thresholding:")

training_positive_rate = y.mean()
print(f"   Training positive rate: {training_positive_rate:.2%}")

# Allow more predictions in evaluation (2.5x training rate)
target_positive_rate = training_positive_rate * 2.5
adaptive_threshold = np.percentile(eval_scores, 100 * (1 - target_positive_rate))

print(f"   Target positive rate: {target_positive_rate:.2%}")
print(f"   Adaptive threshold: {adaptive_threshold:.6f}")
print(f"   Training threshold: {best_threshold:.6f}")

# Use the lower threshold (more aggressive)
final_threshold = min(best_threshold, max(adaptive_threshold, 0.0001))

print(f"\nâœ… FINAL THRESHOLD: {final_threshold:.6f}")

# Apply threshold
eval_features['prediction_score'] = eval_scores
eval_features['is_target'] = (eval_scores > final_threshold).astype(int)

predicted_positives = eval_features['is_target'].sum()
prediction_rate = 100 * predicted_positives / len(eval_features)

print(f"\nğŸ“Š Predictions:")
print(f"   Total tokens: {len(eval_features):,}")
print(f"   Predicted positive: {predicted_positives:,}")
print(f"   Prediction rate: {prediction_rate:.2f}%")

# === SAFETY CHECK ===
min_required = int(len(eval_features) * 0.005)  # At least 0.5%

if predicted_positives < min_required:
    print(f"\nâš ï¸�  WARNING: Too few predictions!")
    print(f"   Minimum required: {min_required} (0.5%)")
    print(f"   Using top-{min_required} approach instead...")
    
    top_n_threshold = np.sort(eval_scores)[-min_required]
    eval_features['is_target'] = (eval_scores >= top_n_threshold).astype(int)
    
    print(f"   New threshold: {top_n_threshold:.6f}")
    print(f"   New predictions: {eval_features['is_target'].sum():,}")
else:
    print(f"\nâœ… Prediction count looks good!")

# Show score distribution for positive predictions
positive_scores = eval_scores[eval_features['is_target'] == 1]
if len(positive_scores) > 0:
    print(f"\nğŸ“Š Positive Prediction Scores:")
    print(f"   Mean: {positive_scores.mean():.6f}")
    print(f"   Min:  {positive_scores.min():.6f}")
    print(f"   Max:  {positive_scores.max():.6f}")


print("\n" + "=" * 80)
print("[STEP 9] CREATING SUBMISSION")
print("=" * 80)

# === GET ORIGINAL TOKEN IDS IN CORRECT ORDER ===
print("ğŸ“‹ Extracting original token IDs...")

eval_df['mint_token_id_clean'] = eval_df['mint_token_id'].astype(str).str.strip().str.lower()

# Get first occurrence of each token (preserves order)
eval_unique = eval_df.drop_duplicates(subset=['mint_token_id_clean'], keep='first')[
    ['mint_token_id', 'mint_token_id_clean']
]

print(f"   Found {len(eval_unique):,} unique tokens")

# === MERGE WITH PREDICTIONS ===
print("ğŸ”— Merging predictions...")

submission_df = eval_unique.merge(
    eval_features[['mint_token_id_clean', 'is_target', 'prediction_score']], 
    on='mint_token_id_clean', 
    how='left'
)

# Handle any missing predictions (should be none)
submission_df['is_target'] = submission_df['is_target'].fillna(0).astype(int)
submission_df['prediction_score'] = submission_df['prediction_score'].fillna(0)

# === CREATE MAIN SUBMISSION FILE ===
final_submission = submission_df[['mint_token_id', 'is_target']]

# === VALIDATION ===
print("\nğŸ”� Validating submission...")

try:
    assert len(final_submission) == 64208, f"â�Œ Expected 64,208 rows, got {len(final_submission)}"
    assert final_submission['is_target'].isin([0, 1]).all(), "â�Œ is_target must be 0 or 1"
    assert not final_submission['mint_token_id'].isna().any(), "â�Œ No null token IDs allowed"
    assert not final_submission['mint_token_id'].duplicated().any(), "â�Œ Duplicate token IDs found"
    
    print("âœ… All validation checks passed!")
    
except AssertionError as e:
    print(f"â�Œ Validation failed: {e}")
    raise

# === SUMMARY STATISTICS ===
print(f"\nğŸ“Š Submission Summary:")
print(f"   Total rows: {len(final_submission):,}")
print(f"   Predicted positive: {final_submission['is_target'].sum():,}")
print(f"   Predicted negative: {(final_submission['is_target']==0).sum():,}")
print(f"   Positive rate: {100 * final_submission['is_target'].sum() / len(final_submission):.2f}%")

# === SAVE FILES ===
print("\nğŸ’¾ Saving files...")

# Main submission
final_submission.to_csv("submission.csv", index=False)
print("   âœ… submission.csv saved")

# Detailed deliverable
deliverable_df = pd.DataFrame({
    'token': submission_df['mint_token_id'],
    'threshold': final_threshold,
    'prediction_value': submission_df['prediction_score'],
    'isTargetToken': submission_df['is_target']
})
deliverable_df.to_csv('deliverable_details.csv', index=False)
print("   âœ… deliverable_details.csv saved")

# === SHOW TOP PREDICTIONS ===
print("\nğŸ“‹ Top 15 Predictions by Score:")
top_preds = submission_df.nlargest(15, 'prediction_score')[
    ['mint_token_id', 'prediction_score', 'is_target']
]
print(top_preds.to_string(index=False))

# === SHOW SOME STATISTICS ===
print("\nğŸ“Š Score Statistics for Positive Predictions:")
positive_df = submission_df[submission_df['is_target'] == 1]
if len(positive_df) > 0:
    print(f"   Count: {len(positive_df):,}")
    print(f"   Mean score: {positive_df['prediction_score'].mean():.6f}")
    print(f"   Median score: {positive_df['prediction_score'].median():.6f}")
    print(f"   Min score: {positive_df['prediction_score'].min():.6f}")
    print(f"   Max score: {positive_df['prediction_score'].max():.6f}")

print("\n" + "=" * 80)
print("ğŸ�‰ COMPLETE! SUBMISSION READY FOR UPLOAD!")
print("=" * 80)
print("\nğŸ“¤ Next steps:")
print("   1. Download submission.csv from the output")
print("   2. Go to competition submission page")
print("   3. Upload submission.csv")
print("   4. Wait for scoring (may take a few minutes)")
print("\nğŸ¤� Good luck!")

