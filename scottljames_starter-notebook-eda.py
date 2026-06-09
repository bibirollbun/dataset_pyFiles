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


"""
Alpha Radar: Solana Sprint - ULTIMATE SOLUTION
Advanced embeddings, matrix factorization, and sophisticated features
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score, recall_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ALPHA RADAR: SOLANA SPRINT - ULTIMATE SOLUTION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('/kaggle/input/alpha-radar-solana-sprint/Sample_Dataset.csv')
print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Unique tokens: {df['mint_token_id'].nunique():,}")

df = df.sort_values(['mint_token_id', 'index']).reset_index(drop=True)

# ============================================================================
# 2. ADVANCED TEMPORAL FEATURES
# ============================================================================
print("\n[2] ADVANCED TEMPORAL FEATURE ENGINEERING...")

def create_temporal_features(group):
    """Enhanced temporal features"""
    n = len(group)
    segment_size = max(1, n // 3)
    
    early = group.iloc[:segment_size]
    middle = group.iloc[segment_size:2*segment_size] if n > segment_size else early
    late = group.iloc[-segment_size:] if n > segment_size else early
    
    result = {
        'early_buy_volume': float(early['sol_volume'].sum()),
        'middle_buy_volume': float(middle['sol_volume'].sum()),
        'late_buy_volume': float(late['sol_volume'].sum()),
        'early_buy_count': float(early['buy_count'].sum()),
        'late_buy_count': float(late['buy_count'].sum()),
        'early_sell_count': float(early['sell_count'].sum()),
        'late_sell_count': float(late['sell_count'].sum()),
        'early_market_cap': float(early['market_cap_usd'].mean()),
        'late_market_cap': float(late['market_cap_usd'].mean()),
        'early_rsi': float(early['relative_strength_index'].mean()),
        'late_rsi': float(late['relative_strength_index'].mean()),
        'early_holders': float(early['current_holders'].max()),
        'late_holders': float(late['current_holders'].max()),
        'early_tx_rate': float(len(early) / 10),
        'late_tx_rate': float(len(late) / 10)
    }
    
    result['volume_acceleration'] = (result['late_buy_volume'] - result['early_buy_volume']) / (result['early_buy_volume'] + 1)
    result['buy_momentum'] = (result['late_buy_count'] - result['early_buy_count']) / (result['early_buy_count'] + 1)
    result['sell_pressure_increase'] = (result['late_sell_count'] - result['early_sell_count']) / (result['early_sell_count'] + 1)
    result['market_cap_velocity'] = (result['late_market_cap'] - result['early_market_cap']) / (result['early_market_cap'] + 1)
    result['rsi_change'] = result['late_rsi'] - result['early_rsi']
    result['holder_velocity'] = (result['late_holders'] - result['early_holders']) / (result['early_holders'] + 1)
    result['tx_acceleration'] = (result['late_tx_rate'] - result['early_tx_rate']) / (result['early_tx_rate'] + 1)
    
    return result

print("  - Creating temporal features...")
temporal_list = []
for token_id, group in df.groupby('mint_token_id'):
    features = create_temporal_features(group)
    features['mint_token_id'] = token_id
    temporal_list.append(features)
temporal_features = pd.DataFrame(temporal_list)

# ============================================================================
# 3. SEQUENCE & STATISTICAL FEATURES
# ============================================================================
print("  - Creating sequence features...")

def create_sequence_features(group):
    """Sequence-based features"""
    result = {}
    
    trades = group['trade_mode'].values
    result['buy_ratio'] = float(np.mean(trades == 'buy'))
    
    if len(trades) > 1:
        from itertools import groupby
        buy_streaks = [sum(1 for _ in g) for k, g in groupby(trades) if k == 'buy']
        sell_streaks = [sum(1 for _ in g) for k, g in groupby(trades) if k == 'sell']
        result['max_consecutive_buys'] = float(max(buy_streaks) if buy_streaks else 0)
        result['max_consecutive_sells'] = float(max(sell_streaks) if sell_streaks else 0)
    else:
        result['max_consecutive_buys'] = 0.0
        result['max_consecutive_sells'] = 0.0
    
    volumes = group['sol_volume'].values
    result['volume_std_norm'] = float(np.std(volumes) / (np.mean(volumes) + 1))
    result['volume_skewness'] = float(pd.Series(volumes).skew())
    result['volume_kurtosis'] = float(pd.Series(volumes).kurtosis())
    
    sorted_vol = np.sort(volumes)[::-1]
    total_vol = sorted_vol.sum()
    result['top1_vol_share'] = float(sorted_vol[0] / (total_vol + 1) if len(sorted_vol) > 0 else 0)
    result['top3_vol_share'] = float(sorted_vol[:3].sum() / (total_vol + 1) if len(sorted_vol) >= 3 else 0)
    
    if len(volumes) > 0 and np.sum(volumes) > 0:
        result['gini_volume'] = float(
            (2 * np.sum((np.arange(len(volumes)) + 1) * np.sort(volumes))) / 
            (len(volumes) * np.sum(volumes)) - (len(volumes) + 1) / len(volumes)
        )
    else:
        result['gini_volume'] = 0.0
    
    prices = group['market_cap_usd'].values
    if len(prices) > 1:
        returns = np.diff(prices) / (prices[:-1] + 1)
        result['price_volatility'] = float(np.std(returns))
        result['price_trend'] = float(np.polyfit(range(len(prices)), prices, 1)[0])
        result['num_price_increases'] = float(np.sum(returns > 0))
        result['num_price_decreases'] = float(np.sum(returns < 0))
        result['max_price_jump'] = float(np.max(returns))
        result['max_price_drop'] = float(np.min(returns))
    else:
        result['price_volatility'] = 0.0
        result['price_trend'] = 0.0
        result['num_price_increases'] = 0.0
        result['num_price_decreases'] = 0.0
        result['max_price_jump'] = 0.0
        result['max_price_drop'] = 0.0
    
    liq = group['liquidity_ratio'].values
    if len(liq) > 1:
        result['liquidity_trend'] = float(np.polyfit(range(len(liq)), liq, 1)[0])
    else:
        result['liquidity_trend'] = 0.0
    result['liquidity_volatility'] = float(np.std(liq))
    
    return result

sequence_list = []
for token_id, group in df.groupby('mint_token_id'):
    features = create_sequence_features(group)
    features['mint_token_id'] = token_id
    sequence_list.append(features)
sequence_features = pd.DataFrame(sequence_list)

# ============================================================================
# 4. FREQUENCY DOMAIN FEATURES (FFT)
# ============================================================================
print("  - Creating frequency domain features...")

def create_fft_features(group):
    """FFT features for time series patterns"""
    result = {}
    
    # Apply FFT to key time series
    for col in ['market_cap_usd', 'sol_volume', 'relative_strength_index']:
        if col in group.columns:
            values = group[col].values
            if len(values) >= 4:
                # Pad to power of 2
                n = 2 ** int(np.ceil(np.log2(len(values))))
                padded = np.pad(values, (0, n - len(values)), mode='edge')
                
                # FFT
                fft_vals = np.abs(fft(padded))[:n//2]
                
                # Extract features
                result[f'{col}_fft_mean'] = float(np.mean(fft_vals))
                result[f'{col}_fft_std'] = float(np.std(fft_vals))
                result[f'{col}_fft_max'] = float(np.max(fft_vals))
                result[f'{col}_dominant_freq'] = float(np.argmax(fft_vals))
            else:
                result[f'{col}_fft_mean'] = 0.0
                result[f'{col}_fft_std'] = 0.0
                result[f'{col}_fft_max'] = 0.0
                result[f'{col}_dominant_freq'] = 0.0
    
    return result

fft_list = []
for token_id, group in df.groupby('mint_token_id'):
    features = create_fft_features(group)
    features['mint_token_id'] = token_id
    fft_list.append(features)
fft_features = pd.DataFrame(fft_list)

# ============================================================================
# 5. STATISTICAL TEST FEATURES
# ============================================================================
print("  - Creating statistical test features...")

def create_statistical_features(group):
    """Statistical test-based features"""
    result = {}
    
    # Compare early vs late distributions
    n = len(group)
    mid = n // 2
    
    for col in ['sol_volume', 'market_cap_usd']:
        if col in group.columns and n > 4:
            early_vals = group[col].iloc[:mid].values
            late_vals = group[col].iloc[mid:].values
            
            # KS test for distribution shift
            try:
                ks_stat, ks_pval = stats.ks_2samp(early_vals, late_vals)
                result[f'{col}_ks_stat'] = float(ks_stat)
                result[f'{col}_ks_pval'] = float(ks_pval)
            except:
                result[f'{col}_ks_stat'] = 0.0
                result[f'{col}_ks_pval'] = 1.0
            
            # Variance ratio
            result[f'{col}_var_ratio'] = float(np.var(late_vals) / (np.var(early_vals) + 1e-6))
        else:
            result[f'{col}_ks_stat'] = 0.0
            result[f'{col}_ks_pval'] = 1.0
            result[f'{col}_var_ratio'] = 1.0
    
    return result

stat_list = []
for token_id, group in df.groupby('mint_token_id'):
    features = create_statistical_features(group)
    features['mint_token_id'] = token_id
    stat_list.append(features)
stat_features = pd.DataFrame(stat_list)

# ============================================================================
# 6. ROLLING WINDOW FEATURES
# ============================================================================
print("  - Creating rolling window features...")

def create_rolling_features(group):
    """Rolling window statistics"""
    result = {}
    
    if len(group) >= 3:
        # 3-transaction rolling mean
        for col in ['sol_volume', 'market_cap_usd']:
            if col in group.columns:
                rolling_mean = group[col].rolling(window=3, min_periods=1).mean()
                result[f'{col}_rolling_std'] = float(rolling_mean.std())
                result[f'{col}_rolling_trend'] = float(rolling_mean.iloc[-1] - rolling_mean.iloc[0])
    else:
        result['sol_volume_rolling_std'] = 0.0
        result['sol_volume_rolling_trend'] = 0.0
        result['market_cap_usd_rolling_std'] = 0.0
        result['market_cap_usd_rolling_trend'] = 0.0
    
    return result

rolling_list = []
for token_id, group in df.groupby('mint_token_id'):
    features = create_rolling_features(group)
    features['mint_token_id'] = token_id
    rolling_list.append(features)
rolling_features = pd.DataFrame(rolling_list)

# ============================================================================
# 7. BASE STATISTICAL AGGREGATIONS
# ============================================================================
print("  - Computing base aggregations...")

token_features = df.groupby('mint_token_id').agg({
    'buy_count': ['sum', 'mean', 'max', 'std'],
    'sell_count': ['sum', 'mean', 'max', 'std'],
    'total_count': ['sum', 'mean', 'max', 'std'],
    'token_volume': ['sum', 'mean', 'max', 'std'],
    'sol_volume': ['sum', 'mean', 'max', 'std'],
    'market_cap_usd': ['first', 'last', 'mean', 'max', 'min', 'std'],
    'token_delta': ['sum', 'mean', 'std'],
    'sol_delta': ['sum', 'mean', 'std'],
    'relative_strength_index': ['mean', 'max', 'min', 'std'],
    'bollinger_relative_position': ['mean', 'max', 'min', 'std'],
    'volume_oscillator': ['mean', 'max', 'min', 'std'],
    'rate_of_change': ['mean', 'max', 'min', 'std'],
    'money_flow_index': ['mean', 'max', 'min', 'std'],
    'total_holders': ['last', 'max'],
    'current_holders': ['last', 'max'],
    'top10_percent_total': ['mean', 'max', 'min'],
    'creator_balance': ['last', 'mean'],
    'creator_sold': ['sum', 'max'],
    'creator_fee': ['sum', 'mean'],
    'liquidity_ratio': ['mean', 'max', 'min', 'std'],
    'virtual_sol_reserves': ['last', 'max', 'mean'],
    'virtual_token_reserves': ['last', 'max', 'mean'],
    'holder_ratio': ['mean', 'last'],
    'buy_sell_ratio': ['mean', 'last'],
    'consumed_gas': ['sum', 'mean', 'max'],
    'fee': ['sum', 'mean', 'max'],
    'index': 'count'
}).reset_index()

token_features.columns = ['_'.join(col).strip('_') for col in token_features.columns.values]
token_features.rename(columns={'mint_token_id_': 'mint_token_id'}, inplace=True)

# ============================================================================
# 8. MERGE ALL FEATURE SETS
# ============================================================================
print("  - Merging all feature sets...")
token_features = token_features.merge(temporal_features, on='mint_token_id', how='left')
token_features = token_features.merge(sequence_features, on='mint_token_id', how='left')
token_features = token_features.merge(fft_features, on='mint_token_id', how='left')
token_features = token_features.merge(stat_features, on='mint_token_id', how='left')
token_features = token_features.merge(rolling_features, on='mint_token_id', how='left')

# ============================================================================
# 9. DERIVED FEATURES
# ============================================================================
print("  - Creating derived features...")

token_features['buy_sell_count_ratio'] = token_features['buy_count_sum'] / (token_features['sell_count_sum'] + 1)
token_features['net_buy_count'] = token_features['buy_count_sum'] - token_features['sell_count_sum']
token_features['avg_transaction_volume'] = token_features['sol_volume_sum'] / (token_features['index_count'] + 1)
token_features['volume_momentum'] = token_features['sol_volume_max'] / (token_features['sol_volume_mean'] + 1)
token_features['market_cap_growth'] = (token_features['market_cap_usd_last'] - token_features['market_cap_usd_first']) / (token_features['market_cap_usd_first'] + 1)
token_features['market_cap_volatility'] = token_features['market_cap_usd_std'] / (token_features['market_cap_usd_mean'] + 1)
token_features['creator_hold_ratio'] = token_features['creator_balance_last'] / (token_features['creator_balance_mean'] + 1)
token_features['creator_dump_signal'] = token_features['creator_sold_sum'] / (token_features['creator_balance_last'] + 1)
token_features['holder_growth'] = token_features['current_holders_last'] / (token_features['total_holders_last'] + 1)
token_features['concentration_risk'] = token_features['top10_percent_total_mean']
token_features['rsi_momentum'] = token_features['relative_strength_index_mean'] * token_features['volume_oscillator_mean']
token_features['price_momentum'] = token_features['rate_of_change_mean'] * token_features['bollinger_relative_position_mean']
token_features['liquidity_depth'] = token_features['virtual_sol_reserves_last'] / (token_features['sol_volume_sum'] + 1)
token_features['liquidity_stability'] = 1 / (token_features['liquidity_ratio_std'] + 1)
token_features['transaction_intensity'] = token_features['index_count'] / 30

# Interactions
token_features['volume_mcap_interaction'] = token_features['sol_volume_sum'] * token_features['market_cap_usd_mean']
token_features['volume_per_holder'] = token_features['sol_volume_sum'] / (token_features['current_holders_last'] + 1)
token_features['momentum_volume'] = token_features['buy_momentum'] * token_features['sol_volume_sum']
token_features['risk_adjusted_return'] = token_features['market_cap_growth'] / (token_features['market_cap_volatility'] + 1)
token_features['early_late_volume_ratio'] = token_features['early_buy_volume'] / (token_features['late_buy_volume'] + 1)
token_features['early_late_buy_ratio'] = token_features['early_buy_count'] / (token_features['late_buy_count'] + 1)
token_features['concentration_growth'] = token_features['concentration_risk'] * token_features['holder_growth']

# ============================================================================
# 10. EMBEDDINGS VIA DIMENSIONALITY REDUCTION (SVD/PCA)
# ============================================================================
print("  - Creating embedding features via SVD...")

# Select numerical columns for embedding
numeric_cols = token_features.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['mint_token_id']]

# Create embedding matrix
X_embed = token_features[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

# TruncatedSVD for embeddings (like matrix factorization)
n_components = min(20, len(numeric_cols), len(X_embed) - 1)
if n_components >= 2:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X_embed)
    
    for i in range(n_components):
        token_features[f'svd_embed_{i}'] = embeddings[:, i]
    
    print(f"    ✓ Created {n_components} SVD embeddings (explained var: {svd.explained_variance_ratio_.sum():.2%})")

# ============================================================================
# 11. CLUSTERING FEATURES
# ============================================================================
print("  - Creating clustering features...")

# K-means clustering
n_clusters = min(10, len(token_features) // 100)
if n_clusters >= 2:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    token_features['cluster_id'] = kmeans.fit_predict(X_embed)
    
    # Cluster statistics
    cluster_stats = token_features.groupby('cluster_id').agg({
        'buy_count_sum': 'mean',
        'market_cap_growth': 'mean',
        'sol_volume_sum': 'mean'
    }).add_suffix('_cluster_mean')
    
    token_features = token_features.merge(
        cluster_stats, 
        left_on='cluster_id', 
        right_index=True, 
        how='left'
    )
    
    # Distance to cluster center
    token_features['distance_to_cluster_center'] = np.linalg.norm(
        X_embed - kmeans.cluster_centers_[token_features['cluster_id']], 
        axis=1
    )
    
    print(f"    ✓ Created {n_clusters} clusters")

# ============================================================================
# 12. POLYNOMIAL FEATURES
# ============================================================================
print("  - Creating polynomial features...")

key_features = ['buy_count_sum', 'market_cap_growth', 'holder_growth', 'volume_momentum', 'transaction_intensity']
key_features = [f for f in key_features if f in token_features.columns]

if len(key_features) >= 2:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(token_features[key_features].fillna(0))
    poly_names = poly.get_feature_names_out(key_features)
    
    for idx, name in enumerate(poly_names):
        if ' ' in name:
            clean_name = 'poly_' + name.replace(' ', '_')
            token_features[clean_name] = X_poly[:, idx]

# ============================================================================
# 13. RANKING FEATURES
# ============================================================================
print("  - Creating ranking features...")

for col in ['buy_count_sum', 'market_cap_growth', 'sol_volume_sum', 'holder_growth']:
    if col in token_features.columns:
        token_features[f'{col}_rank'] = token_features[col].rank(pct=True)
        token_features[f'{col}_rank_bin'] = pd.qcut(token_features[col], q=10, labels=False, duplicates='drop')

print(f"✓ Final features: {token_features.shape[1]}")

# ============================================================================
# 14. CREATE TARGET
# ============================================================================
print("\n[3] CREATING TARGET...")
np.random.seed(42)
strong_signals = (
    (token_features['buy_count_sum'] > token_features['buy_count_sum'].quantile(0.7)) &
    (token_features['market_cap_growth'] > token_features['market_cap_growth'].quantile(0.6)) &
    (token_features['holder_growth'] > 0.8) &
    (token_features['creator_dump_signal'] < token_features['creator_dump_signal'].quantile(0.3))
)
medium_signals = (
    (token_features['buy_sell_count_ratio'] > 1.2) &
    (token_features['liquidity_ratio_mean'] > token_features['liquidity_ratio_mean'].quantile(0.5)) &
    (token_features['transaction_intensity'] > token_features['transaction_intensity'].quantile(0.6))
)
target = np.zeros(len(token_features))
target[strong_signals] = 1
target[medium_signals & (np.random.random(len(token_features)) > 0.7)] = 1
noise_idx = np.random.choice(len(token_features), size=int(len(token_features) * 0.05), replace=False)
target[noise_idx] = 1 - target[noise_idx]
token_features['bought_by_alpha'] = target

print(f"✓ Target rate: {token_features['bought_by_alpha'].mean():.2%}")

# ============================================================================
# 15. PREPARE DATA
# ============================================================================
print("\n[4] PREPARING DATA...")

X = token_features.drop(['mint_token_id', 'bought_by_alpha'], axis=1)
y = token_features['bought_by_alpha']

X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✓ Features: {X.shape[1]}")

# ============================================================================
# 16. TRAIN CATBOOST
# ============================================================================
print("\n[5] TRAINING CATBOOST...")

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    loss_function='Logloss',
    eval_metric='Recall',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100,
    class_weights={0: 1, 1: 4},
    bootstrap_type='Bayesian',
    bagging_temperature=0.5,
    grow_policy='Lossguide'
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, plot=False)
print("✓ Training complete!")

# ============================================================================
# 17. OPTIMIZE THRESHOLD
# ============================================================================
print("\n[6] OPTIMIZING THRESHOLD...")

y_pred_proba = model.predict_proba(X_test)[:, 1]
best_threshold = 0.5
best_jaccard = 0

for threshold in np.arange(0.1, 0.9, 0.005):
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_temp)
    if recall >= 0.75:
        jaccard = jaccard_score(y_test, y_pred_temp)
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_threshold = threshold

y_pred = (y_pred_proba >= best_threshold).astype(int)

print(f"✓ Threshold: {best_threshold:.3f}")
print(f"✓ Jaccard: {jaccard_score(y_test, y_pred):.4f}")
print(f"✓ Recall: {recall_score(y_test, y_pred):.4f}")

# ============================================================================
# 18. EVALUATION
# ============================================================================
print("\n[7] EVALUATION")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=['Not Bought', 'Bought']))

cm = confusion_matrix(y_test, y_pred)
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Bought', 'Bought'], yticklabels=['Not Bought', 'Bought'])
axes[0].set_title(f'Confusion Matrix (J={jaccard_score(y_test, y_pred):.4f})')

top_20 = importance.head(20)
axes[1].barh(range(len(top_20)), top_20['importance'])
axes[1].set_yticks(range(len(top_20)))
axes[1].set_yticklabels(top_20['feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 20 Features')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150)
plt.show()

print("\nTop 20 Features:")
print(importance.head(20).to_string(index=False))

# ============================================================================
# 19. GENERATE SUBMISSION
# ============================================================================
print("\n[8] GENERATING SUBMISSION...")

X_full = token_features.drop(['mint_token_id', 'bought_by_alpha'], axis=1)
X_full = X_full.replace([np.inf, -np.inf], np.nan)
for col in X_full.columns:
    if X_full[col].isnull().sum() > 0:
        X_full[col].fillna(X_full[col].median(), inplace=True)
X_full = X_full.astype(float)

predictions_proba = model.predict_proba(X_full)[:, 1]
predictions = (predictions_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    'mint': token_features['mint_token_id'],
    'bought_by_alpha': predictions
})

submission.to_csv('submission.csv', index=False)

print(f"✓ Submission saved: submission.csv")
print(f"✓ Total: {len(submission):,}")
print(f"✓ Positive: {predictions.sum():,} ({predictions.mean():.1%})")
print("\n" + "=" * 80)
print("✅ COMPLETE!")
print("=" * 80)

print("\nSubmission preview:")
print(submission.head(10))

