import pandas as pd
import numpy as np
import gc
import warnings
from itertools import combinations
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, prange

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 100)

# Color palette for visualizations
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#3B3B3B'
}

print("✅ Libraries imported")




import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import json
from pathlib import Path


# ── Feature selection ────────────────────────────────────────────────────────

FAST_LGB_PARAMS = {
    'learning_rate': 0.05,       # faster than production rate
    'max_depth': 6,
    'n_estimators': 300,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',
    'verbose': -1,
}


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    importance_threshold: float = 0.0001,
    save_path: str = 'results/selected_features.json',
) -> list[str]:
    """
    Train a fast proxy LGB model and return features above an importance threshold.

    Uses gain-based importance (not split count) because gain measures how much
    each feature actually reduces the loss, not just how often it is used.

    Parameters
    ----------
    importance_threshold : Features with mean importance below this fraction of
                           the top feature's importance are dropped.
                           0.0001 = drop anything contributing less than 0.01% of
                           the top feature's contribution. Conservative by design.

    Returns
    -------
    List of selected feature names, ordered by importance descending.
    """
    print("Running feature selection...")

    proxy = lgb.LGBMRegressor(**FAST_LGB_PARAMS)
    proxy.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    importances = pd.Series(
        proxy.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    # Normalise so top feature = 1.0
    importances_norm = importances / (importances.max() + 1e-9)

    selected = importances_norm[importances_norm >= importance_threshold].index.tolist()
    dropped = importances_norm[importances_norm < importance_threshold].index.tolist()

    proxy_val_preds = proxy.predict(X_val)
    proxy_mae = mean_absolute_error(y_val, proxy_val_preds)

    print(f"  Features: {len(X_train.columns)} → {len(selected)} selected, {len(dropped)} dropped")
    print(f"  Proxy model MAE (all features): {proxy_mae:.5f}")
    print(f"  Top 10 features:")
    for feat, imp in importances_norm.head(10).items():
        print(f"    {feat:<45} {imp:.4f}")

    # Persist for reproducibility and README documentation
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'selected': selected,
            'dropped': dropped,
            'n_selected': len(selected),
            'n_dropped': len(dropped),
            'proxy_mae': float(proxy_mae),
            'top_features': importances_norm.head(20).to_dict(),
        }, f, indent=2)

    return selected


# ── DART configuration ───────────────────────────────────────────────────────

DART_LGB_PARAMS = {
    'boosting_type': 'dart',     # key change: dropout regularization
    'learning_rate': 0.05,       # DART needs higher LR — no early stopping possible
    'max_depth': 8,
    'n_estimators': 500,        # fixed; DART cannot use early stopping
    'num_leaves': 128,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'drop_rate': 0.1,            # fraction of trees dropped per round
    'skip_drop': 0.5,            # probability of skipping dropout entirely
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',
    'verbose': -1,
}


def train_dart_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple:
    """
    Train a LightGBM model in DART mode.

    Why DART improves on extreme values:
    Standard gradient boosting adds trees greedily — later trees correct errors
    of earlier ones, which causes the model to over-rely on a small subset of
    high-influence trees. These dominant trees pull predictions toward the mean.

    DART randomly drops a fraction of existing trees during each boosting round,
    forcing new trees to be useful independently rather than as corrections.
    This distributes predictive contribution more evenly and reduces the
    shrinkage effect that collapses predictions toward zero.

    Limitation: DART cannot use early stopping (the dropout makes validation
    loss non-monotonic). n_estimators is therefore fixed.
    """
    print("  Training LightGBM DART...")

    model = lgb.LGBMRegressor(**DART_LGB_PARAMS)
    # Note: no early_stopping callback — incompatible with DART
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"  DART MAE: {mae:.5f}")

    return model, preds, mae


# ── Comparison utility ───────────────────────────────────────────────────────

def compare_prediction_ranges(
    y_true: np.ndarray,
    standard_preds: np.ndarray,
    dart_preds: np.ndarray,
) -> dict:
    """
    Compare how well each model covers the extreme tails of the target.

    This is the key metric for evaluating DART's benefit — not just MAE
    but whether predictions actually reach the tails of the distribution.
    """
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    results = {
        'actual': {f'p{p}': float(np.percentile(y_true, p)) for p in percentiles},
        'standard': {f'p{p}': float(np.percentile(standard_preds, p)) for p in percentiles},
        'dart': {f'p{p}': float(np.percentile(dart_preds, p)) for p in percentiles},
    }

    print("\n  Prediction range comparison:")
    print(f"  {'Percentile':<12} {'Actual':>10} {'Standard':>10} {'DART':>10}")
    print("  " + "-" * 45)
    for p in percentiles:
        key = f'p{p}'
        print(f"  p{p:<11} {results['actual'][key]:>10.2f} "
              f"{results['standard'][key]:>10.2f} "
              f"{results['dart'][key]:>10.2f}")

    return results


# ── Quantile Regression ──────────────────────────────────────────────────────

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Base params for quantile models — lighter than production to keep training fast
QUANTILE_LGB_BASE = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 300,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball (quantile) loss for a single quantile.
    Equivalent to MAE but asymmetric: penalises over-prediction more at low quantiles
    and under-prediction more at high quantiles.
    """
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)))


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list = None,
) -> tuple[dict, dict, np.ndarray]:
    """
    Train one LightGBM model per quantile using pinball loss.

    Why multiple quantiles instead of just predicting the mean:
    A mean-optimised model minimises symmetric squared or absolute error,
    which pulls predictions toward the centre of the distribution. Pinball
    loss is asymmetric — the q=0.9 model is penalised 9x more for
    under-predicting than over-predicting, forcing it to learn the upper tail.
    Training across multiple quantiles gives us a full conditional distribution
    estimate, not just the conditional mean.

    Returns
    -------
    models       : {quantile: fitted LGBMRegressor}
    pinball_scores : {quantile: pinball loss on validation set}
    median_preds : predictions from the q=0.5 model (comparable to MAE model)
    """
    if quantiles is None:
        quantiles = QUANTILES

    print(f"  Training {len(quantiles)} quantile models...")
    models = {}
    pinball_scores = {}
    median_preds = None

    for q in quantiles:
        params = {
            **QUANTILE_LGB_BASE,
            'objective': 'quantile',
            'alpha': q,          # LightGBM's parameter name for the quantile
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        score = _pinball_loss(y_val.values, preds, q)
        models[q] = model
        pinball_scores[q] = score

        if abs(q - 0.5) < 1e-6:
            median_preds = preds

        print(f"    q={q:.2f}  pinball={score:.5f}  "
              f"pred_range=[{preds.min():.2f}, {preds.max():.2f}]")

    return models, pinball_scores, median_preds


def adaptive_quantile_blend(
    X_val: pd.DataFrame,
    quantile_models: dict,
    imbalance_feature: str = 'signed_imbalance',
) -> np.ndarray:
    """
    Blend quantile predictions using adaptive weights based on order book imbalance.

    Core idea: when the order book shows a large signed imbalance (strong buy or
    sell pressure), extreme price moves become more likely. In those cases we
    should weight the tail quantiles (q=0.1 or q=0.9) more heavily. When the
    book is balanced, the median (q=0.5) is most reliable.

    Weighting scheme:
    - Compute a normalised imbalance signal in [-1, 1]
    - Positive signal (buy pressure) → up-weight q=0.75 and q=0.9
    - Negative signal (sell pressure) → up-weight q=0.1 and q=0.25
    - Near-zero signal → concentrate weight on q=0.5

    This is a simple but principled way to use the distribution information
    without requiring a separate meta-learner.

    Parameters
    ----------
    imbalance_feature : column in X_val to use as the imbalance signal.
                        Falls back to equal weighting if column not found.
    """
    quantiles = sorted(quantile_models.keys())
    preds_matrix = np.column_stack([
        quantile_models[q].predict(X_val) for q in quantiles
    ])   # shape: (n_samples, n_quantiles)

    if imbalance_feature not in X_val.columns:
        # fallback: simple average across quantiles
        print("    Imbalance feature not found — using equal quantile weights")
        return preds_matrix.mean(axis=1)

    # Normalise imbalance to [-1, 1] using tanh (soft clipping)
    raw_imbalance = X_val[imbalance_feature].values
    scale = np.percentile(np.abs(raw_imbalance), 95) + 1e-6
    signal = np.tanh(raw_imbalance / scale)   # shape: (n_samples,)

    # Build weight matrix: shape (n_samples, n_quantiles)
    # Base weight: uniform across quantiles
    n_q = len(quantiles)
    weights = np.ones((len(signal), n_q)) / n_q

    q_arr = np.array(quantiles)   # e.g. [0.1, 0.25, 0.5, 0.75, 0.9]

    for i, s in enumerate(signal):
        if s > 0:
            # Buy pressure: shift weight toward upper quantiles
            # Weight proportional to quantile value, scaled by signal strength
            upper_bonus = q_arr * abs(s)
            w = 1/n_q + upper_bonus
        else:
            # Sell pressure: shift weight toward lower quantiles
            lower_bonus = (1 - q_arr) * abs(s)
            w = 1/n_q + lower_bonus
        weights[i] = w / w.sum()   # normalise to sum to 1

    blended = (preds_matrix * weights).sum(axis=1)
    return blended


def evaluate_quantile_coverage(
    y_true: np.ndarray,
    quantile_models: dict,
    X_val: pd.DataFrame,
    standard_preds: np.ndarray,
    dart_preds: np.ndarray,
    blended_preds: np.ndarray,
) -> dict:
    """
    Compare tail coverage across all prediction methods.

    Reports:
    - Prediction range at key percentiles for all methods vs actuals
    - MAE of the blended quantile prediction vs standard and DART
    - Interval coverage: what % of actuals fall within [q10, q90] predictions
    """
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    print("\n  Full prediction range comparison:")
    print(f"  {'Pct':<6} {'Actual':>8} {'Standard':>10} {'DART':>8} {'Quantile':>10}")
    print("  " + "-" * 46)

    results = {}
    for p in percentiles:
        a  = float(np.percentile(y_true, p))
        s  = float(np.percentile(standard_preds, p))
        d  = float(np.percentile(dart_preds, p))
        q  = float(np.percentile(blended_preds, p))
        results[f'p{p}'] = {'actual': a, 'standard': s, 'dart': d, 'quantile_blend': q}
        print(f"  p{p:<5} {a:>8.2f} {s:>10.2f} {d:>8.2f} {q:>10.2f}")

    # MAE comparison
    mae_standard = mean_absolute_error(y_true, standard_preds)
    mae_dart     = mean_absolute_error(y_true, dart_preds)
    mae_blend    = mean_absolute_error(y_true, blended_preds)
    print(f"\n  MAE — Standard: {mae_standard:.5f} | DART: {mae_dart:.5f} | Quantile blend: {mae_blend:.5f}")

    # Interval coverage: % of actuals inside [q10_pred, q90_pred]
    if 0.1 in quantile_models and 0.9 in quantile_models:
        lower = quantile_models[0.1].predict(X_val)
        upper = quantile_models[0.9].predict(X_val)
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        print(f"  Interval coverage [q10, q90]: {coverage:.1%}  (ideal ≈ 80%)")
        results['interval_coverage_q10_q90'] = coverage

    results['mae_standard'] = mae_standard
    results['mae_dart']     = mae_dart
    results['mae_blend']    = mae_blend

    return results



# Configuration
CONFIG = {
    'n_splits': 4,
    'purge_days': 5,
    'xgb_estimators': 2000,
    'lgb_estimators': 2000,
    'early_stopping': 150,
}

print("✅ Configuration set")



def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start_mem:.1f}MB → {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df




# Load data
print("Loading training data...")
df = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')
df = reduce_mem_usage(df)


# ############################################ RAPID MODE (comment out for full training)
# n_dates = df['date_id'].nunique()
# cutoff_date = int(n_dates * 0.05)
# print(f"RAPID MODE: Using first {cutoff_date} dates only.")
# df = df[df['date_id'] < cutoff_date].reset_index(drop=True)



print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['date_id'].min()} to {df['date_id'].max()}")
print(f"Stocks: {df['stock_id'].nunique()}")
print(f"Time buckets per day: {df['seconds_in_bucket'].nunique()}")


def estimate_stock_weights(df):

    print("Estimating stock weights from data...")
    
    # Calculate average matched_size per stock (proxy for market cap)
    stock_stats = df.groupby('stock_id').agg({
        'matched_size': 'mean',
        'bid_size': 'mean',
        'ask_size': 'mean',
        'wap': 'mean'
    })
    
    # Estimate weight as proportion of total matched_size
    total_matched = stock_stats['matched_size'].sum()
    stock_stats['estimated_weight'] = stock_stats['matched_size'] / total_matched
    
    # Normalize to sum to 1
    stock_stats['estimated_weight'] = stock_stats['estimated_weight'] / stock_stats['estimated_weight'].sum()
    
    # Identify lead stocks (top 10 by weight)
    lead_stocks = stock_stats.nlargest(10, 'estimated_weight').index.tolist()
    
    # Create weight dictionary
    weight_dict = stock_stats['estimated_weight'].to_dict()
    
    print(f"  Total stocks: {len(weight_dict)}")
    print(f"  Lead stocks (top 10 by weight): {lead_stocks}")
    print(f"  Top stock weight: {stock_stats['estimated_weight'].max():.4f}")
    print(f"  Min stock weight: {stock_stats['estimated_weight'].min():.6f}")
    
    return weight_dict, lead_stocks, stock_stats



STOCK_WEIGHTS, LEAD_STOCKS, stock_weight_stats = estimate_stock_weights(df)


def plot_target_analysis(df, stock_weights):

    sns.set_context("notebook", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Part 1: Target Analysis', fontsize=20, fontweight='bold', y=1.02)

    # 1. Target Distribution (Scatter Plot)
    ax = axes[0, 0]
    target_clean = df['target'].dropna()
    ax.hist(target_clean, bins=100, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=target_clean.mean(), color='green', linestyle='--', label=f'Mean: {target_clean.mean():.2f}')
    ax.set_xlabel('Target (basis points)')
    ax.set_ylabel('Frequency')
    ax.set_title('Target Distribution')
    ax.legend()

    # 2. Target by Time (Intraday)
    ax = axes[0, 1]
    target_by_time = df.groupby('seconds_in_bucket')['target'].agg(['mean', 'std'])
    ax.fill_between(target_by_time.index, 
                    target_by_time['mean'] - target_by_time['std'],
                    target_by_time['mean'] + target_by_time['std'],
                    alpha=0.3, color=COLORS['primary'])
    ax.plot(target_by_time.index, target_by_time['mean'], color=COLORS['primary'], linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('Target Mean ± Std')
    ax.set_title('2. Target by Time (Intraday)')

    # 3. Daily Target Patterns
    ax = axes[1, 0]
    daily_stats = df.groupby('date_id')['target'].agg(['mean', 'std'])
    ax.fill_between(daily_stats.index, daily_stats['mean'] - daily_stats['std'],
                    daily_stats['mean'] + daily_stats['std'], alpha=0.3, color=COLORS['secondary'])
    ax.plot(daily_stats.index, daily_stats['mean'], color=COLORS['secondary'], linewidth=1)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date ID')
    ax.set_ylabel('Target Mean ± Std')
    ax.set_title('3. Daily Target Patterns')

    # 4. Target Volatility by Stock Weight
    ax = axes[1, 1]
    df_temp = df.copy()
    df_temp['stock_weight'] = df_temp['stock_id'].map(stock_weights)
    
    # Create weight bins
    weight_percentiles = df_temp['stock_weight'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    weight_percentiles = np.unique(weight_percentiles)
    
    if len(weight_percentiles) >= 2:
        df_temp['weight_bin'] = pd.cut(df_temp['stock_weight'], 
                                        bins=weight_percentiles, 
                                        labels=[f'Q{i+1}' for i in range(len(weight_percentiles)-1)],
                                        include_lowest=True)
        target_by_weight = df_temp.groupby('weight_bin', observed=True)['target'].std()
        
        bars = ax.bar(range(len(target_by_weight)), target_by_weight.values, color=COLORS['accent'], alpha=0.7)
        ax.set_xticks(range(len(target_by_weight)))
        ax.set_xticklabels(target_by_weight.index)
        
    ax.set_ylabel('Target Std Dev')
    ax.set_xlabel('Stock Weight Quantile (Q1=Smallest, Q5=Largest)')
    ax.set_title('4. Target Volatility by Stock Weight')
    del df_temp

    plt.tight_layout()
    plt.show()

# Run Part 1
plot_target_analysis(df, STOCK_WEIGHTS)


def plot_microstructure_analysis(df):

    sns.set_context("notebook", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle('Part 2: Market Microstructure Analysis', fontsize=20, fontweight='bold', y=1.02)

    # 1. Imbalance Direction Distribution
    ax = axes[0]
    flag_counts = df['imbalance_buy_sell_flag'].value_counts().sort_index()
    colors_flag = ['red', 'gray', 'green']
    flag_indices = [-1, 0, 1]
    flag_values = [flag_counts.get(i, 0) for i in flag_indices]
    
    bars = ax.bar(flag_indices, flag_values, color=colors_flag, alpha=0.7)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['Sell (-1)', 'Neutral (0)', 'Buy (+1)'])
    ax.set_ylabel('Count')
    ax.set_title('1. Imbalance Direction Distribution')
    
    # 2. Normalized Imbalance Amount per Second
    ax = axes[1]
    df_imb = df.copy()
    # Normalize: imbalance / (imbalance + matched)
    df_imb['norm_imbalance'] = df_imb['imbalance_size'] / (df_imb['imbalance_size'] + df_imb['matched_size'] + 1e-6)
    
    # Group by Second AND Flag
    imb_by_sec = df_imb.groupby(['seconds_in_bucket', 'imbalance_buy_sell_flag'])['norm_imbalance'].mean().unstack()
    
    if 1 in imb_by_sec.columns:
        ax.plot(imb_by_sec.index, imb_by_sec[1], color='green', label='Buy Imbalance', linewidth=2)
    if -1 in imb_by_sec.columns:
        ax.plot(imb_by_sec.index, imb_by_sec[-1], color='red', label='Sell Imbalance', linewidth=2)
        
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('Avg Normalized Imbalance Ratio')
    ax.set_title('2. Imbalance Magnitude vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------

    # 3. Spread Distribution
    ax = axes[2]
    spread = df['ask_price'] - df['bid_price']
    spread_clean = spread[spread > 0].dropna()
    
    ax.hist(spread_clean, bins=100, color=COLORS['success'], alpha=0.7)
    ax.set_xlabel('Spread (Ask - Bid)')
    ax.set_ylabel('Frequency')
    ax.set_title('4. Spread Distribution (Positive Only)')
    ax.set_xlim(0, spread_clean.quantile(0.99))

    plt.tight_layout()
    plt.show()

# Run Part 2
plot_microstructure_analysis(df)


def create_stock_profiles(df):

    print("Creating stock profiles...")
    
    profiles = df.groupby('stock_id').agg({
        # Price characteristics
        'wap': ['mean', 'std', 'min', 'max'],
        'bid_price': ['mean', 'std'],
        'ask_price': ['mean', 'std'],
        
        # Size characteristics
        'matched_size': ['mean', 'median', 'std'],
        'imbalance_size': ['mean', 'median', 'std'],
        'bid_size': ['mean', 'std'],
        'ask_size': ['mean', 'std'],
        
        # Target characteristics
        'target': ['mean', 'std', 'skew', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
        
        # Imbalance direction
        'imbalance_buy_sell_flag': ['mean', 'std'],
    })
    
    profiles.columns = ['_'.join(col).strip() for col in profiles.columns]
    profiles = profiles.rename(columns={
        'target_<lambda_0>': 'target_q05',
        'target_<lambda_1>': 'target_q95'
    })
    
    # Add estimated weight
    profiles['estimated_weight'] = profiles.index.map(STOCK_WEIGHTS)
    
    # Add derived features
    profiles['wap_volatility'] = profiles['wap_std'] / (profiles['wap_mean'] + 1e-6)
    profiles['size_ratio'] = profiles['matched_size_mean'] / (profiles['imbalance_size_mean'] + 1e-6)
    profiles['spread_mean'] = profiles['ask_price_mean'] - profiles['bid_price_mean']
    profiles['target_range'] = profiles['target_q95'] - profiles['target_q05']
    
    # Check for lead stocks
    profiles['is_lead'] = profiles.index.isin(LEAD_STOCKS)
    
    return profiles



stock_profiles = create_stock_profiles(df)
print(f"Stock profiles shape: {stock_profiles.shape}")
stock_profiles.head()


def cluster_stocks(profiles, n_clusters=12):

    print(f"Clustering stocks into {n_clusters} groups...")
    
    # Select features for clustering
    cluster_features = [
        'wap_mean', 'wap_volatility', 
        'matched_size_mean', 'imbalance_size_mean',
        'target_std', 'target_range',
        'estimated_weight', 'spread_mean'
    ]
    
    # Prepare data
    X_cluster = profiles[cluster_features].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    profiles['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_summary = profiles.groupby('cluster').agg({
        'wap_mean': 'mean',
        'wap_volatility': 'mean',
        'target_std': 'mean',
        'estimated_weight': ['mean', 'sum', 'count'],
        'is_lead': 'sum'
    })
    
    return profiles, cluster_summary, scaler


def find_optimal_clusters(profiles, max_k=20):
    print("Finding optimal cluster count...")
    
    cluster_features = [
        'wap_mean', 'wap_volatility', 
        'matched_size_mean', 'imbalance_size_mean',
        'target_std', 'target_range',
        'estimated_weight', 'spread_mean'
    ]
    X = profiles[cluster_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    
    inertias = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return inertias

inertias = find_optimal_clusters(stock_profiles)


stock_profiles, cluster_summary, cluster_scaler = cluster_stocks(stock_profiles, 9)
print("\nCluster Summary:")
print(cluster_summary)


def plot_cluster_analysis(profiles, cluster_summary):

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Stock Cluster Analysis', fontsize=18, fontweight='bold')
    
    n_clusters = profiles['cluster'].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # 1. Cluster sizes
    ax = axes[0, 0]
    cluster_sizes = profiles['cluster'].value_counts().sort_index()
    bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Stocks')
    ax.set_title('Stocks per Cluster')
    for bar, val in zip(bars, cluster_sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. WAP Volatility vs Target Std by cluster
    ax = axes[0, 1]
    for cluster in range(n_clusters):
        mask = profiles['cluster'] == cluster
        ax.scatter(profiles.loc[mask, 'wap_volatility'], 
                   profiles.loc[mask, 'target_std'],
                   c=[colors[cluster]], label=f'C{cluster}', s=80, alpha=0.7)
    ax.set_xlabel('WAP Volatility')
    ax.set_ylabel('Target Std Dev')
    ax.set_title('Volatility Characteristics by Cluster')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # 3. Estimated weight by cluster
    ax = axes[0, 2]
    weight_by_cluster = profiles.groupby('cluster')['estimated_weight'].sum().sort_values(ascending=False)
    ax.bar(range(len(weight_by_cluster)), weight_by_cluster.values, 
           color=[colors[i] for i in weight_by_cluster.index])
    ax.set_xticks(range(len(weight_by_cluster)))
    ax.set_xticklabels([f'C{i}' for i in weight_by_cluster.index])
    ax.set_ylabel('Total Estimated Weight')
    ax.set_title('Weight Concentration by Cluster')
    
    # 4. Cluster characteristics heatmap
    ax = axes[1, 0]
    char_cols = ['wap_volatility', 'target_std', 'estimated_weight', 'matched_size_mean', 'spread_mean']
    cluster_chars = profiles.groupby('cluster')[char_cols].mean()
    # Normalize for visualization
    cluster_chars_norm = (cluster_chars - cluster_chars.min()) / (cluster_chars.max() - cluster_chars.min() + 1e-6)
    
    im = ax.imshow(cluster_chars_norm.T, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax.set_yticks(range(len(char_cols)))
    ax.set_yticklabels(char_cols)
    ax.set_title('Normalized Cluster Characteristics')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 5. Lead stocks distribution
    ax = axes[1, 1]
    lead_by_cluster = profiles[profiles['is_lead']].groupby('cluster').size()
    all_clusters = pd.Series(0, index=range(n_clusters))
    lead_by_cluster = all_clusters.add(lead_by_cluster, fill_value=0)
    ax.bar(lead_by_cluster.index, lead_by_cluster.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Lead Stocks')
    ax.set_title('Lead Stock Distribution')
    
    # 6. Target statistics by cluster
    ax = axes[1, 2]
    target_by_cluster = profiles.groupby('cluster')['target_std'].mean().sort_values(ascending=False)
    ax.barh(range(len(target_by_cluster)), target_by_cluster.values, 
            color=[colors[i] for i in target_by_cluster.index])
    ax.set_yticks(range(len(target_by_cluster)))
    ax.set_yticklabels([f'Cluster {i}' for i in target_by_cluster.index])
    ax.set_xlabel('Average Target Std Dev')
    ax.set_title('Cluster Volatility Ranking')
    
    plt.tight_layout()
    plt.show()



plot_cluster_analysis(stock_profiles, cluster_summary)



def impute_missing(df):
    df = df.copy()
    
    cols = ['imbalance_size', 'reference_price', 'matched_size', 'wap', 'bid_price', 'ask_price']
    df[cols] = df.groupby(['stock_id', 'date_id'])[cols].ffill().bfill()
    
    df['far_price_null'] = df['far_price'].isna().astype(np.int8)
    df['near_price_null'] = df['near_price'].isna().astype(np.int8)
    df['far_price'] = df['far_price'].fillna(df['reference_price'])
    df['near_price'] = df['near_price'].fillna(df['reference_price'])
    
    return df


def create_revealed_targets(df):

    print("Creating revealed targets...")
    
    revealed = df[['date_id', 'seconds_in_bucket', 'stock_id', 'target']].copy()
    revealed = revealed.rename(columns={'target': 'revealed_target'})
    revealed['date_id'] = revealed['date_id'] + 1
    
    return revealed



def create_features(df, stock_profiles, stock_weights, lead_stocks, revealed_targets=None):
    print("Creating features...")
    
    df = impute_missing(df)
    df = df.sort_values(['stock_id', 'date_id', 'seconds_in_bucket']).reset_index(drop=True)
    
    # ================================================================
    # BASIC FEATURES
    # ================================================================
    print("  Basic features...")
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    df['spread'] = df['ask_price'] - df['bid_price']
    df['spread_pct'] = df['spread'] / (df['wap'] + 1e-6)
    df['volume'] = df['bid_size'] + df['ask_size']
    
    # Imbalance features
    df['liquidity_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['volume'] + 1e-6)
    df['size_imbalance'] = df['bid_size'] / (df['ask_size'] + 1e-6)
    df['matched_imbalance'] = (df['imbalance_size'] - df['matched_size']) / (df['matched_size'] + df['imbalance_size'] + 1e-6)
    df['signed_imbalance'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']
    
    # Price comparisons
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    for p1, p2 in combinations(prices, 2):
        df[f'{p1}_{p2}_imb'] = (df[p1] - df[p2]) / (df[p1] + df[p2] + 1e-6)
    
    # Time features
    df['seconds'] = df['seconds_in_bucket'] % 60
    df['minute'] = df['seconds_in_bucket'] // 60
    df['time_pct'] = df['seconds_in_bucket'] / 600
    df['time_to_close'] = 600 - df['seconds_in_bucket']
    df['is_early'] = (df['seconds_in_bucket'] < 180).astype(np.int8)
    df['is_late'] = (df['seconds_in_bucket'] >= 420).astype(np.int8)
    
    # ================================================================
    # STOCK WEIGHT FEATURES
    # ================================================================
    print("  Stock weight features...")
    df['stock_weight'] = df['stock_id'].map(stock_weights).fillna(0).astype(np.float32)
    df['is_lead_stock'] = df['stock_id'].isin(lead_stocks).astype(np.int8)
    
    # ================================================================
    # SHIFT/LAG FEATURES
    # ================================================================
    print("  Lag features...")
    g = df.groupby(['stock_id', 'date_id'])
    
    for window in [1, 2, 3, 5, 10]:
        df[f'wap_shift_{window}'] = g['wap'].shift(window)
        df[f'imbalance_shift_{window}'] = g['imbalance_size'].shift(window)
        df[f'flag_shift_{window}'] = g['imbalance_buy_sell_flag'].shift(window)
        
        df[f'wap_ret_{window}'] = g['wap'].pct_change(window)
        df[f'imbalance_ret_{window}'] = g['imbalance_size'].pct_change(window)
        
        df[f'spread_diff_{window}'] = g['spread'].diff(window)
        df[f'volume_diff_{window}'] = g['volume'].diff(window)
    
    # ================================================================
    # CROSS-SECTIONAL FEATURES (MARKET-LEVEL)
    # ================================================================
    print("  Cross-sectional features...")
    ts = df.groupby(['date_id', 'seconds_in_bucket'])
    
    # Weighted index WAP
    df['weighted_wap'] = df['wap'] * df['stock_weight']
    df['index_wap'] = df.groupby(['date_id', 'seconds_in_bucket'])['weighted_wap'].transform('sum')
    
    # Simple market averages
    df['market_wap'] = ts['wap'].transform('mean')
    df['market_imbalance'] = ts['signed_imbalance'].transform('mean')
    df['market_flag'] = ts['imbalance_buy_sell_flag'].transform('mean')
    
    # Performance vs market
    df['perf_vs_market'] = 10000 * (df['wap'] - df['market_wap'])
    df['perf_vs_index'] = 10000 * (df['wap'] - df['index_wap'])
    
    # Cross-sectional ranks
    df['wap_rank'] = ts['wap'].rank(pct=True)
    df['imbalance_rank'] = ts['signed_imbalance'].rank(pct=True)
    
    # Total imbalance normalization
    total_imb = ts['imbalance_size'].transform('sum')
    df['imbalance_share'] = df['signed_imbalance'] / (total_imb + 1e-6)
    
    # ================================================================
    # INDEX RETURNS (WEIGHTED)
    # ================================================================
    print("  Index return features...")
    for window in [1, 2, 3, 5]:
        wap_change = df[f'wap_ret_{window}'].fillna(0)
        df[f'weighted_change_{window}'] = df['stock_weight'] * (wap_change + 1)
        df[f'index_ret_{window}'] = df.groupby(['date_id', 'seconds_in_bucket'])[f'weighted_change_{window}'].transform('sum')
        
        # Stock return vs index return
        stock_ret = df['wap'] / (df['wap'] - wap_change + 1e-6) - 1
        df[f'rel_perf_{window}'] = 10000 * (stock_ret - df[f'index_ret_{window}'])
        
        del df[f'weighted_change_{window}']
    
    # ================================================================
    # INFERRED PRICE FROM TICK SIZE
    # ================================================================
    print("  Inferred price features...")
    df['price_move'] = g['bid_price'].diff().abs()
    df.loc[df['price_move'] == 0, 'price_move'] = np.nan
    
    df['tick_size'] = g['price_move'].transform(
        lambda x: x.rolling(55, min_periods=1).min()
    )
    df['inferred_price'] = 0.01 / (df['tick_size'] + 1e-9)
    df['inferred_price'] = df['inferred_price'].clip(upper=1000)
    
    # Inferred volumes
    df['inferred_bid_vol'] = df['bid_size'] / (df['inferred_price'] + 1e-6)
    df['inferred_ask_vol'] = df['ask_size'] / (df['inferred_price'] + 1e-6)
    df['inferred_imb_vol'] = df['imbalance_size'] / (df['inferred_price'] + 1e-6)
    
    del df['price_move']
    
    # ================================================================
    # VOLATILITY FEATURES
    # ================================================================
    print("  Volatility features...")
    df['wap_vol_10'] = g['wap_ret_1'].transform(
        lambda x: x.rolling(10, min_periods=1).std()
    )
    
    # ================================================================
    # STOCK PROFILE FEATURES
    # ================================================================
    print("  Stock profile features...")
    profile_cols = ['cluster', 'wap_volatility', 'target_std', 'target_mean', 'estimated_weight']
    for col in profile_cols:
        if col in stock_profiles.columns:
            df[f'stock_{col}'] = df['stock_id'].map(stock_profiles[col].to_dict())
    
    # ================================================================
    # REVEALED TARGET FEATURES
    # ================================================================
    # NEW (fixed) - create a fresh groupby AFTER the merge
    if revealed_targets is not None:
        print("  Revealed target features...")
        df = df.merge(
            revealed_targets[['date_id', 'seconds_in_bucket', 'stock_id', 'revealed_target']],
            on=['date_id', 'seconds_in_bucket', 'stock_id'],
            how='left'
        )
        
        # Interaction feature
        df['revealed_x_flag'] = df['revealed_target'] * df['imbalance_buy_sell_flag']
        
        g_revealed = df.groupby(['stock_id', 'date_id'])
        df['revealed_roll'] = g_revealed['revealed_target'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
    
    # ================================================================
    # CLEANUP
    # ================================================================
    print("  Cleaning up...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    drop_cols = ['weighted_wap', 'tick_size']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"  Total features: {df.shape[1]}")
    return df



# Create revealed targets
revealed_targets = create_revealed_targets(df)

# Create features
df = create_features(df, stock_profiles, STOCK_WEIGHTS, LEAD_STOCKS, revealed_targets)

print(f"\nFinal dataset shape: {df.shape}")



class PurgedGroupTimeSeriesSplit:
    
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, df, groups):
        unique_groups = np.sort(groups.unique())
        n_groups = len(unique_groups)
        fold_size = n_groups // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            val_start = (i + 1) * fold_size
            val_end = val_start + fold_size
            val_groups = unique_groups[val_start:val_end]
            
            train_end = val_start - self.purge_gap
            if train_end <= 0:
                continue
            train_groups = unique_groups[:train_end]
            
            train_idx = groups.isin(train_groups)
            val_idx = groups.isin(val_groups)
            
            yield train_idx, val_idx



def prepare_train_val(df, train_idx, val_idx, stock_profiles):
    
    train_df = df[train_idx].copy()
    val_df = df[val_idx].copy()
    
    # Clip extreme targets
    q_low = train_df['target'].quantile(0.001)
    q_high = train_df['target'].quantile(0.999)
    train_df['target'] = train_df['target'].clip(q_low, q_high)
    
    # Feature columns
    drop_cols = ['row_id', 'time_id', 'date_id', 'target', 'stock_id', 'currently_scored']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    return X_train, y_train, X_val, y_val, val_df, feature_cols



# Model parameters
XGB_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': CONFIG['xgb_estimators'],
    'min_child_weight': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:absoluteerror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

LGB_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': CONFIG['lgb_estimators'],
    'num_leaves': 256,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu',
    'verbose': -1
}



def weighted_zero_sum_adjustment(predictions, val_df, stock_weights):

    adjusted = predictions.copy()
    
    # Create a temporary DataFrame to align indices perfectly
    temp_df = val_df[['date_id', 'seconds_in_bucket', 'stock_id']].copy().reset_index(drop=True)
    temp_df['pred'] = adjusted
    temp_df['weight'] = temp_df['stock_id'].map(stock_weights).fillna(0)
    
    # Calculate weighted mean per timestamp
    # Formula: Sum(Weight * Pred) / Sum(Weight)
    temp_df['weighted_pred'] = temp_df['pred'] * temp_df['weight']
    
    grouped = temp_df.groupby(['date_id', 'seconds_in_bucket'])
    
    # Calculate sums
    group_sums = grouped['weighted_pred'].transform('sum')
    total_weights = grouped['weight'].transform('sum')
    
    # Avoid division by zero
    weighted_means = group_sums / (total_weights + 1e-6)
    
    # If total weight is effectively zero, use simple mean
    simple_means = grouped['pred'].transform('mean')
    weighted_means = np.where(total_weights < 1e-6, simple_means, weighted_means)
    
    # Adjust
    return (temp_df['pred'] - weighted_means).values


# def train_fold(X_train, y_train, X_val, y_val, val_df, stock_weights, fold):
    
#     print(f"\n  Training XGBoost...")
#     xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
#     xgb_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=CONFIG['early_stopping'],
#         verbose=100
#     )
#     xgb_preds = xgb_model.predict(X_val)
#     xgb_mae = mean_absolute_error(y_val, xgb_preds)
#     print(f"  XGBoost MAE: {xgb_mae:.5f} (best_iter: {xgb_model.best_iteration})")
    
#     print(f"\n  Training LightGBM...")
#     lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
#     lgb_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(CONFIG['early_stopping']), lgb.log_evaluation(100)]
#     )
#     lgb_preds = lgb_model.predict(X_val)
#     lgb_mae = mean_absolute_error(y_val, lgb_preds)
#     print(f"  LightGBM MAE: {lgb_mae:.5f} (best_iter: {lgb_model.best_iteration_})")
    
#     # Inside your fold loop, after training XGB and standard LGB:

#     # ── DART ──────────────────────────────────────────────────────
#     dart_model, dart_preds, dart_mae = train_dart_model(
#         X_train, y_train, X_val, y_val
#     )
    
#     # ── Quantile models ───────────────────────────────────────────
#     print("Training quantile models...")
#     quantile_models, pinball_scores, median_preds = train_quantile_models(
#         X_train, y_train, X_val, y_val
#     )
#     quantile_blend = adaptive_quantile_blend(X_val, quantile_models)
#     quantile_mae   = mean_absolute_error(y_val, quantile_blend)
#     print(f"Quantile blend MAE: {quantile_mae:.5f}")
    
#     # ── Coverage evaluation ───────────────────────────────────────
#     coverage_results = evaluate_quantile_coverage(
#         y_val.values, quantile_models, X_val,
#         lgb_preds, dart_preds, quantile_blend,
#     )
    
#     # ── Four-way inverse-MAE ensemble ─────────────────────────────
#     w_xgb      = 1 / (xgb_mae      + 1e-6)
#     w_lgb      = 1 / (lgb_mae      + 1e-6)
#     w_dart     = 1 / (dart_mae     + 1e-6)
#     w_quantile = 1 / (quantile_mae + 1e-6)
#     w_total    = w_xgb + w_lgb + w_dart + w_quantile
    
#     ensemble_preds = (
#         w_xgb      * xgb_preds     +
#         w_lgb      * lgb_preds     +
#         w_dart     * dart_preds    +
#         w_quantile * quantile_blend
#     ) / w_total
#     ensemble_mae = mean_absolute_error(y_val, ensemble_preds)
    
#     adjusted_preds = weighted_zero_sum_adjustment(ensemble_preds, val_df_fold, STOCK_WEIGHTS)
#     adjusted_mae   = mean_absolute_error(y_val, adjusted_preds)
    
#     print(f"4-way Ensemble: {ensemble_mae:.5f} | Adjusted: {adjusted_mae:.5f}")
#     print(f"Weights — XGB: {w_xgb/w_total:.3f}  LGB: {w_lgb/w_total:.3f}  "
#           f"DART: {w_dart/w_total:.3f}  Quantile: {w_quantile/w_total:.3f}")
        
#     return {
#         'xgb_model': xgb_model,
#         'lgb_model': lgb_model,
#         'xgb_preds': xgb_preds,
#         'lgb_preds': lgb_preds,
#         'ensemble_preds': ensemble_preds,
#         'adjusted_preds': adjusted_preds,
#         'xgb_mae': xgb_mae,
#         'lgb_mae': lgb_mae,
#         'ensemble_mae': ensemble_mae,
#         'adjusted_mae': adjusted_mae,
#         'xgb_weight': w_xgb / w_total,
#         'lgb_weight': w_lgb / w_total
#     }



# # Run cross-validation
# print("="*60)
# print("CROSS-VALIDATION")
# print("="*60)

# splitter = PurgedGroupTimeSeriesSplit(n_splits=CONFIG['n_splits'], purge_gap=CONFIG['purge_days'])
# date_ids = df['date_id']

# fold_results = []
# feature_cols = None

# for fold, (train_idx, val_idx) in enumerate(splitter.split(df, date_ids), 1):
#     print(f"\n{'='*60}")
#     print(f"FOLD {fold}/{CONFIG['n_splits']}")
#     print(f"{'='*60}")
    
#     X_train, y_train, X_val, y_val, val_df_fold, feature_cols = prepare_train_val(
#         df, train_idx, val_idx, stock_profiles
#     )
    
#     print(f"Train: {len(X_train):,} samples, Val: {len(X_val):,} samples")
    
#     results = train_fold(X_train, y_train, X_val, y_val, val_df_fold, STOCK_WEIGHTS, fold)
    
#     # Period-wise analysis
#     for period, (t_min, t_max) in [('early', (0, 200)), ('middle', (200, 400)), ('late', (400, 600))]:
#         mask = (val_df_fold['seconds_in_bucket'] >= t_min) & (val_df_fold['seconds_in_bucket'] < t_max)
#         results[f'{period}_mae'] = mean_absolute_error(y_val[mask], results['adjusted_preds'][mask])
    
#     results['val_df'] = val_df_fold
#     results['y_val'] = y_val
#     fold_results.append(results)
    
#     gc.collect()



# ── Paste all of src/optimize.py functions here first ──────────────

splitter = PurgedGroupTimeSeriesSplit(n_splits=CONFIG['n_splits'], purge_gap=CONFIG['purge_days'])
date_ids = df['date_id']
fold_results = []
selected_features = None

for fold, (train_idx, val_idx) in enumerate(splitter.split(df, date_ids), 1):
    print(f"\n{'='*60}\nFOLD {fold}/{CONFIG['n_splits']}\n{'='*60}")

    X_train, y_train, X_val, y_val, val_df_fold, feature_cols = prepare_train_val(
        df, train_idx, val_idx, stock_profiles
    )

    # Feature selection on fold 1 only
    if selected_features is None:
        selected_features = select_features(X_train, y_train, X_val, y_val)

    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]

    # 1. XGBoost
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=CONFIG['early_stopping'])
    xgb_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=100)
    xgb_preds = xgb_model.predict(X_val_sel)
    xgb_mae   = mean_absolute_error(y_val, xgb_preds)

    # 2. Standard LGB
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])
    lgb_preds = lgb_model.predict(X_val_sel)
    lgb_mae   = mean_absolute_error(y_val, lgb_preds)

    # 3. DART LGB
    dart_model, dart_preds, dart_mae = train_dart_model(
        X_train_sel, y_train, X_val_sel, y_val
    )

    # 4. Quantile blend
    quantile_models, pinball_scores, _ = train_quantile_models(
        X_train_sel, y_train, X_val_sel, y_val
    )
    quantile_blend = adaptive_quantile_blend(X_val_sel, quantile_models)
    quantile_mae   = mean_absolute_error(y_val, quantile_blend)

    # Coverage evaluation
    coverage_results = evaluate_quantile_coverage(
        y_val.values, quantile_models, X_val_sel,
        lgb_preds, dart_preds, quantile_blend,
    )

    # Four-way inverse-MAE ensemble
    w_xgb      = 1 / (xgb_mae      + 1e-6)
    w_lgb      = 1 / (lgb_mae      + 1e-6)
    w_dart     = 1 / (dart_mae     + 1e-6)
    w_quantile = 1 / (quantile_mae + 1e-6)
    w_total    = w_xgb + w_lgb + w_dart + w_quantile

    ensemble_preds = (
        w_xgb      * xgb_preds      +
        w_lgb      * lgb_preds      +
        w_dart     * dart_preds     +
        w_quantile * quantile_blend
    ) / w_total
    ensemble_mae = mean_absolute_error(y_val, ensemble_preds)

    adjusted_preds = weighted_zero_sum_adjustment(ensemble_preds, val_df_fold, STOCK_WEIGHTS)
    adjusted_mae   = mean_absolute_error(y_val, adjusted_preds)

    print(f"  XGB: {xgb_mae:.5f} | LGB: {lgb_mae:.5f} | "
          f"DART: {dart_mae:.5f} | Quantile: {quantile_mae:.5f}")
    print(f"  Ensemble: {ensemble_mae:.5f} | Adjusted: {adjusted_mae:.5f}")
    print(f"  Weights — XGB: {w_xgb/w_total:.3f}  LGB: {w_lgb/w_total:.3f}  "
          f"DART: {w_dart/w_total:.3f}  Quantile: {w_quantile/w_total:.3f}")

    result = {
        'xgb_mae':       xgb_mae,
        'lgb_mae':       lgb_mae,
        'dart_mae':      dart_mae,
        'quantile_mae':  quantile_mae,
        'ensemble_mae':  ensemble_mae,
        'adjusted_mae':  adjusted_mae,
        'xgb_weight':    w_xgb      / w_total,
        'lgb_weight':    w_lgb      / w_total,
        'dart_weight':   w_dart     / w_total,
        'quantile_weight': w_quantile / w_total,
        'xgb_model':     xgb_model,
        'lgb_model':     lgb_model,
        'dart_model':    dart_model,
        'adjusted_preds': adjusted_preds,
        'val_df':        val_df_fold,
        'y_val':         y_val,
    }

    for period, (t_min, t_max) in [
        ('early', (0, 200)), ('middle', (200, 400)), ('late', (400, 600))
    ]:
        mask = ((val_df_fold['seconds_in_bucket'] >= t_min) &
                (val_df_fold['seconds_in_bucket'] <  t_max))
        result[f'{period}_mae'] = mean_absolute_error(y_val[mask], adjusted_preds[mask])

    fold_results.append(result)
    gc.collect()


# Inside your fold loop, after training XGB and standard LGB:

# ── DART ──────────────────────────────────────────────────────
dart_model, dart_preds, dart_mae = train_dart_model(
    X_train, y_train, X_val, y_val
)

# ── Quantile models ───────────────────────────────────────────
print("Training quantile models...")
quantile_models, pinball_scores, median_preds = train_quantile_models(
    X_train, y_train, X_val, y_val
)
quantile_blend = adaptive_quantile_blend(X_val, quantile_models)
quantile_mae   = mean_absolute_error(y_val, quantile_blend)
print(f"Quantile blend MAE: {quantile_mae:.5f}")

# ── Coverage evaluation ───────────────────────────────────────
coverage_results = evaluate_quantile_coverage(
    y_val.values, quantile_models, X_val,
    lgb_preds, dart_preds, quantile_blend,
)

# ── Four-way inverse-MAE ensemble ─────────────────────────────
w_xgb      = 1 / (xgb_mae      + 1e-6)
w_lgb      = 1 / (lgb_mae      + 1e-6)
w_dart     = 1 / (dart_mae     + 1e-6)
w_quantile = 1 / (quantile_mae + 1e-6)
w_total    = w_xgb + w_lgb + w_dart + w_quantile

ensemble_preds = (
    w_xgb      * xgb_preds     +
    w_lgb      * lgb_preds     +
    w_dart     * dart_preds    +
    w_quantile * quantile_blend
) / w_total
ensemble_mae = mean_absolute_error(y_val, ensemble_preds)

adjusted_preds = weighted_zero_sum_adjustment(ensemble_preds, val_df_fold, STOCK_WEIGHTS)
adjusted_mae   = mean_absolute_error(y_val, adjusted_preds)

print(f"4-way Ensemble: {ensemble_mae:.5f} | Adjusted: {adjusted_mae:.5f}")
print(f"Weights — XGB: {w_xgb/w_total:.3f}  LGB: {w_lgb/w_total:.3f}  "
      f"DART: {w_dart/w_total:.3f}  Quantile: {w_quantile/w_total:.3f}")


def plot_model_comparison(fold_results):
    """Compare model performances across folds."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    n_folds = len(fold_results)
    folds = list(range(1, n_folds + 1))
    
    xgb_maes = [r['xgb_mae'] for r in fold_results]
    lgb_maes = [r['lgb_mae'] for r in fold_results]
    ens_maes = [r['ensemble_mae'] for r in fold_results]
    adj_maes = [r['adjusted_mae'] for r in fold_results]
    
    # 1. All MAEs by fold
    ax = axes[0, 0]
    x = np.arange(n_folds)
    width = 0.2
    
    ax.bar(x - 1.5*width, xgb_maes, width, label='XGBoost', color=COLORS['primary'])
    ax.bar(x - 0.5*width, lgb_maes, width, label='LightGBM', color=COLORS['secondary'])
    ax.bar(x + 0.5*width, ens_maes, width, label='Ensemble', color=COLORS['accent'])
    ax.bar(x + 1.5*width, adj_maes, width, label='Adjusted', color=COLORS['success'])
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Model and Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Average performance
    ax = axes[0, 1]
    models = ['XGBoost', 'LightGBM', 'Ensemble', 'Adjusted']
    avg_maes = [np.mean(xgb_maes), np.mean(lgb_maes), np.mean(ens_maes), np.mean(adj_maes)]
    std_maes = [np.std(xgb_maes), np.std(lgb_maes), np.std(ens_maes), np.std(adj_maes)]
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    bars = ax.bar(models, avg_maes, yerr=std_maes, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Average MAE')
    ax.set_title('Average Performance (±std)')
    for bar, mae in zip(bars, avg_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Period-wise performance
    ax = axes[0, 2]
    periods = ['Early', 'Middle', 'Late']
    period_avgs = []
    for period in ['early', 'middle', 'late']:
        period_avgs.append(np.mean([r[f'{period}_mae'] for r in fold_results]))
    
    bars = ax.bar(periods, period_avgs, color=[COLORS['success'], COLORS['accent'], COLORS['primary']])
    ax.set_ylabel('Average MAE')
    ax.set_title('MAE by Time Period')
    for bar, mae in zip(bars, period_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{mae:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Improvement from post-processing
    ax = axes[1, 0]
    improvements = [e - a for e, a in zip(ens_maes, adj_maes)]
    colors_imp = [COLORS['success'] if imp > 0 else COLORS['accent'] for imp in improvements]
    ax.bar(folds, improvements, color=colors_imp, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE Improvement')
    ax.set_title('Post-Processing Improvement')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Ensemble weights
    ax = axes[1, 1]
    xgb_weights = [r['xgb_weight'] for r in fold_results]
    lgb_weights = [r['lgb_weight'] for r in fold_results]
    
    ax.bar(folds, xgb_weights, label='XGBoost', color=COLORS['primary'])
    ax.bar(folds, lgb_weights, bottom=xgb_weights, label='LightGBM', color=COLORS['secondary'])
    ax.set_xlabel('Fold')
    ax.set_ylabel('Weight')
    ax.set_title('Ensemble Weights')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    FINAL RESULTS SUMMARY
    {'='*50}
    
    XGBoost:     {np.mean(xgb_maes):.5f} ± {np.std(xgb_maes):.5f}
    LightGBM:    {np.mean(lgb_maes):.5f} ± {np.std(lgb_maes):.5f}
    Ensemble:    {np.mean(ens_maes):.5f} ± {np.std(ens_maes):.5f}
    Adjusted:    {np.mean(adj_maes):.5f} ± {np.std(adj_maes):.5f}
    
    Post-Processing Improvement: {np.mean(improvements):.5f}
    
    Period Breakdown:
      Early (0-200s):   {period_avgs[0]:.5f}
      Middle (200-400s): {period_avgs[1]:.5f}
      Late (400-600s):   {period_avgs[2]:.5f}
    
    Best Fold: {folds[np.argmin(adj_maes)]} ({min(adj_maes):.5f})
    Worst Fold: {folds[np.argmax(adj_maes)]} ({max(adj_maes):.5f})
    """
    ax.text(0.05, 0.95, summary, fontsize=11, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()



plot_model_comparison(fold_results)


def plot_detailed_fold_analysis(fold_results, stock_profiles, fold_idx=-1):
    """Detailed analysis for a specific fold."""
    results = fold_results[fold_idx]
    val_df = results['val_df']
    y_true = results['y_val'].values
    preds = results['adjusted_preds']
    errors = np.abs(y_true - preds)
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 16))
    actual_fold = len(fold_results) if fold_idx == -1 else fold_idx + 1
    fig.suptitle(f'Fold {actual_fold} Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs Actual
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_idx], preds[sample_idx], alpha=0.3, s=10, c=COLORS['primary'])
    lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predictions vs Actual')
    
    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=100, color=COLORS['primary'], alpha=0.7)
    ax.axvline(x=np.median(errors), color='red', linestyle='--', label=f'Median: {np.median(errors):.2f}')
    ax.axvline(x=np.mean(errors), color='green', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # 3. MAE by time
    ax = axes[0, 2]
    time_mae = []
    time_labels = []
    for t in range(0, 600, 10):
        mask = (val_df['seconds_in_bucket'] >= t) & (val_df['seconds_in_bucket'] < t + 10)
        if mask.sum() > 0:
            time_mae.append(mean_absolute_error(y_true[mask], preds[mask]))
            time_labels.append(t)
    
    ax.plot(time_labels, time_mae, color=COLORS['primary'], linewidth=2)
    ax.axvspan(200, 400, alpha=0.2, color='red', label='Middle Period')
    ax.set_xlabel('Seconds in Bucket')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Throughout Auction')
    ax.legend()
    
    # 4. Residuals by predicted value
    ax = axes[0, 3]
    residuals = y_true - preds
    ax.scatter(preds[sample_idx], residuals[sample_idx], alpha=0.3, s=10, c=COLORS['secondary'])
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predictions')
    
    # 5. MAE by stock cluster
    ax = axes[1, 0]
    val_df_copy = val_df.copy()
    val_df_copy['error'] = errors
    val_df_copy['cluster'] = val_df_copy['stock_id'].map(stock_profiles['cluster'].to_dict())
    cluster_mae = val_df_copy.groupby('cluster')['error'].mean().sort_values(ascending=False)
    
    colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(cluster_mae)))
    ax.barh(range(len(cluster_mae)), cluster_mae.values, color=colors_cluster)
    ax.set_yticks(range(len(cluster_mae)))
    ax.set_yticklabels([f'Cluster {c}' for c in cluster_mae.index])
    ax.set_xlabel('MAE')
    ax.set_title('MAE by Stock Cluster')
    ax.invert_yaxis()
    
    # 6. Prediction distribution
    ax = axes[1, 1]
    ax.hist(y_true, bins=100, alpha=0.5, label='Actual', color=COLORS['primary'])
    ax.hist(preds, bins=100, alpha=0.5, label='Predicted', color=COLORS['accent'])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # 7. Zero-sum check
    ax = axes[1, 2]
    ts_means = val_df.copy()
    ts_means['pred'] = preds
    ts_means_grouped = ts_means.groupby(['date_id', 'seconds_in_bucket'])['pred'].mean()
    
    ax.hist(ts_means_grouped, bins=50, color=COLORS['secondary'], alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Mean Prediction per Timestamp')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Zero-Sum Check (std={ts_means_grouped.std():.4f})')
    
    # 8. Top 20 hardest stocks
    ax = axes[1, 3]
    stock_mae = val_df_copy.groupby('stock_id')['error'].mean().sort_values(ascending=False).head(20)
    ax.barh(range(len(stock_mae)), stock_mae.values, color=COLORS['success'])
    ax.set_yticks(range(len(stock_mae)))
    ax.set_yticklabels([f'Stock {s}' for s in stock_mae.index], fontsize=8)
    ax.set_xlabel('MAE')
    ax.set_title('Top 20 Hardest Stocks')
    ax.invert_yaxis()
    
    # 9. Feature importance (XGBoost)
    ax = axes[2, 0]
    importance = results['xgb_model'].feature_importances_
    features = results['xgb_model'].feature_names_in_
    top_idx = np.argsort(importance)[-15:]
    
    ax.barh(range(15), importance[top_idx], color=COLORS['primary'])
    ax.set_yticks(range(15))
    ax.set_yticklabels(features[top_idx], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (XGBoost)')
    
    # 10. Feature importance (LightGBM)
    ax = axes[2, 1]
    importance = results['lgb_model'].feature_importances_
    features = results['lgb_model'].feature_name_
    top_idx = np.argsort(importance)[-15:]
    
    ax.barh(range(15), importance[top_idx], color=COLORS['secondary'])
    ax.set_yticks(range(15))
    ax.set_yticklabels([features[i] for i in top_idx], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Features (LightGBM)')
    
    # 11. MAE by lead vs non-lead
    ax = axes[2, 2]
    val_df_copy['is_lead'] = val_df_copy['stock_id'].isin(LEAD_STOCKS)
    lead_mae = val_df_copy[val_df_copy['is_lead']]['error'].mean()
    nonlead_mae = val_df_copy[~val_df_copy['is_lead']]['error'].mean()
    
    ax.bar(['Lead Stocks', 'Other Stocks'], [lead_mae, nonlead_mae], 
           color=[COLORS['accent'], COLORS['primary']], alpha=0.7)
    ax.set_ylabel('MAE')
    ax.set_title('Lead vs Non-Lead Stocks')
    for i, mae in enumerate([lead_mae, nonlead_mae]):
        ax.text(i, mae + 0.02, f'{mae:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    # 12. Statistics summary
    ax = axes[2, 3]
    ax.axis('off')
    
    summary = f"""
    FOLD {actual_fold} STATISTICS
    {'='*40}
    
    Sample Size: {len(y_true):,}
    
    MAE Scores:
      XGBoost:   {results['xgb_mae']:.5f}
      LightGBM:  {results['lgb_mae']:.5f}
      Ensemble:  {results['ensemble_mae']:.5f}
      Adjusted:  {results['adjusted_mae']:.5f}
    
    Prediction Range:
      Actual: [{y_true.min():.1f}, {y_true.max():.1f}]
      Pred:   [{preds.min():.1f}, {preds.max():.1f}]
    
    Error Statistics:
      Mean:   {errors.mean():.4f}
      Median: {np.median(errors):.4f}
      Max:    {errors.max():.2f}
    
    Coverage:
      ±5 bps:  {(errors <= 5).mean():.1%}
      ±10 bps: {(errors <= 10).mean():.1%}
    """
    ax.text(0.05, 0.95, summary, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.show()



# Plot detailed analysis for last fold
plot_detailed_fold_analysis(fold_results, stock_profiles, fold_idx=-1)



def print_final_summary(fold_results):
    """Print final summary of results."""
    print("="*70)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*70)
    
    xgb_maes = [r['xgb_mae'] for r in fold_results]
    lgb_maes = [r['lgb_mae'] for r in fold_results]
    ens_maes = [r['ensemble_mae'] for r in fold_results]
    adj_maes = [r['adjusted_mae'] for r in fold_results]
    
    print(f"\n{'Model':<15} {'Mean MAE':<12} {'Std':<10} {'Best':<10} {'Worst':<10}")
    print("-"*60)
    print(f"{'XGBoost':<15} {np.mean(xgb_maes):<12.5f} {np.std(xgb_maes):<10.5f} {min(xgb_maes):<10.5f} {max(xgb_maes):<10.5f}")
    print(f"{'LightGBM':<15} {np.mean(lgb_maes):<12.5f} {np.std(lgb_maes):<10.5f} {min(lgb_maes):<10.5f} {max(lgb_maes):<10.5f}")
    print(f"{'Ensemble':<15} {np.mean(ens_maes):<12.5f} {np.std(ens_maes):<10.5f} {min(ens_maes):<10.5f} {max(ens_maes):<10.5f}")
    print(f"{'Adjusted':<15} {np.mean(adj_maes):<12.5f} {np.std(adj_maes):<10.5f} {min(adj_maes):<10.5f} {max(adj_maes):<10.5f}")
    
    print("\n" + "-"*60)
    print("PERIOD BREAKDOWN (Adjusted)")
    print("-"*60)
    for period in ['early', 'middle', 'late']:
        period_maes = [r[f'{period}_mae'] for r in fold_results]
        print(f"{period.capitalize():<10} {np.mean(period_maes):.5f} ± {np.std(period_maes):.5f}")
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)



print_final_summary(fold_results)



import os
import pandas as pd
import numpy as np

def predict(test: pd.DataFrame, revealed_targets: pd.DataFrame, sample_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Called once per time bucket by Kaggle's inference server.
    Must return a DataFrame with columns: row_id, target.
    """
    # Build features on the incoming test slice
    # Note: revealed_targets here is Kaggle's version, not ours
    test_feat = create_features(
        test.copy(),
        stock_profiles,
        STOCK_WEIGHTS,
        LEAD_STOCKS,
        revealed_targets=None,   # no revealed targets at inference time
    )

    # Select the same features used in training
    feature_cols_model = [c for c in selected_features if c in test_feat.columns]
    X_test = test_feat[feature_cols_model].fillna(0)

    # Predict with each model from the last fold (most recent data)
    last = fold_results[-1]

    preds_xgb      = last['xgb_model'].predict(X_test)
    preds_lgb      = last['lgb_model'].predict(X_test)
    preds_dart     = last['dart_model'].predict(X_test)
    preds_quantile = adaptive_quantile_blend(X_test, last['quantile_models'])

    w_xgb      = last['xgb_weight']
    w_lgb      = last['lgb_weight']
    w_dart     = last['dart_weight']
    w_quantile = last['quantile_weight']

    final_preds = (
        w_xgb      * preds_xgb      +
        w_lgb      * preds_lgb      +
        w_dart     * preds_dart     +
        w_quantile * preds_quantile
    )

    # Zero-sum adjustment
    final_preds = weighted_zero_sum_adjustment(final_preds, test, STOCK_WEIGHTS)

    return pd.DataFrame({
        'row_id': test['row_id'],
        'target': final_preds,
    })


# ── Run inference ─────────────────────────────────────────────
import kaggle_evaluation.optiver_inference_server

inference_server = kaggle_evaluation.optiver_inference_server.OptiverInferenceServer(predict)

if os.path.exists('/kaggle/input/optiver-trading-at-the-close/example_test_files/'):
    inference_server.serve()
else:
    inference_server.run_local_test()

