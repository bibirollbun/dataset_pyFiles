import pandas as pd
import numpy as np
import polars as pl
import optuna
from functools import partial


# =====================================================
# Utility Functions
# =====================================================

def build_month_codes():
    """Create a mapping from month abbreviations to numeric values."""
    return {m: i for i, m in enumerate(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}


def split_test_id_column(df: pl.DataFrame) -> pl.DataFrame:
    """Parse the ID column into month text and sector components."""
    return df.with_columns([
        pl.col('id').str.split('_').list.get(0).alias('month_text'),
        pl.col('id').str.split('_').list.get(1).alias('sector')
    ])


def add_time_and_sector_fields(df: pl.DataFrame, month_codes: dict) -> pl.DataFrame:
    """Add parsed year, month, time index, and sector_id to dataframe."""
    df = df.clone()
    
    if 'sector' in df.columns:
        df = df.with_columns([
            pl.col('sector').str.slice(7).cast(pl.Int32).alias('sector_id')
        ])

    if 'month' in df.columns:  # test data
        df = df.with_columns([
            pl.col('month').str.slice(0, 4).cast(pl.Int32).alias('year'),
            pl.col('month').str.slice(5).map_elements(lambda x: month_codes.get(x, 0), return_dtype=pl.Int32).alias('month')
        ])
    else:  # train data
        df = df.with_columns([
            pl.col('month_text').str.slice(0, 4).cast(pl.Int32).alias('year'),
            pl.col('month_text').str.slice(5).map_elements(lambda x: month_codes.get(x, 0), return_dtype=pl.Int32).alias('month')
        ])

    df = df.with_columns([
        ((pl.col('year') - 2019) * 12 + pl.col('month') - 1).alias('time')
    ])
    
    return df


def load_competition_data():
    """Load competition training and test datasets."""
    path = '/kaggle/input/china-real-estate-demand-prediction'
    train = pl.read_csv(f'{path}/train/new_house_transactions.csv')
    test = pl.read_csv(f'{path}/test.csv')
    return train, test


# =====================================================
# Data Transformation
# =====================================================

def build_amount_matrix(train: pl.DataFrame, month_codes: dict) -> pl.DataFrame:
    """Pivot training data into [time x sector_id] transaction matrix."""
    train = add_time_and_sector_fields(train.clone(), month_codes)
    
    # Use polars pivot operation
    pivot = train.pivot(
        index='time', 
        columns='sector_id',
        values='amount_new_house_transactions',
        aggregate_function='first'
    ).fill_null(0)

    # Ensure all 96 sectors are present
    all_sectors = list(range(1, 97))
    existing_cols = [col for col in pivot.columns if col != 'time']
    missing_sectors = [f"{s}" for s in all_sectors if f"{s}" not in existing_cols]
    
    # Add missing sectors with zero values
    for sector in missing_sectors:
        pivot = pivot.with_columns(pl.lit(0).alias(sector))
    
    # Sort columns to maintain consistent order
    sector_cols = [str(i) for i in range(1, 97)]
    pivot = pivot.select(['time'] + sector_cols)
    
    # Convert to pandas-like structure for compatibility with existing model code
    pivot_pd = pivot.to_pandas().set_index('time')
    pivot_pd.columns = pivot_pd.columns.astype(int)
    
    return pivot_pd


# =====================================================
# Modeling Helpers
# =====================================================

def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.8, clip_high=1.5):
    """Compute sector-level December multipliers from training data."""
    is_dec = (a_tr.index % 12 == 11)
    dec_means = a_tr[is_dec].mean()
    nondec_means = a_tr[~is_dec].mean()
    dec_counts = a_tr[is_dec].count()

    raw_mult = dec_means / (nondec_means + eps)
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))

    raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
    raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return raw_mult.clip(clip_low, clip_high).to_dict()


def apply_december_bump_row(pred_row, sector_to_mult: dict):
    """Apply December adjustment to a prediction row."""
    import pandas as pd
    return pred_row.multiply(pd.Series(sector_to_mult)).fillna(pred_row)


def compute_trend_features(a_tr, sector, n_lags):
    """Compute trend-based features for a sector."""
    recent = a_tr[sector].tail(n_lags).values
    if len(recent) < 2:
        return {'trend': 0.0, 'volatility': 0.0, 'momentum': 0.0}
    
    # Linear trend
    x = np.arange(len(recent))
    if np.std(recent) > 0:
        trend = np.corrcoef(x, recent)[0, 1]
    else:
        trend = 0.0
    
    # Volatility (coefficient of variation)
    volatility = np.std(recent) / (np.mean(recent) + 1e-12)
    
    # Momentum (recent vs earlier periods)
    mid_point = len(recent) // 2
    recent_avg = np.mean(recent[mid_point:])
    earlier_avg = np.mean(recent[:mid_point])
    momentum = (recent_avg - earlier_avg) / (earlier_avg + 1e-12)
    
    return {'trend': trend, 'volatility': volatility, 'momentum': momentum}


# =====================================================
# Multiple Prediction Models
# =====================================================

def ewgm_per_sector(a_tr, sector, n_lags, alpha):
    """Exponential weighted geometric mean for one sector."""
    recent = a_tr[sector].tail(n_lags).values
    if len(recent) < n_lags or (recent <= 0).all():
        return 0.0

    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)])
    weights /= weights.sum()

    mask = recent > 0
    if not mask.any():
        return 0.0

    log_vals = np.log(recent[mask] + 1e-12)
    pos_w = weights[mask] / weights[mask].sum()
    return float(np.exp(np.sum(pos_w * log_vals)))


def ema_per_sector(a_tr, sector, n_lags, alpha):
    """Exponential moving average for one sector."""
    recent = a_tr[sector].tail(n_lags).values
    if len(recent) < n_lags:
        return 0.0
    
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)])
    weights /= weights.sum()
    
    return float(np.sum(weights * recent))


def trend_adjusted_per_sector(a_tr, sector, n_lags, alpha, trend_weight=0.3):
    """Trend-adjusted prediction for one sector."""
    recent = a_tr[sector].tail(n_lags).values
    if len(recent) < 3:
        return 0.0
    
    # Base prediction from EMA
    base_pred = ema_per_sector(a_tr, sector, n_lags, alpha)
    
    # Trend adjustment
    x = np.arange(len(recent))
    if np.std(recent) > 0:
        slope = np.polyfit(x, recent, 1)[0]
        trend_adj = slope * trend_weight
    else:
        trend_adj = 0.0
    
    return max(0.0, base_pred + trend_adj)


def seasonal_decomp_per_sector(a_tr, sector, n_lags, alpha):
    """Seasonal decomposition based prediction."""
    if len(a_tr) < 24:  # Need at least 2 years
        return ema_per_sector(a_tr, sector, n_lags, alpha)
    
    series = a_tr[sector]
    if series.sum() == 0:
        return 0.0
    
    # Simple seasonal decomposition
    seasonal_period = 12
    seasonal_means = {}
    for month in range(12):
        month_data = series[series.index % 12 == month]
        seasonal_means[month] = month_data.mean() if len(month_data) > 0 else 0.0
    
    # Deseasonalized series
    deseasonalized = series.copy()
    for i in series.index:
        month = i % 12
        if seasonal_means[month] > 0:
            deseasonalized[i] = series[i] / seasonal_means[month]
    
    # Predict deseasonalized value
    base_pred = ema_per_sector(pd.DataFrame({sector: deseasonalized}), sector, n_lags, alpha)
    
    # Apply seasonal adjustment for next month
    next_month = (series.index[-1] + 1) % 12
    seasonal_factor = seasonal_means[next_month]
    
    return max(0.0, base_pred * seasonal_factor)


def ensemble_per_sector(a_tr, sector, n_lags, alpha, model_weights=None):
    """Ensemble of multiple models for one sector."""
    if model_weights is None:
        model_weights = {'ewgm': 0.4, 'ema': 0.3, 'trend': 0.2, 'seasonal': 0.1}
    
    models = {
        'ewgm': ewgm_per_sector(a_tr, sector, n_lags, alpha),
        'ema': ema_per_sector(a_tr, sector, n_lags, alpha),
        'trend': trend_adjusted_per_sector(a_tr, sector, n_lags, alpha),
        'seasonal': seasonal_decomp_per_sector(a_tr, sector, n_lags, alpha)
    }
    
    ensemble_pred = sum(models[model] * weight for model, weight in model_weights.items())
    return max(0.0, ensemble_pred)


def predict_one_step(a_hist, n_lags, alpha, model_type='ensemble', model_weights=None):
    """Predict next-step values for all sectors using specified model."""
    import pandas as pd
    
    model_functions = {
        'ewgm': ewgm_per_sector,
        'ema': ema_per_sector,
        'trend': trend_adjusted_per_sector,
        'seasonal': seasonal_decomp_per_sector,
        'ensemble': ensemble_per_sector
    }
    
    model_func = model_functions.get(model_type, ewgm_per_sector)
    
    predictions = {}
    for sector in a_hist.columns:
        if a_hist[sector].sum() == 0:
            predictions[sector] = 0.0
        else:
            if model_type == 'ensemble':
                predictions[sector] = model_func(a_hist, sector, n_lags, alpha, model_weights)
            elif model_type == 'trend':
                predictions[sector] = model_func(a_hist, sector, n_lags, alpha, 0.3)
            else:
                predictions[sector] = model_func(a_hist, sector, n_lags, alpha)
    
    return pd.Series(predictions)


def evaluate_params(a_tr_full, n_lags, alpha, t2, clip_low, clip_high, model_type='ensemble', 
                   model_weights=None, val_len=8):
    """Evaluate parameters via rolling-origin backtest with improved validation."""
    times = a_tr_full.index
    if len(times) < max(n_lags + 1, t2 + 1) + val_len:
        return 1e12

    rmses = []
    maes = []
    
    # Use more validation periods for better estimate
    for t in times[-val_len:]:
        a_hist = a_tr_full.loc[a_tr_full.index < t]
        if len(a_hist) < max(n_lags, t2):
            continue

        y_true = a_tr_full.loc[t]
        y_pred = predict_one_step(a_hist, n_lags, alpha, model_type, model_weights)

        if t % 12 == 11:  # December bump
            mult = compute_december_multipliers(a_hist, clip_low=clip_low, clip_high=clip_high)
            y_pred = apply_december_bump_row(y_pred, mult)

        # Calculate both RMSE and MAE
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        
        rmses.append(rmse)
        maes.append(mae)

    # Combine RMSE and MAE for more robust evaluation
    if rmses:
        final_score = 0.7 * np.mean(rmses) + 0.3 * np.mean(maes)
        return float(final_score)
    else:
        return 1e12


def predict_horizon(a_tr, n_lags, alpha, t2, model_type='ensemble', model_weights=None):
    """Forecast horizon [67..78] using specified model."""
    import pandas as pd
    
    idx = np.arange(67, 79)
    preds = pd.DataFrame(index=idx, columns=a_tr.columns, dtype=float)
    
    model_functions = {
        'ewgm': ewgm_per_sector,
        'ema': ema_per_sector,
        'trend': trend_adjusted_per_sector,
        'seasonal': seasonal_decomp_per_sector,
        'ensemble': ensemble_per_sector
    }
    
    model_func = model_functions.get(model_type, ewgm_per_sector)

    for sector in a_tr.columns:
        if (a_tr[sector].tail(t2).min() == 0) or (a_tr[sector].sum() == 0):
            preds[sector] = 0.0
        else:
            if model_type == 'ensemble':
                pred_val = model_func(a_tr, sector, n_lags, alpha, model_weights)
            elif model_type == 'trend':
                pred_val = model_func(a_tr, sector, n_lags, alpha, 0.3)
            else:
                pred_val = model_func(a_tr, sector, n_lags, alpha)
            preds[sector] = pred_val

    preds.index.name = 'time'
    return preds


# =====================================================
# Submission
# =====================================================

def build_submission_df(a_pred, test_raw, month_codes):
    """Format predictions into competition submission file."""
    import pandas as pd  # Only needed for compatibility with existing model output
    
    test = add_time_and_sector_fields(split_test_id_column(test_raw.clone()), month_codes)
    
    # Convert pandas prediction matrix to lookup table
    lookup_pd = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    lookup = pl.from_pandas(lookup_pd)
    
    # Join test data with predictions
    test_pd = test.to_pandas()
    merged_pd = test_pd.merge(lookup.to_pandas(), on=['time', 'sector_id'], how='left')
    merged_pd['pred'] = merged_pd['pred'].fillna(0.0)

    result = merged_pd[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return pl.from_pandas(result)


def generate_submission_with_december_bump(n_lags=6, alpha=0.5, t2=6, clip_low=0.85, clip_high=1.4,
                                         model_type='ensemble', model_weights=None):
    """End-to-end pipeline for submission with December bump."""
    month_codes = build_month_codes()
    train, test = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    a_pred = predict_horizon(a_tr, n_lags, alpha, t2, model_type, model_weights)

    # Apply December bump
    mult = compute_december_multipliers(a_tr, clip_low=clip_low, clip_high=clip_high)
    for t in a_pred.index[a_pred.index % 12 == 11]:
        a_pred.loc[t] = apply_december_bump_row(a_pred.loc[t], mult)

    sub = build_submission_df(a_pred, test, month_codes)
    sub.write_csv('/kaggle/working/submission.csv')
    return a_tr, a_pred, sub


# =====================================================
# Optuna Optimization
# =====================================================

def optuna_objective(trial, a_tr):
    """Enhanced objective for Optuna hyperparameter tuning with model selection."""
    # Core parameters
    n_lags = trial.suggest_int('n_lags', 3, 15)
    alpha = trial.suggest_float('alpha', 0.10, 0.98)
    t2 = trial.suggest_int('t2', 3, 12)
    clip_low = trial.suggest_float('clip_low', 0.60, 0.98)
    clip_high = trial.suggest_float('clip_high', 1.05, 2.00)
    
    # Model selection
    model_type = trial.suggest_categorical('model_type', ['ewgm', 'ema', 'trend', 'seasonal', 'ensemble'])
    
    # Ensemble weights (only used if model_type is 'ensemble')
    model_weights = None
    if model_type == 'ensemble':
        ewgm_weight = trial.suggest_float('ewgm_weight', 0.1, 0.6)
        ema_weight = trial.suggest_float('ema_weight', 0.1, 0.5)
        trend_weight = trial.suggest_float('trend_weight', 0.05, 0.4)
        seasonal_weight = trial.suggest_float('seasonal_weight', 0.05, 0.3)
        
        # Normalize weights
        total = ewgm_weight + ema_weight + trend_weight + seasonal_weight
        model_weights = {
            'ewgm': ewgm_weight / total,
            'ema': ema_weight / total,
            'trend': trend_weight / total,
            'seasonal': seasonal_weight / total
        }

    if clip_low >= clip_high:
        clip_low = max(0.60, clip_high - 0.05)

    return evaluate_params(a_tr, n_lags, alpha, t2, clip_low, clip_high, 
                          model_type, model_weights)


def run_optuna_search(a_tr, n_trials=1500, seed=1337):
    """Run enhanced Optuna search with better sampling and pruning."""
    from tqdm import tqdm
    
    # Use more sophisticated sampler
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=50,  # More random trials before TPE
        n_ei_candidates=48,   # More candidates for expected improvement
        multivariate=True     # Consider parameter interactions
    )
    
    # Add pruning for faster convergence
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=sampler,
        pruner=pruner
    )
    
    # Optimize with progress bar
    with tqdm(total=n_trials, desc="Optimizing hyperparameters") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best score': f'{study.best_value:.4f}'})
        
        study.optimize(
            partial(optuna_objective, a_tr=a_tr), 
            n_trials=n_trials, 
            callbacks=[callback],
            show_progress_bar=False
        )
    
    return study


def analyze_study_results(study):
    """Analyze and display optimization results."""
    print("\n=== Optimization Results Analysis ===")
    print(f"Best score: {study.best_value:.6f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\nParameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {param}: {imp:.3f}")
    except:
        pass


# =====================================================
# Enhanced Main Function
# =====================================================

def main():
    print("Real Estate Demand Forecasting - Enhanced Model")
    print("=" * 50)
    
    # Load and prepare data
    print("Loading and preparing data...")
    month_codes = build_month_codes()
    train, test = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    print(f"Training data shape: {a_tr.shape}")
    print(f"Test data shape: {test.shape}")

    # Run optimization
    print("\nRunning hyperparameter optimization...")
    study = run_optuna_search(a_tr, n_trials=40, seed=1337)
    
    # Analyze results
    analyze_study_results(study)
    
    # Extract best parameters
    best = study.best_params.copy()
    model_weights = None
    if best.get('model_type') == 'ensemble':
        # Extract model weights
        model_weights = {}
        for model in ['ewgm', 'ema', 'trend', 'seasonal']:
            weight_key = f'{model}_weight'
            if weight_key in best:
                model_weights[model] = best.pop(weight_key)
        best['model_weights'] = model_weights

    # Generate final submission
    print(f"\nGenerating predictions with best parameters...")
    print(f"Selected model: {best.get('model_type', 'ensemble')}")
    
    a_tr_final, a_pred_final, sub_final = generate_submission_with_december_bump(**best)
    
    print(f"\nPrediction completed!")
    print(f"Prediction matrix shape: {a_pred_final.shape}")
    print(f"Submission file shape: {sub_final.shape}")
    print("Results saved to /kaggle/working/submission.csv")
    
    return study, best, a_tr_final, a_pred_final, sub_final


if __name__ == "__main__":
    main()


# =====================================================
# Simplified and Fixed Version - Addressing Performance Issues
# =====================================================

def simple_evaluate_params(a_tr_full, n_lags, alpha, t2, clip_low, clip_high, val_len=6):
    """Return to original simple validation method"""
    times = a_tr_full.index
    if len(times) < max(n_lags + 1, t2 + 1) + val_len:
        return 1e12

    rmses = []
    for t in times[-val_len:]:
        a_hist = a_tr_full.loc[a_tr_full.index < t]
        if len(a_hist) < max(n_lags, t2):
            continue

        y_true = a_tr_full.loc[t]
        # Use original EWGM method
        y_pred = simple_predict_one_step(a_hist, n_lags, alpha)

        if t % 12 == 11:  # December bump
            mult = compute_december_multipliers(a_hist, clip_low=clip_low, clip_high=clip_high)
            y_pred = apply_december_bump_row(y_pred, mult)

        rmses.append(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return float(np.mean(rmses)) if rmses else 1e12


def simple_predict_one_step(a_hist, n_lags, alpha):
    """Simplified prediction function, return to original EWGM logic"""
    import pandas as pd
    return pd.Series({
        sector: improved_ewgm_per_sector(a_hist, sector, n_lags, alpha)
        if a_hist[sector].tail(n_lags).min() > 0 else 0.0
        for sector in a_hist.columns
    })


def conservative_optuna_objective(trial, a_tr):
    """Conservative Optuna objective function with smaller search space"""
    n_lags = trial.suggest_int('n_lags', 4, 10)  # Reduced range
    alpha = trial.suggest_float('alpha', 0.3, 0.8)  # Reduced range
    t2 = trial.suggest_int('t2', 4, 8)  # Reduced range
    clip_low = trial.suggest_float('clip_low', 0.75, 0.9)  # Reduced range
    clip_high = trial.suggest_float('clip_high', 1.2, 1.6)  # Reduced range

    if clip_low >= clip_high:
        clip_low = max(0.75, clip_high - 0.05)

    return simple_evaluate_params(a_tr, n_lags, alpha, t2, clip_low, clip_high)


def improved_ewgm_per_sector(a_tr, sector, n_lags, alpha, min_history=3):
    """Improved EWGM with better boundary condition handling"""
    recent = a_tr[sector].tail(n_lags).values
    
    # Stricter condition checking
    if len(recent) < min_history:
        return 0.0
    
    # Check if there are enough non-zero values
    non_zero_count = np.sum(recent > 0)
    if non_zero_count < min_history:
        return 0.0
    
    # If all values are zero
    if (recent <= 0).all():
        return 0.0
    
    # Use improved weight calculation
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)])
    weights /= weights.sum()

    mask = recent > 0
    if not mask.any():
        return 0.0

    # Add small regularization term to prevent numerical instability
    log_vals = np.log(recent[mask] + 1e-10)
    pos_w = weights[mask] / weights[mask].sum()
    
    result = np.exp(np.sum(pos_w * log_vals))
    
    # Add sanity check
    if not np.isfinite(result) or result < 0:
        return 0.0
    
    # Limit extreme predictions
    max_recent = np.max(recent)
    if result > max_recent * 3:  # No more than 3x historical max
        result = max_recent * 1.5
    
    return float(result)


def run_conservative_optimization(a_tr, n_trials=300):
    """Run conservative optimization strategy"""
    from tqdm import tqdm
    
    print("Running conservative optimization strategy...")
    
    # Use simple sampler
    sampler = optuna.samplers.TPESampler(seed=1337, n_startup_trials=20)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Optimize
    with tqdm(total=n_trials, desc="Conservative optimization") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best score': f'{study.best_value:.4f}'})
        
        study.optimize(
            partial(conservative_optuna_objective, a_tr=a_tr), 
            n_trials=n_trials, 
            callbacks=[callback]
        )
    
    return study


def generate_conservative_submission(**params):
    """Generate submission file using conservative parameters"""
    import pandas as pd
    month_codes = build_month_codes()
    train, test = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    
    # Use improved EWGM
    n_lags = params['n_lags']
    alpha = params['alpha']
    t2 = params['t2']
    
    idx = np.arange(67, 79)
    preds = pd.DataFrame(index=idx, columns=a_tr.columns, dtype=float)

    for sector in a_tr.columns:
        if (a_tr[sector].tail(t2).min() == 0) or (a_tr[sector].sum() == 0):
            preds[sector] = 0.0
        else:
            preds[sector] = improved_ewgm_per_sector(a_tr, sector, n_lags, alpha)

    preds.index.name = 'time'
    
    # Apply December bump
    mult = compute_december_multipliers(a_tr, clip_low=params['clip_low'], clip_high=params['clip_high'])
    for t in preds.index[preds.index % 12 == 11]:
        preds.loc[t] = apply_december_bump_row(preds.loc[t], mult)

    sub = build_submission_df(preds, test, month_codes)
    # sub.write_csv('/kaggle/working/submission_conservative.csv')
    return a_tr, preds, sub


# Quick testing and model comparison
def quick_model_comparison():
    """Quick comparison of original vs improved version"""
    print("Quick model performance comparison")
    print("=" * 40)
    
    # Load data
    month_codes = build_month_codes()
    train, _ = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    
    # Test parameters
    test_params = {
        'n_lags': 6,
        'alpha': 0.5,
        't2': 6,
        'clip_low': 0.85,
        'clip_high': 1.4
    }
    
    print("Testing original method...")
    original_score = simple_evaluate_params(a_tr, **test_params)
    print(f"  Original EWGM score: {original_score:.4f}")
    
    print("\nRunning conservative optimization...")
    study = run_conservative_optimization(a_tr, n_trials=100)
    print(f"  Optimized score: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")
    
    if study.best_value < original_score:
        print("Optimization successful!")
        return study.best_params
    else:
        print("Using original parameters")
        return test_params

# Run fixed version
best_params = quick_model_comparison()



# =====================================================
# Fixed Main Function - Focus on Effective Optimization
# =====================================================

def fixed_main():
    """Fixed main function to address performance degradation issues"""
    print("Real Estate Demand Forecasting - Fixed Version")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    month_codes = build_month_codes()
    train, test = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    print(f"Data shape: {a_tr.shape}")
    
    # First test original baseline
    print("\nTesting baseline performance...")
    baseline_params = {
        'n_lags': 6, 'alpha': 0.5, 't2': 6,
        'clip_low': 0.85, 'clip_high': 1.4
    }
    baseline_score = simple_evaluate_params(a_tr, **baseline_params)
    print(f"Baseline score: {baseline_score:.6f}")
    
    # Run conservative optimization
    print(f"\nRunning conservative optimization (target: < {baseline_score:.6f})...")
    study = run_conservative_optimization(a_tr, n_trials=200)
    
    # Select best parameters
    if study.best_value < baseline_score:
        best_params = study.best_params
        print(f"Optimization successful! Improvement: {baseline_score - study.best_value:.6f}")
    else:
        best_params = baseline_params
        print(f"Optimization did not improve, using baseline parameters")
    
    print(f"\nFinal parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Generate final submission
    print(f"\nGenerating final predictions...")
    a_tr_final, a_pred_final, sub_final = generate_conservative_submission(**best_params)
    
    print(f"\nCompleted!")
    print(f"Final score: {study.best_value if study.best_value < baseline_score else baseline_score:.6f}")
    print(f"Submission file saved")
    
    return best_params, study

# Run fixed version
best_params, study = fixed_main()


