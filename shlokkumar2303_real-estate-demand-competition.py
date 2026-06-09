import numpy as np
import pandas as pd
import optuna
from functools import partial
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================
# Utility Functions
# =====================================================

def build_month_codes() -> Dict[str, int]:
    """Create a mapping from month abbreviations to numeric values (1-12)."""
    return {m: i for i, m in enumerate(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}

def split_test_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ID column into month text and sector components."""
    try:
        parts = df['id'].str.split('_', expand=True)
        if parts.shape[1] != 2:
            raise ValueError("ID column must contain exactly one underscore separator.")
        df['month_text'] = parts[0]
        df['sector'] = parts[1]
        return df
    except Exception as e:
        logger.error(f"Error splitting ID column: {e}")
        raise

def add_time_and_sector_fields(df: pd.DataFrame, month_codes: Dict[str, int]) -> pd.DataFrame:
    """Add parsed year, month, time index, and sector_id to DataFrame."""
    try:
        if 'sector' in df.columns:
            df['sector_id'] = df['sector'].str.extract(r'(\d+)$').astype(int)

        if 'month' in df.columns:  # Test data
            df['year'] = df['month'].str[:4].astype(int)
            df['month'] = df['month'].str[5:].map(month_codes)
        else:  # Train data
            df['year'] = df['month_text'].str[:4].astype(int)
            df['month'] = df['month_text'].str[5:].map(month_codes)

        if df[['year', 'month']].isna().any().any():
            raise ValueError("Invalid month or year values detected.")

        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
        return df
    except Exception as e:
        logger.error(f"Error adding time and sector fields: {e}")
        raise

def load_competition_data(data_path: str = '/kaggle/input/china-real-estate-demand-prediction') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition training and test datasets."""
    try:
        train = pd.read_csv(f'{data_path}/train/new_house_transactions.csv')
        test = pd.read_csv(f'{data_path}/test.csv')
        logger.info("Successfully loaded train and test datasets.")
        return train, test
    except FileNotFoundError as e:
        logger.error(f"Data files not found at {data_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# =====================================================
# Data Transformation
# =====================================================

def build_amount_matrix(train: pd.DataFrame, month_codes: Dict[str, int]) -> pd.DataFrame:
    """Pivot training data into [time x sector_id] transaction matrix."""
    try:
        train = add_time_and_sector_fields(train.copy(), month_codes)
        pivot = train.pivot_table(
            index='time', columns='sector_id',
            values='amount_new_house_transactions', aggfunc='sum', fill_value=0
        )
        # Ensure all 96 sectors are present
        all_sectors = np.arange(1, 97)
        pivot = pivot.reindex(columns=all_sectors, fill_value=0)
        logger.info("Amount matrix built successfully.")
        return pivot
    except Exception as e:
        logger.error(f"Error building amount matrix: {e}")
        raise

# =====================================================
# Modeling Helpers
# =====================================================

def compute_december_multipliers(
    a_tr: pd.DataFrame, 
    eps: float = 1e-9, 
    min_dec_obs: int = 1, 
    clip_low: float = 0.8, 
    clip_high: float = 1.5
) -> Dict[int, float]:
    """Compute sector-level December multipliers from training data."""
    try:
        is_dec = (a_tr.index % 12 == 11)
        dec_means = a_tr[is_dec].mean()
        nondec_means = a_tr[~is_dec].mean()
        dec_counts = a_tr[is_dec].count()

        raw_mult = dec_means / (nondec_means + eps)
        overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))

        raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
        raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
        return raw_mult.clip(clip_low, clip_high).to_dict()
    except Exception as e:
        logger.error(f"Error computing December multipliers: {e}")
        raise

def apply_december_bump_row(pred_row: pd.Series, sector_to_mult: Dict[int, float]) -> pd.Series:
    """Apply December adjustment to a prediction row."""
    try:
        return pred_row.multiply(pd.Series(sector_to_mult)).fillna(pred_row)
    except Exception as e:
        logger.error(f"Error applying December bump: {e}")
        raise

def ewgm_per_sector(a_tr: pd.DataFrame, sector: int, n_lags: int, alpha: float) -> float:
    """Exponential weighted geometric mean for one sector."""
    try:
        recent = a_tr[sector].tail(n_lags).values
        if len(recent) < n_lags or (recent <= 0).all():
            return 0.0

        weights = np.array([(1 - alpha) * (alpha ** i) for i in range(n_lags - 1, -1, -1)])
        weights /= weights.sum()

        mask = recent > 0
        if not mask.any():
            return 0.0

        log_vals = np.log(recent[mask] + 1e-12)
        pos_w = weights[mask] / weights[mask].sum()
        return float(np.exp(np.sum(pos_w * log_vals)))
    except Exception as e:
        logger.error(f"Error computing EWGM for sector {sector}: {e}")
        return 0.0

def predict_one_step(a_hist: pd.DataFrame, n_lags: int, alpha: float) -> pd.Series:
    """Predict next-step values for all sectors."""
    try:
        return pd.Series({
            sector: ewgm_per_sector(a_hist, sector, n_lags, alpha)
            for sector in a_hist.columns
        })
    except Exception as e:
        logger.error(f"Error predicting one step: {e}")
        raise

def evaluate_params(
    a_tr_full: pd.DataFrame, 
    n_lags: int, 
    alpha: float, 
    t2: int, 
    clip_low: float, 
    clip_high: float, 
    val_len: int = 6
) -> float:
    """Evaluate parameters via rolling-origin backtest."""
    try:
        times = a_tr_full.index
        if len(times) < max(n_lags + 1, t2 + 1) + val_len:
            logger.warning("Insufficient data for evaluation.")
            return 1e12

        rmses = []
        for t in times[-val_len:]:
            a_hist = a_tr_full.loc[a_tr_full.index < t]
            if len(a_hist) < max(n_lags, t2):
                continue

            y_true = a_tr_full.loc[t]
            y_pred = predict_one_step(a_hist, n_lags, alpha)

            if t % 12 == 11:  # December bump
                mult = compute_december_multipliers(a_hist, clip_low=clip_low, clip_high=clip_high)
                y_pred = apply_december_bump_row(y_pred, mult)

            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            rmses.append(rmse)

        return float(np.mean(rmses)) if rmses else 1e12
    except Exception as e:
        logger.error(f"Error evaluating parameters: {e}")
        return 1e12

def predict_horizon(a_tr: pd.DataFrame, n_lags: int, alpha: float, t2: int) -> pd.DataFrame:
    """Forecast horizon [67..78]."""
    try:
        idx = np.arange(67, 79)
        preds = pd.DataFrame(index=idx, columns=a_tr.columns, dtype=float)
        for sector in a_tr.columns:
            if (a_tr[sector].tail(t2).min() == 0) or (a_tr[sector].sum() == 0):
                preds[sector] = 0.0
            else:
                preds[sector] = ewgm_per_sector(a_tr, sector, n_lags, alpha)
        preds.index.name = 'time'
        logger.info("Prediction horizon generated.")
        return preds
    except Exception as e:
        logger.error(f"Error predicting horizon: {e}")
        raise

# =====================================================
# Submission
# =====================================================

def build_submission_df(a_pred: pd.DataFrame, test_raw: pd.DataFrame, month_codes: Dict[str, int]) -> pd.DataFrame:
    """Format predictions into competition submission file."""
    try:
        test = add_time_and_sector_fields(split_test_id_column(test_raw.copy()), month_codes)
        lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
        merged = test.merge(lookup, on=['time', 'sector_id'], how='left')
        merged['pred'] = merged['pred'].fillna(0.0)
        return merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    except Exception as e:
        logger.error(f"Error building submission DataFrame: {e}")
        raise

def generate_submission_with_december_bump(
    n_lags: int = 6, 
    alpha: float = 0.5, 
    t2: int = 6, 
    clip_low: float = 0.85, 
    clip_high: float = 1.4, 
    data_path: str = '/kaggle/input/china-real-estate-demand-prediction',
    output_path: str = '/kaggle/working/submission.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end pipeline for submission with December bump."""
    try:
        month_codes = build_month_codes()
        train, test = load_competition_data(data_path)
        a_tr = build_amount_matrix(train, month_codes)
        a_pred = predict_horizon(a_tr, n_lags, alpha, t2)

        # Apply December bump
        mult = compute_december_multipliers(a_tr, clip_low=clip_low, clip_high=clip_high)
        for t in a_pred.index[a_pred.index % 12 == 11]:
            a_pred.loc[t] = apply_december_bump_row(a_pred.loc[t], mult)

        sub = build_submission_df(a_pred, test, month_codes)
        sub.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        return a_tr, a_pred, sub
    except Exception as e:
        logger.error(f"Error generating submission: {e}")
        raise

# =====================================================
# Optuna Optimization
# =====================================================

def optuna_objective(trial: optuna.Trial, a_tr: pd.DataFrame) -> float:
    """Objective for Optuna hyperparameter tuning."""
    try:
        n_lags = trial.suggest_int('n_lags', 3, 12)
        alpha = trial.suggest_float('alpha', 0.20, 0.95)
        t2 = trial.suggest_int('t2', 3, 9)
        clip_low = trial.suggest_float('clip_low', 0.70, 0.95)
        clip_high = trial.suggest_float('clip_high', 1.10, 1.80)

        if clip_low >= clip_high:
            clip_low = max(0.70, clip_high - 0.05)

        return evaluate_params(a_tr, n_lags, alpha, t2, clip_low, clip_high)
    except Exception as e:
        logger.error(f"Error in Optuna objective: {e}")
        return 1e12

def run_optuna_search(a_tr: pd.DataFrame, n_trials: int = 1000, seed: int = 1337) -> optuna.Study:
    """Run Optuna search and return the study."""
    try:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(partial(optuna_objective, a_tr=a_tr), n_trials=n_trials, show_progress_bar=True)
        logger.info(f"Optuna search completed with best value: {study.best_value}")
        return study
    except Exception as e:
        logger.error(f"Error running Optuna search: {e}")
        raise

# =====================================================
# Main
# =====================================================

def main(n_trials: int = 512, seed: int = 1337):
    """Main function to run the prediction pipeline."""
    try:
        logger.info("Starting prediction pipeline...")
        month_codes = build_month_codes()
        train, _ = load_competition_data()
        a_tr = build_amount_matrix(train, month_codes)

        # Run Optuna search
        study = run_optuna_search(a_tr, n_trials=n_trials, seed=seed)
        best = study.best_params
        logger.info(f"Best parameters: {best}")

        # Generate submission
        a_tr, a_pred, sub = generate_submission_with_december_bump(**best)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

