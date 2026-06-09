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


# Single Variation Runner with Multiple Iterations for Stability

# ========== DETERMINISTIC SETUP ==========
import os
import random
import numpy as np

# Set environment variables BEFORE importing other libraries
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# ========== CONFIGURATION - CHANGE THIS VALUE ==========
EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45
N_ITERATIONS = 5  # Number of times to run the entire pipeline
# ======================================================

# Imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Feature Engineering
def feature_engineering(df):
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df 

# Configuration
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292",
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    BASE_RANDOM_STATE = 42  # Base seed that will be modified for each iteration

def get_xgb_params(seed):
    """Get XGBoost parameters with specified seed"""
    return {
        'tree_method': 'hist', 
        'device': 'cpu',
        'n_jobs': 1,
        'colsample_bytree': 0.4111224922845363, 
        'colsample_bynode': 0.28869302181383194,
        'gamma': 1.4665430311056709, 
        'learning_rate': 0.014053505540364681, 
        'max_depth': 7, 
        'max_leaves': 40, 
        'n_estimators': 500,
        'reg_alpha': 27.791606770656145, 
        'reg_lambda': 84.90603428439086,
        'subsample': 0.06567,
        'verbosity': 0,
        'random_state': seed,
        'seed': seed,
    }

# Loading Data
def create_time_decay_weights(n: int, decay: float = 0.9, reverse: bool = False) -> np.ndarray:
    """Create time decay weights. If reverse=True, older data gets higher weight."""
    positions = np.arange(n)
    if reverse:
        normalized = 1.0 - (positions / (n - 1))
    else:
        normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

Config.FEATURES += ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
Config.FEATURES = list(set(Config.FEATURES))  # remove duplicates

# Training and Evaluation
def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples)},
    ]

def train_and_evaluate_single_iteration(train_df, test_df, iteration_seed):
    """Run a single iteration of training with a specific seed"""
    print(f"\n{'='*60}")
    print(f"Starting Iteration with seed {iteration_seed}")
    print(f"{'='*60}")
    
    # Set seeds for this iteration
    np.random.seed(iteration_seed)
    random.seed(iteration_seed)
    
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        "xgb": {s["name"]: np.zeros(n_samples) for s in model_slices}
    }
    test_preds = {
        "xgb": {s["name"]: np.zeros(len(test_df)) for s in model_slices}
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            slice_type = s["type"]
            
            if slice_type == "full":
                subset = train_df.reset_index(drop=True)
                rel_idx = train_idx
                sw = full_weights[train_idx]
                
            elif slice_type == "recent":
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                if cutoff > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = full_weights[train_idx]
                    
            elif slice_type == "early":
                subset = train_df.iloc[:cutoff].reset_index(drop=True)
                rel_idx = train_idx[train_idx < cutoff]
                if len(rel_idx) > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = np.array([])

            if len(rel_idx) == 0:
                print(f"  Skipping slice: {slice_name} (no training data in fold)")
                continue

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_valid_np = X_valid.values
            y_valid_np = y_valid.values
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            # Create model with iteration-specific seed
            model = XGBRegressor(**get_xgb_params(iteration_seed + fold))
            model.fit(X_train_np, y_train_np, sample_weight=sw, 
                      eval_set=[(X_valid_np, y_valid_np)], verbose=False)
            
            # Handle predictions based on slice type
            if slice_type == "early":
                mask = valid_idx < cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds["xgb"][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if (~mask).any():
                    oof_preds["xgb"][slice_name][valid_idx[~mask]] = oof_preds["xgb"]["full_data"][valid_idx[~mask]]
            else:
                mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds["xgb"][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if slice_type == "recent" and cutoff > 0 and (~mask).any():
                    oof_preds["xgb"][slice_name][valid_idx[~mask]] = oof_preds["xgb"]["full_data"][valid_idx[~mask]]

            # Test predictions
            test_preds["xgb"][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for slice_name in test_preds["xgb"]:
        test_preds["xgb"][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds

def ensemble_single_iteration(train_df, oof_preds, test_preds):
    """Create ensemble for a single iteration"""
    scores = {}
    for s in oof_preds["xgb"]:
        score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds["xgb"][s])[0]
        scores[s] = score
        print(f"  xgb - {s}: {score:.4f}")
    
    total_score = sum(scores.values())

    # Simple average ensemble
    oof_simple = np.mean(list(oof_preds["xgb"].values()), axis=0)
    test_simple = np.mean(list(test_preds["xgb"].values()), axis=0)
    score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

    # Weighted ensemble based on OOF scores
    oof_weighted = sum(scores[s] / total_score * oof_preds["xgb"][s] for s in scores)
    test_weighted = sum(scores[s] / total_score * test_preds["xgb"][s] for s in scores)
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

    print(f"\nXGB Simple Ensemble Pearson:   {score_simple:.4f}")
    print(f"XGB Weighted Ensemble Pearson: {score_weighted:.4f}")

    # Return the better performing ensemble
    if score_weighted > score_simple:
        return oof_weighted, test_weighted, score_weighted, "weighted"
    else:
        return oof_simple, test_simple, score_simple, "simple"

# Main execution with multiple iterations
def train_with_multiple_iterations(train_df, test_df, submission_df):
    """Run multiple iterations and average the results"""
    
    all_oof_preds = []
    all_test_preds = []
    all_scores = []
    all_types = []
    
    # Run multiple iterations with different seeds
    for iter_num in range(N_ITERATIONS):
        # Use different seed for each iteration
        # You can also use the same seed if you prefer - XGBoost will still have some randomness
        iteration_seed = Config.BASE_RANDOM_STATE + iter_num * 100
        
        print(f"\n\n{'#'*80}")
        print(f"# ITERATION {iter_num + 1} of {N_ITERATIONS}")
        print(f"{'#'*80}")
        
        # Run single iteration
        oof_preds, test_preds = train_and_evaluate_single_iteration(train_df, test_df, iteration_seed)
        
        # Get ensemble for this iteration
        oof_ensemble, test_ensemble, score, ensemble_type = ensemble_single_iteration(train_df, oof_preds, test_preds)
        
        all_oof_preds.append(oof_ensemble)
        all_test_preds.append(test_ensemble)
        all_scores.append(score)
        all_types.append(ensemble_type)
    
    # Average all iterations
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS - Averaging Across All Iterations")
    print(f"{'='*80}")
    
    print(f"\nIndividual iteration scores: {[f'{s:.4f}' for s in all_scores]}")
    print(f"Ensemble types used: {all_types}")
    print(f"Mean score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
    
    # Calculate final predictions as average of all iterations
    final_oof = np.mean(all_oof_preds, axis=0)
    final_test = np.mean(all_test_preds, axis=0)
    
    # Calculate final score
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
    print(f"\nFINAL ensemble score (averaged predictions): {final_score:.4f}")
    
    # Calculate variance of predictions to see stability
    test_std = np.std(all_test_preds, axis=0)
    print(f"Average standard deviation of test predictions: {np.mean(test_std):.6f}")
    print(f"Max standard deviation of test predictions: {np.max(test_std):.6f}")
    
    # Save submission
    filename = f"submission_early_{int(EARLY_PERCENTAGE*100)}pct_{N_ITERATIONS}iter.csv"
    submission_df["prediction"] = final_test
    submission_df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")
    
    return final_score

# Main
if __name__ == "__main__":
    print(f"\nRunning with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    print(f"Number of iterations: {N_ITERATIONS}")
    
    train_df, test_df, submission_df = load_data()
    final_score = train_with_multiple_iterations(train_df, test_df, submission_df)

