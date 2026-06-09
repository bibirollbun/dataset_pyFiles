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


# Single Variation Runner - Set EARLY_PERCENTAGE to desired value

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

# Set all random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========== CONFIGURATION - CHANGE THIS VALUE ==========
EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45
# ======================================================

# Imports
import sys
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Feature Engineering
def feature_engineering(df):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Use explicit order of operations
    df['volume_weighted_sell'] = df['sell_qty'].values * df['volume'].values
    df['buy_sell_ratio'] = df['buy_qty'].values / (df['sell_qty'].values + 1e-8)
    df['selling_pressure'] = df['sell_qty'].values / (df['volume'].values + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'].values - df['sell_qty'].values) / (df['volume'].values + 1e-8)
    
    # Replace inf values
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
    RANDOM_STATE = SEED

# XGBoost parameters for deterministic behavior
XGB_PARAMS = {
    # Core parameters
    'tree_method': 'exact',  # Changed from 'hist' for determinism
    'device': 'cpu',
    'n_jobs': 1,  # Single thread
    
    # Model parameters
    'colsample_bytree': 0.499384, 
    'colsample_bynode': 0.748391,
    'gamma': 7.34723, 
    'learning_rate': 0.4129738, 
    'max_depth': 7, 
    'max_leaves': 40, 
    'n_estimators': 500,
    'reg_alpha': 27.791606770656145, 
    'reg_lambda': 84.90603428439086,
    'subsample': 0.06567,
    
    # Deterministic parameters
    'random_state': Config.RANDOM_STATE,
    'seed': Config.RANDOM_STATE,
    'verbosity': 0,
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]

# Loading Data
def create_time_decay_weights(n: int, decay: float = 0.9, reverse: bool = False) -> np.ndarray:
    """Create time decay weights with deterministic computation."""
    positions = np.arange(n, dtype=np.float64)
    
    if reverse:
        normalized = 1.0 - (positions / (n - 1))
    else:
        normalized = positions / (n - 1)
    
    # Use float64 for consistent precision
    weights = np.power(decay, (1.0 - normalized), dtype=np.float64)
    
    # Normalize weights
    weight_sum = np.sum(weights, dtype=np.float64)
    weights = weights * n / weight_sum
    
    return weights.astype(np.float32)  # Convert back to float32 for memory efficiency

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

def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    
    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
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
                # Use all data
                subset = train_df.reset_index(drop=True)
                rel_idx = train_idx
                sw = full_weights[train_idx]
                
            elif slice_type == "recent":
                # Use data from cutoff to end (recent data)
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                if cutoff > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = full_weights[train_idx]
                    
            elif slice_type == "early":
                # Use data from start to cutoff (early data)
                subset = train_df.iloc[:cutoff].reset_index(drop=True)
                rel_idx = train_idx[train_idx < cutoff]
                if len(rel_idx) > 0:
                    # For early data, we might want to give more weight to later samples within the subset
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = np.array([])
            
            # Skip if no training data available for this slice
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
            
            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train_np, y_train_np, sample_weight=sw, 
                          eval_set=[(X_valid_np, y_valid_np)], verbose=False)
                
                # Handle predictions based on slice type
                if slice_type == "early":
                    # For early slice, only predict on validation samples that were in the training range
                    mask = valid_idx < cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                    # For validation samples outside the early training range, use full_data predictions
                    if (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                else:
                    # For recent slices and full data
                    mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                    if slice_type == "recent" and cutoff > 0 and (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                
                # Test predictions (always use the model regardless of slice type)
                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])
    
    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS
    
    return oof_preds, test_preds, model_slices

# Submission
def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df):
    learner_ensembles = {}
    
    print("\nIndividual Slice Scores:")
    for learner_name in oof_preds:
        scores = {}
        for s in oof_preds[learner_name]:
            # Calculate score only on samples where the model made actual predictions
            # (not filled from other models)
            score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
            scores[s] = score
            print(f"  {learner_name} - {s}: {score:.4f}")
        
        total_score = sum(scores.values())
        
        # Simple average ensemble
        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]
        
        # Weighted ensemble based on OOF scores
        oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
        test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]
        
        print(f"\n{learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")
        
        # Store the better performing ensemble
        if score_weighted > score_simple:
            learner_ensembles[learner_name] = {
                "oof": oof_weighted,
                "test": test_weighted,
                "type": "weighted"
            }
        else:
            learner_ensembles[learner_name] = {
                "oof": oof_simple,
                "test": test_simple,
                "type": "simple"
            }
    
    # Final ensemble across all learners
    final_oof = np.mean([le["oof"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
    
    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")
    print(f"Ensemble types used: {[le['type'] for le in learner_ensembles.values()]}")
    
    # Save with percentage in filename
    filename = f"submission_early_{int(EARLY_PERCENTAGE*100)}pct.csv"
    submission_df["prediction"] = final_test
    submission_df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")

# Main
if __name__ == "__main__":
    print(f"\nRunning with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df)

