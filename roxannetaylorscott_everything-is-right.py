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


# https://www.kaggle.com/code/digixintelligence/euphoria-1-0?scriptVersionId=247586720



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
Credit to the following authors and notebooks/discussions:

https://www.kaggle.com/code/bakuer30/drw-remix-vi - For tuning and improving XGBoost parameters and features
https://www.kaggle.com/competitions/drw-crypto-market-prediction/discussion/581193 - For the idea of temporal weights / time slices
https://www.kaggle.com/competitions/drw-crypto-market-prediction/discussion/584475 - For pointing out that the data at the very beginning may be more valuable too

Enhanced version with:
1. Multiple noise removal percentages (0.01% to 1%)
2. Global clean slice that removes records identified as outliers across multiple time slices
3. Intelligent ensembling based on validation scores
4. Comprehensive analysis of generalization impact
"""

# Multi-Configuration Runner with Multiple Noise Removal Percentages

# ========== CONFIGURATION ==========
EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45
NOISE_REMOVAL_PERCENTAGES = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]  # 0.01%, 0.05%, 0.1%, 0.2%, 0.5%, 1%
MIN_SCORE_THRESHOLD = 0.08  # Minimum score for a slice to be included in intelligent ensemble
MIN_OUTLIER_SLICES = 2  # Minimum number of slices a record must be an outlier in to be globally removed
# ===================================

# Imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import json

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
    RANDOM_STATE = 42

XGB_PARAMS = {
    'tree_method': 'hist', 
    'device': 'gpu',
    'n_jobs': -1,
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
    'random_state': Config.RANDOM_STATE
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]

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
def get_model_slices(n_samples: int, include_clean: bool = True, noise_pct: float = 0.001, include_global_clean: bool = True):
    base_slices = [
        {"name": "full_data", "type": "full", "cutoff": 0, "clean": False, "noise_pct": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples), "clean": False, "noise_pct": 0},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples), "clean": False, "noise_pct": 0},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples), "clean": False, "noise_pct": 0},
    ]
    
    if not include_clean:
        return base_slices
    
    # Add clean versions of each slice
    clean_slices = []
    for s in base_slices:
        clean_slice = s.copy()
        clean_slice["name"] = s["name"] + f"_clean{int(noise_pct*10000)/100}pct"
        clean_slice["clean"] = True
        clean_slice["noise_pct"] = noise_pct
        clean_slices.append(clean_slice)
    
    # Add global clean slice (removes outliers found across multiple slices)
    if include_global_clean:
        global_clean_slice = {
            "name": f"global_clean{int(noise_pct*10000)/100}pct",
            "type": "global_clean",
            "cutoff": 0,
            "clean": True,
            "noise_pct": noise_pct,
            "global_clean": True
        }
        clean_slices.append(global_clean_slice)
    
    return base_slices + clean_slices

def identify_slice_noise(X_train, y_train, sample_weights, noise_pct=0.001):
    """Use full XGB model to identify noisiest records in a slice"""
    # Ensure sample weights match the training data size
    if len(sample_weights) != len(X_train):
        print(f"Warning: Weight size mismatch. Weights: {len(sample_weights)}, Data: {len(X_train)}")
        sample_weights = sample_weights[:len(X_train)]
    
    # Use the exact same parameters as the main model
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    
    # Get predictions and errors
    preds = model.predict(X_train)
    errors = np.abs(y_train - preds)
    
    # Identify top noise_pct as noise
    threshold = np.percentile(errors, (1 - noise_pct) * 100)
    clean_mask = errors <= threshold
    
    # Calculate metrics for analysis
    train_score_before = pearsonr(y_train, preds)[0]
    train_score_after = pearsonr(y_train[clean_mask], preds[clean_mask])[0] if clean_mask.sum() > 0 else 0
    
    return clean_mask, {
        'train_score_before': train_score_before,
        'train_score_after': train_score_after,
        'removed_count': (~clean_mask).sum(),
        'mean_error_removed': errors[~clean_mask].mean() if (~clean_mask).sum() > 0 else 0
    }

def identify_global_outliers(train_df, base_slices, noise_pct, min_outlier_count=2):
    """
    Identify records that are outliers in multiple slices.
    
    This function:
    1. Trains separate models on each base slice (full_data, last_75pct, last_50pct, first_Xpct)
    2. Identifies outliers in each slice based on prediction errors
    3. Counts how many slices each record appears as an outlier
    4. Marks records as global outliers if they appear in min_outlier_count or more slices
    
    The intuition is that records consistently identified as outliers across different
    time windows are likely truly problematic and should be removed globally.
    """
    n_samples = len(train_df)
    outlier_counts = np.zeros(n_samples)
    outlier_details = {}
    
    full_weights = create_time_decay_weights(n_samples)
    
    print(f"\n  Identifying global outliers (noise_pct={noise_pct*100:.2f}%)...")
    
    for s in base_slices:
        slice_type = s["type"]
        cutoff = s["cutoff"]
        
        if slice_type == "full":
            subset = train_df.reset_index(drop=True)
            indices = np.arange(n_samples)
            sw = full_weights
        elif slice_type == "recent":
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            indices = np.arange(cutoff, n_samples)
            sw = create_time_decay_weights(len(subset))
        elif slice_type == "early":
            subset = train_df.iloc[:cutoff].reset_index(drop=True)
            indices = np.arange(cutoff)
            sw = create_time_decay_weights(len(subset))
        
        if len(subset) == 0:
            continue
            
        X = subset[Config.FEATURES].values
        y = subset[Config.LABEL_COLUMN].values
        
        # Identify outliers in this slice
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X, y, sample_weight=sw, verbose=False)
        preds = model.predict(X)
        errors = np.abs(y - preds)
        
        threshold = np.percentile(errors, (1 - noise_pct) * 100)
        outlier_mask = errors > threshold
        
        # Map back to original indices
        slice_outliers = indices[outlier_mask]
        outlier_counts[slice_outliers] += 1
        
        outlier_details[s["name"]] = {
            "n_outliers": outlier_mask.sum(),
            "outlier_indices": slice_outliers.tolist()
        }
        
        print(f"    {s['name']}: {outlier_mask.sum()} outliers found")
    
    # Create global clean mask (keep records that are outliers in fewer than min_outlier_count slices)
    global_clean_mask = outlier_counts < min_outlier_count
    n_global_outliers = (outlier_counts >= min_outlier_count).sum()
    
    print(f"  Total global outliers (in {min_outlier_count}+ slices): {n_global_outliers}")
    
    return global_clean_mask, outlier_details

def train_and_evaluate(train_df, test_df, model_slices):
    n_samples = len(train_df)
    
    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }
    
    # Store detailed metrics
    slice_metrics = {s["name"]: {"folds": [], "noise_analysis": []} for s in model_slices}
    
    # For global clean slice, first identify global outliers
    global_clean_masks = {}
    for s in model_slices:
        if s.get("global_clean", False):
            base_slices = [sl for sl in model_slices if not sl["clean"]]
            global_clean_mask, outlier_details = identify_global_outliers(
                train_df, base_slices, s["noise_pct"], MIN_OUTLIER_SLICES
            )
            global_clean_masks[s["noise_pct"]] = global_clean_mask
            slice_metrics[s["name"]]["outlier_details"] = outlier_details

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
            is_clean = s["clean"]
            noise_pct = s.get("noise_pct", 0)
            is_global_clean = s.get("global_clean", False)
            
            if is_global_clean:
                # Use global clean mask
                global_mask = global_clean_masks[noise_pct]
                
                # Get training indices that pass the global clean mask
                train_mask = global_mask[train_idx]
                clean_train_idx = train_idx[train_mask]
                
                if len(clean_train_idx) == 0:
                    print(f"  Skipping slice: {slice_name} (no training data after global cleaning)")
                    continue
                
                X_train = train_df.iloc[clean_train_idx][Config.FEATURES]
                y_train = train_df.iloc[clean_train_idx][Config.LABEL_COLUMN]
                
                # Get weights for the clean indices
                sw = full_weights[clean_train_idx]
                # Renormalize weights
                sw = sw * len(sw) / sw.sum()
                
                X_train_np = X_train.values
                y_train_np = y_train.values
                
                print(f"  Training slice: {slice_name}, samples: {len(X_train)} (global outlier removal)")
                
            elif slice_type == "full":
                X_train = train_df.iloc[train_idx][Config.FEATURES]
                y_train = train_df.iloc[train_idx][Config.LABEL_COLUMN]
                sw = full_weights[train_idx]
                
            elif slice_type == "recent":
                mask = train_idx >= cutoff
                filtered_idx = train_idx[mask]
                if len(filtered_idx) == 0:
                    print(f"  Skipping slice: {slice_name} (no training data in fold)")
                    continue
                    
                X_train = train_df.iloc[filtered_idx][Config.FEATURES]
                y_train = train_df.iloc[filtered_idx][Config.LABEL_COLUMN]
                
                # Create weights for the subset
                subset_positions = filtered_idx - cutoff
                subset_weights = create_time_decay_weights(len(train_df) - cutoff)
                sw = subset_weights[subset_positions]
                    
            elif slice_type == "early":
                mask = train_idx < cutoff
                filtered_idx = train_idx[mask]
                if len(filtered_idx) == 0:
                    print(f"  Skipping slice: {slice_name} (no training data in fold)")
                    continue
                    
                X_train = train_df.iloc[filtered_idx][Config.FEATURES]
                y_train = train_df.iloc[filtered_idx][Config.LABEL_COLUMN]
                
                # Create weights for the subset
                subset_weights = create_time_decay_weights(cutoff)
                sw = subset_weights[filtered_idx]

            if not is_global_clean:
                X_train_np = X_train.values
                y_train_np = y_train.values
            
            X_valid_np = X_valid.values
            y_valid_np = y_valid.values
            
            noise_analysis = None
            
            # For clean slices (but not global clean), identify and remove noise
            if is_clean and not is_global_clean:
                original_size = len(X_train)
                clean_mask, noise_analysis = identify_slice_noise(X_train_np, y_train_np, sw, noise_pct)
                
                X_train_np = X_train_np[clean_mask]
                y_train_np = y_train_np[clean_mask]
                sw = sw[clean_mask]
                
                print(f"  Training slice: {slice_name}, samples: {len(X_train_np)} (removed {original_size - len(X_train_np)} noisy records)")
                slice_metrics[slice_name]["noise_analysis"].append(noise_analysis)
            elif not is_global_clean:
                print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train_np, y_train_np, sample_weight=sw, 
                          eval_set=[(X_valid_np, y_valid_np)], verbose=False)
                
                # Get validation predictions
                if is_global_clean or slice_type == "full":
                    # For global clean and full data, predict on all validation samples
                    preds = model.predict(X_valid)
                    oof_preds[learner["name"]][slice_name][valid_idx] = preds
                    
                    valid_score = pearsonr(y_valid, preds)[0]
                    slice_metrics[slice_name]["folds"].append({
                        "fold": fold,
                        "valid_score": valid_score,
                        "n_valid": len(valid_idx)
                    })
                    
                elif slice_type == "early":
                    mask = valid_idx < cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        preds = model.predict(train_df.iloc[idxs][Config.FEATURES])
                        oof_preds[learner["name"]][slice_name][idxs] = preds
                        
                        valid_score = pearsonr(train_df.iloc[idxs][Config.LABEL_COLUMN], preds)[0]
                        slice_metrics[slice_name]["folds"].append({
                            "fold": fold,
                            "valid_score": valid_score,
                            "n_valid": len(idxs)
                        })
                    
                    if (~mask).any():
                        base_name = "full_data"
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]][base_name][valid_idx[~mask]]
                else:
                    mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                    if mask.any():
                        idxs = valid_idx[mask]
                        preds = model.predict(train_df.iloc[idxs][Config.FEATURES])
                        oof_preds[learner["name"]][slice_name][idxs] = preds
                        
                        valid_score = pearsonr(train_df.iloc[idxs][Config.LABEL_COLUMN], preds)[0]
                        slice_metrics[slice_name]["folds"].append({
                            "fold": fold,
                            "valid_score": valid_score,
                            "n_valid": len(idxs)
                        })
                    
                    if slice_type == "recent" and cutoff > 0 and (~mask).any():
                        base_name = "full_data"
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]][base_name][valid_idx[~mask]]

                # Test predictions
                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, slice_metrics

# Analysis and Submission
def analyze_generalization(slice_metrics):
    """Analyze how noise removal affects generalization"""
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)
    
    for slice_name, metrics in slice_metrics.items():
        if metrics["folds"]:
            valid_scores = [f["valid_score"] for f in metrics["folds"]]
            avg_valid = np.mean(valid_scores)
            std_valid = np.std(valid_scores)
            
            print(f"\n{slice_name}:")
            print(f"  Average validation score: {avg_valid:.4f} (±{std_valid:.4f})")
            
            if metrics["noise_analysis"]:
                train_before = np.mean([n["train_score_before"] for n in metrics["noise_analysis"]])
                train_after = np.mean([n["train_score_after"] for n in metrics["noise_analysis"]])
                avg_removed = np.mean([n["removed_count"] for n in metrics["noise_analysis"]])
                
                print(f"  Training score before cleaning: {train_before:.4f}")
                print(f"  Training score after cleaning: {train_after:.4f}")
                print(f"  Average samples removed: {avg_removed:.0f}")
                print(f"  Generalization gap (train-valid): {train_after - avg_valid:.4f}")
            
            # Special handling for global clean slice
            if "outlier_details" in metrics:
                print(f"  Global outlier analysis:")
                for slice_name_detail, details in metrics["outlier_details"].items():
                    print(f"    - {slice_name_detail}: {details['n_outliers']} outliers")

def create_intelligent_ensemble(train_df, oof_preds, test_preds, slice_scores, min_score=0.08):
    """Create ensemble using only slices above threshold"""
    print(f"\nCreating intelligent ensemble (min score: {min_score})")
    
    learner_ensembles = {}
    
    for learner_name in oof_preds:
        # Filter slices by score
        good_slices = {s: score for s, score in slice_scores[learner_name].items() if score >= min_score}
        
        if not good_slices:
            print(f"Warning: No slices above threshold for {learner_name}, using all slices")
            good_slices = slice_scores[learner_name]
        
        print(f"\n{learner_name}: Using {len(good_slices)}/{len(slice_scores[learner_name])} slices")
        for s, score in good_slices.items():
            print(f"  - {s}: {score:.4f}")
        
        total_score = sum(good_slices.values())
        
        # Weighted ensemble of good slices
        oof_intelligent = sum(good_slices[s] / total_score * oof_preds[learner_name][s] for s in good_slices)
        test_intelligent = sum(good_slices[s] / total_score * test_preds[learner_name][s] for s in good_slices)
        
        score_intelligent = pearsonr(train_df[Config.LABEL_COLUMN], oof_intelligent)[0]
        
        learner_ensembles[learner_name] = {
            "oof": oof_intelligent,
            "test": test_intelligent,
            "score": score_intelligent,
            "n_slices": len(good_slices)
        }
    
    return learner_ensembles

def save_submission(submission_df, predictions, filename_suffix, description):
    """Save submission with clear naming"""
    filename = f"submission_{filename_suffix}.csv"
    submission_copy = submission_df.copy()
    submission_copy["prediction"] = predictions
    submission_copy.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")
    print(f"Description: {description}")
    return filename

# Main execution
def run_all_experiments(train_df, test_df, submission_df):
    """Run all experiments with different noise removal percentages"""
    
    all_results = {}
    
    # 1. First create baseline (original 4 slices only)
    print("\n" + "="*80)
    print("CREATING BASELINE (Original 4 slices only)")
    print("="*80)
    
    baseline_slices = get_model_slices(len(train_df), include_clean=False)
    oof_baseline, test_baseline, metrics_baseline = train_and_evaluate(train_df, test_df, baseline_slices)
    
    # Calculate baseline scores
    baseline_scores = {}
    for learner_name in oof_baseline:
        baseline_scores[learner_name] = {}
        for s in oof_baseline[learner_name]:
            score = pearsonr(train_df[Config.LABEL_COLUMN], oof_baseline[learner_name][s])[0]
            baseline_scores[learner_name][s] = score
    
    # Create baseline ensemble
    final_baseline = np.mean([
        np.mean(list(test_baseline[learner_name].values()), axis=0) 
        for learner_name in test_baseline
    ], axis=0)
    
    save_submission(submission_df, final_baseline, 
                   f"baseline_4slices_early{int(EARLY_PERCENTAGE*100)}pct",
                   "Baseline ensemble of 4 original time slices (no cleaning)")
    
    all_results["baseline"] = {
        "scores": baseline_scores,
        "final_score": pearsonr(train_df[Config.LABEL_COLUMN], 
                               np.mean([np.mean(list(oof_baseline[learner_name].values()), axis=0) 
                                       for learner_name in oof_baseline], axis=0))[0]
    }
    
    # 2. Run experiments with different noise removal percentages
    for noise_pct in NOISE_REMOVAL_PERCENTAGES:
        print(f"\n" + "="*80)
        print(f"EXPERIMENT: Noise Removal = {noise_pct*100:.2f}%")
        print("="*80)
        
        # Get slices with this noise removal percentage
        model_slices = get_model_slices(len(train_df), include_clean=True, noise_pct=noise_pct)
        
        # Train and evaluate
        oof_preds, test_preds, slice_metrics = train_and_evaluate(train_df, test_df, model_slices)
        
        # Analyze generalization
        analyze_generalization(slice_metrics)
        
        # Calculate all slice scores
        slice_scores = {}
        for learner_name in oof_preds:
            slice_scores[learner_name] = {}
            for s in oof_preds[learner_name]:
                score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
                slice_scores[learner_name][s] = score
        
        # Save results
        experiment_key = f"noise_{int(noise_pct*10000)/100}pct"
        all_results[experiment_key] = {
            "noise_pct": noise_pct,
            "scores": slice_scores,
            "metrics": slice_metrics
        }
        
        # Create different ensemble combinations
        # Each ensemble represents a different hypothesis about which slices are most valuable
        
        # a) All slices - simple average (hypothesis: all perspectives equally valuable)
        n_total_slices = len(model_slices)
        final_all_simple = np.mean([
            np.mean(list(test_preds[learner_name].values()), axis=0) 
            for learner_name in test_preds
        ], axis=0)
        
        save_submission(submission_df, final_all_simple,
                       f"all{n_total_slices}slices_simple_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                       f"Simple average of all {n_total_slices} slices with {noise_pct*100:.2f}% noise removal")
        
        # b) All slices - weighted by score
        final_all_weighted = np.zeros(len(test_df))
        total_weight = 0
        for learner_name in test_preds:
            for slice_name, preds in test_preds[learner_name].items():
                weight = slice_scores[learner_name][slice_name]
                final_all_weighted += weight * preds
                total_weight += weight
        final_all_weighted /= total_weight
        
        save_submission(submission_df, final_all_weighted,
                       f"all{n_total_slices}slices_weighted_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                       f"Score-weighted average of all {n_total_slices} slices with {noise_pct*100:.2f}% noise removal")
        
        # c) Clean slices only (excluding global clean)
        clean_preds = []
        for learner_name in test_preds:
            learner_clean = [test_preds[learner_name][s] for s in test_preds[learner_name] 
                           if "_clean" in s and "global_clean" not in s]
            if learner_clean:
                clean_preds.append(np.mean(learner_clean, axis=0))
        
        if clean_preds:
            final_clean_only = np.mean(clean_preds, axis=0)
            save_submission(submission_df, final_clean_only,
                           f"clean4slices_only_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                           f"Clean slices only (4 slices) with {noise_pct*100:.2f}% noise removal")
        
        # d) Global clean only
        global_clean_name = f"global_clean{int(noise_pct*10000)/100}pct"
        if global_clean_name in test_preds[list(test_preds.keys())[0]]:
            final_global_clean = np.mean([
                test_preds[learner_name][global_clean_name]
                for learner_name in test_preds
            ], axis=0)
            
            save_submission(submission_df, final_global_clean,
                           f"globalclean_only_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                           f"Global clean only with {noise_pct*100:.2f}% noise removal")
        
        # e) All clean slices including global
        all_clean_preds = []
        for learner_name in test_preds:
            learner_all_clean = [test_preds[learner_name][s] for s in test_preds[learner_name] 
                                if "_clean" in s]
            if learner_all_clean:
                all_clean_preds.append(np.mean(learner_all_clean, axis=0))
        
        if all_clean_preds:
            final_all_clean = np.mean(all_clean_preds, axis=0)
            save_submission(submission_df, final_all_clean,
                           f"allclean5slices_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                           f"All clean slices (5 total) with {noise_pct*100:.2f}% noise removal")
        
        # f) Intelligent ensemble (above threshold)
        intelligent_ensembles = create_intelligent_ensemble(
            train_df, oof_preds, test_preds, slice_scores, MIN_SCORE_THRESHOLD
        )
        
        final_intelligent = np.mean([
            le["test"] for le in intelligent_ensembles.values()
        ], axis=0)
        
        save_submission(submission_df, final_intelligent,
                       f"intelligent_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                       f"Intelligent ensemble (score>{MIN_SCORE_THRESHOLD}) with {noise_pct*100:.2f}% noise removal")
        
        # g) Top 3 slices only
        top_3_slices = {}
        for learner_name in slice_scores:
            sorted_slices = sorted(slice_scores[learner_name].items(), key=lambda x: x[1], reverse=True)[:3]
            top_3_slices[learner_name] = dict(sorted_slices)
        
        final_top3 = np.zeros(len(test_df))
        total_weight = 0
        for learner_name in test_preds:
            for slice_name, score in top_3_slices[learner_name].items():
                final_top3 += score * test_preds[learner_name][slice_name]
                total_weight += score
        final_top3 /= total_weight
        
        save_submission(submission_df, final_top3,
                       f"top3slices_noise{int(noise_pct*10000)/100}pct_early{int(EARLY_PERCENTAGE*100)}pct",
                       f"Top 3 performing slices with {noise_pct*100:.2f}% noise removal")
    
    # 3. Save summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary = {
        "early_percentage": EARLY_PERCENTAGE,
        "experiments": {}
    }
    
    for exp_name, exp_data in all_results.items():
        if exp_name == "baseline":
            summary["experiments"][exp_name] = {
                "final_score": exp_data["final_score"],
                "n_slices": 4
            }
        else:
            # Calculate average scores for clean vs original
            avg_clean = np.mean([
                score for learner_scores in exp_data["scores"].values()
                for slice_name, score in learner_scores.items()
                if "_clean" in slice_name and "global_clean" not in slice_name
            ])
            avg_original = np.mean([
                score for learner_scores in exp_data["scores"].values()
                for slice_name, score in learner_scores.items()
                if "_clean" not in slice_name
            ])
            
            # Get global clean score if available
            global_clean_scores = [
                score for learner_scores in exp_data["scores"].values()
                for slice_name, score in learner_scores.items()
                if "global_clean" in slice_name
            ]
            avg_global_clean = np.mean(global_clean_scores) if global_clean_scores else None
            
            summary["experiments"][exp_name] = {
                "noise_pct": exp_data["noise_pct"],
                "avg_clean_score": avg_clean,
                "avg_original_score": avg_original,
                "avg_global_clean_score": avg_global_clean,
                "improvement": avg_clean - avg_original
            }
    
    # Save summary as JSON
    with open(f"experiment_summary_early{int(EARLY_PERCENTAGE*100)}pct.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nExperiment Summary:")
    print(f"{'Experiment':<20} {'Clean Avg':<12} {'Original Avg':<12} {'Global Clean':<12} {'Improvement':<12}")
    print("-" * 68)
    
    for exp_name, exp_info in summary["experiments"].items():
        if exp_name == "baseline":
            print(f"{exp_name:<20} {'N/A':<12} {exp_info['final_score']:.4f}")
        else:
            clean_avg = f"{exp_info['avg_clean_score']:.4f}"
            orig_avg = f"{exp_info['avg_original_score']:.4f}"
            global_avg = f"{exp_info['avg_global_clean_score']:.4f}" if exp_info['avg_global_clean_score'] else "N/A"
            improvement = f"{exp_info['improvement']:.4f}"
            print(f"{exp_name:<20} {clean_avg:<12} {orig_avg:<12} {global_avg:<12} {improvement:<12}")

# Main
if __name__ == "__main__":
    print(f"\nRunning experiments with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    print(f"Testing noise removal percentages: {[p*100 for p in NOISE_REMOVAL_PERCENTAGES]}%")
    print(f"Global outliers: Records appearing as outliers in {MIN_OUTLIER_SLICES}+ slices")
    print(f"\nThis will create:")
    print(f"  - 1 baseline file (4 original slices)")
    print(f"  - 7 files per noise level × {len(NOISE_REMOVAL_PERCENTAGES)} levels = {7 * len(NOISE_REMOVAL_PERCENTAGES)} files")
    print(f"  - 1 JSON summary")
    print(f"  - Total: {1 + 7 * len(NOISE_REMOVAL_PERCENTAGES) + 1} files")
    
    train_df, test_df, submission_df = load_data()
    run_all_experiments(train_df, test_df, submission_df)

