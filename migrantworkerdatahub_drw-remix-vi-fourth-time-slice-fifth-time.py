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


# Complete Working Code with 60% Similarity Slice

# Imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor

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
    
    # Similarity configuration
    SIMILARITY_PCT = 0.60  # 60% of training data

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

# Similarity Finding Functions
def find_similar_records_combined(train_df, features, label_col, early_pct=0.30, recent_pct=0.55, similarity_pct=0.60):
    """
    Find records that behave similarly to both early and recent periods.
    This combines multiple methods for robust selection of 60% of data.
    """
    n_samples = len(train_df)
    n_select = int(similarity_pct * n_samples)
    
    early_cutoff = int(early_pct * n_samples)
    recent_cutoff = int((1 - recent_pct) * n_samples)
    
    print(f"\nFinding {n_select} similar records ({similarity_pct*100:.0f}% of {n_samples} total records)")
    print(f"Early period: first {early_cutoff} records")
    print(f"Recent period: last {n_samples - recent_cutoff} records")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df[features])
    y_scaled = StandardScaler().fit_transform(train_df[[label_col]]).ravel()
    
    # Initialize scores array
    similarity_scores = np.zeros(n_samples)
    
    # Method 1: Prediction Agreement
    print("\nMethod 1: Prediction Agreement...")
    
    # Train models on early and recent data
    early_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                              random_state=42, verbosity=0)
    early_model.fit(train_df.iloc[:early_cutoff][features], 
                   train_df.iloc[:early_cutoff][label_col])
    
    recent_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=42, verbosity=0)
    recent_model.fit(train_df.iloc[recent_cutoff:][features], 
                    train_df.iloc[recent_cutoff:][label_col])
    
    # Get predictions
    early_preds = early_model.predict(train_df[features])
    recent_preds = recent_model.predict(train_df[features])
    actual_values = train_df[label_col].values
    
    # Calculate agreement
    pred_diff = np.abs(early_preds - recent_preds)
    pred_agreement = 1.0 - (pred_diff / (np.std(actual_values) + 1e-8))
    
    # Calculate error similarity
    early_errors = np.abs(actual_values - early_preds)
    recent_errors = np.abs(actual_values - recent_preds)
    error_similarity = 1.0 - np.abs(early_errors - recent_errors) / (np.std(actual_values) + 1e-8)
    
    # Both models should be reasonably accurate
    accuracy_score = np.exp(-(early_errors + recent_errors) / (2 * np.std(actual_values)))
    
    # Combine scores
    method1_scores = (pred_agreement * 0.4 + error_similarity * 0.3 + accuracy_score * 0.3)
    similarity_scores += method1_scores / 4  # Will average across 4 methods
    
    # Method 2: Feature-Target Correlation Similarity
    print("Method 2: Feature-Target Correlation Similarity...")
    
    # Calculate correlations in sliding windows
    window_size = max(100, int(0.02 * n_samples))
    correlation_scores = np.zeros(n_samples)
    
    # Get reference correlations
    early_corrs = train_df.iloc[:early_cutoff][features].corrwith(
        train_df.iloc[:early_cutoff][label_col])
    recent_corrs = train_df.iloc[recent_cutoff:][features].corrwith(
        train_df.iloc[recent_cutoff:][label_col])
    
    for i in range(n_samples):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_samples, i + window_size // 2)
        
        if end_idx - start_idx > 50:
            local_corrs = train_df.iloc[start_idx:end_idx][features].corrwith(
                train_df.iloc[start_idx:end_idx][label_col])
            
            # Compare to both early and recent correlations
            early_corr_sim = 1.0 - np.mean(np.abs(local_corrs - early_corrs))
            recent_corr_sim = 1.0 - np.mean(np.abs(local_corrs - recent_corrs))
            
            correlation_scores[i] = np.sqrt(early_corr_sim * recent_corr_sim)
    
    similarity_scores += correlation_scores / 4
    
    # Method 3: Statistical Distribution Similarity
    print("Method 3: Statistical Distribution Similarity...")
    
    # Calculate distribution characteristics
    early_mean = np.mean(X_scaled[:early_cutoff], axis=0)
    early_std = np.std(X_scaled[:early_cutoff], axis=0)
    recent_mean = np.mean(X_scaled[recent_cutoff:], axis=0)
    recent_std = np.std(X_scaled[recent_cutoff:], axis=0)
    
    distribution_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Distance to early distribution
        early_dist = np.sqrt(np.mean(((X_scaled[i] - early_mean) / (early_std + 1e-8))**2))
        # Distance to recent distribution  
        recent_dist = np.sqrt(np.mean(((X_scaled[i] - recent_mean) / (recent_std + 1e-8))**2))
        
        # We want records close to both distributions
        distribution_scores[i] = 1.0 / (1.0 + early_dist * recent_dist)
    
    similarity_scores += distribution_scores / 4
    
    # Method 4: K-Nearest Neighbors Consistency
    print("Method 4: K-Nearest Neighbors Consistency...")
    
    k = 30
    early_knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    early_knn.fit(train_df.iloc[:early_cutoff][features], 
                 train_df.iloc[:early_cutoff][label_col])
    
    recent_knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    recent_knn.fit(train_df.iloc[recent_cutoff:][features], 
                  train_df.iloc[recent_cutoff:][label_col])
    
    # Get predictions
    early_knn_preds = early_knn.predict(train_df[features])
    recent_knn_preds = recent_knn.predict(train_df[features])
    
    # Similarity based on prediction consistency
    knn_pred_diff = np.abs(early_knn_preds - recent_knn_preds)
    knn_scores = 1.0 - (knn_pred_diff / (np.std(actual_values) + 1e-8))
    
    similarity_scores += knn_scores / 4
    
    # Apply smoothing to avoid selecting isolated points
    print("\nApplying spatial smoothing...")
    from scipy.ndimage import gaussian_filter1d
    similarity_scores = gaussian_filter1d(similarity_scores, sigma=10)
    
    # Select top 60% records
    selected_indices = np.argsort(similarity_scores)[-n_select:]
    selected_indices = np.sort(selected_indices)
    
    # Print selection statistics
    early_selected = np.sum(selected_indices < early_cutoff)
    middle_selected = np.sum((selected_indices >= early_cutoff) & (selected_indices < recent_cutoff))
    recent_selected = np.sum(selected_indices >= recent_cutoff)
    
    print(f"\nSelection distribution:")
    print(f"  From early period: {early_selected} ({early_selected/n_select*100:.1f}%)")
    print(f"  From middle period: {middle_selected} ({middle_selected/n_select*100:.1f}%)")
    print(f"  From recent period: {recent_selected} ({recent_selected/n_select*100:.1f}%)")
    
    # Analyze quality of selection
    selected_scores = similarity_scores[selected_indices]
    print(f"\nSimilarity scores:")
    print(f"  Mean score of selected: {np.mean(selected_scores):.4f}")
    print(f"  Min score of selected: {np.min(selected_scores):.4f}")
    print(f"  Mean score of all: {np.mean(similarity_scores):.4f}")
    
    return selected_indices

# Training and Evaluation
def get_model_slices(n_samples: int, similar_indices: np.ndarray):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0, "indices": None},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples), "indices": None},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples), "indices": None},
        {"name": "first_35pct", "type": "early", "cutoff": int(0.35 * n_samples), "indices": None},
        {"name": "similar_60pct", "type": "similar", "cutoff": 0, "indices": similar_indices},
    ]

def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    
    # Find similar records (60% of data)
    similar_indices = find_similar_records_combined(
        train_df, Config.FEATURES, Config.LABEL_COLUMN, 
        early_pct=0.30, recent_pct=0.55, similarity_pct=Config.SIMILARITY_PCT
    )
    
    model_slices = get_model_slices(n_samples, similar_indices)

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
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = np.array([])
                    
            elif slice_type == "similar":
                # Use similar records (non-continuous)
                similar_indices = s["indices"]
                # Find which training indices are in the similar set
                train_in_similar = np.isin(train_idx, similar_indices)
                if np.any(train_in_similar):
                    # Get the actual indices that are both in train and similar
                    actual_train_idx = train_idx[train_in_similar]
                    subset = train_df.iloc[similar_indices].reset_index(drop=True)
                    # Map actual_train_idx to positions in subset
                    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(similar_indices)}
                    rel_idx = np.array([idx_map[idx] for idx in actual_train_idx if idx in idx_map])
                    # Use time decay weights based on original positions
                    original_positions = np.array([idx for idx in actual_train_idx if idx in idx_map])
                    sw = create_time_decay_weights(n_samples)[original_positions]
                else:
                    rel_idx = np.array([])
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
                        
                elif slice_type == "similar":
                    # For similar slice, predict on all validation samples
                    all_preds = model.predict(train_df.iloc[valid_idx][Config.FEATURES])
                    oof_preds[learner["name"]][slice_name][valid_idx] = all_preds
                    
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
        
        # Custom ensemble - adjust weights based on slice performance
        custom_weights = {}
        for s in scores:
            if s == 'similar_60pct':
                # Give similarity slice weight based on its relative performance
                avg_other_scores = np.mean([v for k, v in scores.items() if k != 'similar_60pct'])
                if scores[s] > avg_other_scores:
                    custom_weights[s] = scores[s] * 1.2  # 20% bonus if above average
                else:
                    custom_weights[s] = scores[s] * 0.8  # 20% penalty if below average
            else:
                custom_weights[s] = scores[s]
        
        total_custom = sum(custom_weights.values())
        oof_custom = sum(custom_weights[s] / total_custom * oof_preds[learner_name][s] for s in scores)
        test_custom = sum(custom_weights[s] / total_custom * test_preds[learner_name][s] for s in scores)
        score_custom = pearsonr(train_df[Config.LABEL_COLUMN], oof_custom)[0]

        print(f"\n{learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")
        print(f"{learner_name.upper()} Custom Ensemble Pearson:   {score_custom:.4f}")

        # Store the best performing ensemble
        best_score = max(score_simple, score_weighted, score_custom)
        if best_score == score_custom:
            learner_ensembles[learner_name] = {
                "oof": oof_custom,
                "test": test_custom,
                "type": "custom"
            }
        elif best_score == score_weighted:
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

    submission_df["prediction"] = final_test
    submission_df.to_csv("submission_with_60pct_similarity.csv", index=False)
    print("\nSaved: submission_with_60pct_similarity.csv")
    
    # Also save individual slice predictions for analysis
    print("\nSaving individual slice predictions...")
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            slice_submission = submission_df.copy()
            slice_submission["prediction"] = test_preds[learner_name][slice_name]
            filename = f"submission_{learner_name}_{slice_name}.csv"
            slice_submission.to_csv(filename, index=False)
            print(f"  Saved: {filename}")

# Main
if __name__ == "__main__":
    print("="*80)
    print("CRYPTO PREDICTION WITH 60% SIMILARITY SLICE")
    print("="*80)
    
    # Load data
    train_df, test_df, submission_df = load_data()
    
    # Train and evaluate
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    
    # Create submissions
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)

