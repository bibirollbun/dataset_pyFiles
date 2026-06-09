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


#!/usr/bin/env python3
"""
DRW Crypto Market Prediction - Complete Pipeline
================================================
1. Progressive feature selection with XGBoost
2. Training on full dataset with selected features
3. Time-decay weighted ensemble
4. Submission generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import pearsonr
import warnings
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed
np.random.seed(42)

# Check if we're in Kaggle environment
KAGGLE_INPUT_PATH = '/kaggle/input/drw-crypto-market-prediction'
if os.path.exists(KAGGLE_INPUT_PATH):
    DATA_PATH = KAGGLE_INPUT_PATH
else:
    DATA_PATH = '.'

# =========================
# Part 1: Progressive Feature Selection
# =========================

class ProgressiveXGBoostSelector:
    """
    Progressive feature selection using XGBoost with increasing sample sizes
    """
    
    def __init__(self, sample_sizes=[50000, 100000, 200000], selection_ratio=0.5):
        self.sample_sizes = sample_sizes
        self.selection_ratio = selection_ratio
        self.stage_results = []
        self.selected_features = None
        self.xgb_models = []
        
    def remove_bad_features(self, X, feature_names):
        """Remove features with no variance or all missing values"""
        n_samples, n_features = X.shape
        valid_mask = np.ones(n_features, dtype=bool)
        
        for i in range(n_features):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            
            # Check if feature has any finite values
            if not np.any(finite_mask):
                valid_mask[i] = False
                continue
                
            finite_vals = col[finite_mask]
            
            # Check for zero variance
            if np.var(finite_vals) < 1e-10 or len(np.unique(finite_vals)) == 1:
                valid_mask[i] = False
                continue
                
            # Check if too sparse (>99% zeros)
            if np.sum(finite_vals == 0) / len(finite_vals) > 0.99:
                valid_mask[i] = False
        
        # Return cleaned data and convert feature names to list
        cleaned_features = np.array(feature_names)[valid_mask]
        return X[:, valid_mask], list(cleaned_features), valid_mask
    
    def rank_transform(self, X):
        """Apply rank transformation to all features"""
        X_ranked = np.zeros_like(X)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            
            if np.any(finite_mask):
                # Apply rank transformation to finite values
                ranks = stats.rankdata(col[finite_mask], method='average')
                percentiles = (ranks - 1) / (len(ranks) - 1)
                X_ranked[finite_mask, i] = percentiles
                X_ranked[~finite_mask, i] = 0.5
            else:
                X_ranked[:, i] = 0.5
        
        return X_ranked
    
    def train_xgboost_and_get_importance(self, X, y, feature_names, stage_num):
        """Train XGBoost and calculate feature importance"""
        print(f"\nTraining XGBoost for Stage {stage_num}...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Ensure feature_names is a list
        if isinstance(feature_names, np.ndarray):
            feature_names = list(feature_names)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # XGBoost parameters for feature selection
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # Train model
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=evallist,
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        # Get predictions for evaluation
        y_pred = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        pearson_corr, _ = stats.pearsonr(y_val, y_pred)
        
        print(f"Stage {stage_num} Performance:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  Pearson: {pearson_corr:.6f}")
        
        # Get feature importance
        importance_dict = model.get_score(importance_type='gain')
        
        # Create importance dataframe
        importance_df = pd.DataFrame([
            {'feature': f, 'importance': importance_dict.get(f, 0)}
            for f in feature_names
        ])
        
        # Add additional importance metrics
        importance_df['importance_norm'] = importance_df['importance'] / importance_df['importance'].sum()
        importance_df['rank'] = importance_df['importance'].rank(ascending=False)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Store results
        self.xgb_models.append(model)
        
        return importance_df, {
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson_corr,
            'n_features': len(feature_names),
            'sample_size': len(X)
        }
    
    def select_top_features(self, importance_df, ratio=0.5):
        """Select top features based on importance"""
        n_select = max(1, int(len(importance_df) * ratio))
        top_features = importance_df.head(n_select)['feature'].values
        return list(top_features)
    
    def run_progressive_selection(self, X_full, y_full, feature_names):
        """Run the progressive feature selection pipeline"""
        print("="*80)
        print("PROGRESSIVE FEATURE SELECTION WITH XGBOOST")
        print("="*80)
        
        current_features = list(feature_names)
        stage_summaries = []
        
        for stage_num, sample_size in enumerate(self.sample_sizes, 1):
            print(f"\n{'='*60}")
            print(f"STAGE {stage_num}: {sample_size:,} SAMPLES")
            print('='*60)
            
            # Sample data
            actual_sample_size = min(sample_size, len(X_full))
            if actual_sample_size < len(X_full):
                # Use most recent samples
                sample_indices = np.arange(len(X_full) - actual_sample_size, len(X_full))
            else:
                sample_indices = np.arange(len(X_full))
            
            # Get current feature indices
            feature_indices = [i for i, f in enumerate(feature_names) if f in current_features]
            
            # Extract data
            X_stage = X_full[sample_indices][:, feature_indices]
            y_stage = y_full[sample_indices]
            
            print(f"Stage {stage_num} data shape: {X_stage.shape}")
            
            # Remove bad features
            X_clean, features_clean, valid_mask = self.remove_bad_features(
                X_stage, current_features
            )
            
            print(f"After cleaning: {X_clean.shape} ({len(current_features) - len(features_clean)} features removed)")
            
            # Rank transform
            X_ranked = self.rank_transform(X_clean)
            
            # Train XGBoost and get importance
            importance_df, metrics = self.train_xgboost_and_get_importance(
                X_ranked, y_stage, features_clean, stage_num
            )
            
            # Select top features
            selected_features = self.select_top_features(importance_df, self.selection_ratio)
            
            print(f"\nStage {stage_num} Summary:")
            print(f"  Features in: {len(features_clean)}")
            print(f"  Features selected: {len(selected_features)}")
            print(f"  Top 5 features: {selected_features[:5]}")
            
            # Store stage results
            stage_summary = {
                'stage': stage_num,
                'sample_size': actual_sample_size,
                'features_in': len(features_clean),
                'features_selected': len(selected_features),
                'metrics': metrics,
                'importance_df': importance_df,
                'selected_features': selected_features,
                'top_10_features': list(importance_df.head(10)['feature'].values)
            }
            stage_summaries.append(stage_summary)
            self.stage_results.append(stage_summary)
            
            # Update features for next stage
            current_features = selected_features
            
            # Early stopping if too few features
            if len(current_features) < 10:
                print(f"\nStopping early: Only {len(current_features)} features remaining")
                break
        
        self.selected_features = current_features
        
        return current_features, stage_summaries

# =========================
# Part 2: Full Dataset Training Configuration
# =========================

class Config:
    TRAIN_PATH = os.path.join(DATA_PATH, "train.parquet")
    TEST_PATH = os.path.join(DATA_PATH, "test.parquet")
    SUBMISSION_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

# Optimized XGBoost parameters for full dataset training
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "cpu",  # Change to "gpu" if GPU is available
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]

# =========================
# Part 3: Utility Functions
# =========================

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    """Create time decay weights for samples"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data_with_features(selected_features):
    """Load data with only selected features"""
    # Load full data
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # Filter to selected features
    train_features = [f for f in selected_features if f in train_df.columns]
    test_features = [f for f in selected_features if f in test_df.columns]
    
    print(f"\nUsing {len(train_features)} features for training")
    print(f"Market features included: {[f for f in train_features if f in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']]}")
    
    # Select features and label
    X_train = train_df[train_features]
    y_train = train_df[Config.LABEL_COLUMN]
    X_test = test_df[test_features]
    
    print(f"\nLoaded data - Train: {X_train.shape}, Test: {X_test.shape}, Submission: {submission_df.shape}")
    
    return X_train, y_train, X_test, submission_df, train_features

def get_model_slices(n_samples: int):
    """Define model training slices"""
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)}
    ]

# =========================
# Part 4: Training and Evaluation
# =========================

def train_and_evaluate(X_train, y_train, X_test, feature_names):
    """Train models with time decay and multiple slices"""
    n_samples = len(X_train)
    model_slices = get_model_slices(n_samples)
    
    # Initialize prediction storage
    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(X_test)) for s in model_slices}
        for learner in LEARNERS
    }
    
    # Create weights
    full_weights = create_time_decay_weights(n_samples)
    
    # K-Fold cross-validation
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_valid = X_train.iloc[valid_idx]
        y_valid = y_train.iloc[valid_idx]
        
        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            
            # Get slice of data
            subset_X = X_train.iloc[cutoff:].reset_index(drop=True)
            subset_y = y_train.iloc[cutoff:].reset_index(drop=True)
            
            # Adjust indices for slice
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            
            X_train_slice = subset_X.iloc[rel_idx]
            y_train_slice = subset_y.iloc[rel_idx]
            
            # Get weights for slice
            if cutoff > 0:
                sw = create_time_decay_weights(len(subset_X))[rel_idx]
            else:
                sw = full_weights[train_idx]
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train_slice)}")
            
            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                
                # Train model
                model.fit(
                    X_train_slice, 
                    y_train_slice, 
                    sample_weight=sw,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )
                
                # Out-of-fold predictions
                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(X_train.iloc[idxs])
                
                # Handle predictions for samples before cutoff
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = \
                        oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                
                # Test predictions
                test_preds[learner["name"]][slice_name] += model.predict(X_test)
    
    # Normalize test predictions by number of folds
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS
    
    return oof_preds, test_preds, model_slices

# =========================
# Part 5: Ensemble & Submission
# =========================

def ensemble_and_submit(y_train, oof_preds, test_preds, submission_df):
    """Create ensemble predictions and generate submission"""
    learner_ensembles = {}
    
    for learner_name in oof_preds:
        # Calculate scores for each slice
        scores = {}
        for s in oof_preds[learner_name]:
            scores[s] = pearsonr(y_train, oof_preds[learner_name][s])[0]
        
        total_score = sum(scores.values())
        
        # Simple average ensemble
        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(y_train, oof_simple)[0]
        
        # Weighted average ensemble
        oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
        test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
        score_weighted = pearsonr(y_train, oof_weighted)[0]
        
        print(f"\n{learner_name.upper()} Model Scores:")
        for slice_name, score in scores.items():
            print(f"  {slice_name}: {score:.4f}")
        print(f"  Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"  Weighted Ensemble Pearson: {score_weighted:.4f}")
        
        # Store ensemble predictions
        learner_ensembles[learner_name] = {
            "oof_simple": oof_simple,
            "test_simple": test_simple,
            "oof_weighted": oof_weighted,
            "test_weighted": test_weighted
        }
    
    # Final ensemble across all learners
    final_oof = np.mean([le["oof_simple"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test_simple"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(y_train, final_oof)[0]
    
    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")
    
    # Create submission
    submission_df["prediction"] = final_test
    submission_df.to_csv("submission.csv", index=False)
    print("\nSaved: submission.csv")
    
    return final_score, learner_ensembles

# =========================
# Main Execution Pipeline
# =========================

def main():
    """Main execution function"""
    print("="*80)
    print("DRW CRYPTO MARKET PREDICTION - COMPLETE PIPELINE")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    # Step 1: Load initial data for feature selection
    print("\n" + "="*60)
    print("STEP 1: PROGRESSIVE FEATURE SELECTION")
    print("="*60)
    
    train_path = os.path.join(DATA_PATH, 'train.parquet')
    df = pd.read_parquet(train_path)
    print(f"Total dataset size: {len(df):,} rows")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Initial shape: {X.shape}")
    print(f"Target statistics - Mean: {np.mean(y):.6f}, Std: {np.std(y):.6f}")
    
    # Run progressive feature selection
    selector = ProgressiveXGBoostSelector(
        sample_sizes=[50000, 100000, 200000],
        selection_ratio=0.5
    )
    
    start_time = time.time()
    final_features, stage_summaries = selector.run_progressive_selection(X, y, feature_cols)
    selection_time = time.time() - start_time
    
    print(f"\nFeature selection completed in {selection_time:.2f} seconds")
    print(f"Selected {len(final_features)} features from {len(feature_cols)}")
    
    # Save selected features
    pd.DataFrame({'feature': final_features}).to_csv('selected_features.csv', index=False)
    print("Selected features saved to 'selected_features.csv'")
    
    # Step 2: Train on full dataset with selected features
    print("\n" + "="*60)
    print("STEP 2: FULL DATASET TRAINING WITH SELECTED FEATURES")
    print("="*60)
    
    # Load data with selected features
    X_train, y_train, X_test, submission_df, train_features = load_data_with_features(final_features)
    
    # Train models
    start_time = time.time()
    oof_preds, test_preds, model_slices = train_and_evaluate(X_train, y_train, X_test, train_features)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Step 3: Create ensemble and submission
    print("\n" + "="*60)
    print("STEP 3: ENSEMBLE AND SUBMISSION")
    print("="*60)
    
    final_score, learner_ensembles = ensemble_and_submit(y_train, oof_preds, test_preds, submission_df)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Total processing time: {selection_time + training_time:.2f} seconds")
    print(f"Feature selection time: {selection_time:.2f} seconds")
    print(f"Model training time: {training_time:.2f} seconds")
    print(f"Features used: {len(final_features)} (from {len(feature_cols)} original)")
    print(f"Final CV Pearson score: {final_score:.4f}")
    print(f"\nTop 10 selected features:")
    for i, feat in enumerate(final_features[:10], 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("="*80)
    
    return selector, final_features, final_score

if __name__ == "__main__":
    selector, final_features, final_score = main()

