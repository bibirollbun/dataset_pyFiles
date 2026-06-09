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
import shap
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import time
import warnings
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
warnings.filterwarnings('ignore')

class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Progress tracking files
    progress_file = "feature_selection_progress.json"
    results_file = "feature_selection_results_incremental.csv"
    batch_size = 50  # Number of features to test per run
    
    # Parallel processing settings
    n_workers = min(4, multiprocessing.cpu_count())  # Number of parallel workers
    use_parallel = True  # Toggle parallel processing
    
    # Early stopping settings
    early_stopping_enabled = True
    early_stopping_patience = 20  # Stop if no improvement in last N features
    early_stopping_min_improvement = 0.0001  # Minimum improvement threshold

def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe

def create_time_weights(n_samples, decay_factor=0.95):
    """
    Create exponentially decaying weights based on sample position.
    More recent samples (higher indices) get higher weights.
    decay_factor controls the rate of decay (0.95 = 5% decay per time unit)
    """
    positions = np.arange(n_samples)
    # Normalize positions to [0, 1] range
    normalized_positions = positions / (n_samples - 1)
    # Apply exponential weighting
    weights = decay_factor ** (1 - normalized_positions)
    # Normalize weights to sum to n_samples (maintains scale)
    weights = weights * n_samples / weights.sum()
    return weights

def evaluate_feature_set(train_data, features, sample_weights, xgb_params, n_folds=5):
    """
    Evaluate a feature set using cross-validation and return the average Pearson correlation
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_data))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        X_train = train_data.iloc[train_idx][features]
        y_train = train_data.iloc[train_idx]["label"]
        X_valid = train_data.iloc[valid_idx][features]
        y_valid = train_data.iloc[valid_idx]["label"]
        
        # Extract sample weights for this fold's training data
        train_weights = sample_weights[train_idx]
        
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=25,
            verbose=0
        )
        
        oof_preds[valid_idx] = model.predict(X_valid)
    
    pearson_score = pearsonr(train_data["label"], oof_preds)[0]
    return pearson_score, oof_preds

def evaluate_single_feature(feature, train_data, selected_features, sample_weights, xgb_params, baseline_score):
    """
    Evaluate a single feature (for parallel processing)
    """
    test_features = selected_features + [feature]
    
    try:
        start_time = time.time()
        score, _ = evaluate_feature_set(train_data, test_features, sample_weights, xgb_params)
        elapsed_time = time.time() - start_time
        
        improvement = score - baseline_score
        
        return {
            'feature': feature,
            'score': score,
            'improvement': improvement,
            'time': elapsed_time,
            'success': True
        }
    except Exception as e:
        return {
            'feature': feature,
            'score': None,
            'improvement': None,
            'time': None,
            'success': False,
            'error': str(e)
        }

def check_early_stopping(results, patience, min_improvement):
    """
    Check if early stopping criteria is met
    """
    if len(results) < patience:
        return False, ""
    
    # Get last 'patience' results
    recent_results = results[-patience:]
    
    # Check if all recent improvements are below threshold
    recent_improvements = [r['improvement'] for r in recent_results if r.get('improvement') is not None]
    
    if len(recent_improvements) == patience and all(imp < min_improvement for imp in recent_improvements):
        avg_recent_improvement = np.mean(recent_improvements)
        message = f"Early stopping triggered: No significant improvements (>{min_improvement}) in last {patience} features. "
        message += f"Average recent improvement: {avg_recent_improvement:.6f}"
        return True, message
    
    return False, ""

def load_progress():
    """Load previous progress if it exists"""
    if os.path.exists(CFG.progress_file):
        with open(CFG.progress_file, 'r') as f:
            return json.load(f)
    return None

def save_progress(progress_data):
    """Save current progress"""
    with open(CFG.progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_previous_results():
    """Load previous results if they exist"""
    if os.path.exists(CFG.results_file):
        return pd.read_csv(CFG.results_file)
    return None

# Load data
print("Loading data...")
train_full = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test_full = pd.read_parquet(CFG.test_path).reset_index(drop=True)
sample = pd.read_csv(CFG.sample_sub_path)

# Define initial selected features
selected_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
]

# Identify all available features
all_features = [col for col in train_full.columns if col not in ["timestamp", "label"]]
print(f"Total available features: {len(all_features)}")

# Identify features not in current selection
additional_features = [f for f in all_features if f not in selected_features]
print(f"Total features to test: {len(additional_features)}")

# Load previous progress if exists
progress = load_progress()
if progress:
    print("\n" + "="*60)
    print("RESUMING FROM PREVIOUS PROGRESS")
    print("="*60)
    tested_features = progress.get('tested_features', [])
    baseline_score = progress.get('baseline_score', None)
    best_score = progress.get('best_score', baseline_score)
    best_feature = progress.get('best_feature', None)
    best_features_list = progress.get('best_features_list', selected_features.copy())
    early_stopped = progress.get('early_stopped', False)
    
    # Remove already tested features
    additional_features = [f for f in additional_features if f not in tested_features]
    print(f"Features already tested: {len(tested_features)}")
    print(f"Remaining features to test: {len(additional_features)}")
    print(f"Current best score: {best_score:.6f}")
    if best_feature:
        print(f"Current best additional feature: {best_feature}")
    if early_stopped:
        print("Note: Previous run was early stopped")
else:
    print("\n" + "="*60)
    print("STARTING FRESH RUN")
    print("="*60)
    tested_features = []
    baseline_score = None
    best_score = None
    best_feature = None
    best_features_list = selected_features.copy()
    early_stopped = False

# Prepare data with all features for memory efficiency
train = train_full[all_features + ["label"]]
test = test_full[all_features]

# Reduce memory usage
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# Generate sample weights
sample_weights = create_time_weights(len(train), decay_factor=0.95)

# XGBoost parameters
xgb_params = {
    "tree_method": "gpu_hist" if CFG.n_workers == 1 else "hist",  # Use CPU hist for parallel
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0
}

# Evaluate baseline model if not already done
if baseline_score is None:
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL")
    print("="*60)
    start_time = time.time()
    baseline_score, baseline_oof = evaluate_feature_set(train, selected_features, sample_weights, xgb_params)
    baseline_time = time.time() - start_time
    print(f"Baseline Pearson Correlation: {baseline_score:.6f}")
    print(f"Time taken: {baseline_time:.2f} seconds")
    
    # Initialize best score if not set
    if best_score is None:
        best_score = baseline_score

# Load previous results
previous_results = load_previous_results()
if previous_results is not None:
    results = previous_results.to_dict('records')
    print(f"\nLoaded {len(results)} previous results")
else:
    results = []

# Check for early stopping if enabled
if CFG.early_stopping_enabled and not early_stopped:
    should_stop, stop_message = check_early_stopping(results, CFG.early_stopping_patience, CFG.early_stopping_min_improvement)
    if should_stop:
        print("\n" + "="*60)
        print("EARLY STOPPING TRIGGERED")
        print("="*60)
        print(stop_message)
        early_stopped = True
        additional_features = []  # Stop testing new features

# Select batch of features to test
features_to_test = additional_features[:CFG.batch_size]
if not features_to_test:
    print("\n" + "="*60)
    print("ALL FEATURES HAVE BEEN TESTED OR EARLY STOPPED!")
    print("="*60)
    print(f"Best score achieved: {best_score:.6f}")
    if best_feature:
        print(f"Best additional feature: {best_feature}")
    # Skip to final model training
else:
    # Feature selection for current batch
    print("\n" + "="*60)
    print(f"TESTING BATCH OF {len(features_to_test)} FEATURES")
    if CFG.use_parallel and CFG.n_workers > 1:
        print(f"Using parallel processing with {CFG.n_workers} workers")
    print("="*60)
    
    batch_start_time = time.time()
    
    if CFG.use_parallel and CFG.n_workers > 1:
        # Parallel evaluation
        eval_func = partial(
            evaluate_single_feature,
            train_data=train,
            selected_features=selected_features,
            sample_weights=sample_weights,
            xgb_params=xgb_params,
            baseline_score=baseline_score
        )
        
        with ProcessPoolExecutor(max_workers=CFG.n_workers) as executor:
            # Submit all features for evaluation
            future_to_feature = {executor.submit(eval_func, feature): feature 
                               for feature in features_to_test}
            
            # Process completed evaluations
            for idx, future in enumerate(as_completed(future_to_feature)):
                feature = future_to_feature[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        print(f"\nCompleted {idx+1}/{len(features_to_test)}: {feature}")
                        print(f"  Score: {result['score']:.6f} | Improvement: {result['improvement']:+.6f} | Time: {result['time']:.2f}s")
                        
                        # Store results
                        results.append(result)
                        
                        # Update best if improved
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_feature = feature
                            best_features_list = selected_features + [feature]
                            print(f"  *** NEW BEST FEATURE! ***")
                    else:
                        print(f"\nFailed {idx+1}/{len(features_to_test)}: {feature}")
                        print(f"  Error: {result.get('error', 'Unknown error')}")
                    
                    tested_features.append(feature)
                    
                except Exception as e:
                    print(f"\nError processing {feature}: {str(e)}")
                    tested_features.append(feature)
    else:
        # Sequential evaluation (original code)
        for idx, feature in enumerate(features_to_test):
            print(f"\nTesting feature {idx+1}/{len(features_to_test)} (Total tested: {len(tested_features) + idx + 1}): {feature}")
            
            result = evaluate_single_feature(
                feature, train, selected_features, sample_weights, 
                xgb_params, baseline_score
            )
            
            if result['success']:
                print(f"  Score: {result['score']:.6f} | Improvement: {result['improvement']:+.6f} | Time: {result['time']:.2f}s")
                results.append(result)
                
                # Update best if improved
                if result['score'] > best_score:
                    best_score = result['score']
                    best_feature = feature
                    best_features_list = selected_features + [feature]
                    print(f"  *** NEW BEST FEATURE! ***")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
            tested_features.append(feature)
    
    batch_elapsed_time = time.time() - batch_start_time
    print(f"\nBatch completed in {batch_elapsed_time:.2f} seconds")
    
    # Check for early stopping after batch
    if CFG.early_stopping_enabled:
        should_stop, stop_message = check_early_stopping(results, CFG.early_stopping_patience, CFG.early_stopping_min_improvement)
        if should_stop:
            print("\n" + "="*60)
            print("EARLY STOPPING TRIGGERED")
            print("="*60)
            print(stop_message)
            early_stopped = True
    
    # Save progress
    progress_data = {
        'tested_features': tested_features,
        'baseline_score': baseline_score,
        'best_score': best_score,
        'best_feature': best_feature,
        'best_features_list': best_features_list,
        'early_stopped': early_stopped,
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_progress(progress_data)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    results_df.to_csv(CFG.results_file, index=False)
    print(f"\nResults saved to {CFG.results_file}")

# Show summary of all results so far
if results:
    results_df = pd.DataFrame(results)
    # Filter out failed results for summary
    successful_results = results_df[results_df['score'].notna()]
    
    if len(successful_results) > 0:
        successful_results = successful_results.sort_values('score', ascending=False)
        
        print("\n" + "="*60)
        print("CURRENT FEATURE SELECTION SUMMARY")
        print("="*60)
        print(f"\nBaseline Score: {baseline_score:.6f}")
        print(f"Best Score: {best_score:.6f}")
        print(f"Features tested so far: {len(tested_features)}")
        print(f"Features remaining: {len(additional_features) - len(features_to_test)}")
        
        if best_feature:
            print(f"\nBest Additional Feature: {best_feature}")
            print(f"Improvement: {best_score - baseline_score:+.6f}")
        
        # Show top 10 features
        print("\nTop 10 Features by Score (from all tested features):")
        print(successful_results.head(10)[['feature', 'score', 'improvement']].to_string(index=False))
        
        # Show early stopping status
        if CFG.early_stopping_enabled:
            recent_improvements = successful_results.tail(CFG.early_stopping_patience)['improvement'].values
            if len(recent_improvements) > 0:
                avg_recent_improvement = np.mean(recent_improvements)
                print(f"\nAverage improvement in last {len(recent_improvements)} features: {avg_recent_improvement:.6f}")

# Check if all features have been tested or early stopped
if len(additional_features) == 0 or early_stopped:
    print("\n" + "="*60)
    print("FEATURE SELECTION COMPLETE - TRAINING FINAL MODEL")
    print("="*60)
    
    # Reset to GPU for final training if available
    xgb_params['tree_method'] = 'gpu_hist'
    
    # Train final model with best feature set
    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    feature_importances = {}
    
    for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"\n### Fold {i + 1} ###")
        
        X_train = train.iloc[train_idx][best_features_list]
        y_train = train.iloc[train_idx]["label"]
        X_valid = train.iloc[valid_idx][best_features_list]
        y_valid = train.iloc[valid_idx]["label"]
        X_test = test[best_features_list]
        
        # Extract sample weights for this fold's training data
        train_weights = sample_weights[train_idx]
        
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=25,
            verbose=100
        )
        
        oof_preds[valid_idx] = model.predict(X_valid)
        test_preds += model.predict(X_test)
        
        fold_score = pearsonr(y_valid, oof_preds[valid_idx])[0]
        print(f"Fold {i + 1} Pearson Correlation: {fold_score:.6f}")
        
        # Store feature importances
        for feat, imp in zip(best_features_list, model.feature_importances_):
            if feat not in feature_importances:
                feature_importances[feat] = []
            feature_importances[feat].append(imp)
    
    # Calculate final score
    final_score = pearsonr(train["label"], oof_preds)[0]
    print(f"\nFinal Out-of-Fold Pearson Correlation: {final_score:.6f}")
    
    # Average test predictions
    test_preds /= FOLDS
    
    # Calculate average feature importances
    avg_importances = {feat: np.mean(imps) for feat, imps in feature_importances.items()}
    importance_df = pd.DataFrame({
        'feature': list(avg_importances.keys()),
        'importance': list(avg_importances.values())
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # SHAP analysis
    print("\nGenerating SHAP analysis...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    shap_values = explainer.shap_values(X_test[:1000])  # Use subset for efficiency
    shap.summary_plot(shap_values, X_test[:1000])
    
    # Save submission
    sample["prediction"] = test_preds
    sample.to_csv("submission.csv", index=False)
    print("\nPredictions saved to submission.csv")
    print(sample.head())
    
    # Save comprehensive feature summary
    feature_summary_data = {
        'baseline_features': selected_features,
        'baseline_score': baseline_score,
        'best_additional_feature': best_feature if best_feature else 'None',
        'final_score': final_score,
        'improvement': final_score - baseline_score,
        'total_features_tested': len(tested_features),
        'early_stopped': early_stopped,
        'final_feature_count': len(best_features_list)
    }
    
    with open("feature_selection_summary.json", 'w') as f:
        json.dump(feature_summary_data, f, indent=2)
    
    # Save feature importances
    importance_df.to_csv("feature_importances.csv", index=False)
    
    print("\nFeature selection summary saved to feature_selection_summary.json")
    print("Feature importances saved to feature_importances.csv")
    
    # Clean up progress file since we're done
    if os.path.exists(CFG.progress_file):
        os.remove(CFG.progress_file)
        print("\nProgress file cleaned up")
else:
    print("\n" + "="*60)
    print("BATCH COMPLETE - MORE FEATURES REMAIN")
    print("="*60)
    print(f"Run the script again to test the next {CFG.batch_size} features")
    print(f"Progress has been saved to {CFG.progress_file}")
    if CFG.early_stopping_enabled:
        print(f"\nEarly stopping is enabled (patience={CFG.early_stopping_patience}, min_improvement={CFG.early_stopping_min_improvement})")

