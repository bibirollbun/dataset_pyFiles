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
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

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

def adjust_xgb_params_for_features(base_params, n_features):
    """
    Dynamically adjust XGBoost parameters based on number of features
    to prevent overfitting when adding more features
    """
    params = base_params.copy()
    
    # As we add more features, we need more regularization
    if n_features > 25:
        # Increase regularization
        regularization_factor = 1 + (n_features - 25) * 0.1
        params['reg_alpha'] = base_params['reg_alpha'] * regularization_factor
        params['reg_lambda'] = base_params['reg_lambda'] * regularization_factor
        
        # Slightly reduce tree complexity
        params['max_depth'] = max(12, base_params['max_depth'] - (n_features - 25) // 10)
        params['max_leaves'] = max(8, base_params['max_leaves'] - (n_features - 25) // 15)
        
        # Increase minimum child weight
        params['min_child_weight'] = base_params['min_child_weight'] + (n_features - 25) // 5
        
        # Reduce column sampling
        params['colsample_bytree'] = max(0.5, base_params['colsample_bytree'] - (n_features - 25) * 0.01)
        
        print(f"\nAdjusted parameters for {n_features} features:")
        print(f"  reg_alpha: {base_params['reg_alpha']:.1f} -> {params['reg_alpha']:.1f}")
        print(f"  reg_lambda: {base_params['reg_lambda']:.1f} -> {params['reg_lambda']:.1f}")
        print(f"  max_depth: {base_params['max_depth']} -> {params['max_depth']}")
        print(f"  colsample_bytree: {base_params['colsample_bytree']:.3f} -> {params['colsample_bytree']:.3f}")
    
    return params

def robust_feature_selection(train_df, test_df, current_features, target='label', max_new_features=5):
    """
    Robust feature selection that validates features using cross-validation
    and checks for redundancy
    """
    print("\n" + "="*60)
    print("ROBUST FEATURE SELECTION WITH OVERFITTING PREVENTION")
    print("="*60)
    
    # Get all X features not in current selection
    all_x_features = [col for col in train_df.columns if col.startswith('X')]
    available_features = [f for f in all_x_features if f not in current_features]
    
    print(f"Current features: {len(current_features)}")
    print(f"Available X features to evaluate: {len(available_features)}")
    
    # Prepare data
    X_current = train_df[current_features]
    y = train_df[target]
    
    # Calculate baseline performance with current features
    print("\nCalculating baseline performance...")
    baseline_model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    baseline_scores = cross_val_score(baseline_model, X_current, y, cv=5, 
                                    scoring='neg_mean_squared_error')
    baseline_score = -np.mean(baseline_scores)
    print(f"Baseline MSE with {len(current_features)} features: {baseline_score:.6f}")
    
    # Evaluate each candidate feature
    feature_evaluations = []
    
    print("\nEvaluating candidate features...")
    for i, feat in enumerate(available_features[:200]):  # Limit to 200 for speed
        if i % 20 == 0:
            print(f"Progress: {i}/{min(200, len(available_features))} features evaluated")
        
        try:
            # Clean the feature
            train_feat = train_df[feat].replace([np.inf, -np.inf], np.nan)
            test_feat = test_df[feat].replace([np.inf, -np.inf], np.nan)
            
            # Skip if too many missing values
            if train_feat.isna().sum() > len(train_feat) * 0.3:
                continue
            
            train_feat = train_feat.fillna(train_feat.median())
            test_feat = test_feat.fillna(test_feat.median())
            
            # Skip if constant
            if train_feat.std() < 1e-8:
                continue
            
            # 1. Check redundancy with existing features
            max_corr_existing = 0
            for existing_feat in current_features[:20]:  # Check top 20 features
                if existing_feat in train_df.columns:
                    corr = abs(pearsonr(train_feat, train_df[existing_feat])[0])
                    max_corr_existing = max(max_corr_existing, corr)
            
            # Skip if too correlated with existing features
            if max_corr_existing > 0.95:
                continue
            
            # 2. Calculate robust correlation using different methods
            pearson_corr = abs(pearsonr(train_feat, y)[0])
            spearman_corr = abs(spearmanr(train_feat, y)[0])
            
            # 3. Cross-validated importance
            X_with_feat = X_current.copy()
            X_with_feat[feat] = train_feat
            
            # Quick CV to test if feature improves performance
            cv_model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
            cv_scores = cross_val_score(cv_model, X_with_feat, y, cv=3, 
                                      scoring='neg_mean_squared_error')
            cv_score = -np.mean(cv_scores)
            
            # Calculate improvement
            improvement = (baseline_score - cv_score) / baseline_score
            
            # 4. Stability checks
            # Check correlation stability across time windows
            n_windows = 3
            window_size = len(train_df) // n_windows
            window_corrs = []
            
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size if w < n_windows - 1 else len(train_df)
                window_data = train_df.iloc[start:end]
                if len(window_data) > 100:
                    w_corr = abs(pearsonr(window_data[feat], window_data[target])[0])
                    window_corrs.append(w_corr)
            
            corr_stability = 1 - (np.std(window_corrs) / (np.mean(window_corrs) + 1e-8)) if window_corrs else 0
            
            # 5. Permutation test (simplified)
            n_perms = 5
            perm_corrs = []
            for _ in range(n_perms):
                y_perm = np.random.permutation(y)
                perm_corr = abs(pearsonr(train_feat, y_perm)[0])
                perm_corrs.append(perm_corr)
            
            # Feature should have much higher correlation than random
            signal_ratio = pearson_corr / (np.mean(perm_corrs) + 1e-8)
            
            feature_evaluations.append({
                'feature': feat,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'cv_improvement': improvement,
                'redundancy': max_corr_existing,
                'stability': corr_stability,
                'signal_ratio': signal_ratio,
                'combined_score': (
                    0.25 * pearson_corr +
                    0.15 * spearman_corr +
                    0.30 * max(0, improvement) +
                    0.10 * (1 - max_corr_existing) +
                    0.10 * corr_stability +
                    0.10 * min(signal_ratio / 5, 1)  # Cap signal ratio contribution
                )
            })
            
        except Exception as e:
            continue
    
    if not feature_evaluations:
        print("No valid features found!")
        return []
    
    # Sort by combined score
    eval_df = pd.DataFrame(feature_evaluations).sort_values('combined_score', ascending=False)
    
    # Select features that actually improve performance
    selected_new_features = []
    for _, row in eval_df.iterrows():
        if row['cv_improvement'] > 0.001 and row['signal_ratio'] > 2.0 and len(selected_new_features) < max_new_features:
            selected_new_features.append(row['feature'])
    
    print(f"\nSelected {len(selected_new_features)} new features that improve performance")
    
    if selected_new_features:
        print("\nTop selected features:")
        for feat in selected_new_features[:5]:
            feat_info = eval_df[eval_df['feature'] == feat].iloc[0]
            print(f"\n{feat}:")
            print(f"  Correlation: {feat_info['pearson_corr']:.4f}")
            print(f"  CV Improvement: {feat_info['cv_improvement']*100:.2f}%")
            print(f"  Redundancy: {feat_info['redundancy']:.4f}")
            print(f"  Signal Ratio: {feat_info['signal_ratio']:.2f}")
    
    # Save detailed analysis
    eval_df.to_csv("robust_feature_evaluation.csv", index=False)
    
    return selected_new_features

# ==================== MAIN EXECUTION ====================

# Load data
train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
sample = pd.read_csv(CFG.sample_sub_path)

# Original selected features
original_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
]

# Robust feature selection
new_features = robust_feature_selection(train, test, original_features, max_new_features=3)

# Decide whether to add features based on validation
if new_features:
    print(f"\nAdding {len(new_features)} validated features")
    selected_features = original_features + new_features
else:
    print("\nNo features passed validation. Using original feature set.")
    selected_features = original_features

print(f"\nFinal feature count: {len(selected_features)}")

# Select features and reduce memory
train = train[selected_features + ["label"]]
test = test[selected_features]

train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

print("Train=", train.shape)
print("Test=", test.shape)
print("Sample=", sample.shape)

RMV = ["label"]
FEATURES = [c for c in train.columns if c not in RMV]
print(f"There are {len(FEATURES)} FEATURES")

# Define cross-validation
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Base XGBoost parameters
base_xgb_params = {
    "tree_method": "gpu_hist",
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

# Adjust parameters based on feature count
xgb_params = adjust_xgb_params_for_features(base_xgb_params, len(FEATURES))

# Define model configurations
model_configs = [
    {"name": "Model 1 (100% Full Data)", "percent": 1.00},
    {"name": "Model 2 (90% Recent)", "percent": 0.90},
    {"name": "Model 3 (80% Recent)", "percent": 0.80},
    {"name": "Model 4 (70% Recent)", "percent": 0.70},
    {"name": "Model 5 (60% Recent)", "percent": 0.60},
    {"name": "Model 6 (50% Recent)", "percent": 0.50},
    {"name": "Model 7 (40% Recent)", "percent": 0.40}
]

# Initialize predictions for all models
n_models = len(model_configs)
oof_preds_all = [np.zeros(len(train)) for _ in range(n_models)]
test_preds_all = [np.zeros(len(test)) for _ in range(n_models)]

# Generate sample weights for Model 1 (full data)
sample_weights_full = create_time_weights(len(train), decay_factor=0.95)
print(f"\nModel 1 - Full data sample weights range: [{sample_weights_full.min():.4f}, {sample_weights_full.max():.4f}]")
print(f"Model 1 - Full data sample weights mean: {sample_weights_full.mean():.4f}")

# Calculate cutoffs for each model
cutoffs = []
for config in model_configs:
    if config["percent"] == 1.00:
        cutoffs.append(0)
    else:
        cutoff_idx = int(len(train) * (1 - config["percent"]))
        cutoffs.append(cutoff_idx)
        print(f"\n{config['name']} - Using most recent {len(train) - cutoff_idx} samples ({int(config['percent']*100)}% of data)")

# Cross-validation loop
for fold_num, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print("\n" + "#" * 50)
    print(f"### Fold {fold_num + 1}")
    print("#" * 50)
    
    X_valid = train.iloc[valid_idx][FEATURES]
    y_valid = train.iloc[valid_idx]["label"]
    X_test = test[FEATURES]
    
    # Train each model
    for model_idx, (config, cutoff) in enumerate(zip(model_configs, cutoffs)):
        print(f"\n--- {config['name']} ---")
        
        if config["percent"] == 1.00:
            # Model 1: Full data with time weights
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["label"]
            train_weights = sample_weights_full[train_idx]
        else:
            # Other models: Recent data subsets
            train_idx_recent = train_idx[train_idx >= cutoff]
            train_idx_recent_adjusted = train_idx_recent - cutoff
            train_recent = train.iloc[cutoff:].reset_index(drop=True)
            
            X_train = train_recent.iloc[train_idx_recent_adjusted][FEATURES]
            y_train = train_recent.iloc[train_idx_recent_adjusted]["label"]
            
            sample_weights_recent = create_time_weights(len(train_recent), decay_factor=0.95)
            train_weights = sample_weights_recent[train_idx_recent_adjusted]
        
        # Train the model
        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=25,
            verbose=200
        )
        
        # Make predictions
        if config["percent"] == 1.00:
            oof_preds_all[model_idx][valid_idx] = model.predict(X_valid)
        else:
            valid_idx_in_range = valid_idx[valid_idx >= cutoff]
            if len(valid_idx_in_range) > 0:
                X_valid_subset = train.iloc[valid_idx_in_range][FEATURES]
                oof_preds_all[model_idx][valid_idx_in_range] = model.predict(X_valid_subset)
            
            valid_idx_out_range = valid_idx[valid_idx < cutoff]
            if len(valid_idx_out_range) > 0:
                oof_preds_all[model_idx][valid_idx_out_range] = oof_preds_all[0][valid_idx_out_range]
        
        test_preds_all[model_idx] += model.predict(X_test)

# Average test predictions across folds
for i in range(n_models):
    test_preds_all[i] /= FOLDS

# Calculate individual model scores
pearson_scores = []
for i, config in enumerate(model_configs):
    score = pearsonr(train["label"], oof_preds_all[i])[0]
    pearson_scores.append(score)

print("\n" + "=" * 50)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 50)
for config, score in zip(model_configs, pearson_scores):
    print(f"{config['name']} Pearson Correlation: {score:.4f}")

# Create ensemble predictions
ensemble_oof_preds = np.mean(oof_preds_all, axis=0)
ensemble_test_preds = np.mean(test_preds_all, axis=0)
ensemble_pearson_score = pearsonr(train["label"], ensemble_oof_preds)[0]

print("\n" + "=" * 50)
print("ENSEMBLE PERFORMANCE")
print("=" * 50)
print(f"Ensemble (Equal Weight) Pearson Correlation: {ensemble_pearson_score:.4f}")

# Performance-weighted ensemble
total_score = sum(pearson_scores)
weights = [score / total_score for score in pearson_scores]

weighted_ensemble_oof = np.zeros(len(train))
weighted_ensemble_test = np.zeros(len(test))

for i in range(n_models):
    weighted_ensemble_oof += weights[i] * oof_preds_all[i]
    weighted_ensemble_test += weights[i] * test_preds_all[i]

weighted_ensemble_score = pearsonr(train["label"], weighted_ensemble_oof)[0]

print(f"\nWeighted Ensemble Performance:")
for config, weight in zip(model_configs, weights):
    print(f"  {config['name']} weight: {weight:.3f}")
print(f"  Weighted Ensemble Pearson Correlation: {weighted_ensemble_score:.4f}")

# Use the better ensemble for final predictions
if weighted_ensemble_score > ensemble_pearson_score:
    final_test_preds = weighted_ensemble_test
    print("\nUsing weighted ensemble for final predictions")
else:
    final_test_preds = ensemble_test_preds
    print("\nUsing simple average ensemble for final predictions")

# SHAP analysis
print("\nGenerating SHAP analysis...")
model1_for_shap = XGBRegressor(**xgb_params)
model1_for_shap.fit(
    train[FEATURES], train["label"],
    sample_weight=sample_weights_full,
    verbose=0
)
explainer = shap.TreeExplainer(model1_for_shap, feature_perturbation="tree_path_dependent", model_output="raw")
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Save predictions
sample["prediction"] = final_test_preds
sample.to_csv("submission.csv", index=False)
print("\nPredictions saved to submission.csv")
print(sample.head())

# Save detailed results
results_data = {
    'model': [config['name'] for config in model_configs] + ['Simple Ensemble', 'Weighted Ensemble'],
    'pearson_correlation': pearson_scores + [ensemble_pearson_score, weighted_ensemble_score],
    'weight_in_final': [weight if weighted_ensemble_score > ensemble_pearson_score else 1/n_models 
                        for weight in weights] + [np.nan, np.nan]
}

ensemble_results = pd.DataFrame(results_data)
ensemble_results.to_csv("ensemble_results.csv", index=False)
print("\nEnsemble results saved to ensemble_results.csv")
print(ensemble_results)

# Save feature information
feature_info = pd.DataFrame({
    'feature': FEATURES,
    'is_original': [feat in original_features for feat in FEATURES]
})
feature_info.to_csv("final_features.csv", index=False)
print(f"\nFinal feature list saved. Total features: {len(FEATURES)}")

