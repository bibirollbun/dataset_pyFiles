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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

manual_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
]

# XGBoost parameters (same for all models)
xgb_params = {
    "tree_method": "hist",
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

def calculate_corr(all_data,row_to_display):
    """calculate the pearson R of the label and other labels, 
    return the rank of pearson r (ranked results)"""
    results = []

    for col in all_data.columns:
        if col == 'label':
            continue
        if all_data[col].nunique() <= 1:
            continue
        try:
            r, p = pearsonr(all_data[col], all_data['label'])
            results.append((col, r, p))
        except Exception as e:
            print(f"Skipping {col}: {e}")
    
    correlation_df = pd.DataFrame(results, columns=["feature", "pearson_r", "p_value"])
    correlation_df["abs_r"] = correlation_df["pearson_r"].abs()
    correlation_df = correlation_df.sort_values("abs_r", ascending=False).drop("abs_r", axis=1)

    print(correlation_df.head(row_to_display))

    # print(results)
    return correlation_df

# Create time-based sample weights, set decay factors
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
    # more recent data will have higher weights

def integrate_process_features(manual_features,correlation_df, all_data,n_components):
    """require correlation_df[0]>= 50 and has 'feature' column,
    require original data df, returns the final df with the 25 features maintained"""

    # 2. top 50 featurs in pearson test
    top_pearson_features = correlation_df.head(50)["feature"].tolist()

    # 3. merge
    merged_features = list(set(manual_features + top_pearson_features))

    # print(f"concatenated features count: {len(merged_features)}")

    # # PCA & to 25 
    # X = all_data[merged_features]
    # X_scaled = StandardScaler().fit_transform(X)

    # pca = PCA(n_components)  # 保留 25 个主成分
    # X_pca = pca.fit_transform(X_scaled)

    # print(f"original dimension: {X.shape[1]}, after PCA: {X_pca.shape[1]}")
    return merged_features



train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
sample = pd.read_csv(CFG.sample_sub_path)

# reduce mem_usage
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")
print("Train=", train.shape)
print("Test=", test.shape)
print("Sample=", sample.shape)

# calculate correlation matrix, select features best correlate with label
correlate_matrix = calculate_corr(train,50)

# return a new dataset with 25 features
merged_feature = integrate_process_features(manual_features,correlate_matrix,train, 25)
RMV = ["label"] # set features
FEATURES = [c for c in train.columns if c not in RMV] # set features removal
FEATURES = [c for c in FEATURES if c in merged_feature]

print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")

# Define cross-validation
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# initialize predictions for all models
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
        cutoffs.append(0)  # Full data starts from index 0
    else:
        cutoff_idx = int(len(train) * (1 - config["percent"]))
        cutoffs.append(cutoff_idx)
        print(f"\n{config['name']} - Using most recent {len(train) - cutoff_idx} samples ({int(config['percent']*100)}% of data)")

for fold_num, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print("\n" + "#" * 50)
    print(f"### Fold {fold_num + 1}")
    print("#" * 50)

    X_valid = train.iloc[valid_idx][FEATURES]
    y_valid = train.iloc[valid_idx]["label"]
    X_test = test[FEATURES]

    # Kfolds
    for model_idx, (config, cutoff) in enumerate(zip(model_configs, cutoffs)):
        print(f"\n--- {config['name']} ---")

        if config["percent"] == 1.00:
            # Model 1: Full data with time weights
            X_train = train.iloc[train_idx][FEATURES]
            y_train = train.iloc[train_idx]["label"]
            
            # Extract sample weights for this fold's training data
            train_weights = sample_weights_full[train_idx]
        else:
            # Other models: Recent data subsets
            # Filter train indices to only include those from the recent data
            train_idx_recent = train_idx[train_idx >= cutoff]
            
            # Adjust indices to start from 0 for the recent subset
            train_idx_recent_adjusted = train_idx_recent - cutoff
            
            # Get the recent subset of training data
            train_recent = train.iloc[cutoff:].reset_index(drop=True)
            
            X_train = train_recent.iloc[train_idx_recent_adjusted][FEATURES]
            y_train = train_recent.iloc[train_idx_recent_adjusted]["label"]
            
            # Create time weights for the recent data subset
            sample_weights_recent = create_time_weights(len(train_recent), decay_factor=0.95)
            train_weights = sample_weights_recent[train_idx_recent_adjusted]
        # Train Step
        model = XGBRegressor(**xgb_params, early_stopping_rounds=25)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_valid, y_valid)],
            verbose=200
        )

        # predict validation set
        if config["percent"] == 1.00:
            # Model 1 predicts for all validation indices
            oof_preds_all[model_idx][valid_idx] = model.predict(X_valid)
        else:
            valid_idx_in_range = valid_idx[valid_idx >= cutoff]
            if len(valid_idx_in_range) > 0:
                X_valid_subset = train.iloc[valid_idx_in_range][FEATURES]
                oof_preds_all[model_idx][valid_idx_in_range] = model.predict(X_valid_subset)
            
            # use Model 1 predictions 
            valid_idx_out_range = valid_idx[valid_idx < cutoff]
            if len(valid_idx_out_range) > 0:
                oof_preds_all[model_idx][valid_idx_out_range] = oof_preds_all[0][valid_idx_out_range]
        
        # predict test set
        test_preds_all[model_idx] += model.predict(X_test)


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
# Simple average ensemble
ensemble_oof_preds = np.mean(oof_preds_all, axis=0)
ensemble_test_preds = np.mean(test_preds_all, axis=0)

# Calculate ensemble score
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

# SHAP analysis (using Model 1 as representative)
print("\nGenerating SHAP analysis...")
# Note: We need to retrain model1 for SHAP since we're outside the CV loop
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

