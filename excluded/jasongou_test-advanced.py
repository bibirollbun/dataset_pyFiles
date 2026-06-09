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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

# ==================== UTILITY FUNCTIONS ====================

def reduce_mem_usage(dataframe, dataset):    
    print(f'Reducing memory usage for: {dataset}')
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
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')
    
    return dataframe

def create_time_weights(n_samples, decay_factor=0.95):
    """Create exponentially decaying weights for time-based importance."""
    positions = np.arange(n_samples)
    normalized_positions = positions / (n_samples - 1)
    weights = decay_factor ** (1 - normalized_positions)
    weights = weights * n_samples / weights.sum()
    return weights

def remove_highly_correlated_features(df, features, threshold=0.95):
    """Remove features with high correlation to reduce redundancy."""
    print(f"\nRemoving highly correlated features (threshold={threshold})...")
    corr_matrix = df[features].corr().abs()
    upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the feature with lower correlation to target if available
                if 'label' in df.columns:
                    corr_i = abs(df[corr_matrix.columns[i]].corr(df['label']))
                    corr_j = abs(df[corr_matrix.columns[j]].corr(df['label']))
                    to_drop.add(corr_matrix.columns[i] if corr_i < corr_j else corr_matrix.columns[j])
                else:
                    to_drop.add(corr_matrix.columns[j])
    
    print(f"Removing {len(to_drop)} highly correlated features")
    return [f for f in features if f not in to_drop]

# ==================== ENHANCED FEATURE ENGINEERING ====================

def create_market_microstructure_features(df):
    """Create domain-specific features for crypto market microstructure."""
    print("\nCreating market microstructure features...")
    new_features = {}
    
    # Order book features
    if all(col in df.columns for col in ['bid_qty', 'ask_qty']):
        new_features['order_book_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
        new_features['order_book_spread'] = np.log1p(df['ask_qty']) - np.log1p(df['bid_qty'])
        new_features['order_book_depth'] = df['bid_qty'] + df['ask_qty']
        new_features['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    
    # Trade flow features
    if all(col in df.columns for col in ['buy_qty', 'sell_qty']):
        new_features['trade_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        new_features['net_trade_flow'] = df['buy_qty'] - df['sell_qty']
        new_features['trade_intensity'] = df['buy_qty'] + df['sell_qty']
        new_features['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    
    # Volume features
    if 'volume' in df.columns:
        new_features['log_volume'] = np.log1p(df['volume'])
        if 'trade_intensity' in new_features:
            new_features['avg_trade_size'] = df['volume'] / (new_features['trade_intensity'] + 1e-8)
    
    # Combined liquidity indicators
    if all(f in new_features for f in ['order_book_depth', 'trade_intensity']):
        new_features['liquidity_ratio'] = new_features['trade_intensity'] / (new_features['order_book_depth'] + 1e-8)
    
    return pd.DataFrame(new_features, index=df.index)

def create_pca_features(train_df, test_df, features, n_components=30):
    """Create PCA features based on EDA insights."""
    print(f"\nCreating PCA features (n_components={n_components})...")
    
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    # Create PCA feature names
    pca_features = [f'PCA_{i+1}' for i in range(n_components)]
    
    # Convert to DataFrame
    train_pca_df = pd.DataFrame(train_pca, columns=pca_features, index=train_df.index)
    test_pca_df = pd.DataFrame(test_pca, columns=pca_features, index=test_df.index)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return train_pca_df, test_pca_df

def create_cluster_features(train_df, test_df, features, n_clusters=5):
    """Create cluster-based features from EDA insights."""
    print(f"\nCreating cluster features (n_clusters={n_clusters})...")
    
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(train_scaled)
    test_clusters = kmeans.predict(test_scaled)
    
    # Calculate distances to each cluster center
    train_distances = kmeans.transform(train_scaled)
    test_distances = kmeans.transform(test_scaled)
    
    # Create cluster features
    cluster_features = {}
    cluster_features['cluster'] = train_clusters
    
    for i in range(n_clusters):
        cluster_features[f'dist_to_cluster_{i}'] = train_distances[:, i]
    
    train_cluster_df = pd.DataFrame(cluster_features, index=train_df.index)
    
    # Test features
    test_cluster_features = {}
    test_cluster_features['cluster'] = test_clusters
    
    for i in range(n_clusters):
        test_cluster_features[f'dist_to_cluster_{i}'] = test_distances[:, i]
    
    test_cluster_df = pd.DataFrame(test_cluster_features, index=test_df.index)
    
    return train_cluster_df, test_cluster_df

def create_interaction_features(df, top_features, max_interactions=50):
    """Create interaction features for top performing features."""
    print(f"\nCreating interaction features for top {len(top_features)} features...")
    new_features = {}
    
    # Create interactions only for top features to manage memory
    interaction_count = 0
    for i, feat1 in enumerate(top_features):
        if interaction_count >= max_interactions:
            break
        for j, feat2 in enumerate(top_features[i+1:], i+1):
            if interaction_count >= max_interactions:
                break
            
            # Multiplicative interaction
            new_features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            # Ratio interaction (with safety for division)
            denominator = df[feat2].replace(0, np.nan)
            new_features[f'{feat1}_div_{feat2}'] = df[feat1] / denominator
            new_features[f'{feat1}_div_{feat2}'].fillna(0, inplace=True)
            
            interaction_count += 2
    
    print(f"Created {len(new_features)} interaction features")
    return pd.DataFrame(new_features, index=df.index)

# ==================== FEATURE SELECTION ====================

def select_features_ensemble(train_df, test_df, target='label'):
    """Enhanced feature selection based on EDA insights."""
    print("\n" + "="*60)
    print("ENHANCED FEATURE SELECTION")
    print("="*60)
    
    # Define feature groups based on EDA
    # Top correlated features from EDA
    top_correlated = ['X21', 'X20', 'X28', 'X863', 'X29', 'X19', 'X27', 'X22', 
                      'X858', 'X219', 'X860', 'X531', 'X287', 'X289', 'X291']
    
    # Features selected by all methods in EDA
    consensus_features = ['X175', 'X179', 'X137', 'X197', 'X22', 'X40', 'X181', 
                         'X28', 'X169', 'X198', 'X173']
    
    # Original selected features from simpler model
    original_selected = ["X863", "X856", "X344", "X598", "X862", "X385", "X852", 
                        "X603", "X860", "X674", "X415", "X345", "X137", "X855", 
                        "X174", "X302", "X178", "X532", "X168", "X612"]
    
    # Market features
    market_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Combine all important features
    all_important = list(set(top_correlated + consensus_features + original_selected))
    
    # Get additional X features not in our important list
    all_x_features = [col for col in train_df.columns if col.startswith('X')]
    remaining_x_features = [f for f in all_x_features if f not in all_important]
    
    # Calculate correlations for remaining features
    print("Evaluating remaining features...")
    y = train_df[target]
    feature_scores = []
    
    for feat in remaining_x_features[:200]:  # Evaluate top 200 remaining
        try:
            corr = abs(pearsonr(train_df[feat], y)[0])
            feature_scores.append((feat, corr))
        except:
            continue
    
    # Sort and select top additional features
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    additional_features = [feat for feat, _ in feature_scores[:15]]
    
    # Final feature list
    selected_features = all_important + additional_features + market_features
    selected_features = list(set(selected_features))  # Remove duplicates
    
    # Remove highly correlated features
    selected_features = remove_highly_correlated_features(
        train_df, selected_features, threshold=0.95
    )
    
    print(f"\nSelected {len(selected_features)} features")
    print(f"  - Top correlated: {len([f for f in selected_features if f in top_correlated])}")
    print(f"  - Consensus features: {len([f for f in selected_features if f in consensus_features])}")
    print(f"  - Market features: {len([f for f in selected_features if f in market_features])}")
    
    return selected_features

# ==================== MODEL CONFIGURATIONS ====================

def get_model_params():
    """Define optimized parameters for each boosting algorithm."""
    
    xgb_params = {
        "tree_method": "gpu_hist" if "gpu" in str(pd.__version__) else "hist",
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "max_depth": 15,
        "max_leaves": 20,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.6,
        "colsample_bynode": 0.5,
        "gamma": 2.0,
        "reg_alpha": 40,
        "reg_lambda": 80,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }
    
    catboost_params = {
        "iterations": 1500,
        "learning_rate": 0.02,
        "depth": 10,
        "l2_leaf_reg": 50,
        "min_data_in_leaf": 30,
        "random_strength": 2.0,
        "bagging_temperature": 0.8,
        "border_count": 128,
        "subsample": 0.8,
        "sampling_frequency": "PerTree",
        "random_state": 42,
        "verbose": False,
        "thread_count": -1,
        "task_type": "GPU" if "gpu" in str(pd.__version__) else "CPU"
    }
    
    lgbm_params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 12,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 40,
        "reg_lambda": 80,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "metric": "rmse",
        "device": "gpu" if "gpu" in str(pd.__version__) else "cpu"
    }
    
    return xgb_params, catboost_params, lgbm_params

# ==================== TRAINING PIPELINE ====================

def train_model_ensemble(train_df, test_df, features, target='label'):
    """Train ensemble of XGBoost, CatBoost, and LightGBM models."""
    
    print("\n" + "="*70)
    print("TRAINING MULTI-ALGORITHM ENSEMBLE")
    print("="*70)
    
    # Get model parameters
    xgb_params, catboost_params, lgbm_params = get_model_params()
    
    # Define cross-validation
    FOLDS = 5
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    # Model configurations for time-based training
    time_configs = [
        {"name": "Full Data", "percent": 1.00},
        {"name": "Recent 80%", "percent": 0.80},
        {"name": "Recent 60%", "percent": 0.60},
        {"name": "Recent 40%", "percent": 0.40}
    ]
    
    # Initialize predictions storage
    model_types = ['xgb', 'catboost', 'lgbm']
    oof_predictions = {model: {config['name']: np.zeros(len(train_df)) 
                              for config in time_configs} 
                      for model in model_types}
    test_predictions = {model: {config['name']: np.zeros(len(test_df)) 
                               for config in time_configs} 
                       for model in model_types}
    
    # Train models
    for fold_num, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_num + 1}")
        print(f"{'='*50}")
        
        X_valid = train_df.iloc[valid_idx][features]
        y_valid = train_df.iloc[valid_idx][target]
        X_test = test_df[features]
        
        for config in time_configs:
            print(f"\n--- {config['name']} ---")
            
            # Calculate data subset
            if config["percent"] == 1.00:
                X_train = train_df.iloc[train_idx][features]
                y_train = train_df.iloc[train_idx][target]
                sample_weights = create_time_weights(len(X_train), decay_factor=0.95)
            else:
                cutoff_idx = int(len(train_df) * (1 - config["percent"]))
                train_idx_recent = train_idx[train_idx >= cutoff_idx]
                train_idx_adjusted = train_idx_recent - cutoff_idx
                train_recent = train_df.iloc[cutoff_idx:].reset_index(drop=True)
                
                X_train = train_recent.iloc[train_idx_adjusted][features]
                y_train = train_recent.iloc[train_idx_adjusted][target]
                sample_weights = create_time_weights(len(X_train), decay_factor=0.95)
            
            # Train XGBoost
            print("Training XGBoost...")
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=50,
                verbose=False
            )
            oof_predictions['xgb'][config['name']][valid_idx] = xgb_model.predict(X_valid)
            test_predictions['xgb'][config['name']] += xgb_model.predict(X_test) / FOLDS
            
            # Train CatBoost
            print("Training CatBoost...")
            catboost_model = CatBoostRegressor(**catboost_params)
            catboost_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                verbose=False
            )
            oof_predictions['catboost'][config['name']][valid_idx] = catboost_model.predict(X_valid)
            test_predictions['catboost'][config['name']] += catboost_model.predict(X_test) / FOLDS
            
            # Train LightGBM
            print("Training LightGBM...")
            lgbm_model = LGBMRegressor(**lgbm_params)
            lgbm_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, y_valid)],
                callbacks=[],
                eval_metric='rmse'
            )
            oof_predictions['lgbm'][config['name']][valid_idx] = lgbm_model.predict(X_valid)
            test_predictions['lgbm'][config['name']] += lgbm_model.predict(X_test) / FOLDS
    
    return oof_predictions, test_predictions

# ==================== MAIN EXECUTION ====================

print("\n" + "="*80)
print("ENHANCED CRYPTO PRICE PREDICTION WITH MULTI-ALGORITHM ENSEMBLE")
print("="*80)

# Load data
print("\nLoading data...")
train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
sample = pd.read_csv(CFG.sample_sub_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Feature selection based on EDA insights
selected_features = select_features_ensemble(train, test)

# Create market microstructure features
train_market = create_market_microstructure_features(train)
test_market = create_market_microstructure_features(test)

# Create PCA features (using 30 components based on EDA showing 23 for 90% variance)
train_pca, test_pca = create_pca_features(train, test, selected_features, n_components=30)

# Create cluster features (5 clusters based on EDA)
train_clusters, test_clusters = create_cluster_features(train, test, selected_features, n_clusters=5)

# Create interaction features for top features
top_features_for_interaction = selected_features[:15]  # Top 15 features
train_interactions = create_interaction_features(train, top_features_for_interaction)
test_interactions = create_interaction_features(test, top_features_for_interaction)

# Combine all features
print("\nCombining all features...")
train_enhanced = pd.concat([
    train[selected_features + ['label']], 
    train_market, 
    train_pca, 
    train_clusters,
    train_interactions
], axis=1)

test_enhanced = pd.concat([
    test[selected_features], 
    test_market, 
    test_pca, 
    test_clusters,
    test_interactions
], axis=1)

# Get final feature list (excluding label)
final_features = [col for col in train_enhanced.columns if col != 'label']
print(f"\nFinal feature count: {len(final_features)}")

# Reduce memory usage
train_enhanced = reduce_mem_usage(train_enhanced, "train_enhanced")
test_enhanced = reduce_mem_usage(test_enhanced, "test_enhanced")

# Train ensemble models
oof_predictions, test_predictions = train_model_ensemble(
    train_enhanced, test_enhanced, final_features
)

# Calculate individual model performances
print("\n" + "="*60)
print("MODEL PERFORMANCE EVALUATION")
print("="*60)

y_true = train_enhanced['label']
performance_scores = {}

for model_type in ['xgb', 'catboost', 'lgbm']:
    print(f"\n{model_type.upper()} Performance:")
    performance_scores[model_type] = {}
    
    for config_name in oof_predictions[model_type]:
        score = pearsonr(y_true, oof_predictions[model_type][config_name])[0]
        performance_scores[model_type][config_name] = score
        print(f"  {config_name}: {score:.4f}")

# Create final ensemble
print("\n" + "="*60)
print("CREATING FINAL ENSEMBLE")
print("="*60)

# Flatten all predictions for ensemble
all_oof_preds = []
all_test_preds = []
all_scores = []

for model_type in ['xgb', 'catboost', 'lgbm']:
    for config_name in oof_predictions[model_type]:
        all_oof_preds.append(oof_predictions[model_type][config_name])
        all_test_preds.append(test_predictions[model_type][config_name])
        all_scores.append(performance_scores[model_type][config_name])

# Calculate weights based on performance
all_scores = np.array(all_scores)
weights = all_scores / all_scores.sum()

# Create weighted ensemble
final_oof = np.zeros(len(train_enhanced))
final_test = np.zeros(len(test_enhanced))

for i, (oof_pred, test_pred, weight) in enumerate(zip(all_oof_preds, all_test_preds, weights)):
    final_oof += weight * oof_pred
    final_test += weight * test_pred

# Calculate final ensemble score
final_score = pearsonr(y_true, final_oof)[0]
print(f"\nFinal Weighted Ensemble Score: {final_score:.4f}")

# Also try simple average ensemble
simple_oof = np.mean(all_oof_preds, axis=0)
simple_test = np.mean(all_test_preds, axis=0)
simple_score = pearsonr(y_true, simple_oof)[0]
print(f"Simple Average Ensemble Score: {simple_score:.4f}")

# Use the better ensemble
if final_score > simple_score:
    final_predictions = final_test
    print("\nUsing weighted ensemble for final predictions")
else:
    final_predictions = simple_test
    print("\nUsing simple average ensemble for final predictions")

# Feature importance analysis using best single model
print("\nGenerating feature importance analysis...")
# Train a single XGBoost model on full data for SHAP
xgb_params, _, _ = get_model_params()
xgb_full = XGBRegressor(**xgb_params)
xgb_full.fit(train_enhanced[final_features], y_true, verbose=False)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': final_features,
    'importance': xgb_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20))

# Save predictions
sample["prediction"] = final_predictions
sample.to_csv("submission.csv", index=False)
print("\nPredictions saved to submission.csv")
print(sample.head())

# Save detailed results
results_summary = pd.DataFrame({
    'Model_Config': [f"{model}_{config}" for model in ['xgb', 'catboost', 'lgbm'] 
                     for config in performance_scores[model].keys()],
    'Pearson_Score': [performance_scores[model][config] 
                      for model in ['xgb', 'catboost', 'lgbm'] 
                      for config in performance_scores[model].keys()],
    'Weight_in_Ensemble': [w for w in weights]
})
results_summary = results_summary.sort_values('Pearson_Score', ascending=False)
results_summary.to_csv("ensemble_results_enhanced.csv", index=False)
print("\nDetailed results saved to ensemble_results_enhanced.csv")

# Save feature list
pd.DataFrame({'feature': final_features}).to_csv("final_features_enhanced.csv", index=False)
print(f"\nFinal feature list saved. Total features used: {len(final_features)}")

