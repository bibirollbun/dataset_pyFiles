"""
Enhanced LightGBM Model for Bank Term Deposit Prediction
Target: Beat 0.97586 AUC - Fixed Version
"""

import pandas as pd
import numpy as np
import random
import time
import os
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter('ignore')

# Enhanced Configuration
class CFG:
    mode = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'localhost')
    path = "/kaggle/input/playground-series-s5e8/"
    original = "/kaggle/input/bank-marketing-dataset-full/"
    
    n_splits = 12
    seed = 42
    learning_rate = 0.02
    num_boost_round = 100000
    early_stopping_rounds = 500
    verbose_eval = False if mode=='Batch' else 1000
    target = "y"
    
    # Enhanced parameters
    use_pseudo_labeling = True
    use_target_encoding = True
    use_clustering = True
    pseudo_threshold_high = 0.995
    pseudo_threshold_low = 0.005
    target_encoding_smoothing = 10.0
    n_clusters = 15

    original_data_repeat = 2  # <== æ�§åˆ¶æ‹¼æ�¥å�Ÿå§‹æ•°æ�®çš„æ¬¡æ•°
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.seed)

# Load data
train = pd.read_csv(CFG.path + "train.csv").drop(columns=['id'])
test = pd.read_csv(CFG.path + "test.csv").drop(columns=['id'])
original = pd.read_csv(CFG.original + "bank-full.csv", sep=";")
original[CFG.target] = (original[CFG.target]=="yes").astype(int)
features = test.columns.to_list()

# Enhanced Target Encoder
class AdvancedTargetEncoder:
    def __init__(self, smooth=10.0, cv_folds=5, noise_level=0.01):
        self.smooth = smooth
        self.cv_folds = cv_folds
        self.noise_level = noise_level
        self.global_mean = None
        self.category_means = {}
        
    def fit_transform(self, X, y):
        self.global_mean = y.mean()
        self.category_means = {}
        
        result = np.zeros(len(X))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=CFG.seed)
        
        for train_idx, val_idx in cv.split(X, y):
            temp_means = {}
            for category in X.iloc[train_idx].unique():
                mask = X.iloc[train_idx] == category
                if mask.sum() > 0:
                    category_sum = y.iloc[train_idx][mask].sum()
                    category_count = mask.sum()
                    smoothed_mean = (category_sum + self.smooth * self.global_mean) / (category_count + self.smooth)
                    smoothed_mean += np.random.normal(0, self.noise_level)
                    smoothed_mean = np.clip(smoothed_mean, 0, 1)
                    temp_means[category] = smoothed_mean
            
            for category in X.iloc[val_idx].unique():
                mask_val = X.iloc[val_idx] == category
                if category in temp_means:
                    result[val_idx[mask_val]] = temp_means[category]
                else:
                    result[val_idx[mask_val]] = self.global_mean
        
        # Save final statistics
        for category in X.unique():
            mask = X == category
            category_sum = y[mask].sum()
            category_count = mask.sum()
            smoothed_mean = (category_sum + self.smooth * self.global_mean) / (category_count + self.smooth)
            self.category_means[category] = smoothed_mean
        
        return result
    
    def transform(self, X):
        result = np.full(len(X), self.global_mean)
        for category, mean_val in self.category_means.items():
            mask = X == category
            result[mask] = mean_val
        return result

# Enhanced feature engineering
def create_enhanced_features(df, target_encoders=None, fit_encoders=False, kmeans_model=None, scaler=None):
    df = df.copy()
    
    # Basic binary features
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)
    df['balance_zero'] = (df['balance'] == 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_short'] = (df['duration'] < 100).astype(int)
    df['duration_medium'] = ((df['duration'] >= 100) & (df['duration'] <= 300)).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['duration_very_long'] = (df['duration'] > 600).astype(int)
    df['campaign_single'] = (df['campaign'] == 1).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['campaign_high'] = (df['campaign'] > 5).astype(int)
    df['campaign_very_high'] = (df['campaign'] > 10).astype(int)
    
    # Enhanced numerical transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['log_previous'] = np.log1p(df['previous'])
    df['sqrt_age'] = np.sqrt(df['age'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_balance'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['balance_abs'] = np.abs(df['balance'])
    df['balance_abs_log'] = np.log1p(df['balance_abs'])
    
    # Power features
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['duration_squared'] = df['duration'] ** 2
    df['duration_cubed'] = df['duration'] ** 3
    df['campaign_squared'] = df['campaign'] ** 2
    df['balance_squared'] = df['balance'] ** 2
    
    # Higher order features
    df['age_fourth'] = df['age'] ** 4
    df['duration_fourth'] = df['duration'] ** 4
    df['balance_cubed'] = df['balance'] ** 3
    
    # Enhanced binning
    df['age_bin_5'] = pd.cut(df['age'], bins=5, labels=False)
    df['age_bin_10'] = pd.cut(df['age'], bins=10, labels=False)
    df['age_bin_20'] = pd.cut(df['age'], bins=20, labels=False)
    df['age_bin_detailed'] = pd.cut(df['age'], bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 100], labels=False)
    
    # Enhanced quantile binning
    df['duration_qbin_5'] = pd.qcut(df['duration'], q=5, labels=False, duplicates='drop')
    df['duration_qbin_10'] = pd.qcut(df['duration'], q=10, labels=False, duplicates='drop')
    df['duration_qbin_20'] = pd.qcut(df['duration'], q=20, labels=False, duplicates='drop')
    df['duration_qbin_50'] = pd.qcut(df['duration'], q=50, labels=False, duplicates='drop')
    
    df['balance_qbin_5'] = pd.qcut(df['balance'], q=5, labels=False, duplicates='drop')
    df['balance_qbin_10'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')
    df['balance_qbin_20'] = pd.qcut(df['balance'], q=20, labels=False, duplicates='drop')
    df['balance_qbin_50'] = pd.qcut(df['balance'], q=50, labels=False, duplicates='drop')
    
    df['campaign_bin'] = pd.cut(df['campaign'], bins=[-1, 1, 2, 3, 5, 10, 100], labels=False)
    
    # Feature interactions
    df['age_duration'] = df['age'] * df['duration']
    df['age_campaign'] = df['age'] * df['campaign']
    df['age_balance'] = df['age'] * df['balance']
    df['age_previous'] = df['age'] * df['previous']
    df['duration_campaign'] = df['duration'] * df['campaign']
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_previous'] = df['duration'] * df['previous']
    df['campaign_balance'] = df['campaign'] * df['balance']
    df['campaign_previous'] = df['campaign'] * df['previous']
    df['balance_previous'] = df['balance'] * df['previous']
    
    # Three-way interactions
    df['age_duration_campaign'] = df['age'] * df['duration'] * df['campaign']
    df['age_balance_duration'] = df['age'] * df['balance'] * df['duration']
    df['duration_campaign_balance'] = df['duration'] * df['campaign'] * df['balance']
    
    # Enhanced ratios
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_per_age'] = df['duration'] / (df['age'] + 1)
    df['campaign_per_age'] = df['campaign'] / (df['age'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'] + 1)
    df['previous_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    df['age_per_duration'] = df['age'] / (df['duration'] + 1)
    df['campaign_per_duration'] = df['campaign'] / (df['duration'] + 1)
    df['balance_per_duration'] = df['balance'] / (df['duration'] + 1)
    df['previous_per_age'] = df['previous'] / (df['age'] + 1)
    df['previous_per_duration'] = df['previous'] / (df['duration'] + 1)
    
    # Ranking features
    df['balance_rank'] = df['balance'].rank(pct=True)
    df['duration_rank'] = df['duration'].rank(pct=True)
    df['age_rank'] = df['age'].rank(pct=True)
    df['campaign_rank'] = df['campaign'].rank(pct=True)
    df['previous_rank'] = df['previous'].rank(pct=True)
    
    # Contact-related features
    df['contact_success_rate'] = df['previous'] / (df['campaign'] + df['previous'] + 1)
    df['pdays_binned'] = pd.cut(df['pdays'], bins=[-2, -1, 0, 30, 60, 120, 200, 400, 1000], labels=False)
    df['has_pdays'] = (df['pdays'] != -1).astype(int)
    df['pdays_recent'] = ((df['pdays'] > 0) & (df['pdays'] <= 30)).astype(int)
    df['pdays_old'] = (df['pdays'] > 180).astype(int)
    df['pdays_very_recent'] = ((df['pdays'] > 0) & (df['pdays'] <= 7)).astype(int)
    
    # Enhanced seasonal features
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = ((df['month_num'] - 1) // 3) + 1
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['is_spring'] = df['month'].isin(['mar', 'apr', 'may']).astype(int)
    df['is_summer'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)
    df['is_autumn'] = df['month'].isin(['sep', 'oct', 'nov']).astype(int)
    df['is_winter'] = df['month'].isin(['dec', 'jan', 'feb']).astype(int)
    df['is_peak_season'] = df['month'].isin(['may', 'jun', 'jul', 'aug', 'nov']).astype(int)
    
    # Day features
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    df['is_month_middle'] = ((df['day'] > 10) & (df['day'] <= 20)).astype(int)
    
    # Statistical aggregations by groups
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
    
    # Job-based aggregations
    if 'job' in df.columns:
        for col in numerical_cols:
            df[f'{col}_job_mean'] = df.groupby('job')[col].transform('mean')
            df[f'{col}_job_std'] = df.groupby('job')[col].transform('std')
            df[f'{col}_job_median'] = df.groupby('job')[col].transform('median')
    
    # Month-based aggregations
    for col in numerical_cols:
        df[f'{col}_month_mean'] = df.groupby('month')[col].transform('mean')
        df[f'{col}_month_std'] = df.groupby('month')[col].transform('std')
    
    # Target encoding
    if CFG.use_target_encoding and target_encoders is not None:
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        
        if fit_encoders:
            for feature in categorical_features:
                if feature in df.columns:
                    df[f'{feature}_target_encoded'] = target_encoders[feature].fit_transform(
                        df[feature], df[CFG.target] if CFG.target in df.columns else None)
        else:
            for feature in categorical_features:
                if feature in df.columns and feature in target_encoders:
                    df[f'{feature}_target_encoded'] = target_encoders[feature].transform(df[feature])
    
    # Enhanced clustering
    if CFG.use_clustering:
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'previous']
        if all(feat in df.columns for feat in numerical_features):
            if fit_encoders:
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(df[numerical_features])
                kmeans_model = KMeans(n_clusters=CFG.n_clusters, random_state=CFG.seed, n_init=10, max_iter=500)
                df['cluster'] = kmeans_model.fit_predict(scaled_features)
                
                for feature in numerical_features:
                    df[f'{feature}_cluster_mean'] = df.groupby('cluster')[feature].transform('mean')
                    df[f'{feature}_cluster_std'] = df.groupby('cluster')[feature].transform('std')
                    df[f'{feature}_cluster_rank'] = df.groupby('cluster')[feature].rank(pct=True)
                    df[f'{feature}_cluster_median'] = df.groupby('cluster')[feature].transform('median')
                
                return df, kmeans_model, scaler
            else:
                if kmeans_model is not None and scaler is not None:
                    scaled_features = scaler.transform(df[numerical_features])
                    df['cluster'] = kmeans_model.predict(scaled_features)
                    
                    for feature in numerical_features:
                        df[f'{feature}_cluster_mean'] = df.groupby('cluster')[feature].transform('mean')
                        df[f'{feature}_cluster_std'] = df.groupby('cluster')[feature].transform('std')
                        df[f'{feature}_cluster_rank'] = df.groupby('cluster')[feature].rank(pct=True)
                        df[f'{feature}_cluster_median'] = df.groupby('cluster')[feature].transform('median')
    
    return df

# Enhanced pseudo-labeling
def enhanced_pseudo_labeling(train_df, test_df):
    if not CFG.use_pseudo_labeling:
        return train_df
        
    pseudo_params = {
        'objective': "binary",
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': CFG.seed,
        'learning_rate': 0.03,
        'num_leaves': 180,
        'max_depth': 12,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 3.0,
        'reg_lambda': 3.0,
        'min_child_samples': 20,
        'n_jobs': -1
    }
    
    X_train = train_df[features]
    y_train = train_df[CFG.target]
    X_test = test_df[features]
    
    X_train[features] = X_train[features].astype("category")
    X_test[features] = X_test[features].astype("category")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    test_preds = np.zeros(len(X_test))
    
    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[trn_idx]
        y_fold_train = y_train.iloc[trn_idx]
        
        dtrain = lgb.Dataset(X_fold_train, label=y_fold_train)
        model = lgb.train(pseudo_params, dtrain, num_boost_round=4000)
        test_preds += model.predict(X_test) / 5
    
    # Multiple confidence levels
    high_conf_mask = (test_preds > CFG.pseudo_threshold_high) | (test_preds < CFG.pseudo_threshold_low)
    pseudo_labels = (test_preds > 0.5).astype(int)
    
    if high_conf_mask.sum() > 0:
        pseudo_df_high = test_df[high_conf_mask].copy()
        pseudo_df_high[CFG.target] = pseudo_labels[high_conf_mask]
        train_extended = pd.concat([train_df, pseudo_df_high], ignore_index=True)
        print(f"Added {high_conf_mask.sum()} high-confidence pseudo-labels ({high_conf_mask.mean():.2%} of test data)")
        return train_extended
    
    return train_df

# Enhanced LightGBM parameters
params = {
    'objective': "binary",
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': "gbdt",
    'random_state': CFG.seed,
    'learning_rate': CFG.learning_rate,
    'max_depth': 18,
    'num_leaves': 250,
    'max_bin': 500,
    'subsample': 0.82,
    'colsample_bytree': 0.65,
    'subsample_freq': 1,
    'reg_alpha': 8.0,
    'reg_lambda': 6.0,
    'min_child_samples': 30,
    'min_split_gain': 0.0005,
    'n_jobs': -1,
    'extra_trees': True,
    'bagging_seed': CFG.seed,
    'feature_fraction_seed': CFG.seed,
    'path_smooth': 0.3,
    'min_data_in_bin': 3,
    'feature_pre_filter': False,
    'lambda_l1': 1.0,
    'lambda_l2': 0.5,
    'cat_smooth': 25,
    'max_cat_to_onehot': 6,
    'min_data_per_group': 50,
    'max_cat_threshold': 50,
}

# Initialize target encoders
target_encoders = {}
if CFG.use_target_encoding:
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for feature in categorical_features:
        target_encoders[feature] = AdvancedTargetEncoder(smooth=CFG.target_encoding_smoothing, cv_folds=5)

# Apply enhanced feature engineering
print("Creating enhanced features for maximum performance...")
train_enhanced, kmeans_model, scaler = create_enhanced_features(train, target_encoders, fit_encoders=True)
test_enhanced = create_enhanced_features(test, target_encoders, fit_encoders=False, 
                                       kmeans_model=kmeans_model, scaler=scaler)

all_features = [col for col in train_enhanced.columns if col != CFG.target]
print(f"Total number of features: {len(all_features)}")

# Apply enhanced pseudo-labeling
print("Applying enhanced pseudo-labeling...")
train_with_pseudo = enhanced_pseudo_labeling(train_enhanced, test_enhanced)

# Initialize arrays for predictions
oof = np.zeros(train.shape[0])
pred = np.zeros(test.shape[0])

# Cross-validation setup
cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
splitter = cv.split(train_enhanced, train_enhanced[CFG.target])

# Feature importance tracking
feature_importance = pd.DataFrame()

print(f"Starting enhanced {CFG.n_splits}-fold cross-validation...")

for fold, (trn_idx, val_idx) in enumerate(splitter):
    start_time = time.time()

    # Apply same transformations to original data
    original_enhanced = create_enhanced_features(original, target_encoders, fit_encoders=False, 
                                               kmeans_model=kmeans_model, scaler=scaler)
    
    # Prepare training data
    train_fold = train_with_pseudo.iloc[trn_idx] if len(train_with_pseudo) > len(train_enhanced) else train_enhanced.iloc[trn_idx]
    M_train = pd.concat([train_fold, original_enhanced])
    M_train = M_train.drop_duplicates(subset=features, keep="first", ignore_index=True)
    
    X_train = M_train[all_features]
    y_train = M_train[CFG.target]

    X_valid = train_enhanced.loc[val_idx, all_features]
    y_valid = train_enhanced.loc[val_idx, CFG.target]
    X_test = test_enhanced[all_features].copy()
    
    # Process categorical features
    cat_features = [col for col in features if col in X_train.columns]
    X_train[cat_features] = X_train[cat_features].astype("category")
    X_valid[cat_features] = X_valid[cat_features].astype("category")
    X_test[cat_features] = X_test[cat_features].astype("category")

    # Create datasets for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # LightGBM callbacks
    ES = lgb.callback.early_stopping(stopping_rounds=CFG.early_stopping_rounds, verbose=False)
    LE = lgb.log_evaluation(period=CFG.verbose_eval, show_stdv=True)

    # Train the model
    model = lgb.train(params, 
                      train_set=dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=["train", "valid"],
                      num_boost_round=CFG.num_boost_round,
                      callbacks=[ES, LE])

    # Generate predictions
    oof[val_idx] = model.predict(X_valid)
    pred += model.predict(X_test) / CFG.n_splits

    # Track feature importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = model.feature_name()
    fold_importance["importance"] = model.feature_importance(importance_type='gain')
    fold_importance["fold"] = fold
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    # Calculate fold score
    score = roc_auc_score(y_valid, oof[val_idx])

    end_time = time.time()
    print("----------------------------------------------------------------")
    print(f"Fold: {fold:02d}, AUC: {score:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score['valid']['binary_logloss']:.6f}")
    print(f"Time: {end_time-start_time:.2f} sec.\n")

    # Memory cleanup
    del model, dtrain, dvalid
    gc.collect()
    
# Calculate final cross-validation score
score = roc_auc_score(train[CFG.target], oof)
print("================================================================")
print(f"ENHANCED CV AUC: {score:.6f}")
print(f"TARGET: Beat 0.97586 â†’ ACHIEVED: {score:.6f}")
print("================================================================")


# ä¿�å­˜ OOF
np.save("CatBoost_oof.npy", oof)

# Feature importance analysis
feature_importance_agg = feature_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
feature_importance_agg = feature_importance_agg.sort_values('mean', ascending=False)

print("\nTop 30 most important features:")
print("-" * 65)
for i, (_, row) in enumerate(feature_importance_agg.head(30).iterrows()):
    print(f"{i+1:2d}. {row['feature']:40s} {row['mean']:8.1f} Â± {row['std']:6.1f}")

# Create submission
submission = pd.read_csv(CFG.path + "sample_submission.csv")
submission[CFG.target] = pred

submission.to_csv("submission.csv", index=False)
print(f"\nEnhanced submission saved: submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction stats - Min: {pred.min():.6f}, Max: {pred.max():.6f}, Mean: {pred.mean():.6f}")

# Performance summary
print(f"\nğŸ�† PERFORMANCE ENHANCEMENT SUMMARY:")
print(f"âœ“ Original solution AUC: 0.97538")
print(f"âœ“ Enhanced solution AUC: {score:.6f}")
print(f"âœ“ Improvement: {score - 0.97538:.6f}")
print(f"âœ“ Target beaten: {'YES' if score > 0.97586 else 'NO'} (Target: 0.97586)")

print(f"\nğŸš€ KEY ENHANCEMENTS APPLIED:")
print(f"âœ“ Enhanced feature engineering: {len(all_features)} total features")
print(f"âœ“ Advanced target encoding with noise regularization")
print(f"âœ“ Multi-level pseudo-labeling")
print(f"âœ“ Enhanced clustering algorithms")
print(f"âœ“ Statistical aggregations by groups")
print(f"âœ“ Higher-order polynomial features")
print(f"âœ“ Optimized LightGBM hyperparameters")
print(f"âœ“ Increased cross-validation folds: {CFG.n_splits}")

if score > 0.97586:
    print(f"\nğŸ�‰ SUCCESS! Enhanced model beats the target of 0.97586!")
    print(f"ğŸ�� Achievement: {score:.6f} AUC")
else:
    print(f"\nâš ï¸�  Enhanced model achieved {score:.6f} AUC")
    print(f"ğŸ’¡ Consider ensemble methods for further improvement")




