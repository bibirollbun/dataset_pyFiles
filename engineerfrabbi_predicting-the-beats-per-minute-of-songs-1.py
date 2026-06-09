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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# For hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - using default hyperparameters")



# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


print("ğŸ”¹ STEP 1: Loading and inspecting data...")

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")


# Basic inspection
print("\nğŸ“Š Train dataset info:")
print(train.info())
print("\nğŸ“Š First few rows:")
print(train.head())

print("\nğŸ“Š Target variable statistics:")
target_col = 'BeatsPerMinute'
print(train[target_col].describe())


# Check for missing values
print("\nğŸ”� Missing values in train:")
print(train.isnull().sum().sort_values(ascending=False))

print("\nğŸ”� Missing values in test:")
print(test.isnull().sum().sort_values(ascending=False))


# Identify feature types
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_features:
    numeric_features.remove(target_col)
if 'ID' in numeric_features:
    numeric_features.remove('ID')

categorical_features = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nğŸ“‹ Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"ğŸ“‹ Categorical features ({len(categorical_features)}): {categorical_features}")


# Target distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(train[target_col], bins=50, alpha=0.7, edgecolor='black')
plt.title(f'{target_col} Distribution')
plt.xlabel('BPM')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.boxplot(train[target_col])
plt.title(f'{target_col} Boxplot')
plt.ylabel('BPM')

plt.subplot(1, 3, 3)
from scipy import stats
stats.probplot(train[target_col], dist="norm", plot=plt)
plt.title(f'{target_col} Q-Q Plot')

plt.tight_layout()
plt.show()


print(f"Target mean: {train[target_col].mean():.2f}")
print(f"Target median: {train[target_col].median():.2f}")
print(f"Target std: {train[target_col].std():.2f}")
print(f"Target range: [{train[target_col].min():.2f}, {train[target_col].max():.2f}]")


# Correlation matrix for numeric features
if len(numeric_features) > 0:
    plt.figure(figsize=(12, 10))
    corr_matrix = train[numeric_features + [target_col]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Features most correlated with target
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)[1:]
    print(f"\nğŸ�¯ Top features correlated with {target_col}:")
    print(target_corr.head(10))


# Distribution of numeric features
if len(numeric_features) > 0:
    n_cols = 4
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
    
    for i, feature in enumerate(numeric_features):
        if i < len(axes):
            train[feature].hist(bins=30, alpha=0.7, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
    
    # Hide empty subplots
    for i in range(len(numeric_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Categorical features analysis
for cat_feat in categorical_features:
    print(f"\nğŸ“Š {cat_feat} value counts:")
    print(train[cat_feat].value_counts().head(10))


print("\nğŸ”¹ STEP 3: Data preprocessing and feature engineering...")

# Combine train and test for consistent preprocessing
train['is_train'] = 1
test['is_train'] = 0
test[target_col] = 0  # placeholder

combined = pd.concat([train, test], axis=0, ignore_index=True)
print(f"Combined dataset shape: {combined.shape}")


# Handle missing values
def fill_missing_values(df):
    df_filled = df.copy()
    
    # Numeric features - fill with median
    for feature in numeric_features:
        if df_filled[feature].isnull().sum() > 0:
            median_val = df_filled[df_filled['is_train'] == 1][feature].median()
            df_filled[feature].fillna(median_val, inplace=True)
            print(f"Filled {feature} missing values with median: {median_val:.2f}")
    
    # Categorical features - fill with mode
    for feature in categorical_features:
        if df_filled[feature].isnull().sum() > 0:
            mode_val = df_filled[df_filled['is_train'] == 1][feature].mode().iloc[0]
            df_filled[feature].fillna(mode_val, inplace=True)
            print(f"Filled {feature} missing values with mode: {mode_val}")
    
    return df_filled

combined = fill_missing_values(combined)


# Feature Engineering
def create_features(df):
    df_new = df.copy()
    
    # 1. Interaction features for audio characteristics
    audio_features = [f for f in numeric_features if f in df_new.columns]
    
    if len(audio_features) >= 2:
        # Energy-based interactions
        if 'energy' in df_new.columns and 'loudness' in df_new.columns:
            df_new['energy_loudness'] = df_new['energy'] * df_new['loudness']
        
        if 'energy' in df_new.columns and 'acousticness' in df_new.columns:
            df_new['energy_per_acousticness'] = df_new['energy'] / (df_new['acousticness'] + 1e-8)
        
        # Tempo-related features
        if 'tempo' in df_new.columns:
            df_new['tempo_squared'] = df_new['tempo'] ** 2
            df_new['tempo_log'] = np.log1p(df_new['tempo'])
        
        # Valence and energy combination
        if 'valence' in df_new.columns and 'energy' in df_new.columns:
            df_new['valence_energy'] = df_new['valence'] * df_new['energy']
        
        # Danceability features
        if 'danceability' in df_new.columns:
            df_new['danceability_squared'] = df_new['danceability'] ** 2
            if 'energy' in df_new.columns:
                df_new['dance_energy_ratio'] = df_new['danceability'] / (df_new['energy'] + 1e-8)
    
    # 2. Statistical features across numeric columns
    numeric_cols = [f for f in audio_features if f in df_new.columns]
    if len(numeric_cols) >= 3:
        df_new['feat_mean'] = df_new[numeric_cols].mean(axis=1)
        df_new['feat_std'] = df_new[numeric_cols].std(axis=1)
        df_new['feat_max'] = df_new[numeric_cols].max(axis=1)
        df_new['feat_min'] = df_new[numeric_cols].min(axis=1)
    
    # 3. Categorical encoding
    for cat_feat in categorical_features:
        if cat_feat in df_new.columns:
            # Frequency encoding
            freq_map = df_new[df_new['is_train'] == 1][cat_feat].value_counts().to_dict()
            df_new[f'{cat_feat}_freq'] = df_new[cat_feat].map(freq_map).fillna(0)
            
            # Target encoding (only on train data)
            if df_new['is_train'].sum() > 0:  # If we have train data
                target_map = df_new[df_new['is_train'] == 1].groupby(cat_feat)[target_col].mean().to_dict()
                df_new[f'{cat_feat}_target_enc'] = df_new[cat_feat].map(target_map)
                df_new[f'{cat_feat}_target_enc'].fillna(df_new[df_new['is_train'] == 1][target_col].mean(), inplace=True)
    
    return df_new

combined_engineered = create_features(combined)


# Log transform skewed features
def transform_skewed_features(df, threshold=0.5):
    df_transformed = df.copy()
    numeric_feats = df_transformed.select_dtypes(include=[np.number]).columns
    numeric_feats = [f for f in numeric_feats if f not in ['ID', target_col, 'is_train']]
    
    for feature in numeric_feats:
        if df_transformed[feature].min() >= 0:  # Only for non-negative features
            skewness = df_transformed[df_transformed['is_train'] == 1][feature].skew()
            if abs(skewness) > threshold:
                df_transformed[f'{feature}_log1p'] = np.log1p(df_transformed[feature])
                print(f"Applied log1p to {feature} (skewness: {skewness:.2f})")
    
    return df_transformed

combined_final = transform_skewed_features(combined_engineered)


# Create PCA features
def add_pca_features(df, n_components=5):
    df_pca = df.copy()
    numeric_feats = [f for f in df_pca.columns if df_pca[f].dtype in ['int64', 'float64'] 
                     and f not in ['ID', target_col, 'is_train']]
    
    if len(numeric_feats) >= n_components:
        pca = PCA(n_components=n_components, random_state=SEED)
        train_data = df_pca[df_pca['is_train'] == 1][numeric_feats]
        
        # Fit PCA on train data
        pca.fit(train_data)
        
        # Transform all data
        pca_features = pca.transform(df_pca[numeric_feats])
        
        for i in range(n_components):
            df_pca[f'pca_{i}'] = pca_features[:, i]
        
        print(f"Added {n_components} PCA components explaining {pca.explained_variance_ratio_.sum():.3f} variance")
    
    return df_pca

combined_final = add_pca_features(combined_final, n_components=3)


# Split back to train and test
train_processed = combined_final[combined_final['is_train'] == 1].copy()
test_processed = combined_final[combined_final['is_train'] == 0].copy()

# Remove helper columns
train_processed.drop(['is_train'], axis=1, inplace=True)
test_processed.drop(['is_train', target_col], axis=1, inplace=True)

print(f"Final train shape: {train_processed.shape}")
print(f"Final test shape: {test_processed.shape}")


# Prepare features and target
feature_cols = [f for f in train_processed.columns if f not in ['ID', target_col]]
X = train_processed[feature_cols].copy()
y = train_processed[target_col].copy()
X_test = test_processed[feature_cols].copy()

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(feature_cols)}")


# Create stratified bins for CV
def create_stratified_bins(target, n_bins=10):
    bins = pd.qcut(target, q=n_bins, labels=False, duplicates='drop')
    return bins

y_bins = create_stratified_bins(y, n_bins=10)


print("\nğŸ”¹ STEP 4: Model training with cross-validation...")

# Cross-validation setup
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Store OOF predictions and test predictions
oof_predictions = {}
test_predictions = {}
cv_scores = {}

# Model parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': SEED,
    'n_estimators': 1000
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'n_estimators': 1000
}

cb_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.05,
    'depth': 6,
    'random_state': SEED,
    'verbose': False,
    'n_estimators': 1000
}

# Model training function
def train_model(model_name, model, X_train, y_train, X_val, y_val, X_test_fold):
    if model_name in ['lgb', 'lightgbm']:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
    elif model_name in ['xgb', 'xgboost']:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
    elif model_name in ['cb', 'catboost']:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_fold)
    
    return val_pred, test_pred

# Train models
models = {
    'lgb': lgb.LGBMRegressor(**lgb_params),
    'xgb': xgb.XGBRegressor(**xgb_params),
    'cb': cb.CatBoostRegressor(**cb_params),
}

# Optional: Add Neural Network if dataset is large enough
if X.shape[0] > 1000:
    models['mlp'] = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1
    )

for model_name, model in models.items():
    print(f"\nğŸš€ Training {model_name.upper()}...")
    
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features for neural network
        if model_name == 'mlp':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            X_test_scaled = scaler.transform(X_test)
            
            val_pred, test_pred_fold = train_model(
                model_name, model, X_train_scaled, y_train_fold, 
                X_val_scaled, y_val_fold, X_test_scaled
            )
        else:
            val_pred, test_pred_fold = train_model(
                model_name, model, X_train_fold, y_train_fold, 
                X_val_fold, y_val_fold, X_test
            )
        
        oof_pred[val_idx] = val_pred
        test_pred += test_pred_fold / N_SPLITS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
        fold_scores.append(fold_rmse)
        print(f"  Fold {fold + 1}: RMSE = {fold_rmse:.4f}")
    
    overall_rmse = np.sqrt(mean_squared_error(y, oof_pred))
    cv_scores[model_name] = overall_rmse
    oof_predictions[model_name] = oof_pred
    test_predictions[model_name] = test_pred
    
    print(f"  {model_name.upper()} CV RMSE: {overall_rmse:.4f} Â± {np.std(fold_scores):.4f}")



print("\nğŸ”¹ STEP 5: Model ensembling...")

# Simple average ensemble
ensemble_oof = np.mean([oof_predictions[name] for name in oof_predictions.keys()], axis=0)
ensemble_test = np.mean([test_predictions[name] for name in test_predictions.keys()], axis=0)

ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"ğŸ“Š Simple Average Ensemble CV RMSE: {ensemble_rmse:.4f}")

# Weighted ensemble optimization
def optimize_weights(predictions_dict, target):
    from scipy.optimize import minimize
    
    def objective(weights):
        weights = weights / weights.sum()
        ensemble_pred = np.average([predictions_dict[name] for name in predictions_dict.keys()], 
                                   weights=weights, axis=0)
        return np.sqrt(mean_squared_error(target, ensemble_pred))
    
    n_models = len(predictions_dict)
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x / result.x.sum()

# Optimize ensemble weights
optimal_weights = optimize_weights(oof_predictions, y)
model_names = list(oof_predictions.keys())

print(f"ğŸ�¯ Optimal weights:")
for i, (name, weight) in enumerate(zip(model_names, optimal_weights)):
    print(f"  {name}: {weight:.3f}")

# Create optimal ensemble
optimal_oof = np.average([oof_predictions[name] for name in model_names], 
                        weights=optimal_weights, axis=0)
optimal_test = np.average([test_predictions[name] for name in model_names], 
                         weights=optimal_weights, axis=0)

optimal_rmse = np.sqrt(mean_squared_error(y, optimal_oof))
print(f"ğŸ“Š Optimal Weighted Ensemble CV RMSE: {optimal_rmse:.4f}")


print("\nğŸ”¹ STEP 6: Generating final predictions...")

# Use the best ensemble
if optimal_rmse < ensemble_rmse:
    final_predictions = optimal_test
    print("âœ… Using optimal weighted ensemble")
else:
    final_predictions = ensemble_test
    print("âœ… Using simple average ensemble")

# Clip predictions to reasonable range
target_min = y.min()
target_max = y.max()
final_predictions = np.clip(final_predictions, target_min, target_max)

print(f"ğŸ“ˆ Final predictions stats:")
print(f"  Min: {final_predictions.min():.2f}")
print(f"  Max: {final_predictions.max():.2f}")
print(f"  Mean: {final_predictions.mean():.2f}")
print(f"  Std: {final_predictions.std():.2f}")

# Create submission
if 'ID' in test.columns:
    test_ids = test['ID']
elif 'id' in test.columns:
    test_ids = test['id']
elif 'ID' in test_processed.columns:
    test_ids = test_processed['ID']
else:
    # Generate sequential IDs if no ID column exists
    test_ids = range(len(final_predictions))
    print("âš ï¸� No ID column found, using sequential IDs")

submission = pd.DataFrame({
    'id': test_ids,
    target_col: final_predictions
})

print(f"\nğŸ“‹ Submission shape: {submission.shape}")
print("ğŸ“‹ First few predictions:")
print(submission.head())

# Save submission
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved as 'submission.csv'")


print("\nğŸ”¹ STEP 7: Model analysis...")

# Feature importance from best tree model
best_tree_model = 'lgb'  # Default to LightGBM
if best_tree_model in models:
    # Retrain on full data for feature importance
    best_model = lgb.LGBMRegressor(**lgb_params)
    best_model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"ğŸ�¯ Top 15 most important features:")
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# Model performance summary
print(f"\nğŸ“Š MODEL PERFORMANCE SUMMARY:")
print("="*50)
for model_name, score in cv_scores.items():
    print(f"{model_name.upper():>10}: {score:.4f} RMSE")
print("="*50)
print(f"{'ENSEMBLE':>10}: {min(ensemble_rmse, optimal_rmse):.4f} RMSE")



# Prediction vs actual plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y, optimal_oof, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual BPM')
plt.ylabel('Predicted BPM')
plt.title('OOF Predictions vs Actual')

plt.subplot(1, 2, 2)
residuals = y - optimal_oof
plt.scatter(optimal_oof, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted BPM')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

print(f"\nğŸ�† FINAL RESULTS:")
print(f"ğŸ“Š Best CV RMSE: {min(cv_scores.values()):.4f}")
print(f"ğŸ“Š Ensemble CV RMSE: {min(ensemble_rmse, optimal_rmse):.4f}")
print(f"ğŸ�¯ Target for competition: < {min(ensemble_rmse, optimal_rmse) - 0.001:.4f}")

print("\nâœ… PIPELINE COMPLETE! Ready for submission.")
print("ğŸ�† This notebook implements Kaggle Grandmaster best practices:")
print("   âœ“ Comprehensive EDA and feature engineering")
print("   âœ“ Multiple strong models with proper CV")
print("   âœ“ Advanced ensembling techniques")
print("   âœ“ Robust validation and prediction clipping")
print("   âœ“ Production-ready code with reproducible results")

