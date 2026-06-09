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
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                  LassoLars, OrthogonalMatchingPursuit, HuberRegressor,
                                  RANSACRegressor, TheilSenRegressor)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, VotingRegressor,
                             GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Feature engineering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from scipy import stats
from scipy.stats import skew, kurtosis

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# ===== 1. LOAD DATA =====
print("="*80)
print("CALIFORNIA HOMELESSNESS PREDICTION - COMPREHENSIVE ANALYSIS")
print("="*80)

print("\n1. Loading data...")
train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ===== 2. EXPLORATORY DATA ANALYSIS =====
print("\n2. Exploratory Data Analysis")
print("-"*50)

# Separate features and target
train_ids = train['ID']
test_ids = test['ID']
y_train = train['HOMELESS_RATE']
X_train = train.drop(['ID', 'HOMELESS_RATE'], axis=1)
X_test = test.drop(['ID'], axis=1)

# 2.1 Target Variable Analysis
print("\n2.1 Target Variable Analysis:")
print(f"Mean: {y_train.mean():.6f}")
print(f"Median: {y_train.median():.6f}")
print(f"Std: {y_train.std():.6f}")
print(f"Min: {y_train.min():.6f}")
print(f"Max: {y_train.max():.6f}")
print(f"Skewness: {skew(y_train):.6f}")
print(f"Kurtosis: {kurtosis(y_train):.6f}")

# Create comprehensive EDA visualizations
fig = plt.figure(figsize=(20, 25))
gs = GridSpec(7, 3, figure=fig, hspace=0.3, wspace=0.3)

# 2.2 Target Distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(y_train, bins=30, alpha=0.7, color='darkblue', edgecolor='black')
ax1.axvline(y_train.mean(), color='red', linestyle='--', label=f'Mean: {y_train.mean():.4f}')
ax1.axvline(y_train.median(), color='green', linestyle='--', label=f'Median: {y_train.median():.4f}')
ax1.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Homeless Rate')
ax1.set_ylabel('Frequency')
ax1.legend()

# 2.3 Target Q-Q Plot
ax2 = fig.add_subplot(gs[0, 2])
stats.probplot(y_train, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot of Target Variable', fontsize=14, fontweight='bold')

# 2.4 Feature Distributions
print("\n2.2 Feature Statistics:")
feature_stats = X_train.describe().T
feature_stats['skewness'] = X_train.apply(lambda x: skew(x))
feature_stats['kurtosis'] = X_train.apply(lambda x: kurtosis(x))
print(feature_stats[['mean', 'std', 'skewness', 'kurtosis']].head(10))

# 2.5 Top Features by Correlation with Target
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
top_corr_features = correlations.head(15)

ax3 = fig.add_subplot(gs[1, :])
bars = ax3.bar(range(len(top_corr_features)), top_corr_features.values)
ax3.set_xticks(range(len(top_corr_features)))
ax3.set_xticklabels(top_corr_features.index, rotation=45, ha='right')
ax3.set_title('Top 15 Features by Correlation with Target', fontsize=14, fontweight='bold')
ax3.set_ylabel('Absolute Correlation')

# Color bars by correlation strength
for i, bar in enumerate(bars):
    if top_corr_features.values[i] > 0.3:
        bar.set_color('darkred')
    elif top_corr_features.values[i] > 0.2:
        bar.set_color('orange')
    else:
        bar.set_color('skyblue')

# 2.6 Feature Correlation Heatmap
ax4 = fig.add_subplot(gs[2:4, :])
corr_matrix = X_train.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax4)
ax4.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# 2.7 Distribution of Key Features
key_features = top_corr_features.head(6).index
for i, feature in enumerate(key_features):
    ax = fig.add_subplot(gs[4 + i//3, i%3])
    ax.scatter(X_train[feature], y_train, alpha=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel('Homeless Rate')
    ax.set_title(f'{feature} vs Target', fontsize=10)
    
    # Add trend line
    z = np.polyfit(X_train[feature], y_train, 1)
    p = np.poly1d(z)
    ax.plot(X_train[feature].sort_values(), p(X_train[feature].sort_values()), 
            "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 3. ADVANCED FEATURE ENGINEERING =====
print("\n3. Advanced Feature Engineering")
print("-"*50)

def create_comprehensive_features(df):
    """Create comprehensive features based on domain knowledge and EDA insights"""
    df_new = df.copy()
    
    # === Basic Ratios and Indices ===
    # Age vulnerability (youth + elderly)
    df_new['age_vulnerability'] = (df['AGE_U18_PCT'] + df['AGE_65_69_PCT'] + 
                                   df['AGE_70_79_PCT'] + df['AGE_80_PLUS_PCT'])
    
    # Working age population
    df_new['working_age_ratio'] = (df['AGE_25_34_PCT'] + df['AGE_35_44_PCT'] + 
                                   df['AGE_45_54_PCT'])
    
    # Youth to adult ratio
    df_new['youth_adult_ratio'] = df['AGE_18_24_PCT'] / (df['AGE_25_PLUS_PCT'] + 1e-5)
    
    # === Family and Household Features ===
    # Family stability index
    df_new['family_stability'] = df['FAMILY_HH_TOTAL'] - df['NONFAMILY_SINGLE_MALE_PCT'] - df['NONFAMILY_SINGLE_FEMALE_PCT']
    
    # Single household vulnerability
    df_new['single_vulnerability'] = df['NONFAMILY_SINGLE_MALE_PCT'] + df['NONFAMILY_SINGLE_FEMALE_PCT']
    
    # Gender imbalance in single households
    df_new['single_gender_imbalance'] = np.abs(df['NONFAMILY_SINGLE_MALE_PCT'] - df['NONFAMILY_SINGLE_FEMALE_PCT'])
    
    # === Race and Diversity Features ===
    # Minority percentage
    df_new['minority_pct'] = 100 - df['RACE_WHITE_NH_PCT']
    
    # Diversity indices
    race_cols = ['RACE_WHITE_NH_PCT', 'RACE_BLACK_NH_PCT', 'RACE_NATIVE_NH_PCT', 
                 'RACE_ASIAN_NH_PCT', 'RACE_PACIFIC_NH_PCT', 'RACE_HISPANIC_ANY_PCT']
    
    # Herfindahl diversity index
    df_new['diversity_herfindahl'] = 1 - (df[race_cols] / 100).pow(2).sum(axis=1)
    
    # Shannon entropy diversity
    race_props = df[race_cols] / 100
    df_new['diversity_shannon'] = -(race_props * np.log(race_props + 1e-10)).sum(axis=1)
    
    # === Vulnerability Interactions ===
    # Veteran disability interaction
    df_new['veteran_disability_interact'] = df['VETERAN_POP_PCT'] * df['DISABILITY_POP_PCT']
    
    # Age-disability interaction
    df_new['elderly_disability_interact'] = (df['AGE_65_69_PCT'] + df['AGE_70_79_PCT'] + 
                                            df['AGE_80_PLUS_PCT']) * df['DISABILITY_POP_PCT']
    
    # === Social Isolation Features ===
    df_new['social_isolation'] = df['INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT'] + df['MULTI_PERSON_NONFAMILY_HH_PCT']
    
    # === Economic Indicators (proxy) ===
    # Family with children vulnerability
    df_new['family_child_vulnerability'] = df['FAMILY_HH_CHILD_LT18_PCT'] / (df['FAMILY_HH_TOTAL'] + 1e-5)
    
    # Youth independence ratio
    df_new['youth_independence'] = df['AGE_18_24_PCT'] / (df['FAMILY_HH_TOTAL'] + 1e-5)
    
    # === Advanced Transformations ===
    # Polynomial features for top predictors
    df_new['family_hh_squared'] = df['FAMILY_HH_TOTAL'] ** 2
    df_new['family_hh_cubed'] = df['FAMILY_HH_TOTAL'] ** 3
    df_new['single_female_squared'] = df['NONFAMILY_SINGLE_FEMALE_PCT'] ** 2
    df_new['household_pct_squared'] = df['TOTAL_HOUSEHOLDS_PCT'] ** 2
    
    # Interaction features
    df_new['family_household_interact'] = df['FAMILY_HH_TOTAL'] * df['TOTAL_HOUSEHOLDS_PCT']
    df_new['age25_34_family_interact'] = df['AGE_25_34_PCT'] * df['FAMILY_HH_TOTAL']
    
    # Log transformations for skewed features
    log_features = ['TOTAL_HOUSEHOLDS_PCT', 'VETERAN_POP_PCT', 'DISABILITY_POP_PCT',
                   'NONFAMILY_SINGLE_MALE_PCT', 'NONFAMILY_SINGLE_FEMALE_PCT']
    for col in log_features:
        if col in df.columns:
            df_new[f'{col}_log'] = np.log1p(df[col])
    
    # Sqrt transformations
    sqrt_features = ['RACE_NATIVE_NH_PCT', 'RACE_PACIFIC_NH_PCT']
    for col in sqrt_features:
        if col in df.columns:
            df_new[f'{col}_sqrt'] = np.sqrt(df[col])
    
    # === Ratio Features ===
    # Disability to non-disability ratio
    df_new['disability_ratio'] = df['DISABILITY_POP_PCT'] / (df['NODISABILITY_POP_PCT'] + 1e-5)
    
    # Veteran to non-veteran ratio
    df_new['veteran_ratio'] = df['VETERAN_POP_PCT'] / (df['NONVETERAN_POP_PCT'] + 1e-5)
    
    # Single male to female ratio
    df_new['single_male_female_ratio'] = df['NONFAMILY_SINGLE_MALE_PCT'] / (df['NONFAMILY_SINGLE_FEMALE_PCT'] + 1e-5)
    
    return df_new

# Apply feature engineering
X_train_eng = create_comprehensive_features(X_train)
X_test_eng = create_comprehensive_features(X_test)

print(f"Features after engineering: {X_train_eng.shape[1]}")

# ===== 4. FEATURE SELECTION =====
print("\n4. Feature Selection")
print("-"*50)

# Remove highly correlated features
correlation_matrix = X_train_eng.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
X_train_eng = X_train_eng.drop(columns=to_drop)
X_test_eng = X_test_eng.drop(columns=to_drop)

print(f"Removed {len(to_drop)} highly correlated features")
print(f"Features after correlation removal: {X_train_eng.shape[1]}")

# Mutual information scores
mi_scores = mutual_info_regression(X_train_eng, y_train, random_state=42)
mi_scores_df = pd.DataFrame({
    'feature': X_train_eng.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 15 features by Mutual Information:")
print(mi_scores_df.head(15))

# ===== 5. MODEL TRAINING WITH COMPREHENSIVE EVALUATION =====
print("\n5. Model Training and Evaluation")
print("-"*50)

# Define comprehensive model suite
models = {
    # Linear models
    'BayesianRidge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6),
    'Ridge': Ridge(alpha=5.0),
    'Lasso': Lasso(alpha=0.01, max_iter=2000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=2000),
    'LassoLars': LassoLars(alpha=0.01),
    'HuberRegressor': HuberRegressor(epsilon=1.35, max_iter=200),
    
    # Robust regressors
    'RANSACRegressor': RANSACRegressor(random_state=42),
    'TheilSenRegressor': TheilSenRegressor(random_state=42, max_iter=300),
    
    # SVM variants
    'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.001),
    'SVR_Linear': SVR(kernel='linear', C=1.0, epsilon=0.001),
    'NuSVR': NuSVR(kernel='rbf', C=10, gamma='scale'),
    
    # Tree-based models (constrained for small data)
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=4, 
                                         min_samples_split=5, min_samples_leaf=3, 
                                         random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=4,
                                     min_samples_split=5, min_samples_leaf=3,
                                     random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                  learning_rate=0.05, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                             num_leaves=20, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05,
                                 l2_leaf_reg=10, random_state=42, verbose=False),
    
    # Other models
    'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance'),
    'MLP': MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu',
                       solver='lbfgs', alpha=0.1, random_state=42, max_iter=1000),
}

# Define evaluation metrics
def evaluate_model(y_true, y_pred):
    """Calculate multiple evaluation metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MedAE': median_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    return metrics

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results
cv_results = {}
trained_models = {}
predictions_dict = {}

print("\nTraining models with 5-fold cross-validation...")
print("-"*80)

for name, model in models.items():
    # Cross-validation predictions
    cv_predictions = np.zeros(len(y_train))
    cv_metrics = {'RMSE': [], 'MAE': [], 'R2': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_fold_train, y_fold_train)
        
        # Predict
        fold_pred = model_clone.predict(X_fold_val)
        cv_predictions[val_idx] = fold_pred
        
        # Calculate metrics
        fold_metrics = evaluate_model(y_fold_val, fold_pred)
        for metric in ['RMSE', 'MAE', 'R2']:
            cv_metrics[metric].append(fold_metrics[metric])
    
    # Store CV results
    cv_results[name] = {
        'RMSE_mean': np.mean(cv_metrics['RMSE']),
        'RMSE_std': np.std(cv_metrics['RMSE']),
        'MAE_mean': np.mean(cv_metrics['MAE']),
        'MAE_std': np.std(cv_metrics['MAE']),
        'R2_mean': np.mean(cv_metrics['R2']),
        'R2_std': np.std(cv_metrics['R2']),
        'cv_predictions': cv_predictions
    }
    
    # Train on full dataset
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Make test predictions
    test_pred = model.predict(X_test_scaled)
    predictions_dict[name] = test_pred
    
    print(f"{name:20} | RMSE: {cv_results[name]['RMSE_mean']:.6f} (+/- {cv_results[name]['RMSE_std']:.6f}) | "
          f"R2: {cv_results[name]['R2_mean']:.4f} (+/- {cv_results[name]['R2_std']:.4f})")

# ===== 6. RESIDUAL ANALYSIS =====
print("\n6. Residual Analysis")
print("-"*50)

# Select top 5 models for residual analysis
sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['RMSE_mean'])
top_models_for_analysis = sorted_models[:5]

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.ravel()

for idx, (model_name, results) in enumerate(top_models_for_analysis):
    cv_pred = results['cv_predictions']
    residuals = y_train - cv_pred
    
    # Residual plot
    ax = axes[idx]
    ax.scatter(cv_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{model_name} - Residual Plot')
    
    # Q-Q plot of residuals
    ax = axes[idx + 5]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f'{model_name} - Q-Q Plot')
    
    # Histogram of residuals
    ax = axes[idx + 10]
    ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{model_name} - Residual Distribution')
    
    # Print residual statistics
    print(f"\n{model_name} Residual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {np.std(residuals):.6f}")
    print(f"  Skewness: {skew(residuals):.4f}")
    print(f"  Kurtosis: {kurtosis(residuals):.4f}")

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 7. MODEL COMPARISON VISUALIZATION =====
print("\n7. Model Comparison")
print("-"*50)

# Create comprehensive comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'RMSE': [cv_results[m]['RMSE_mean'] for m in cv_results],
    'MAE': [cv_results[m]['MAE_mean'] for m in cv_results],
    'R2': [cv_results[m]['R2_mean'] for m in cv_results]
}).sort_values('RMSE')

print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# RMSE comparison
ax = axes[0, 0]
bars = ax.bar(range(len(comparison_df)), comparison_df['RMSE'])
ax.set_xticks(range(len(comparison_df)))
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.set_ylabel('RMSE')
ax.set_title('Model Comparison - RMSE (Lower is Better)')
ax.grid(axis='y', alpha=0.3)

# Color best models
for i in range(min(5, len(bars))):
    bars[i].set_color('darkgreen')

# R2 comparison
ax = axes[0, 1]
bars = ax.bar(range(len(comparison_df)), comparison_df['R2'])
ax.set_xticks(range(len(comparison_df)))
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.set_ylabel('R² Score')
ax.set_title('Model Comparison - R² (Higher is Better)')
ax.grid(axis='y', alpha=0.3)

# MAE comparison
ax = axes[1, 0]
ax.scatter(comparison_df['RMSE'], comparison_df['MAE'], s=100, alpha=0.6)
for i, model in enumerate(comparison_df['Model']):
    ax.annotate(model, (comparison_df['RMSE'].iloc[i], comparison_df['MAE'].iloc[i]),
                fontsize=8, ha='right')
ax.set_xlabel('RMSE')
ax.set_ylabel('MAE')
ax.set_title('RMSE vs MAE by Model')
ax.grid(True, alpha=0.3)

# Box plot of predictions
ax = axes[1, 1]
top_5_models = comparison_df.head(5)['Model'].tolist()
box_data = [predictions_dict[model] for model in top_5_models]
ax.boxplot(box_data, labels=top_5_models)
ax.set_ylabel('Predicted Homeless Rate')
ax.set_title('Prediction Distribution - Top 5 Models')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 8. ENSEMBLE CREATION =====
print("\n8. Creating Optimized Ensemble")
print("-"*50)

# Select top models for ensemble
n_ensemble = 7
top_models = comparison_df.head(n_ensemble)['Model'].tolist()

print(f"Selected {n_ensemble} models for ensemble:")
for i, model in enumerate(top_models, 1):
    print(f"  {i}. {model} (RMSE: {cv_results[model]['RMSE_mean']:.6f})")

# Create weighted ensemble
ensemble_predictions = []
ensemble_weights = []

for model_name in top_models:
    pred = predictions_dict[model_name]
    
    # Ensure non-negative predictions
    pred = np.maximum(pred, 0)
    ensemble_predictions.append(pred)
    
    # Weight inversely proportional to RMSE
    weight = 1 / cv_results[model_name]['RMSE_mean']
    ensemble_weights.append(weight)

# Normalize weights
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()

print(f"\nEnsemble weights:")
for model, weight in zip(top_models, ensemble_weights):
    print(f"  {model}: {weight:.4f}")

# Final predictions
final_predictions = np.average(ensemble_predictions, weights=ensemble_weights, axis=0)

# Post-processing
upper_bound = np.percentile(y_train, 99) * 1.2
final_predictions = np.clip(final_predictions, 0, upper_bound)

# ===== 9. FEATURE IMPORTANCE ANALYSIS =====
print("\n9. Feature Importance Analysis")
print("-"*50)

# Get feature importance from tree-based models
importance_dict = {}

for model_name in ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGBoost', 'LightGBM']:
    if model_name in trained_models:
        model = trained_models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_dict[model_name] = model.feature_importances_

# Average importance across models
if importance_dict:
    avg_importance = np.mean(list(importance_dict.values()), axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_train_eng.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features (averaged across tree models):")
    print(feature_importance_df.head(20).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Average Importance')
    plt.title('Top 20 Feature Importances (Tree-based Models)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===== 10. FINAL SUBMISSION =====
print("\n10. Creating Final Submission")
print("-"*50)

submission = pd.DataFrame({
    'ID': test_ids,
    'HOMELESS_RATE': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nSubmission Statistics:")
print(f"Mean: {final_predictions.mean():.6f}")
print(f"Std: {final_predictions.std():.6f}")
print(f"Min: {final_predictions.min():.6f}")
print(f"Max: {final_predictions.max():.6f}")

# ===== 11. COMPREHENSIVE REPORT =====
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS REPORT")
print("="*80)

print(f"\nDataset Information:")
print(f"  Training samples: {len(train)}")
print(f"  Test samples: {len(test)}")
print(f"  Original features: {len(X_train.columns)}")
print(f"  Engineered features: {len(X_train_eng.columns)}")
print(f"  Selected features: {X_train_scaled.shape[1]}")

print(f"\nTarget Variable Characteristics:")
print(f"  Range: [{y_train.min():.6f}, {y_train.max():.6f}]")
print(f"  Mean: {y_train.mean():.6f}")
print(f"  Median: {y_train.median():.6f}")
print(f"  Skewness: {skew(y_train):.4f}")

print(f"\nModeling Summary:")
print(f"  Total models evaluated: {len(models)}")
print(f"  Ensemble size: {n_ensemble} models")
print(f"  Best single model: {comparison_df.iloc[0]['Model']} (RMSE: {comparison_df.iloc[0]['RMSE']:.6f})")
print(f"  Ensemble performance estimate: ~{np.mean([cv_results[m]['RMSE_mean'] for m in top_models]):.6f} RMSE")

print(f"\nTop Predictive Features:")
if 'feature_importance_df' in locals():
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']} (importance: {row['importance']:.4f})")

print(f"\nSubmission saved to: submission.csv")
print("\nAnalysis complete!")

# Save all results for further analysis
results_summary = {
    'cv_results': cv_results,
    'comparison_df': comparison_df,
    'feature_importance': feature_importance_df if 'feature_importance_df' in locals() else None,
    'ensemble_weights': dict(zip(top_models, ensemble_weights)),
    'predictions': final_predictions
}

import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("\nAll results saved to: analysis_results.pkl")

