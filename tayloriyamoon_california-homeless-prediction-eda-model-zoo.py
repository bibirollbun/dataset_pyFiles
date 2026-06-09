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
California Homelessness Prediction Challenge - Complete Solution
Fully working implementation with zero errors
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Core imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Linear Models
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, BayesianRidge, HuberRegressor,
    Lars, LassoLars
)

# Tree-based Models
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor

# Other Models
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neural_network import MLPRegressor

# Gradient Boosting Libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print(" CALIFORNIA HOMELESSNESS PREDICTION - COMPLETE SOLUTION")
print("=" * 80)

# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
print("\n[STEP 1] Loading data...")

train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/sample_submission.csv')

print(f"âœ“ Train shape: {train.shape}")
print(f"âœ“ Test shape: {test.shape}")

# Separate features and target
train_ids = train['ID']
test_ids = test['ID']
y_train = train['HOMELESS_RATE']
X_train = train.drop(['ID', 'HOMELESS_RATE'], axis=1)
X_test = test.drop(['ID'], axis=1)

print(f"\nTarget statistics:")
print(f"  Mean: {y_train.mean():.6f}")
print(f"  Std: {y_train.std():.6f}")
print(f"  Min: {y_train.min():.6f}")
print(f"  Max: {y_train.max():.6f}")
print(f"  Zeros: {(y_train == 0).sum()}")

# ================================================================================
# SECTION 2: FEATURE ENGINEERING
# ================================================================================
print("\n[STEP 2] Feature Engineering...")

def create_features(df):
    """Create comprehensive feature set with error handling"""
    df_new = df.copy()
    
    # Age vulnerability indices
    df_new['age_vulnerability'] = (
        df['AGE_U18_PCT'] + 
        df['AGE_65_69_PCT'] + 
        df['AGE_70_79_PCT'] + 
        df['AGE_80_PLUS_PCT']
    )
    
    df_new['working_age_ratio'] = (
        df['AGE_25_34_PCT'] + 
        df['AGE_35_44_PCT'] + 
        df['AGE_45_54_PCT']
    )
    
    # Safe division with small epsilon to avoid division by zero
    df_new['youth_ratio'] = df['AGE_18_24_PCT'] / (df['AGE_25_PLUS_PCT'] + 1e-5)
    
    # Household metrics
    df_new['family_stability'] = (
        df['FAMILY_HH_TOTAL'] - 
        df['NONFAMILY_SINGLE_MALE_PCT'] - 
        df['NONFAMILY_SINGLE_FEMALE_PCT']
    )
    
    df_new['single_vulnerability'] = (
        df['NONFAMILY_SINGLE_MALE_PCT'] + 
        df['NONFAMILY_SINGLE_FEMALE_PCT']
    )
    
    df_new['social_isolation'] = (
        df['INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT'] + 
        df['MULTI_PERSON_NONFAMILY_HH_PCT']
    )
    
    # Diversity metrics
    df_new['minority_pct'] = 100 - df['RACE_WHITE_NH_PCT']
    
    race_cols = [
        'RACE_WHITE_NH_PCT', 'RACE_BLACK_NH_PCT', 'RACE_NATIVE_NH_PCT',
        'RACE_ASIAN_NH_PCT', 'RACE_PACIFIC_NH_PCT', 'RACE_HISPANIC_ANY_PCT'
    ]
    df_new['diversity_index'] = 1 - (df[race_cols] / 100).pow(2).sum(axis=1)
    
    # Special populations
    df_new['veteran_disability'] = df['VETERAN_POP_PCT'] * df['DISABILITY_POP_PCT'] / 100
    
    df_new['family_child_ratio'] = (
        df['FAMILY_HH_CHILD_LT18_PCT'] / (df['FAMILY_HH_TOTAL'] + 1e-5)
    )
    
    # Polynomial features for key predictors
    key_features = ['FAMILY_HH_TOTAL', 'NONFAMILY_SINGLE_FEMALE_PCT', 'RACE_NATIVE_NH_PCT']
    for feat in key_features:
        if feat in df.columns:
            df_new[f'{feat}_squared'] = df[feat] ** 2
            df_new[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
    
    # Log transformations
    log_features = ['TOTAL_HOUSEHOLDS_PCT', 'VETERAN_POP_PCT', 'DISABILITY_POP_PCT']
    for col in log_features:
        if col in df.columns:
            df_new[f'{col}_log'] = np.log1p(np.abs(df[col]))
    
    # Interaction features
    df_new['age_disability_interact'] = df['AGE_65_69_PCT'] * df['DISABILITY_POP_PCT'] / 100
    df_new['single_youth_interact'] = df['NONFAMILY_SINGLE_MALE_PCT'] * df['AGE_18_24_PCT'] / 100
    
    # Ratio features - FIXED: Use df_new['working_age_ratio'] instead of df['working_age_ratio']
    df_new['dependency_ratio'] = (
        (df['AGE_U18_PCT'] + df['AGE_65_69_PCT'] + df['AGE_70_79_PCT'] + df['AGE_80_PLUS_PCT']) /
        (df_new['working_age_ratio'] + 1e-5)  # Fixed to use df_new and add epsilon
    )
    
    # Additional domain-specific features
    df_new['elderly_ratio'] = (
        (df['AGE_65_69_PCT'] + df['AGE_70_79_PCT'] + df['AGE_80_PLUS_PCT']) / 
        (df['AGE_25_PLUS_PCT'] + 1e-5)
    )
    
    df_new['youth_elderly_ratio'] = (
        df['AGE_18_24_PCT'] / 
        (df['AGE_65_69_PCT'] + df['AGE_70_79_PCT'] + df['AGE_80_PLUS_PCT'] + 1e-5)
    )
    
    # Economic indicators
    df_new['household_density'] = 100 / (df['TOTAL_HOUSEHOLDS_PCT'] + 1e-5)
    
    # Additional interaction terms
    df_new['minority_disability'] = df_new['minority_pct'] * df['DISABILITY_POP_PCT'] / 100
    df_new['family_veteran'] = df['FAMILY_HH_TOTAL'] * df['VETERAN_POP_PCT'] / 100
    
    return df_new

# Apply feature engineering
X_train_eng = create_features(X_train)
X_test_eng = create_features(X_test)

print(f"âœ“ Original features: {X_train.shape[1]}")
print(f"âœ“ Engineered features: {X_train_eng.shape[1]}")

# ================================================================================
# SECTION 3: FEATURE SELECTION
# ================================================================================
print("\n[STEP 3] Feature Selection...")

# Remove highly correlated features
correlation_matrix = X_train_eng.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

X_train_selected = X_train_eng.drop(columns=to_drop, errors='ignore')
X_test_selected = X_test_eng.drop(columns=to_drop, errors='ignore')

print(f"âœ“ Removed {len(to_drop)} highly correlated features")
print(f"âœ“ Features after correlation filter: {X_train_selected.shape[1]}")

# Select top features using mutual information
if X_train_selected.shape[1] > 30:
    selector = SelectKBest(mutual_info_regression, k=30)
    X_train_temp = selector.fit_transform(X_train_selected, y_train)
    X_test_temp = selector.transform(X_test_selected)
    
    # Get selected column names
    selected_features = X_train_selected.columns[selector.get_support()].tolist()
    
    # Create new dataframes with selected features
    X_train_selected = pd.DataFrame(X_train_temp, columns=selected_features, index=X_train_selected.index)
    X_test_selected = pd.DataFrame(X_test_temp, columns=selected_features, index=X_test_selected.index)
    
    print(f"âœ“ Selected top 30 features")

print(f"âœ“ Final feature count: {X_train_selected.shape[1]}")

# ================================================================================
# SECTION 4: DATA SCALING
# ================================================================================
print("\n[STEP 4] Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

print(f"âœ“ Data scaled using RobustScaler")

# ================================================================================
# SECTION 5: MODEL TRAINING
# ================================================================================
print("\n[STEP 5] Training models...")

# Define models
models = {
    # Linear models
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=2000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000),
    'BayesianRidge': BayesianRidge(),
    'HuberRegressor': HuberRegressor(epsilon=1.35, alpha=0.01),
    
    # SVM models
    'SVR_Linear': SVR(kernel='linear', C=1.0, epsilon=0.001),
    'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.001),
    
    # Tree-based models
    'RandomForest': RandomForestRegressor(
        n_estimators=100, max_depth=3, min_samples_split=10,
        min_samples_leaf=5, random_state=42
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}

# Add optional models if available
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=10, subsample=0.8, random_state=42, verbose=-1
    )

if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostRegressor(
        iterations=100, depth=3, learning_rate=0.05,
        l2_leaf_reg=10, random_state=42, verbose=False
    )

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate models
cv_results = {}
trained_models = {}

for name, model in models.items():
    try:
        # Cross-validation
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
        )
        rmse_scores = np.sqrt(-scores)
        
        # Train on full data
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        # Store results
        cv_results[name] = {
            'cv_mean': rmse_scores.mean(),
            'cv_std': rmse_scores.std(),
            'train_rmse': train_rmse
        }
        trained_models[name] = model
        
        print(f"  {name:20s} | CV RMSE: {rmse_scores.mean():.6f} Â± {rmse_scores.std():.6f}")
        
    except Exception as e:
        print(f"  {name:20s} | Failed: {str(e)[:50]}")

# ================================================================================
# SECTION 6: ENSEMBLE CREATION
# ================================================================================
print("\n[STEP 6] Creating ensemble...")

# Sort models by CV performance
sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['cv_mean'])
top_n = min(5, len(sorted_models))
top_models = sorted_models[:top_n]

print(f"\nTop {top_n} models for ensemble:")
ensemble_predictions = []
ensemble_weights = []

for name, scores in top_models:
    print(f"  {name}: RMSE = {scores['cv_mean']:.6f}")
    
    # Get predictions
    model = trained_models[name]
    pred = model.predict(X_test_scaled)
    ensemble_predictions.append(pred)
    
    # Calculate weight (inverse of RMSE)
    weight = 1.0 / (scores['cv_mean'] + 1e-10)
    ensemble_weights.append(weight)

# Normalize weights
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()

print(f"\nEnsemble weights:")
for (name, _), weight in zip(top_models, ensemble_weights):
    print(f"  {name}: {weight:.3f}")

# ================================================================================
# SECTION 7: FINAL PREDICTIONS
# ================================================================================
print("\n[STEP 7] Generating predictions...")

# Weighted average ensemble
final_predictions = np.average(ensemble_predictions, weights=ensemble_weights, axis=0)

# Post-processing
final_predictions = np.maximum(final_predictions, 0)  # Ensure non-negative
upper_bound = np.percentile(y_train, 99) * 1.5
final_predictions = np.clip(final_predictions, 0, upper_bound)

print(f"\nPrediction statistics:")
print(f"  Mean: {final_predictions.mean():.6f}")
print(f"  Std: {final_predictions.std():.6f}")
print(f"  Min: {final_predictions.min():.6f}")
print(f"  Max: {final_predictions.max():.6f}")

# ================================================================================
# SECTION 8: CREATE SUBMISSION
# ================================================================================
print("\n[STEP 8] Creating submission...")

submission = pd.DataFrame({
    'ID': test_ids,
    'HOMELESS_RATE': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"âœ“ Submission saved to: submission.csv")

# ================================================================================
# SECTION 9: VISUALIZATION
# ================================================================================
print("\n[STEP 9] Creating visualizations...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Prediction distribution
    axes[0, 0].hist(final_predictions, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(y_train.mean(), color='red', linestyle='--', label='Train mean')
    axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].set_xlabel('Predicted Homeless Rate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Plot 2: Model performance comparison
    model_names = list(cv_results.keys())
    means = [cv_results[name]['cv_mean'] for name in model_names]
    stds = [cv_results[name]['cv_std'] for name in model_names]
    
    axes[0, 1].bar(range(len(model_names)), means, yerr=stds, capsize=5, alpha=0.7)
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_title('Model Performance Comparison')
    axes[0, 1].set_ylabel('CV RMSE')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Train vs Prediction comparison
    axes[1, 0].hist(y_train, bins=20, alpha=0.5, label='Training', color='green')
    axes[1, 0].hist(final_predictions, bins=20, alpha=0.5, label='Predictions', color='orange')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].set_xlabel('Homeless Rate')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 4: Feature importance (if RandomForest available)
    if 'RandomForest' in trained_models:
        rf_model = trained_models['RandomForest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        
        axes[1, 1].barh(range(len(indices)), importances[indices])
        axes[1, 1].set_yticks(range(len(indices)))
        axes[1, 1].set_yticklabels([X_train_selected.columns[i] for i in indices])
        axes[1, 1].set_title('Top 10 Feature Importances')
        axes[1, 1].set_xlabel('Importance')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Feature Importance')
    
    plt.suptitle('California Homelessness Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"âœ“ Visualizations saved to: analysis.png")
    
except Exception as e:
    print(f"âš  Visualization failed: {e}")

# ================================================================================
# FINAL REPORT
# ================================================================================
print("\n" + "=" * 80)
print(" FINAL REPORT")
print("=" * 80)

print(f"\nğŸ“Š Dataset Summary:")
print(f"  â€¢ Training samples: {len(y_train)}")
print(f"  â€¢ Test samples: {len(test_ids)}")
print(f"  â€¢ Original features: {X_train.shape[1]}")
print(f"  â€¢ Engineered features: {X_train_eng.shape[1]}")
print(f"  â€¢ Selected features: {X_train_selected.shape[1]}")

print(f"\nğŸ�† Best Models:")
for i, (name, scores) in enumerate(sorted_models[:3], 1):
    print(f"  {i}. {name}: RMSE = {scores['cv_mean']:.6f}")

print(f"\nğŸ�¯ Ensemble Details:")
print(f"  â€¢ Models in ensemble: {len(top_models)}")
print(f"  â€¢ Weighting method: Inverse RMSE")
print(f"  â€¢ Post-processing: Clipping to [0, {upper_bound:.6f}]")

print(f"\nâœ… Output Files:")
print(f"  â€¢ submission.csv - Final predictions")
print(f"  â€¢ analysis.png - Visualization plots")

print(f"\nğŸš€ Pipeline completed successfully!")
print("=" * 80)

