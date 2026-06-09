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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================================================================================
COMPLETE SMOOTH ENSEMBLE PIPELINE FOR CHOLESTEROL PREDICTION
======================================================================================
All-in-one working implementation with smooth model transitions
======================================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Essential imports
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Optional advanced models
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("âš ï¸�  XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
    print("âš ï¸�  LightGBM not available")

# Other imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from scipy.spatial.distance import cdist
import time
from datetime import datetime

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("ğŸš€ COMPLETE SMOOTH ENSEMBLE PIPELINE FOR CHOLESTEROL PREDICTION")
print("=" * 100)

# ========================================
# SMOOTH ENSEMBLE COORDINATOR CLASS
# ========================================

class SimpleSmoothEnsemble(BaseEstimator, RegressorMixin):
    """
    Simplified smooth ensemble that creates weighted combinations of models
    based on local performance and model agreement
    """
    
    def __init__(self, base_models=None, n_neighbors=20, temperature=1.0, random_state=42):
        self.base_models = base_models or {}
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.random_state = random_state
        
        # To be fitted
        self.X_train_ = None
        self.y_train_ = None
        self.model_predictions_ = {}
        self.local_weights_ = {}
        self.scaler_ = None
        
    def fit(self, X, y):
        """Fit the smooth ensemble"""
        print("\nğŸ�¯ Fitting Smooth Ensemble...")
        
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Store predictions from each model
        print("  Computing model predictions...")
        for name, model in self.base_models.items():
            self.model_predictions_[name] = model.predict(X)
        
        # Compute local weights for each model
        print("  Computing local model weights...")
        self._compute_local_weights(X, y)
        
        return self
    
    def _compute_local_weights(self, X, y):
        """Compute weights based on local model performance"""
        # Use KNN to find local neighborhoods
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(X)
        
        # For each model, compute local performance
        for name in self.base_models.keys():
            predictions = self.model_predictions_[name]
            local_weights = np.zeros(len(X))
            
            for i in range(len(X)):
                # Find neighbors
                distances, indices = nn.kneighbors([X[i]])
                indices = indices[0]
                
                # Compute local error
                local_errors = np.abs(predictions[indices] - y.iloc[indices] if hasattr(y, 'iloc') else y[indices])
                
                # Convert error to weight (inverse relationship)
                # Add small epsilon to avoid division by zero
                local_weights[i] = 1.0 / (np.mean(local_errors) + 1e-6)
            
            self.local_weights_[name] = local_weights
    
    def predict(self, X):
        """Make predictions with smooth weighting"""
        # Find nearest training points for each test point
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.X_train_)
        
        predictions = []
        
        for i in range(len(X)):
            # Find nearest neighbors
            distances, indices = nn.kneighbors([X[i]])
            indices = indices[0]
            
            # Get weights for this region
            weights = []
            model_preds = []
            
            for name, model in self.base_models.items():
                # Average weight from neighbors
                local_weight = np.mean(self.local_weights_[name][indices])
                weights.append(local_weight)
                
                # Model prediction
                pred = model.predict([X[i]])[0]
                model_preds.append(pred)
            
            # Apply softmax to weights with temperature
            weights = np.array(weights)
            weights = softmax(weights / self.temperature)
            
            # Weighted prediction
            final_pred = np.sum(weights * np.array(model_preds))
            predictions.append(final_pred)
        
        return np.array(predictions)
    
    def predict_with_details(self, X):
        """Predict with additional details about model contributions"""
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.X_train_)
        
        predictions = []
        all_weights = []
        disagreements = []
        
        for i in range(len(X)):
            distances, indices = nn.kneighbors([X[i]])
            indices = indices[0]
            
            weights = []
            model_preds = []
            
            for name, model in self.base_models.items():
                local_weight = np.mean(self.local_weights_[name][indices])
                weights.append(local_weight)
                pred = model.predict([X[i]])[0]
                model_preds.append(pred)
            
            weights = np.array(weights)
            weights = softmax(weights / self.temperature)
            
            final_pred = np.sum(weights * np.array(model_preds))
            predictions.append(final_pred)
            all_weights.append(weights)
            
            # Compute disagreement as std of predictions
            disagreements.append(np.std(model_preds))
        
        return np.array(predictions), np.array(all_weights), np.array(disagreements)

# ========================================
# DATA LOADING AND PREPARATION
# ========================================
print("\nğŸ“‚ LOADING DATA...")

# Load data
train_df = pd.read_csv('/kaggle/input/ndsc-regression-cholesterol-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/ndsc-regression-cholesterol-prediction/test.csv')

print(f"âœ… Train samples: {len(train_df)}")
print(f"âœ… Test samples: {len(test_df)}")

# Target column
target_column = 'Cholesterol Total (mg/dL)'

# ========================================
# FEATURE ENGINEERING
# ========================================
print("\nğŸ”§ FEATURE ENGINEERING...")

def engineer_features(df):
    """Create features for cholesterol prediction"""
    df_feat = df.copy()
    
    # Handle categorical
    if 'Jenis Kelamin' in df_feat.columns:
        df_feat['is_male'] = (df_feat['Jenis Kelamin'] == 'M').astype(int)
        df_feat.drop('Jenis Kelamin', axis=1, inplace=True)
    
    if 'Tempat lahir' in df_feat.columns:
        # Simple frequency encoding
        if 'birthplace_freq' not in globals():
            global birthplace_freq
            birthplace_freq = df_feat['Tempat lahir'].value_counts().to_dict()
        df_feat['birthplace_encoded'] = df_feat['Tempat lahir'].map(birthplace_freq).fillna(0)
        df_feat.drop('Tempat lahir', axis=1, inplace=True)
    
    # BMI categories
    if 'IMT (kg/m2)' in df_feat.columns:
        df_feat['bmi_underweight'] = (df_feat['IMT (kg/m2)'] < 18.5).astype(int)
        df_feat['bmi_normal'] = ((df_feat['IMT (kg/m2)'] >= 18.5) & (df_feat['IMT (kg/m2)'] < 25)).astype(int)
        df_feat['bmi_overweight'] = ((df_feat['IMT (kg/m2)'] >= 25) & (df_feat['IMT (kg/m2)'] < 30)).astype(int)
        df_feat['bmi_obese'] = (df_feat['IMT (kg/m2)'] >= 30).astype(int)
    
    # Blood pressure features
    if all(col in df_feat.columns for col in ['Tekanan darah  (S)', 'Tekanan darah  (D)']):
        df_feat['pulse_pressure'] = df_feat['Tekanan darah  (S)'] - df_feat['Tekanan darah  (D)']
        df_feat['mean_arterial_pressure'] = (df_feat['Tekanan darah  (S)'] + 2 * df_feat['Tekanan darah  (D)']) / 3
        df_feat['hypertension'] = ((df_feat['Tekanan darah  (S)'] >= 140) | (df_feat['Tekanan darah  (D)'] >= 90)).astype(int)
    
    # Glucose features
    if 'Glukosa Puasa (mg/dL)' in df_feat.columns:
        df_feat['glucose_high'] = (df_feat['Glukosa Puasa (mg/dL)'] >= 100).astype(int)
        df_feat['glucose_log'] = np.log1p(df_feat['Glukosa Puasa (mg/dL)'])
    
    # Triglyceride features
    if 'Trigliserida (mg/dL)' in df_feat.columns:
        df_feat['trig_high'] = (df_feat['Trigliserida (mg/dL)'] >= 150).astype(int)
        df_feat['trig_log'] = np.log1p(df_feat['Trigliserida (mg/dL)'])
    
    # Ratios
    if all(col in df_feat.columns for col in ['Lingkar perut (cm)', 'Tinggi badan (cm)']):
        df_feat['waist_height_ratio'] = df_feat['Lingkar perut (cm)'] / df_feat['Tinggi badan (cm)']
    
    # Age groups
    if 'Usia' in df_feat.columns:
        df_feat['age_group'] = pd.cut(df_feat['Usia'], bins=[0, 30, 40, 50, 60, 100], labels=[1, 2, 3, 4, 5]).astype(float)
        df_feat['age_squared'] = df_feat['Usia'] ** 2
    
    # Fill any missing values
    df_feat = df_feat.fillna(df_feat.median())
    
    return df_feat

# Apply feature engineering
X = engineer_features(train_df.drop(target_column, axis=1))
y = train_df[target_column]

test_ids = test_df['id'] if 'id' in test_df.columns else range(len(test_df))
X_test = engineer_features(test_df.drop('id', axis=1) if 'id' in test_df.columns else test_df)

# Ensure same columns
common_cols = sorted(list(set(X.columns) & set(X_test.columns)))
X = X[common_cols]
X_test = X_test[common_cols]

print(f"âœ… Features created: {len(common_cols)}")

# ========================================
# DATA PREPROCESSING
# ========================================
print("\nğŸ”§ DATA PREPROCESSING...")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"âœ… Train shape: {X_train_scaled.shape}")
print(f"âœ… Validation shape: {X_val_scaled.shape}")

# ========================================
# TRAIN MODEL ZOO
# ========================================
print("\nğŸ¦� TRAINING MODEL ZOO...")

base_models = {}

# 1. Random Forest
print("  Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
base_models['RandomForest'] = rf

# 2. Gradient Boosting
print("  Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED)
gb.fit(X_train_scaled, y_train)
base_models['GradientBoosting'] = gb

# 3. Extra Trees
print("  Training Extra Trees...")
et = ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
et.fit(X_train_scaled, y_train)
base_models['ExtraTrees'] = et

# 4. Ridge
print("  Training Ridge...")
ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge.fit(X_train_scaled, y_train)
base_models['Ridge'] = ridge

# 5. ElasticNet
print("  Training ElasticNet...")
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_SEED)
elastic.fit(X_train_scaled, y_train)
base_models['ElasticNet'] = elastic

# 6. KNN
print("  Training KNN...")
knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn.fit(X_train_scaled, y_train)
base_models['KNN'] = knn

# 7. XGBoost (if available)
if HAS_XGB:
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_SEED)
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], 
                  early_stopping_rounds=50, verbose=False)
    base_models['XGBoost'] = xgb_model

# 8. LightGBM (if available)
if HAS_LGB:
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, 
                                   random_state=RANDOM_SEED, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    base_models['LightGBM'] = lgb_model

print(f"\nâœ… Trained {len(base_models)} models")

# Evaluate individual models
print("\nğŸ“Š INDIVIDUAL MODEL PERFORMANCE:")
model_scores = {}
for name, model in base_models.items():
    pred = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    model_scores[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, RÂ²={r2:.4f}")

# ========================================
# ANALYZE MODEL DISAGREEMENTS
# ========================================
print("\nğŸ”� ANALYZING MODEL DISAGREEMENTS...")

# Get validation predictions
val_predictions = {}
for name, model in base_models.items():
    val_predictions[name] = model.predict(X_val_scaled)

# Compute pairwise disagreements
print("\nğŸ“Š Pairwise model disagreements (avg absolute difference):")
model_names = list(base_models.keys())
for i, model1 in enumerate(model_names):
    for j, model2 in enumerate(model_names):
        if i < j:
            disagreement = np.mean(np.abs(val_predictions[model1] - val_predictions[model2]))
            print(f"  {model1} vs {model2}: {disagreement:.2f}")

# ========================================
# CREATE SMOOTH ENSEMBLE
# ========================================
print("\nğŸŒŠ CREATING SMOOTH ENSEMBLE...")

# Retrain models on full training data
print("\nğŸ”§ Retraining models on full dataset...")
X_full_scaled = scaler.fit_transform(X)
y_full = y

for name, model in base_models.items():
    if name in ['XGBoost', 'LightGBM']:
        # These models need special handling
        if name == 'XGBoost' and HAS_XGB:
            model.fit(X_full_scaled, y_full)
        elif name == 'LightGBM' and HAS_LGB:
            model.fit(X_full_scaled, y_full)
    else:
        model.fit(X_full_scaled, y_full)

# Create smooth ensemble
print("\nğŸ�¯ Fitting smooth ensemble coordinator...")
smooth_ensemble = SimpleSmoothEnsemble(
    base_models=base_models,
    n_neighbors=20,
    temperature=0.5,  # Lower = sharper transitions, Higher = smoother
    random_state=RANDOM_SEED
)

smooth_ensemble.fit(X_full_scaled, y_full)

# ========================================
# MAKE PREDICTIONS
# ========================================
print("\nğŸ”® MAKING PREDICTIONS...")

# Standard predictions
X_test_final_scaled = scaler.transform(X_test)
smooth_predictions, weights, disagreements = smooth_ensemble.predict_with_details(X_test_final_scaled)

print(f"\nğŸ“Š Prediction statistics:")
print(f"  Mean: {smooth_predictions.mean():.2f}")
print(f"  Std: {smooth_predictions.std():.2f}")
print(f"  Min: {smooth_predictions.min():.2f}")
print(f"  Max: {smooth_predictions.max():.2f}")

print(f"\nğŸ“Š Model disagreement statistics:")
print(f"  Mean disagreement: {disagreements.mean():.2f}")
print(f"  Max disagreement: {disagreements.max():.2f}")
print(f"  High disagreement samples (>P90): {np.sum(disagreements > np.percentile(disagreements, 90))}")

# Average model contributions
print(f"\nğŸ�¨ Average model contributions:")
avg_weights = weights.mean(axis=0)
for i, name in enumerate(model_names):
    print(f"  {name}: {avg_weights[i]:.1%}")

# ========================================
# HANDLE HIGH UNCERTAINTY REGIONS
# ========================================
print("\nâš ï¸�  HANDLING HIGH UNCERTAINTY REGIONS...")

# Identify high disagreement samples
high_disagreement_mask = disagreements > np.percentile(disagreements, 80)
n_high_disagreement = np.sum(high_disagreement_mask)

if n_high_disagreement > 0:
    print(f"\n  Found {n_high_disagreement} high-disagreement predictions")
    
    # Conservative adjustment: pull toward training mean
    training_mean = y.mean()
    adjustment_factor = 0.2
    
    smooth_predictions[high_disagreement_mask] = (
        smooth_predictions[high_disagreement_mask] * (1 - adjustment_factor) +
        training_mean * adjustment_factor
    )
    
    print(f"  Applied conservative adjustment to {n_high_disagreement} predictions")

# ========================================
# POST-PROCESSING
# ========================================
print("\nğŸ”§ POST-PROCESSING...")

# Clip to reasonable range
lower_bound = y.quantile(0.01)
upper_bound = y.quantile(0.99)
smooth_predictions = np.clip(smooth_predictions, lower_bound, upper_bound)

print(f"  Clipped predictions to [{lower_bound:.1f}, {upper_bound:.1f}]")

# ========================================
# CREATE SUBMISSIONS
# ========================================
print("\nğŸ’¾ CREATING SUBMISSIONS...")

# Main submission
submission = pd.DataFrame({
    'id': test_ids,
    target_column: smooth_predictions
})

submission.to_csv('submission_smooth_ensemble.csv', index=False)
print("âœ… Saved: submission_smooth_ensemble.csv")

# Submission with details
submission_detailed = submission.copy()
submission_detailed['disagreement'] = disagreements
submission_detailed['is_high_uncertainty'] = high_disagreement_mask.astype(int)

# Add individual model predictions for analysis
for i, name in enumerate(model_names):
    submission_detailed[f'weight_{name}'] = weights[:, i]

submission_detailed.to_csv('submission_detailed.csv', index=False)
print("âœ… Saved: submission_detailed.csv")

print(f"\nğŸ“‹ Submission preview:")
print(submission.head(10))

# ========================================
# VISUALIZATIONS
# ========================================
print("\nğŸ“Š CREATING VISUALIZATIONS...")

# 1. Model weights distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Average model contributions
ax = axes[0, 0]
ax.bar(model_names, avg_weights)
ax.set_xlabel('Model')
ax.set_ylabel('Average Weight')
ax.set_title('Average Model Contributions')
ax.tick_params(axis='x', rotation=45)

# Plot 2: Disagreement distribution
ax = axes[0, 1]
ax.hist(disagreements, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(disagreements.mean(), color='red', linestyle='--', label=f'Mean: {disagreements.mean():.2f}')
ax.set_xlabel('Disagreement Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Model Disagreements')
ax.legend()

# Plot 3: Weight variation
ax = axes[1, 0]
for i, name in enumerate(model_names[:5]):  # Top 5 models
    ax.plot(weights[:100, i], label=name, alpha=0.7)  # First 100 samples
ax.set_xlabel('Sample Index')
ax.set_ylabel('Model Weight')
ax.set_title('Model Weight Variation (First 100 Samples)')
ax.legend()

# Plot 4: Predictions vs disagreement
ax = axes[1, 1]
scatter = ax.scatter(smooth_predictions, disagreements, alpha=0.5, c=disagreements, cmap='viridis')
ax.set_xlabel('Prediction')
ax.set_ylabel('Disagreement Score')
ax.set_title('Predictions vs Model Disagreement')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig('smooth_ensemble_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 100)
print("ğŸ�‰ SMOOTH ENSEMBLE PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 100)

print(f"""
Pipeline Summary:
----------------
â€¢ Models trained: {len(base_models)}
â€¢ Smooth weight transitions: âœ…
â€¢ Disagreement-based uncertainty: âœ…
â€¢ High-uncertainty adjustments: âœ…

Model Performance:
-----------------
â€¢ Best individual model: {min(model_scores, key=lambda x: model_scores[x]['rmse'])}
â€¢ Best RMSE: {min(model_scores.values(), key=lambda x: x['rmse'])['rmse']:.4f}

Ensemble Results:
----------------
â€¢ Predictions mean: {smooth_predictions.mean():.2f}
â€¢ Predictions std: {smooth_predictions.std():.2f}
â€¢ High disagreement samples: {n_high_disagreement}

Key Features:
------------
âœ¨ Smooth transitions between models based on local performance
âœ¨ Automatic handling of disagreement regions
âœ¨ Conservative adjustments in uncertain areas
âœ¨ Interpretable model contributions

Files Created:
-------------
â€¢ submission_smooth_ensemble.csv - Main submission
â€¢ submission_detailed.csv - Detailed analysis
â€¢ smooth_ensemble_analysis.png - Visualizations

Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

print("\nâœ¨ The smooth ensemble successfully balances model strengths and handles uncertainties!")
print("=" * 100)

# Clean up
import gc
gc.collect()

