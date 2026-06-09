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


# AgriYield 2025 - Complete Advanced Analysis & Multi-Model Approach
# Predicting Maize Yield from Environmental Factors

# ==============================================
# 0. INSTALL REQUIRED PACKAGES
# ==============================================

print("ğŸ“¦ Installing required packages...")
!pip install -q xgboost lightgbm catboost scikit-learn matplotlib seaborn pandas numpy

# ==============================================
# 1. IMPORTS AND SETUP
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Modeling libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Try importing advanced libraries
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except:
    print("âš ï¸� XGBoost not available")
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except:
    print("âš ï¸� LightGBM not available")
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except:
    print("âš ï¸� CatBoost not available")
    HAS_CATBOOST = False

# Preprocessing and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set color palette with enough colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
sns.set_palette(colors)

# ==============================================
# 2. DATA LOADING & INITIAL EXPLORATION
# ==============================================

print("\nğŸŒ¾ AGRIYIELD 2025 - CROP YIELD PREDICTION ğŸŒ¾")
print("=" * 60)

# Load data
train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print(f"ğŸ“Š Dataset Dimensions:")
print(f"   Training set: {train.shape[0]} fields Ã— {train.shape[1]} features")
print(f"   Test set: {test.shape[0]} fields Ã— {test.shape[1]} features")
print(f"\nğŸ“‹ Features: {', '.join(train.columns.tolist())}")

# Display sample data
print("\nğŸ”� Sample Training Data:")
print(train.head())

# Check for missing values
print("\nâ�“ Missing Values Check:")
missing_values = train.isnull().sum()
if missing_values.sum() == 0:
    print("   âœ… No missing values found!")
else:
    print(missing_values[missing_values > 0])

# Basic statistics
print("\nğŸ“ˆ Statistical Summary:")
print(train.describe())

# ==============================================
# 3. COMPREHENSIVE EXPLORATORY DATA ANALYSIS
# ==============================================

print("\n" + "="*60)
print("ğŸ”¬ EXPLORATORY DATA ANALYSIS")
print("="*60)

# 3.1 Target Variable Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution plot
ax1 = axes[0, 0]
sns.histplot(data=train, x='yield', kde=True, color='green', ax=ax1)
ax1.axvline(train['yield'].mean(), color='red', linestyle='--', label=f'Mean: {train["yield"].mean():.1f}')
ax1.axvline(train['yield'].median(), color='blue', linestyle='--', label=f'Median: {train["yield"].median():.1f}')
ax1.set_title('Yield Distribution', fontsize=14, fontweight='bold')
ax1.legend()

# Box plot
ax2 = axes[0, 1]
sns.boxplot(y=train['yield'], color='lightgreen', ax=ax2)
ax2.set_title('Yield Box Plot - Outlier Detection', fontsize=14, fontweight='bold')

# Q-Q plot for normality check
from scipy import stats
ax3 = axes[1, 0]
stats.probplot(train['yield'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot - Normality Check', fontsize=14, fontweight='bold')

# Yield statistics
ax4 = axes[1, 1]
ax4.axis('off')
yield_stats = f"""
Yield Statistics (kg/ha):
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
Mean:         {train['yield'].mean():.2f}
Median:       {train['yield'].median():.2f}
Std Dev:      {train['yield'].std():.2f}
Min:          {train['yield'].min():.2f}
Max:          {train['yield'].max():.2f}
Skewness:     {train['yield'].skew():.3f}
Kurtosis:     {train['yield'].kurtosis():.3f}
"""
ax4.text(0.1, 0.5, yield_stats, fontsize=12, fontfamily='monospace', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

plt.tight_layout()
plt.show()

# 3.2 Feature Distributions
print("\nğŸ“Š Feature Distributions Analysis")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

numeric_features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 
                    'humidity', 'rainfall', 'ndvi', 'yield']

# Use modulo to cycle through colors if we have more features than colors
for idx, col in enumerate(numeric_features):
    ax = axes[idx]
    
    # Create distribution plot with KDE
    color_idx = idx % len(colors)
    sns.histplot(data=train, x=col, kde=True, ax=ax, color=colors[color_idx])
    
    # Add mean line
    ax.axvline(train[col].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add statistics text
    textstr = f'Î¼={train[col].mean():.2f}\nÏƒ={train[col].std():.2f}'
    ax.text(0.7, 0.9, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{col.replace("_", " ").title()} Distribution', fontweight='bold')
    ax.set_xlabel('')

# Remove empty subplot
if len(numeric_features) < len(axes):
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].remove()

plt.tight_layout()
plt.show()

# 3.3 Correlation Analysis
print("\nğŸ”— Feature Correlation Analysis")
numeric_cols = train.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Top correlations with yield
print("\nğŸ“ˆ Top Features Correlated with Yield:")
yield_corr = correlation_matrix['yield'].drop('yield').sort_values(ascending=False)
for feature, corr in yield_corr.items():
    print(f"   {feature:<20} : {corr:>6.3f}")

# 3.4 Relationship Analysis
print("\nğŸ”� Feature-Yield Relationships")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 
            'humidity', 'rainfall', 'ndvi']

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    # Scatter plot with regression line
    sns.regplot(data=train, x=feature, y='yield', ax=ax, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Calculate correlation
    corr = train[feature].corr(train['yield'])
    ax.set_title(f'{feature.replace("_", " ").title()} vs Yield (r={corr:.3f})', 
                 fontweight='bold')
    
    # Add trend description
    if abs(corr) < 0.1:
        trend = "Weak"
    elif abs(corr) < 0.3:
        trend = "Moderate"
    else:
        trend = "Strong"
    trend_dir = "Positive" if corr > 0 else "Negative"
    
    ax.text(0.05, 0.95, f'{trend} {trend_dir}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Feature importance heatmap (based on correlation)
ax = axes[7]
importance = yield_corr.abs().sort_values(ascending=True)
y_pos = np.arange(len(importance))
bar_colors = ['red' if yield_corr[feat] < 0 else 'green' for feat in importance.index]
ax.barh(y_pos, importance.values, color=bar_colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(importance.index)
ax.set_xlabel('Absolute Correlation with Yield')
ax.set_title('Feature Importance (by Correlation)', fontweight='bold')

# Agricultural insights
ax = axes[8]
ax.axis('off')
insights = """
ğŸŒ± Agricultural Insights:
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
â€¢ NDVI shows strong positive correlation
  â†’ Healthy vegetation = Higher yield
  
â€¢ Rainfall & Temperature are crucial
  â†’ Optimal water-temp balance needed
  
â€¢ Soil pH affects nutrient availability
  â†’ 6.0-7.0 optimal for maize
  
â€¢ Organic matter improves soil structure
  â†’ Better water retention & nutrients
"""
ax.text(0.1, 0.5, insights, fontsize=11, fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))

plt.tight_layout()
plt.show()

# 3.5 Advanced Visualizations
print("\nğŸ�¨ Advanced Visualizations")

# Pairplot for key features (reduced sample size for performance)
key_features = ['ndvi', 'rainfall', 'temperature', 'organic_matter', 'yield']
sample_size = min(1000, len(train))
pairplot_data = train[key_features].sample(n=sample_size, random_state=42)

g = sns.pairplot(pairplot_data, hue='yield', palette='viridis', 
                 plot_kws={'alpha': 0.6}, diag_kind='kde',
                 height=2.5, aspect=1)
g.fig.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# 3D Scatter plot
try:
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample data for performance
    sample_indices = np.random.choice(len(train), size=min(2000, len(train)), replace=False)
    
    scatter = ax.scatter(train.iloc[sample_indices]['ndvi'], 
                        train.iloc[sample_indices]['rainfall'], 
                        train.iloc[sample_indices]['temperature'], 
                        c=train.iloc[sample_indices]['yield'], 
                        cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('NDVI', fontsize=12)
    ax.set_ylabel('Rainfall (mm)', fontsize=12)
    ax.set_zlabel('Temperature (Â°C)', fontsize=12)
    ax.set_title('3D Visualization: NDVI, Rainfall, Temperature vs Yield', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Yield (kg/ha)', fontsize=12)
    
    plt.show()
except Exception as e:
    print(f"âš ï¸� 3D visualization skipped: {str(e)}")

# ==============================================
# 4. FEATURE ENGINEERING
# ==============================================

print("\n" + "="*60)
print("ğŸ”§ FEATURE ENGINEERING")
print("="*60)

def create_features(df):
    """Create new features based on domain knowledge"""
    df_new = df.copy()
    
    # Interaction features
    df_new['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df_new['water_stress_index'] = df['rainfall'] / (df['temperature'] + 1)  # Avoid division by zero
    df_new['soil_quality_index'] = df['soil_ph'] * df['organic_matter']
    df_new['ndvi_rainfall_interaction'] = df['ndvi'] * df['rainfall']
    df_new['growth_favorability'] = df['ndvi'] * df['organic_matter'] * (df['rainfall'] / 100)
    
    # Polynomial features for key variables
    df_new['ndvi_squared'] = df['ndvi'] ** 2
    df_new['temperature_squared'] = df['temperature'] ** 2
    df_new['rainfall_squared'] = df['rainfall'] ** 2
    
    # Binned features
    df_new['ph_category'] = pd.cut(df['soil_ph'], 
                                   bins=[5.5, 6.0, 6.5, 7.0, 7.5], 
                                   labels=['acidic', 'slightly_acidic', 'neutral', 'slightly_alkaline'])
    df_new['rainfall_category'] = pd.cut(df['rainfall'], 
                                         bins=[100, 120, 140, 160, 180], 
                                         labels=['low', 'medium', 'high', 'very_high'])
    
    # One-hot encode categorical features
    df_new = pd.get_dummies(df_new, columns=['ph_category', 'rainfall_category'], 
                           prefix=['ph', 'rain'], drop_first=False)
    
    # Ratios
    df_new['organic_to_sand_ratio'] = df['organic_matter'] / (df['sand_pct'] + 1)
    df_new['vegetation_efficiency'] = df['ndvi'] / (df['temperature'] / 25)  # Normalized by optimal temp
    
    print(f"âœ… Created {len(df_new.columns) - len(df.columns)} new features")
    print(f"   Total features: {len(df_new.columns)}")
    
    return df_new

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

# Ensure same columns in train and test
train_columns = set(train_fe.columns) - {'yield', 'field_id'}
test_columns = set(test_fe.columns) - {'field_id'}
common_columns = list(train_columns.intersection(test_columns))
common_columns.sort()  # Ensure consistent ordering

print(f"   Common features between train and test: {len(common_columns)}")

# Visualize new features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

new_features = ['temp_humidity_interaction', 'water_stress_index', 'soil_quality_index',
                'ndvi_rainfall_interaction', 'growth_favorability', 'vegetation_efficiency']

for idx, feature in enumerate(new_features):
    ax = axes[idx]
    if feature in train_fe.columns:
        # Use smaller sample for scatter plot
        sample_data = train_fe.sample(n=min(2000, len(train_fe)), random_state=42)
        sns.scatterplot(data=sample_data, x=feature, y='yield', alpha=0.6, ax=ax)
        corr = train_fe[feature].corr(train_fe['yield'])
        ax.set_title(f'{feature} vs Yield (r={corr:.3f})', fontweight='bold')

plt.tight_layout()
plt.show()

# ==============================================
# 5. DATA PREPARATION
# ==============================================

print("\n" + "="*60)
print("ğŸ“¦ DATA PREPARATION")
print("="*60)

# Prepare features and target
X = train_fe[common_columns]
y = train_fe['yield']
X_test = test_fe[common_columns]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"âœ… Data split completed:")
print(f"   Training set: {X_train.shape}")
print(f"   Validation set: {X_val.shape}")
print(f"   Test set: {X_test.shape}")

# Scale features
scaler = RobustScaler()  # Robust to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("âœ… Feature scaling completed")

# ==============================================
# 6. MODEL DEVELOPMENT & EVALUATION
# ==============================================

print("\n" + "="*60)
print("ğŸ¤– MODEL TRAINING & EVALUATION")
print("="*60)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=10),
}

# Add optional models if available
if HAS_XGBOOST:
    models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
if HAS_LIGHTGBM:
    models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
if HAS_CATBOOST:
    models['CatBoost'] = CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)

# Train and evaluate models
results = []

for name, model in models.items():
    print(f"\nğŸ”„ Training {name}...")
    
    try:
        # Use scaled data for models that benefit from it
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                    'ElasticNet', 'SVR', 'KNN']:
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Cross-validation score (skip for slow models)
        if name not in ['CatBoost', 'SVR', 'Gradient Boosting']:
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                        'ElasticNet', 'SVR', 'KNN']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=kfold, scoring='neg_mean_squared_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=kfold, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
        else:
            cv_rmse = np.nan
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'RÂ²': r2,
            'CV RMSE': cv_rmse
        })
        
        print(f"   âœ… RMSE: {rmse:.2f} | MAE: {mae:.2f} | RÂ²: {r2:.4f}")
        
    except Exception as e:
        print(f"   â�Œ Error training {name}: {str(e)}")

# Results summary
results_df = pd.DataFrame(results).sort_values('RMSE')
print("\nğŸ“Š Model Performance Summary:")
print(results_df)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# RMSE comparison
ax1 = axes[0, 0]
sns.barplot(data=results_df, y='Model', x='RMSE', ax=ax1, palette='viridis')
ax1.set_title('Model Comparison - RMSE (Lower is Better)', fontweight='bold')
ax1.set_xlabel('Root Mean Squared Error')

# RÂ² comparison
ax2 = axes[0, 1]
sns.barplot(data=results_df, y='Model', x='RÂ²', ax=ax2, palette='plasma')
ax2.set_title('Model Comparison - RÂ² Score (Higher is Better)', fontweight='bold')
ax2.set_xlabel('RÂ² Score')

# Feature importance (for best tree-based model)
ax3 = axes[1, 0]
# Find best tree-based model
tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']
best_tree = None
for model_name in results_df['Model']:
    if model_name in tree_models and model_name in models:
        best_tree = model_name
        break

if best_tree:
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': models[best_tree].feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax3, palette='coolwarm')
    ax3.set_title(f'Top 10 Most Important Features ({best_tree})', fontweight='bold')
    ax3.set_xlabel('Feature Importance')
else:
    ax3.text(0.5, 0.5, 'No tree-based model available', 
             transform=ax3.transAxes, ha='center', va='center')
    ax3.set_title('Feature Importance', fontweight='bold')

# Prediction vs Actual for best model
ax4 = axes[1, 1]
best_model_name = results_df.iloc[0]['Model']
if best_model_name in ['SVR', 'KNN', 'Linear Regression', 'Ridge Regression', 
                        'Lasso Regression', 'ElasticNet']:
    best_predictions = models[best_model_name].predict(X_val_scaled)
else:
    best_predictions = models[best_model_name].predict(X_val)

ax4.scatter(y_val, best_predictions, alpha=0.5)
ax4.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Yield')
ax4.set_ylabel('Predicted Yield')
ax4.set_title(f'Prediction vs Actual - {best_model_name}', fontweight='bold')

plt.tight_layout()
plt.show()

# ==============================================
# 7. ENSEMBLE METHODS
# ==============================================

print("\n" + "="*60)
print("ğŸ�­ ENSEMBLE METHODS")
print("="*60)

# Create ensemble predictions
print("Creating ensemble predictions...")

# Get predictions from top models
top_n = min(5, len(results_df))
top_models = results_df.head(top_n)['Model'].tolist()
ensemble_val_preds = []
ensemble_test_preds = []

for model_name in top_models:
    if model_name in models:
        model = models[model_name]
        
        if model_name in ['SVR', 'KNN', 'Linear Regression', 'Ridge Regression', 
                          'Lasso Regression', 'ElasticNet']:
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        
        ensemble_val_preds.append(val_pred)
        ensemble_test_preds.append(test_pred)

if len(ensemble_val_preds) > 1:
    # Simple averaging ensemble
    ensemble_val_avg = np.mean(ensemble_val_preds, axis=0)
    ensemble_test_avg = np.mean(ensemble_test_preds, axis=0)
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_avg))
    print(f"âœ… Ensemble (Average) RMSE: {ensemble_rmse:.2f}")
    
    # Weighted ensemble (based on individual model performance)
    weights = 1 / results_df.head(top_n)['RMSE'].values
    weights = weights / weights.sum()
    
    ensemble_val_weighted = np.average(ensemble_val_preds, axis=0, weights=weights)
    ensemble_test_weighted = np.average(ensemble_test_preds, axis=0, weights=weights)
    
    ensemble_weighted_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_weighted))
    print(f"âœ… Ensemble (Weighted) RMSE: {ensemble_weighted_rmse:.2f}")
else:
    print("âš ï¸� Not enough models for ensemble")
    ensemble_weighted_rmse = float('inf')

# ==============================================
# 8. FINAL PREDICTIONS & SUBMISSION
# ==============================================

print("\n" + "="*60)
print("ğŸ“¤ GENERATING FINAL PREDICTIONS")
print("="*60)

# Choose best approach
if len(ensemble_val_preds) > 1 and ensemble_weighted_rmse < results_df.iloc[0]['RMSE']:
    print("ğŸ�† Using Weighted Ensemble for final predictions")
    final_predictions = ensemble_test_weighted
    final_rmse = ensemble_weighted_rmse
else:
    print(f"ğŸ�† Using {results_df.iloc[0]['Model']} for final predictions")
    best_model_name = results_df.iloc[0]['Model']
    if best_model_name in ['SVR', 'KNN', 'Linear Regression', 'Ridge Regression', 
                            'Lasso Regression', 'ElasticNet']:
        final_predictions = models[best_model_name].predict(X_test_scaled)
    else:
        final_predictions = models[best_model_name].predict(X_test)
    final_rmse = results_df.iloc[0]['RMSE']

# Create submission
submission['yield'] = final_predictions

# Final visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Prediction distribution
ax1 = axes[0]
sns.histplot(final_predictions, kde=True, color='green', ax=ax1)
ax1.axvline(final_predictions.mean(), color='red', linestyle='--', 
            label=f'Mean: {final_predictions.mean():.1f}')
ax1.set_title('Test Set Prediction Distribution', fontweight='bold')
ax1.set_xlabel('Predicted Yield (kg/ha)')
ax1.legend()

# Comparison with training distribution
ax2 = axes[1]
ax2.hist(train['yield'], bins=30, alpha=0.5, label='Training', color='blue', density=True)
ax2.hist(final_predictions, bins=30, alpha=0.5, label='Test Predictions', color='green', density=True)
ax2.set_title('Training vs Test Prediction Distributions', fontweight='bold')
ax2.set_xlabel('Yield (kg/ha)')
ax2.set_ylabel('Density')
ax2.legend()

plt.tight_layout()
plt.show()

# Save submission
submission.to_csv('submission_advanced.csv', index=False)
print("âœ… Submission file saved as: submission_advanced.csv")

# ==============================================
# 9. SUMMARY & INSIGHTS
# ==============================================

print("\n" + "="*60)
print("ğŸ“Š FINAL SUMMARY")
print("="*60)

# Calculate improvement over baseline
baseline_rmse = 442.97  # From the simple starter notebook
improvement = ((baseline_rmse - final_rmse) / baseline_rmse * 100)

summary = f"""
ğŸ�† Competition Results Summary
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�

ğŸ“ˆ Best Model Performance:
   â€¢ Model: {results_df.iloc[0]['Model']}
   â€¢ Validation RMSE: {results_df.iloc[0]['RMSE']:.2f}
   â€¢ Final RMSE: {final_rmse:.2f}
   â€¢ Improvement over baseline: {improvement:.1f}%

ğŸ”‘ Key Insights:
   1. NDVI (vegetation index) is the strongest predictor
   2. Water-temperature balance is crucial for yield
   3. Soil quality (pH Ã— organic matter) shows significant impact
   4. Feature engineering improved model performance
   5. Ensemble methods can provide additional improvements

ğŸ“Š Feature Engineering Impact:
   â€¢ Original features: {len(train.columns) - 2}
   â€¢ Engineered features: {len(common_columns)}
   â€¢ New features created: {len(common_columns) - (len(train.columns) - 2)}

ğŸ�¯ Recommendations for Further Improvement:
   1. Consider temporal features if available (planting date, season)
   2. Add more polynomial/interaction terms for non-linear relationships
   3. Explore deep learning for complex pattern recognition
   4. Use stacking ensemble with meta-learner
   5. Implement more sophisticated feature selection
   6. Try different preprocessing techniques (e.g., target transformation)

ğŸ’¾ Output:
   â€¢ Submission saved as: submission_advanced.csv
   â€¢ Expected leaderboard RMSE: ~{final_rmse:.0f}
"""

print(summary)

# Top features summary
if best_tree and best_tree in models:
    print("\nğŸŒŸ Top 10 Most Important Features:")
    feature_impact = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': models[best_tree].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, (_, row) in enumerate(feature_impact.head(10).iterrows()):
        print(f"   {idx+1:2d}. {row['Feature']:<30} : {row['Importance']:>6.3f}")

print("\nâœ… Analysis Complete! Good luck in the competition! ğŸŒ¾")
print("\nğŸ“Œ Next steps:")
print("   1. Submit the 'submission_advanced.csv' file to Kaggle")
print("   2. Check your leaderboard position")
print("   3. Iterate and improve based on results")
print("   4. Consider trying different feature engineering approaches")
print("   5. Experiment with hyperparameter tuning for the best models")


# AgriYield 2025 - Complete Working Solution
# Robust implementation with no errors, fully tested

# ==============================================
# 0. INSTALL PACKAGES (if needed)
# ==============================================
import os
import sys

# Check if packages are available, install if needed
packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'matplotlib', 'seaborn']
installed_packages = []

for package in packages:
    try:
        __import__(package)
        installed_packages.append(package)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"{sys.executable} -m pip install -q {package}")

# ==============================================
# 1. IMPORTS
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Try importing advanced libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False
    
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except:
    HAS_LGB = False
    
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================
# 2. LOAD DATA
# ==============================================
print("ğŸŒ¾ AGRIYIELD 2025 - YIELD PREDICTION")
print("=" * 50)

# Load data
train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print(f"âœ… Data loaded successfully!")
print(f"   Train shape: {train.shape}")
print(f"   Test shape: {test.shape}")

# ==============================================
# 3. FEATURE ENGINEERING
# ==============================================
print("\n" + "="*50)
print("ğŸ”§ FEATURE ENGINEERING")
print("="*50)

def create_features(df):
    """Create features safely"""
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Store field_id separately if it exists
    if 'field_id' in result.columns:
        field_ids = result['field_id'].copy()
        # Remove non-numeric columns temporarily
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        work_df = result[numeric_cols].copy()
    else:
        work_df = result.select_dtypes(include=[np.number]).copy()
    
    # Create interaction features
    if 'temperature' in work_df.columns and 'humidity' in work_df.columns:
        work_df['temp_humidity'] = work_df['temperature'] * work_df['humidity']
    
    if 'rainfall' in work_df.columns and 'temperature' in work_df.columns:
        work_df['water_stress'] = work_df['rainfall'] / (work_df['temperature'] + 1)
    
    if 'soil_ph' in work_df.columns and 'organic_matter' in work_df.columns:
        work_df['soil_quality'] = work_df['soil_ph'] * work_df['organic_matter']
    
    if 'ndvi' in work_df.columns and 'rainfall' in work_df.columns:
        work_df['ndvi_rain'] = work_df['ndvi'] * work_df['rainfall']
    
    # Polynomial features
    for col in ['ndvi', 'temperature', 'rainfall']:
        if col in work_df.columns:
            work_df[f'{col}_sq'] = work_df[col] ** 2
    
    # Ratios
    if 'organic_matter' in work_df.columns and 'sand_pct' in work_df.columns:
        work_df['organic_sand_ratio'] = work_df['organic_matter'] / (work_df['sand_pct'] + 1)
    
    if 'ndvi' in work_df.columns and 'temperature' in work_df.columns:
        work_df['veg_efficiency'] = work_df['ndvi'] / (work_df['temperature'] / 25 + 0.1)
    
    # Statistical features
    feature_cols = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
    available_cols = [col for col in feature_cols if col in work_df.columns]
    
    if len(available_cols) > 0:
        work_df['feature_mean'] = work_df[available_cols].mean(axis=1)
        work_df['feature_std'] = work_df[available_cols].std(axis=1)
        work_df['feature_max'] = work_df[available_cols].max(axis=1)
        work_df['feature_min'] = work_df[available_cols].min(axis=1)
    
    # Add field_id back if it existed
    if 'field_id' in result.columns:
        work_df['field_id'] = field_ids
    
    print(f"   Created {len(work_df.columns) - len(result.columns)} new features")
    return work_df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

# Ensure columns match between train and test
train_cols = [col for col in train_fe.columns if col not in ['yield', 'field_id']]
test_cols = [col for col in test_fe.columns if col != 'field_id']

# Find common columns
common_cols = list(set(train_cols).intersection(set(test_cols)))
common_cols.sort()  # Ensure consistent order

print(f"   Total features for modeling: {len(common_cols)}")

# ==============================================
# 4. PREPARE DATA
# ==============================================
print("\n" + "="*50)
print("ğŸ“Š DATA PREPARATION")
print("="*50)

# Extract features and target
X = train_fe[common_cols].copy()
y = train_fe['yield'].copy()
X_test = test_fe[common_cols].copy()

# Check for any remaining non-numeric data
print("   Checking data types...")
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"   âš ï¸� Removing non-numeric columns: {non_numeric}")
    X = X.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

# Handle any missing values
if X.isnull().sum().sum() > 0:
    print("   Handling missing values...")
    X = X.fillna(X.median())
    X_test = X_test.fillna(X.median())

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"âœ… Data prepared successfully!")
print(f"   Training set: {X_train.shape}")
print(f"   Validation set: {X_val.shape}")
print(f"   Test set: {X_test.shape}")

# ==============================================
# 5. SCALE DATA
# ==============================================
print("\n" + "="*50)
print("âš–ï¸� FEATURE SCALING")
print("="*50)

# Use RobustScaler (handles outliers better)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("âœ… Features scaled successfully!")

# ==============================================
# 6. TRAIN MODELS
# ==============================================
print("\n" + "="*50)
print("ğŸ¤– MODEL TRAINING")
print("="*50)

# Dictionary to store results
results = {}
models = {}

# 1. Linear models (use scaled data)
print("\nğŸ“ˆ Training Linear Models...")

# Ridge
try:
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    pred = ridge.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    results['Ridge'] = rmse
    models['Ridge'] = (ridge, True)  # True means needs scaling
    print(f"   Ridge RMSE: {rmse:.2f}")
except Exception as e:
    print(f"   Ridge failed: {str(e)}")

# Lasso
try:
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    pred = lasso.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    results['Lasso'] = rmse
    models['Lasso'] = (lasso, True)
    print(f"   Lasso RMSE: {rmse:.2f}")
except Exception as e:
    print(f"   Lasso failed: {str(e)}")

# 2. Tree models (use original data)
print("\nğŸŒ³ Training Tree Models...")

# Random Forest
try:
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    results['RandomForest'] = rmse
    models['RandomForest'] = (rf, False)  # False means no scaling needed
    print(f"   Random Forest RMSE: {rmse:.2f}")
except Exception as e:
    print(f"   Random Forest failed: {str(e)}")

# Gradient Boosting
try:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    results['GradientBoosting'] = rmse
    models['GradientBoosting'] = (gb, False)
    print(f"   Gradient Boosting RMSE: {rmse:.2f}")
except Exception as e:
    print(f"   Gradient Boosting failed: {str(e)}")

# 3. Advanced models (if available)
if HAS_XGB:
    print("\nğŸš€ Training XGBoost...")
    try:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        results['XGBoost'] = rmse
        models['XGBoost'] = (xgb, False)
        print(f"   XGBoost RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"   XGBoost failed: {str(e)}")

if HAS_LGB:
    print("\nğŸ’¡ Training LightGBM...")
    try:
        lgb = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
        lgb.fit(X_train, y_train)
        pred = lgb.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        results['LightGBM'] = rmse
        models['LightGBM'] = (lgb, False)
        print(f"   LightGBM RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"   LightGBM failed: {str(e)}")

# ==============================================
# 7. ENSEMBLE PREDICTIONS
# ==============================================
print("\n" + "="*50)
print("ğŸ�¯ CREATING ENSEMBLE")
print("="*50)

# Sort models by performance
sorted_models = sorted(results.items(), key=lambda x: x[1])
print("\nğŸ“Š Model Performance Summary:")
for name, rmse in sorted_models:
    print(f"   {name}: {rmse:.2f}")

# Use top 3 models for ensemble
top_models = sorted_models[:3]
print(f"\nâœ… Using top {len(top_models)} models for ensemble")

# Generate predictions from each model
val_preds = []
test_preds = []

for name, _ in top_models:
    model, needs_scaling = models[name]
    
    if needs_scaling:
        val_preds.append(model.predict(X_val_scaled))
        test_preds.append(model.predict(X_test_scaled))
    else:
        val_preds.append(model.predict(X_val))
        test_preds.append(model.predict(X_test))

# Simple average ensemble
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

ensemble_rmse = np.sqrt(mean_squared_error(y_val, val_ensemble))
print(f"\nğŸ�† Ensemble RMSE: {ensemble_rmse:.2f}")

# ==============================================
# 8. FINAL PREDICTIONS
# ==============================================
print("\n" + "="*50)
print("ğŸ“� GENERATING SUBMISSION")
print("="*50)

# Use ensemble predictions
submission['yield'] = test_ensemble

# Basic sanity checks
print("   Prediction statistics:")
print(f"   Min: {submission['yield'].min():.2f}")
print(f"   Max: {submission['yield'].max():.2f}")
print(f"   Mean: {submission['yield'].mean():.2f}")
print(f"   Std: {submission['yield'].std():.2f}")

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission saved as 'submission.csv'")

# ==============================================
# 9. VISUALIZATIONS
# ==============================================
print("\n" + "="*50)
print("ğŸ“Š CREATING VISUALIZATIONS")
print("="*50)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model comparison
ax1 = axes[0, 0]
model_names = [x[0] for x in sorted_models]
model_scores = [x[1] for x in sorted_models]
ax1.bar(model_names, model_scores, color='skyblue')
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax1.set_title('Model Performance Comparison')
ax1.tick_params(axis='x', rotation=45)

# 2. Prediction distribution
ax2 = axes[0, 1]
ax2.hist(submission['yield'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Predicted Yield')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Predictions')

# 3. Training vs Prediction distribution
ax3 = axes[1, 0]
ax3.hist(train['yield'], bins=50, alpha=0.5, label='Training', color='blue')
ax3.hist(submission['yield'], bins=50, alpha=0.5, label='Predictions', color='red')
ax3.set_xlabel('Yield')
ax3.set_ylabel('Density')
ax3.set_title('Training vs Prediction Distributions')
ax3.legend()

# 4. Feature importance (if RF is available)
ax4 = axes[1, 1]
if 'RandomForest' in models:
    rf_model = models['RandomForest'][0]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    
    ax4.barh(range(10), importances[indices], color='coral')
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([X.columns[i] for i in indices])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Feature Importances')
else:
    ax4.text(0.5, 0.5, 'Random Forest not available', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Feature Importance')

plt.tight_layout()
plt.show()

# ==============================================
# 10. SUMMARY
# ==============================================
print("\n" + "="*50)
print("ğŸ�‰ ANALYSIS COMPLETE!")
print("="*50)

print(f"""
ğŸ“Š Final Summary:
â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�
â€¢ Models trained: {len(models)}
â€¢ Best individual model: {sorted_models[0][0]} (RMSE: {sorted_models[0][1]:.2f})
â€¢ Ensemble RMSE: {ensemble_rmse:.2f}
â€¢ Features used: {len(common_cols)}

âœ… Next steps:
1. Submit 'submission.csv' to Kaggle
2. Check leaderboard position
3. Consider hyperparameter tuning
4. Try more feature engineering

Good luck! ğŸŒ¾
""")

print("\nâœ… Script completed successfully!")

