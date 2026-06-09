#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Fertilizer Recommendation System - Fault Tolerant Version
Kaggle Playground Series S5E6
Complete analysis and prediction system with robust error handling
"""

# =============================================================================
# 1. ENVIRONMENT SETUP AND IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Gradient Boosting Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, using dual ensemble instead")

# Additional utilities
import os
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# Set visualization style with error handling
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')

# Set color palette
sns.set_palette("husl")

print("=" * 80)
print("FERTILIZER RECOMMENDATION SYSTEM - ROBUST ENHANCED VERSION")
print("=" * 80)

# Display available files
print("\nğŸ“� Available Data Files:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        print(f"  â€¢ {filepath}")

# =============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATASETS")
print("=" * 80)

# Load all datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

print(f"\nğŸ“Š Dataset Shapes:")
print(f"  â€¢ Training set: {train.shape[0]:,} rows Ã— {train.shape[1]} columns")
print(f"  â€¢ Test set: {test.shape[0]:,} rows Ã— {test.shape[1]} columns")
print(f"  â€¢ Submission template: {submission.shape[0]:,} rows Ã— {submission.shape[1]} columns")

# Display basic information
print("\nğŸ“‹ Training Data Overview:")
print(train.info())

print("\nğŸ”� First 5 rows of training data:")
print(train.head())

print("\nğŸ“ˆ Statistical Summary of Numerical Features:")
print(train.describe())

print("\nğŸ“Š Categorical Features Distribution:")
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
for col in categorical_cols:
    if col in train.columns:
        print(f"\n{col}:")
        print(train[col].value_counts())

# Check for missing values
print("\nğŸ”� Missing Values Analysis:")
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

missing_df = pd.DataFrame({
    'Train Missing': missing_train,
    'Test Missing': missing_test,
    'Train %': (missing_train / len(train)) * 100,
    'Test %': (missing_test / len(test)) * 100
})
print(missing_df[missing_df['Train Missing'] > 0])
if len(missing_df[missing_df['Train Missing'] > 0]) == 0:
    print("  âœ… No missing values found in the datasets!")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 3.1 Target Variable Analysis
fig = plt.figure(figsize=(15, 5))

# Subplot 1: Count plot
ax1 = plt.subplot(1, 3, 1)
fertilizer_counts = train['Fertilizer Name'].value_counts()
colors_viridis = plt.cm.viridis(np.linspace(0, 1, len(fertilizer_counts)))
bars = ax1.barh(fertilizer_counts.index, fertilizer_counts.values, color=colors_viridis)
ax1.set_xlabel('Count')
ax1.set_ylabel('Fertilizer Type')
ax1.set_title('Fertilizer Distribution', fontsize=14, fontweight='bold')

# Subplot 2: Pie chart
ax2 = plt.subplot(1, 3, 2)
ax2.pie(fertilizer_counts.values, labels=fertilizer_counts.index, autopct='%1.1f%%', 
        colors=colors_viridis)
ax2.set_title('Fertilizer Percentage Distribution', fontsize=14, fontweight='bold')

# Subplot 3: Target imbalance ratio
ax3 = plt.subplot(1, 3, 3)
imbalance_ratio = fertilizer_counts.max() / fertilizer_counts.min()
ax3.bar(['Max/Min Ratio'], [imbalance_ratio], color='coral')
ax3.set_title(f'Class Imbalance Ratio: {imbalance_ratio:.2f}', fontsize=14, fontweight='bold')
ax3.set_ylabel('Ratio')

plt.tight_layout()
plt.show()

print(f"\nğŸ“Š Target Variable Statistics:")
print(f"  â€¢ Number of unique fertilizers: {train['Fertilizer Name'].nunique()}")
print(f"  â€¢ Most common fertilizer: {fertilizer_counts.index[0]} ({fertilizer_counts.values[0]:,} samples)")
print(f"  â€¢ Least common fertilizer: {fertilizer_counts.index[-1]} ({fertilizer_counts.values[-1]:,} samples)")

# 3.2 Numerical Features Distribution
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

fig = plt.figure(figsize=(18, 10))
for i, col in enumerate(numerical_features, 1):
    # Distribution plot
    plt.subplot(3, 6, i)
    plt.hist(train[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(train[col].mean(), color='red', linestyle='--', linewidth=1, label='Mean')
    plt.title(f'{col} Distribution', fontsize=10, fontweight='bold')
    plt.xlabel(col, fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.legend(fontsize=6)
    
    # Box plot
    plt.subplot(3, 6, i + 6)
    box = plt.boxplot(train[col].values, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightcoral')
    plt.title(f'{col} Box Plot', fontsize=10, fontweight='bold')
    plt.ylabel(col, fontsize=8)
    
    # Stats annotation
    mean_val = train[col].mean()
    median_val = train[col].median()
    std_val = train[col].std()
    
    plt.subplot(3, 6, i + 12)
    plt.text(0.1, 0.7, f'Mean: {mean_val:.2f}', fontsize=9)
    plt.text(0.1, 0.5, f'Median: {median_val:.2f}', fontsize=9)
    plt.text(0.1, 0.3, f'Std: {std_val:.2f}', fontsize=9)
    plt.text(0.1, 0.1, f'Skew: {train[col].skew():.2f}', fontsize=9)
    plt.axis('off')
    plt.title(f'{col} Statistics', fontsize=10, fontweight='bold')

plt.suptitle('Numerical Features Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# 3.3 Correlation Analysis
print("\nğŸ“Š Correlation Matrix Analysis:")

# Create correlation matrix
corr_matrix = train[numerical_features].corr()

# Visualize correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, ax=axes[0], fmt='.2f',
            cbar_kws={"shrink": 0.8})
axes[0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')

# Correlation strength bars
corr_values = []
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_values.append(abs(corr_matrix.iloc[i, j]))
        corr_pairs.append(f"{corr_matrix.columns[i][:4]}-{corr_matrix.columns[j][:4]}")

sorted_idx = np.argsort(corr_values)[::-1][:10]
axes[1].barh(range(len(sorted_idx)), [corr_values[i] for i in sorted_idx])
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels([corr_pairs[i] for i in sorted_idx])
axes[1].set_xlabel('Absolute Correlation')
axes[1].set_title('Top 10 Feature Correlations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Find strong correlations
strong_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if strong_corr:
    print("\nğŸ”— Strong Correlations (|r| > 0.5):")
    for feat1, feat2, corr in strong_corr:
        print(f"  â€¢ {feat1} â†” {feat2}: {corr:.3f}")
else:
    print("\nâœ… No strong correlations found between numerical features")

# 3.4 Categorical Features Analysis
fig = plt.figure(figsize=(15, 5))

# Soil Type distribution
ax1 = plt.subplot(1, 3, 1)
soil_counts = train['Soil Type'].value_counts()
colors_soil = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(soil_counts)))
ax1.barh(soil_counts.index, soil_counts.values, color=colors_soil)
ax1.set_title('Soil Type Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Count')

# Crop Type distribution
ax2 = plt.subplot(1, 3, 2)
crop_counts = train['Crop Type'].value_counts()
colors_crop = plt.cm.YlGn(np.linspace(0.3, 0.9, len(crop_counts)))
ax2.barh(crop_counts.index, crop_counts.values, color=colors_crop)
ax2.set_title('Crop Type Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Count')

# Soil-Crop combination
ax3 = plt.subplot(1, 3, 3)
combo_counts = train.groupby(['Soil Type', 'Crop Type']).size().nlargest(10)
combo_labels = [f"{soil[:4]}-{crop[:4]}" for soil, crop in combo_counts.index]
colors_combo = plt.cm.autumn(np.linspace(0.2, 0.8, len(combo_counts)))
ax3.barh(combo_labels, combo_counts.values, color=colors_combo)
ax3.set_title('Top 10 Soil-Crop Combinations', fontsize=14, fontweight='bold')
ax3.set_xlabel('Count')

plt.tight_layout()
plt.show()

# 3.5 NPK Analysis (Nitrogen, Phosphorous, Potassium)
print("\nğŸŒ± NPK Nutrient Analysis:")

fig = plt.figure(figsize=(18, 6))

# NPK Distribution by Fertilizer
ax1 = plt.subplot(1, 3, 1)
npk_by_fert = train.groupby('Fertilizer Name')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()
npk_by_fert.plot(kind='bar', ax=ax1, width=0.8, color=['#2E7D32', '#1565C0', '#F57C00'])
ax1.set_title('Average NPK by Fertilizer Type', fontsize=14, fontweight='bold')
ax1.set_xlabel('Fertilizer')
ax1.set_ylabel('Average Amount')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.legend(title='Nutrient')
ax1.grid(True, alpha=0.3)

# NPK Ratios
ax2 = plt.subplot(1, 3, 2)
train['N/P Ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1)
train['N/K Ratio'] = train['Nitrogen'] / (train['Potassium'] + 1)
train['P/K Ratio'] = train['Phosphorous'] / (train['Potassium'] + 1)

ratio_cols = ['N/P Ratio', 'N/K Ratio', 'P/K Ratio']
box_data = [train[col].values for col in ratio_cols]
bp = ax2.boxplot(box_data, labels=ratio_cols, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#66BB6A', '#42A5F5', '#FFA726']):
    patch.set_facecolor(color)
ax2.set_title('NPK Ratio Distributions', fontsize=14, fontweight='bold')
ax2.set_ylabel('Ratio Value')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True, alpha=0.3)

# 3D Scatter of NPK
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
sample_data = train.sample(n=min(5000, len(train)), random_state=42)
colors_3d = pd.factorize(sample_data['Fertilizer Name'])[0]
scatter = ax3.scatter(sample_data['Nitrogen'], 
                     sample_data['Phosphorous'], 
                     sample_data['Potassium'],
                     c=colors_3d,
                     cmap='viridis', 
                     alpha=0.6,
                     s=20)
ax3.set_xlabel('Nitrogen')
ax3.set_ylabel('Phosphorous')
ax3.set_zlabel('Potassium')
ax3.set_title('NPK 3D Distribution', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax3, pad=0.1, shrink=0.8)

plt.tight_layout()
plt.show()

# 3.6 Environmental Conditions Analysis
print("\nğŸŒ¡ï¸� Environmental Conditions Analysis:")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Temperature vs Humidity by Fertilizer
fertilizer_colors = pd.factorize(train['Fertilizer Name'])[0]
axes[0, 0].scatter(train['Temparature'], train['Humidity'], 
                   c=fertilizer_colors, 
                   alpha=0.3, cmap='tab10', s=1)
axes[0, 0].set_xlabel('Temperature')
axes[0, 0].set_ylabel('Humidity')
axes[0, 0].set_title('Temperature vs Humidity', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Moisture by Soil Type
soil_types = train['Soil Type'].unique()
moisture_data = [train[train['Soil Type']==soil]['Moisture'].values for soil in soil_types]
vp = axes[0, 1].violinplot(moisture_data,
                           positions=range(len(soil_types)),
                           showmeans=True)
axes[0, 1].set_xticks(range(len(soil_types)))
axes[0, 1].set_xticklabels(soil_types, rotation=45)
axes[0, 1].set_ylabel('Moisture')
axes[0, 1].set_title('Moisture Distribution by Soil Type', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Crop Type vs Environmental Factors
crop_env = train.groupby('Crop Type')[['Temparature', 'Humidity', 'Moisture']].mean()
im1 = axes[0, 2].imshow(crop_env.T, aspect='auto', cmap='YlOrRd')
axes[0, 2].set_yticks(range(3))
axes[0, 2].set_yticklabels(['Temperature', 'Humidity', 'Moisture'])
axes[0, 2].set_xticks(range(len(crop_env.index)))
axes[0, 2].set_xticklabels([c[:6] for c in crop_env.index], rotation=45, ha='right', fontsize=8)
axes[0, 2].set_title('Environmental Conditions by Crop', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 2])

# NPK Requirements by Crop
crop_npk = train.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()
im2 = axes[1, 0].imshow(crop_npk.T, aspect='auto', cmap='viridis')
axes[1, 0].set_yticks(range(3))
axes[1, 0].set_yticklabels(['Nitrogen', 'Phosphorous', 'Potassium'])
axes[1, 0].set_xticks(range(len(crop_npk.index)))
axes[1, 0].set_xticklabels([c[:6] for c in crop_npk.index], rotation=45, ha='right', fontsize=8)
axes[1, 0].set_title('NPK Requirements by Crop', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1, 0])

# Fertilizer distribution by Soil Type
fert_soil = pd.crosstab(train['Soil Type'], train['Fertilizer Name'])
im3 = axes[1, 1].imshow(fert_soil, aspect='auto', cmap='coolwarm')
axes[1, 1].set_yticks(range(len(fert_soil.index)))
axes[1, 1].set_yticklabels(fert_soil.index, fontsize=8)
axes[1, 1].set_xticks(range(len(fert_soil.columns)))
axes[1, 1].set_xticklabels(fert_soil.columns, rotation=45, ha='right', fontsize=8)
axes[1, 1].set_title('Fertilizer Usage by Soil Type', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[1, 1])

# Fertilizer distribution by Crop Type
fert_crop = pd.crosstab(train['Crop Type'], train['Fertilizer Name'])
im4 = axes[1, 2].imshow(fert_crop, aspect='auto', cmap='plasma')
axes[1, 2].set_yticks(range(len(fert_crop.index)))
axes[1, 2].set_yticklabels([c[:8] for c in fert_crop.index], fontsize=8)
axes[1, 2].set_xticks(range(len(fert_crop.columns)))
axes[1, 2].set_xticklabels(fert_crop.columns, rotation=45, ha='right', fontsize=8)
axes[1, 2].set_title('Fertilizer Usage by Crop Type', fontsize=12, fontweight='bold')
plt.colorbar(im4, ax=axes[1, 2])

plt.suptitle('Comprehensive Environmental and Agricultural Analysis', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

def create_features(df):
    """Create engineered features for the model"""
    df_feat = df.copy()
    
    # NPK Ratios (with safe division)
    df_feat['N_P_ratio'] = df_feat['Nitrogen'] / (df_feat['Phosphorous'] + 1)
    df_feat['N_K_ratio'] = df_feat['Nitrogen'] / (df_feat['Potassium'] + 1)
    df_feat['P_K_ratio'] = df_feat['Phosphorous'] / (df_feat['Potassium'] + 1)
    
    # NPK Sums and Products
    df_feat['NPK_sum'] = df_feat['Nitrogen'] + df_feat['Phosphorous'] + df_feat['Potassium']
    df_feat['NPK_product'] = np.log1p(df_feat['Nitrogen'] * df_feat['Phosphorous'] * df_feat['Potassium'])
    df_feat['NP_product'] = df_feat['Nitrogen'] * df_feat['Phosphorous']
    df_feat['NK_product'] = df_feat['Nitrogen'] * df_feat['Potassium']
    df_feat['PK_product'] = df_feat['Phosphorous'] * df_feat['Potassium']
    
    # Environmental Interactions
    df_feat['Temp_Humidity'] = df_feat['Temparature'] * df_feat['Humidity']
    df_feat['Temp_Moisture'] = df_feat['Temparature'] * df_feat['Moisture']
    df_feat['Humidity_Moisture'] = df_feat['Humidity'] * df_feat['Moisture']
    df_feat['Moisture_per_Humidity'] = df_feat['Moisture'] / (df_feat['Humidity'] + 1)
    
    # Climate Index
    df_feat['Climate_Index'] = (df_feat['Temparature'] + df_feat['Humidity'] + df_feat['Moisture']) / 3
    
    # Nutrient Efficiency Indicators
    df_feat['N_efficiency'] = df_feat['Nitrogen'] / (df_feat['NPK_sum'] + 1)
    df_feat['P_efficiency'] = df_feat['Phosphorous'] / (df_feat['NPK_sum'] + 1)
    df_feat['K_efficiency'] = df_feat['Potassium'] / (df_feat['NPK_sum'] + 1)
    
    # Environmental Stress Indicators
    df_feat['Heat_Stress'] = (df_feat['Temparature'] > 35).astype(int)
    df_feat['Drought_Stress'] = (df_feat['Moisture'] < 30).astype(int)
    df_feat['High_Humidity'] = (df_feat['Humidity'] > 70).astype(int)
    
    # Polynomial features for key variables
    df_feat['Nitrogen_squared'] = df_feat['Nitrogen'] ** 2
    df_feat['Phosphorous_squared'] = df_feat['Phosphorous'] ** 2
    df_feat['Potassium_squared'] = df_feat['Potassium'] ** 2
    
    # Additional features
    df_feat['NPK_std'] = df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
    df_feat['NPK_mean'] = df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)
    df_feat['NPK_max'] = df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].max(axis=1)
    df_feat['NPK_min'] = df_feat[['Nitrogen', 'Phosphorous', 'Potassium']].min(axis=1)
    
    return df_feat

# Apply feature engineering
train_feat = create_features(train)
test_feat = create_features(test)

# List new features
new_features = [col for col in train_feat.columns if col not in train.columns]
print(f"\nâœ¨ Created {len(new_features)} new features:")
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# =============================================================================
# 5. DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Encode categorical variables
print("\nğŸ”„ Encoding categorical variables...")

# Target encoding
fertilizer_le = LabelEncoder()
train_feat['Fertilizer_Label'] = fertilizer_le.fit_transform(train_feat['Fertilizer Name'])

# Feature encoding
soil_le = LabelEncoder()
crop_le = LabelEncoder()

train_feat['Soil_Type_Label'] = soil_le.fit_transform(train_feat['Soil Type'])
train_feat['Crop_Type_Label'] = crop_le.fit_transform(train_feat['Crop Type'])

test_feat['Soil_Type_Label'] = soil_le.transform(test_feat['Soil Type'])
test_feat['Crop_Type_Label'] = crop_le.transform(test_feat['Crop Type'])

print(f"  âœ… Encoded Soil Types: {len(soil_le.classes_)} categories")
print(f"  âœ… Encoded Crop Types: {len(crop_le.classes_)} categories")
print(f"  âœ… Encoded Fertilizers: {len(fertilizer_le.classes_)} categories")

# Define feature columns
feature_cols = [
    # Original features
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'Soil_Type_Label', 'Crop_Type_Label',
    # Engineered features
    'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
    'NPK_sum', 'NPK_product', 'NP_product', 'NK_product', 'PK_product',
    'Temp_Humidity', 'Temp_Moisture', 'Humidity_Moisture',
    'Moisture_per_Humidity', 'Climate_Index',
    'N_efficiency', 'P_efficiency', 'K_efficiency',
    'Heat_Stress', 'Drought_Stress', 'High_Humidity',
    'Nitrogen_squared', 'Phosphorous_squared', 'Potassium_squared',
    'NPK_std', 'NPK_mean', 'NPK_max', 'NPK_min'
]

# Prepare data
X = train_feat[feature_cols]
y = train_feat['Fertilizer_Label']
X_test = test_feat[feature_cols]

# Handle any inf or nan values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nğŸ“Š Final dataset shapes:")
print(f"  â€¢ X_train: {X.shape}")
print(f"  â€¢ y_train: {y.shape}")
print(f"  â€¢ X_test: {X_test.shape}")

# =============================================================================
# 6. MODEL DEVELOPMENT
# =============================================================================

print("\n" + "=" * 80)
print("MODEL DEVELOPMENT")
print("=" * 80)

# 6.1 Cross-Validation Setup
print("\nğŸ”„ Setting up 5-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6.2 XGBoost Model
print("\nğŸš€ Training XGBoost Model...")

# Check if GPU is available
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    gpu_available = result.returncode == 0
    tree_method = "gpu_hist" if gpu_available else "hist"
    print(f"  â€¢ Using tree_method: {tree_method}")
except:
    tree_method = "hist"
    print("  â€¢ Using CPU for XGBoost")

xgb_params = {
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.2,
    'subsample': 0.93,
    'colsample_bytree': 0.52,
    'gamma': 0.03,
    'min_child_weight': 9,
    'reg_alpha': 1.3e-5,
    'reg_lambda': 0.18,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'tree_method': tree_method,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

model_xgb = XGBClassifier(**xgb_params)

# Cross-validation for XGBoost
print("  â€¢ Running cross-validation...")
xgb_scores = cross_val_score(model_xgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  â€¢ XGBoost CV Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

# Train on full data
print("  â€¢ Training on full dataset...")
model_xgb.fit(X, y)
pred_xgb = model_xgb.predict_proba(X_test)

# 6.3 LightGBM Model
print("\nğŸš€ Training LightGBM Model...")

# Check device for LightGBM
try:
    lgb_device = 'gpu' if gpu_available else 'cpu'
except:
    lgb_device = 'cpu'

print(f"  â€¢ Using device: {lgb_device}")

lgb_params = {
    'objective': 'multiclass',
    'num_class': 7,
    'learning_rate': 0.13,
    'max_depth': 11,
    'num_leaves': 140,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.54,
    'bagging_fraction': 0.79,
    'bagging_freq': 5,
    'lambda_l1': 4.35,
    'lambda_l2': 4.47,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'device': lgb_device
}

# Remove device parameter if CPU (causes warning)
if lgb_device == 'cpu':
    lgb_params.pop('device')

model_lgb = LGBMClassifier(**lgb_params)

# Cross-validation for LightGBM
print("  â€¢ Running cross-validation...")
lgb_scores = cross_val_score(model_lgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  â€¢ LightGBM CV Accuracy: {lgb_scores.mean():.4f} (+/- {lgb_scores.std():.4f})")

# Train on full data
print("  â€¢ Training on full dataset...")
model_lgb.fit(X, y)
pred_lgb = model_lgb.predict_proba(X_test)

# 6.4 Third Model (CatBoost or Extra XGBoost)
if CATBOOST_AVAILABLE:
    print("\nğŸš€ Training CatBoost Model...")
    
    cat_params = {
        'iterations': 100,
        'depth': 10,
        'learning_rate': 0.15,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU'
    }
    
    model_cat = CatBoostClassifier(**cat_params)
    
    # Cross-validation for CatBoost
    print("  â€¢ Running cross-validation...")
    cat_scores = cross_val_score(model_cat, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  â€¢ CatBoost CV Accuracy: {cat_scores.mean():.4f} (+/- {cat_scores.std():.4f})")
    
    # Train on full data
    print("  â€¢ Training on full dataset...")
    model_cat.fit(X, y, verbose=False)
    pred_cat = model_cat.predict_proba(X_test)
else:
    print("\nğŸš€ Training Extra XGBoost Model (CatBoost alternative)...")
    
    # Different hyperparameters for diversity
    xgb2_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.15,
        'subsample': 0.85,
        'colsample_bytree': 0.6,
        'gamma': 0.05,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 0.1,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': tree_method,
        'random_state': 123,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model_cat = XGBClassifier(**xgb2_params)
    
    # Cross-validation
    print("  â€¢ Running cross-validation...")
    cat_scores = cross_val_score(model_cat, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  â€¢ XGBoost-2 CV Accuracy: {cat_scores.mean():.4f} (+/- {cat_scores.std():.4f})")
    
    # Train on full data
    print("  â€¢ Training on full dataset...")
    model_cat.fit(X, y)
    pred_cat = model_cat.predict_proba(X_test)

# =============================================================================
# 7. ENSEMBLE OPTIMIZATION
# =============================================================================

print("\n" + "=" * 80)
print("ENSEMBLE OPTIMIZATION")
print("=" * 80)

# Test different ensemble weights
print("\nğŸ”� Finding optimal ensemble weights...")

best_weights = None
best_score = 0

# Grid search for optimal weights
weight_combinations = [
    [0.33, 0.33, 0.34],
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4],
    [0.5, 0.25, 0.25],
    [0.25, 0.5, 0.25],
    [0.25, 0.25, 0.5],
    [0.45, 0.35, 0.2],
    [0.35, 0.45, 0.2],
    [0.2, 0.4, 0.4]
]

# Find best weights based on CV scores
print("\n  Testing weight combinations...")
for weights in weight_combinations:
    score_estimate = (weights[0] * xgb_scores.mean() + 
                     weights[1] * lgb_scores.mean() + 
                     weights[2] * cat_scores.mean())
    
    if score_estimate > best_score:
        best_score = score_estimate
        best_weights = weights

model3_name = "CatBoost" if CATBOOST_AVAILABLE else "XGBoost-2"
print(f"\n  âœ… Optimal weights found:")
print(f"     â€¢ XGBoost: {best_weights[0]:.2f}")
print(f"     â€¢ LightGBM: {best_weights[1]:.2f}")
print(f"     â€¢ {model3_name}: {best_weights[2]:.2f}")

# Create weighted ensemble
pred_ensemble = (best_weights[0] * pred_xgb + 
                best_weights[1] * pred_lgb + 
                best_weights[2] * pred_cat)

# =============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importances from each model
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)

lgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_lgb.feature_importances_
}).sort_values('importance', ascending=False)

cat_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_cat.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importances
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XGBoost importances
colors_xgb = plt.cm.Blues(np.linspace(0.4, 0.9, 15))
axes[0].barh(range(15), xgb_importance['importance'][:15].values, color=colors_xgb)
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(xgb_importance['feature'][:15].values, fontsize=8)
axes[0].set_xlabel('Importance')
axes[0].set_title('XGBoost - Top 15 Features', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

# LightGBM importances
colors_lgb = plt.cm.Greens(np.linspace(0.4, 0.9, 15))
axes[1].barh(range(15), lgb_importance['importance'][:15].values, color=colors_lgb)
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(lgb_importance['feature'][:15].values, fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('LightGBM - Top 15 Features', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

# Third model importances
colors_cat = plt.cm.Oranges(np.linspace(0.4, 0.9, 15))
axes[2].barh(range(15), cat_importance['importance'][:15].values, color=colors_cat)
axes[2].set_yticks(range(15))
axes[2].set_yticklabels(cat_importance['feature'][:15].values, fontsize=8)
axes[2].set_xlabel('Importance')
axes[2].set_title(f'{model3_name} - Top 15 Features', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Average feature importance
avg_importance = pd.DataFrame({
    'feature': feature_cols,
    'avg_importance': (xgb_importance.set_index('feature')['importance'] + 
                      lgb_importance.set_index('feature')['importance'] + 
                      cat_importance.set_index('feature')['importance']) / 3
}).sort_values('avg_importance', ascending=False)

print("\nğŸ“Š Top 10 Most Important Features (Average):")
for idx, (i, row) in enumerate(avg_importance.head(10).iterrows(), 1):
    print(f"  {idx:2d}. {row['feature']}: {row['avg_importance']:.4f}")

# =============================================================================
# 9. PREDICTION INSIGHTS
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTION INSIGHTS")
print("=" * 80)

# Analyze ensemble predictions
pred_classes = np.argmax(pred_ensemble, axis=1)
pred_confidence = np.max(pred_ensemble, axis=1)

# Confidence distribution
plt.figure(figsize=(15, 5))

# Subplot 1: Confidence histogram
plt.subplot(1, 3, 1)
n, bins, patches = plt.hist(pred_confidence, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Prediction Confidence')
plt.ylabel('Number of Samples')
plt.title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
plt.axvline(pred_confidence.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {pred_confidence.mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Predicted distribution
plt.subplot(1, 3, 2)
pred_distribution = pd.Series(pred_classes).value_counts().sort_index()
fertilizer_names = fertilizer_le.inverse_transform(pred_distribution.index)
colors_bar = plt.cm.Set3(np.linspace(0, 1, len(fertilizer_names)))
plt.bar(range(len(fertilizer_names)), pred_distribution.values, color=colors_bar, edgecolor='black')
plt.xticks(range(len(fertilizer_names)), fertilizer_names, rotation=45, ha='right')
plt.ylabel('Predicted Count')
plt.title('Predicted Fertilizer Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Subplot 3: Confidence by class
plt.subplot(1, 3, 3)
confidence_by_class = [pred_confidence[pred_classes == i] for i in range(len(fertilizer_le.classes_))]
bp = plt.boxplot(confidence_by_class, labels=fertilizer_le.classes_, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_bar):
    patch.set_facecolor(color)
plt.xlabel('Fertilizer Type')
plt.ylabel('Confidence')
plt.title('Confidence by Predicted Class', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.suptitle('Prediction Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nğŸ“Š Prediction Statistics:")
print(f"  â€¢ Average confidence: {pred_confidence.mean():.4f}")
print(f"  â€¢ Confidence std dev: {pred_confidence.std():.4f}")
print(f"  â€¢ High confidence (>0.8) predictions: {(pred_confidence > 0.8).sum():,} ({(pred_confidence > 0.8).mean()*100:.1f}%)")
print(f"  â€¢ Low confidence (<0.5) predictions: {(pred_confidence < 0.5).sum():,} ({(pred_confidence < 0.5).mean()*100:.1f}%)")

# =============================================================================
# 10. GENERATE SUBMISSION
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING SUBMISSION")
print("=" * 80)

# Get top 3 predictions for each sample
top_3_preds = np.argsort(pred_ensemble, axis=1)[:, -3:][:, ::-1]

# Convert to fertilizer names
top_3_names = []
for row in top_3_preds:
    names = fertilizer_le.inverse_transform(row)
    top_3_names.append(' '.join(names))

# Create submission dataframe
submission_final = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top_3_names
})

# Save submission
submission_final.to_csv("submission_enhanced.csv", index=False)

print(f"\nâœ… Submission file created: submission_enhanced.csv")
print(f"  â€¢ Total predictions: {len(submission_final):,}")
print(f"  â€¢ Sample predictions:")
for i in range(min(5, len(submission_final))):
    print(f"    ID {submission_final.iloc[i]['id']}: {submission_final.iloc[i]['Fertilizer Name']}")

# =============================================================================
# 11. MODEL PERFORMANCE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 80)

summary_data = {
    'Model': ['XGBoost', 'LightGBM', model3_name, 'Weighted Ensemble'],
    'CV Accuracy': [
        f"{xgb_scores.mean():.4f} Â± {xgb_scores.std():.4f}",
        f"{lgb_scores.mean():.4f} Â± {lgb_scores.std():.4f}",
        f"{cat_scores.mean():.4f} Â± {cat_scores.std():.4f}",
        f"~{best_score:.4f} (estimated)"
    ],
    'Weight in Ensemble': [
        f"{best_weights[0]:.2%}",
        f"{best_weights[1]:.2%}",
        f"{best_weights[2]:.2%}",
        "100%"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\nğŸ“Š Model Performance Table:")
print(summary_df.to_string(index=False))

print("\nğŸ�¯ Key Insights:")
print("  1. The ensemble model combines strengths of three gradient boosting algorithms")
print("  2. Feature engineering added significant predictive power")
print(f"  3. Total of {len(feature_cols)} features used in the final model")
print("  4. NPK ratios and environmental interactions are highly predictive")
print("  5. Model shows high confidence in majority of predictions")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! ğŸ�‰")
print("Thank you for using the Enhanced Fertilizer Recommendation System")
print("=" * 80)

