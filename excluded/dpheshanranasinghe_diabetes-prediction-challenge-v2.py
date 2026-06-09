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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
# (though pandas has built-in .corr() which is usually enough)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# 1. Load your data first
df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')  

# 1.1 Check Data Distributions

# Basic info about the dataset
print("Dataset Shape:", df.shape)
print("\n" + "="*50)
print("Data Types and Non-Null Counts:")
print(df.info())
print("\n" + "="*50)

# Statistical summary of numerical features
print("Statistical Summary:")
print(df.describe())
print("\n" + "="*50)

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
print("\n" + "="*50)

# Check unique values for categorical columns
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                     'smoking_status', 'employment_status']

print("Categorical Variables - Unique Values:")
for col in categorical_cols:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())
    print("-"*30)


# Visualize distribution of numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('id')  # Remove ID column

# Create subplots for all numerical features
n_cols = 4
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=10)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

# Hide empty subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


# Focus on target variable distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['diagnosed_diabetes'], bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.title('Distribution of Diagnosed Diabetes (Target)', fontsize=12, fontweight='bold')
plt.xlabel('Diabetes Probability')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['diagnosed_diabetes'], vert=True)
plt.title('Boxplot of Diagnosed Diabetes', fontsize=12, fontweight='bold')
plt.ylabel('Diabetes Probability')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nTarget Variable Statistics:")
print(f"Mean: {df['diagnosed_diabetes'].mean():.4f}")
print(f"Median: {df['diagnosed_diabetes'].median():.4f}")
print(f"Std: {df['diagnosed_diabetes'].std():.4f}")
print(f"Min: {df['diagnosed_diabetes'].min():.4f}")
print(f"Max: {df['diagnosed_diabetes'].max():.4f}")


# ============================================================
# FEATURE ENGINEERING & PREPROCESSING
# ============================================================

# Make a copy of the dataframe to preserve original
df_processed = df.copy()

# Remove ID column (not a feature)
df_processed = df_processed.drop('id', axis=1)

print("Starting Feature Engineering...")
print("="*50)

# ============================================================
# 1. HANDLE PHYSICAL ACTIVITY OUTLIERS
# ============================================================
# Cap at 99th percentile to handle extreme outliers
percentile_99 = df_processed['physical_activity_minutes_per_week'].quantile(0.99)
print(f"\n99th percentile for physical activity: {percentile_99:.2f} minutes/week")

df_processed['physical_activity_minutes_per_week'] = df_processed['physical_activity_minutes_per_week'].clip(upper=percentile_99)
print(f"Capped physical activity values above {percentile_99:.2f}")

# ============================================================
# 2. ENCODE CATEGORICAL VARIABLES
# ============================================================

# 2.1 Label Encoding for ORDINAL variables (have natural order)
from sklearn.preprocessing import LabelEncoder

# Income Level (has clear order: Low -> High)
income_mapping = {
    'Low': 0,
    'Lower-Middle': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4
}
df_processed['income_level'] = df_processed['income_level'].map(income_mapping)
print(f"\nâœ“ Income Level: Label encoded (0=Low to 4=High)")

# Education Level (has clear order)
education_mapping = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}
df_processed['education_level'] = df_processed['education_level'].map(education_mapping)
print(f"âœ“ Education Level: Label encoded (0=No formal to 3=Postgraduate)")

# 2.2 One-Hot Encoding for NOMINAL variables (no natural order)
print("\nApplying One-Hot Encoding for nominal variables...")

# Gender
gender_dummies = pd.get_dummies(df_processed['gender'], prefix='gender', drop_first=True)
df_processed = pd.concat([df_processed, gender_dummies], axis=1)
df_processed = df_processed.drop('gender', axis=1)
print(f"âœ“ Gender: One-hot encoded ({len(gender_dummies.columns)} columns)")

# Ethnicity
ethnicity_dummies = pd.get_dummies(df_processed['ethnicity'], prefix='ethnicity', drop_first=True)
df_processed = pd.concat([df_processed, ethnicity_dummies], axis=1)
df_processed = df_processed.drop('ethnicity', axis=1)
print(f"âœ“ Ethnicity: One-hot encoded ({len(ethnicity_dummies.columns)} columns)")

# Smoking Status
smoking_dummies = pd.get_dummies(df_processed['smoking_status'], prefix='smoking', drop_first=True)
df_processed = pd.concat([df_processed, smoking_dummies], axis=1)
df_processed = df_processed.drop('smoking_status', axis=1)
print(f"âœ“ Smoking Status: One-hot encoded ({len(smoking_dummies.columns)} columns)")

# Employment Status
employment_dummies = pd.get_dummies(df_processed['employment_status'], prefix='employment', drop_first=True)
df_processed = pd.concat([df_processed, employment_dummies], axis=1)
df_processed = df_processed.drop('employment_status', axis=1)
print(f"âœ“ Employment Status: One-hot encoded ({len(employment_dummies.columns)} columns)")

print("\n" + "="*50)
print("Encoding Complete!")
print(f"Original features: {len(df.columns)}")
print(f"Processed features: {len(df_processed.columns)}")
print("="*50)

# ============================================================
# 3. CHECK PROCESSED DATA
# ============================================================
print("\nProcessed Data Info:")
print(df_processed.info())
print("\nFirst few rows:")
print(df_processed.head())

# Check column names
print("\nAll column names after preprocessing:")
print(df_processed.columns.tolist())


# ============================================================
# DATA SPLITTING & SCALING
# ============================================================

print("Starting Data Splitting and Scaling...")
print("="*50)

# ============================================================
# 1. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

X = df_processed.drop('diagnosed_diabetes', axis=1)
y = df_processed['diagnosed_diabetes']

print(f"\nOriginal Dataset:")
print(f"Total samples: {len(X)}")
print(f"Total features: {X.shape[1]}")
print(f"Target distribution:")
print(f"  Class 0 (No diabetes): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"  Class 1 (Diabetes): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")

# ============================================================
# 2. TRAIN/VAL/TEST SPLIT (80/10/10)
# ============================================================

# First split: 80% train, 20% temp (which will become val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.20,  # 20% for val+test
    random_state=42,
    stratify=y  # Maintains class distribution
)

# Second split: Split the 20% into 10% val and 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,  # 50% of 20% = 10% of total
    random_state=42,
    stratify=y_temp
)

print(f"\nâœ“ Data Split Complete!")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nTarget distribution in splits:")
print(f"  Train - Class 1: {(y_train == 1).sum()/len(y_train)*100:.2f}%")
print(f"  Val   - Class 1: {(y_val == 1).sum()/len(y_val)*100:.2f}%")
print(f"  Test  - Class 1: {(y_test == 1).sum()/len(y_test)*100:.2f}%")

# ============================================================
# 3. IDENTIFY NUMERICAL COLUMNS FOR SCALING
# ============================================================

# Get numerical columns (exclude boolean one-hot encoded columns)
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Boolean columns (one-hot encoded) - don't scale these
boolean_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

print(f"\nâœ“ Feature Types Identified:")
print(f"  Numerical features to scale: {len(numerical_cols)}")
print(f"  Boolean features (no scaling): {len(boolean_cols)}")

print(f"\nNumerical columns: {numerical_cols}")
print(f"\nBoolean columns: {boolean_cols}")

# ============================================================
# 4. STANDARDIZE NUMERICAL FEATURES
# ============================================================

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on TRAINING data only
scaler.fit(X_train[numerical_cols])

# Transform all three sets using the same scaler
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convert boolean columns to int (0/1) for neural network compatibility
for col in boolean_cols:
    X_train_scaled[col] = X_train_scaled[col].astype(int)
    X_val_scaled[col] = X_val_scaled[col].astype(int)
    X_test_scaled[col] = X_test_scaled[col].astype(int)

print(f"\nâœ“ Scaling Complete!")
print(f"  Fitted scaler on training data")
print(f"  Transformed train, val, and test sets")

# ============================================================
# 5. VERIFY SCALING
# ============================================================

print(f"\n" + "="*50)
print("Scaling Verification (Training Set Numerical Features):")
print("="*50)

# Check mean and std of scaled training data (should be ~0 and ~1)
scaled_stats = X_train_scaled[numerical_cols].describe().loc[['mean', 'std']]
print(scaled_stats)

print(f"\nâœ“ All means should be close to 0")
print(f"âœ“ All standard deviations should be close to 1")

# ============================================================
# 6. FINAL SHAPES
# ============================================================

print(f"\n" + "="*50)
print("Final Dataset Shapes:")
print("="*50)
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_val_scaled: {X_val_scaled.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

print("\nâœ“ Data is ready for correlation analysis and model training!")


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("Starting Correlation Analysis...")
print("="*50)

# ============================================================
# 1. COMPUTE CORRELATION MATRIX
# ============================================================

# Use the scaled training data for correlation analysis
# Add target back for correlation with features
train_with_target = X_train_scaled.copy()
train_with_target['diagnosed_diabetes'] = y_train.values

# Compute correlation matrix
correlation_matrix = train_with_target.corr()

print(f"\nâœ“ Correlation matrix computed")
print(f"  Shape: {correlation_matrix.shape}")

# ============================================================
# 2. CORRELATION WITH TARGET VARIABLE
# ============================================================

# Get correlations with target, sorted by absolute value
target_corr = correlation_matrix['diagnosed_diabetes'].drop('diagnosed_diabetes')
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print(f"\n" + "="*50)
print("TOP 15 FEATURES CORRELATED WITH DIABETES:")
print("="*50)
for i, (feature, corr_value) in enumerate(target_corr_sorted.head(15).items(), 1):
    actual_corr = target_corr[feature]
    print(f"{i:2d}. {feature:40s} {actual_corr:+.4f}")

# ============================================================
# 3. VISUALIZE CORRELATION HEATMAP (Full Matrix)
# ============================================================

plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            center=0,
            annot=False,  # Too many features to annotate
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - All Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ============================================================
# 4. VISUALIZE TOP FEATURES CORRELATION WITH TARGET
# ============================================================

# Plot top 15 features correlated with target
plt.figure(figsize=(12, 8))
top_15_features = target_corr_sorted.head(15)
colors = ['green' if target_corr[f] > 0 else 'red' for f in top_15_features.index]

plt.barh(range(len(top_15_features)), 
         [target_corr[f] for f in top_15_features.index],
         color=colors,
         alpha=0.7,
         edgecolor='black')
plt.yticks(range(len(top_15_features)), top_15_features.index)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.title('Top 15 Features Correlated with Diabetes Diagnosis', 
          fontsize=14, fontweight='bold', pad=15)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 5. IDENTIFY MULTICOLLINEARITY
# ============================================================

print(f"\n" + "="*50)
print("HIGHLY CORRELATED FEATURE PAIRS (Multicollinearity Check):")
print("="*50)
print("Threshold: |correlation| > 0.8\n")

# Get upper triangle of correlation matrix (to avoid duplicates)
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Find feature pairs with high correlation
high_corr_pairs = []
for column in upper_triangle.columns:
    high_corr = upper_triangle[column][abs(upper_triangle[column]) > 0.8]
    for idx, value in high_corr.items():
        if idx != 'diagnosed_diabetes' and column != 'diagnosed_diabetes':
            high_corr_pairs.append((column, idx, value))

if high_corr_pairs:
    for i, (feat1, feat2, corr_val) in enumerate(sorted(high_corr_pairs, 
                                                         key=lambda x: abs(x[2]), 
                                                         reverse=True), 1):
        print(f"{i:2d}. {feat1:35s} <--> {feat2:35s} = {corr_val:+.4f}")
else:
    print("No highly correlated feature pairs found (|corr| > 0.8)")

# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print(f"\n" + "="*50)
print("CORRELATION SUMMARY:")
print("="*50)
print(f"Features with |correlation| > 0.3 with target: {(target_corr_sorted > 0.3).sum()}")
print(f"Features with |correlation| > 0.2 with target: {(target_corr_sorted > 0.2).sum()}")
print(f"Features with |correlation| > 0.1 with target: {(target_corr_sorted > 0.1).sum()}")
print(f"Features with |correlation| < 0.05 with target: {(target_corr_sorted < 0.05).sum()}")

print("\nâœ“ Correlation analysis complete!")


# ============================================================
# FEATURE SELECTION - Remove Multicollinear Feature
# ============================================================

print("Feature Selection: Removing Multicollinear Features...")
print("="*50)

# Drop cholesterol_total from all datasets
features_to_drop = ['cholesterol_total']

X_train_final = X_train_scaled.drop(features_to_drop, axis=1)
X_val_final = X_val_scaled.drop(features_to_drop, axis=1)
X_test_final = X_test_scaled.drop(features_to_drop, axis=1)

print(f"\nâœ“ Dropped features: {features_to_drop}")
print(f"\nFinal feature count: {X_train_final.shape[1]}")
print(f"\nFinal Dataset Shapes:")
print(f"  X_train_final: {X_train_final.shape}")
print(f"  X_val_final: {X_val_final.shape}")
print(f"  X_test_final: {X_test_final.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val: {y_val.shape}")
print(f"  y_test: {y_test.shape}")

print(f"\nâœ“ Data is ready for Neural Network training!")
print("="*50)

# Display final feature list
print("\nFinal Features (30 total):")
print("-"*50)
for i, col in enumerate(X_train_final.columns, 1):
    print(f"{i:2d}. {col}")


# ============================================================
# 9. BUILD & TRAIN MULTIPLE MODELS (ENSEMBLE APPROACH)
# ============================================================

print("Building and Training Ensemble Models...")
print("="*70)

import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

# Store all models and their performance
models = {}
model_scores = {}

# ============================================================
# MODEL 1: NEURAL NETWORK
# ============================================================

print("\n" + "="*70)
print("MODEL 1: NEURAL NETWORK")
print("="*70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Build Neural Network architecture (YOUR EXACT ARCHITECTURE)
nn_model = Sequential([
    # Hidden Layer 1
    Dense(256, activation='relu', input_shape=(X_train_final.shape[1],), 
          name='dense_1'),
    Dropout(0.3, name='dropout_1'),
    
    # Hidden Layer 2
    Dense(128, activation='relu', name='dense_2'),
    Dropout(0.3, name='dropout_2'),
    
    # Hidden Layer 3
    Dense(64, activation='relu', name='dense_3'),
    Dropout(0.2, name='dropout_3'),
    
    # Hidden Layer 4
    Dense(32, activation='relu', name='dense_4'),
    Dropout(0.2, name='dropout_4'),
    
    # Output Layer
    Dense(1, activation='sigmoid', name='output')
])

print("\nâœ“ Neural Network Architecture:")
nn_model.summary()

# Compile the model
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print("\nâœ“ Model compiled successfully!")
print(f"  Optimizer: Adam")
print(f"  Loss: Binary Crossentropy")
print(f"  Metrics: Accuracy, AUC, Precision, Recall")

# Setup callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_diabetes_model.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

print("\nâœ“ Callbacks configured:")
print(f"  Early Stopping: patience=10, monitor=val_loss")
print(f"  Model Checkpoint: saves best model based on val_auc")

# Train Neural Network
print("\n" + "="*70)
print("STARTING TRAINING...")
print("="*70)

history = nn_model.fit(
    X_train_final, y_train,
    validation_data=(X_val_final, y_val),
    epochs=100,
    batch_size=512,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("\nâœ“ Training completed!")
print("="*70)

# Evaluate on validation set
nn_val_preds = nn_model.predict(X_val_final).flatten()
nn_val_loss = log_loss(y_val, nn_val_preds)
nn_val_auc = roc_auc_score(y_val, nn_val_preds)

models['Neural Network'] = nn_model
model_scores['Neural Network'] = {
    'val_loss': nn_val_loss,
    'val_auc': nn_val_auc
}

print(f"\nâœ“ Neural Network trained!")
print(f"  Validation Log Loss: {nn_val_loss:.4f}")
print(f"  Validation AUC: {nn_val_auc:.4f}")

# ============================================================
# MODEL 2: XGBOOST
# ============================================================

print("\n" + "="*70)
print("MODEL 2: XGBOOST")
print("="*70)

# Build XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50
)

print("XGBoost Parameters:")
print(xgb_model.get_params())

# Train XGBoost
print("\nTraining XGBoost...")
xgb_model.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    verbose=50
)

# Evaluate on validation set
xgb_val_preds = xgb_model.predict_proba(X_val_final)[:, 1]
xgb_val_loss = log_loss(y_val, xgb_val_preds)
xgb_val_auc = roc_auc_score(y_val, xgb_val_preds)

models['XGBoost'] = xgb_model
model_scores['XGBoost'] = {
    'val_loss': xgb_val_loss,
    'val_auc': xgb_val_auc
}

print(f"\nâœ“ XGBoost trained!")
print(f"  Validation Log Loss: {xgb_val_loss:.4f}")
print(f"  Validation AUC: {xgb_val_auc:.4f}")

# ============================================================
# MODEL 3: LIGHTGBM
# ============================================================

print("\n" + "="*70)
print("MODEL 3: LIGHTGBM")
print("="*70)

# Build LightGBM model
lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)

print("LightGBM Parameters:")
print(lgb_model.get_params())

# Train LightGBM
print("\nTraining LightGBM...")
lgb_model.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# Evaluate on validation set
lgb_val_preds = lgb_model.predict_proba(X_val_final)[:, 1]
lgb_val_loss = log_loss(y_val, lgb_val_preds)
lgb_val_auc = roc_auc_score(y_val, lgb_val_preds)

models['LightGBM'] = lgb_model
model_scores['LightGBM'] = {
    'val_loss': lgb_val_loss,
    'val_auc': lgb_val_auc
}

print(f"\nâœ“ LightGBM trained!")
print(f"  Validation Log Loss: {lgb_val_loss:.4f}")
print(f"  Validation AUC: {lgb_val_auc:.4f}")

# ============================================================
# CREATE ENSEMBLE
# ============================================================

print("\n" + "="*70)
print("CREATING ENSEMBLE")
print("="*70)

# Get predictions from all models on validation set
ensemble_val_preds = (nn_val_preds + xgb_val_preds + lgb_val_preds) / 3

# Evaluate ensemble
ensemble_val_loss = log_loss(y_val, ensemble_val_preds)
ensemble_val_auc = roc_auc_score(y_val, ensemble_val_preds)

model_scores['Ensemble (Average)'] = {
    'val_loss': ensemble_val_loss,
    'val_auc': ensemble_val_auc
}

print(f"\nâœ“ Ensemble created (simple averaging)")
print(f"  Validation Log Loss: {ensemble_val_loss:.4f}")
print(f"  Validation AUC: {ensemble_val_auc:.4f}")

# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON (Validation Set)")
print("="*70)

comparison_df = pd.DataFrame(model_scores).T
comparison_df = comparison_df.sort_values('val_auc', ascending=False)

print("\n" + comparison_df.to_string())

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# AUC comparison
axes[0].barh(comparison_df.index, comparison_df['val_auc'], color='skyblue', edgecolor='black')
axes[0].set_xlabel('Validation AUC', fontsize=12)
axes[0].set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(comparison_df['val_auc']):
    axes[0].text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')

# Log Loss comparison
axes[1].barh(comparison_df.index, comparison_df['val_loss'], color='coral', edgecolor='black')
axes[1].set_xlabel('Validation Log Loss (lower is better)', fontsize=12)
axes[1].set_title('Model Log Loss Comparison', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(comparison_df['val_loss']):
    axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# FIND BEST WEIGHTED ENSEMBLE (OPTIONAL)
# ============================================================

print("\n" + "="*70)
print("SEARCHING FOR OPTIMAL WEIGHTED ENSEMBLE")
print("="*70)

best_auc = ensemble_val_auc
best_weights = [1/3, 1/3, 1/3]

# Try different weight combinations
print("\nTrying different weight combinations...")
for w_nn in [0.2, 0.3, 0.4, 0.5]:
    for w_xgb in [0.2, 0.3, 0.4, 0.5]:
        w_lgb = 1 - w_nn - w_xgb
        if w_lgb >= 0.1 and w_lgb <= 0.6:
            weighted_preds = (w_nn * nn_val_preds + 
                            w_xgb * xgb_val_preds + 
                            w_lgb * lgb_val_preds)
            auc = roc_auc_score(y_val, weighted_preds)
            
            if auc > best_auc:
                best_auc = auc
                best_weights = [w_nn, w_xgb, w_lgb]
                print(f"  New best: NN={w_nn:.1f}, XGB={w_xgb:.1f}, LGB={w_lgb:.1f} -> AUC={auc:.4f}")

print(f"\nâœ“ Best weighted ensemble found!")
print(f"  Neural Network weight: {best_weights[0]:.2f}")
print(f"  XGBoost weight: {best_weights[1]:.2f}")
print(f"  LightGBM weight: {best_weights[2]:.2f}")
print(f"  Best Validation AUC: {best_auc:.4f}")

# Store best weights for later use
ensemble_weights = {
    'nn': best_weights[0],
    'xgb': best_weights[1],
    'lgb': best_weights[2]
}

print("\n" + "="*70)
print("âœ“ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*70)


# ============================================================
# 5. VISUALIZE TRAINING HISTORY (NEURAL NETWORK)
# ============================================================

print("\nGenerating Neural Network training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Neural Network Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Neural Network Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: AUC
axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
axes[1, 0].set_title('Neural Network AUC', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Precision & Recall
axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
axes[1, 1].set_title('Neural Network Precision & Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("âœ“ Neural Network visualizations complete!")

# ============================================================
# 6. VISUALIZE FEATURE IMPORTANCE (TREE-BASED MODELS)
# ============================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Get feature importance from XGBoost
xgb_importance = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Get feature importance from LightGBM
lgb_importance = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot feature importance for both models
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# XGBoost Feature Importance (Top 15)
top_xgb = xgb_importance.head(15)
axes[0].barh(range(len(top_xgb)), top_xgb['importance'], color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(top_xgb)))
axes[0].set_yticklabels(top_xgb['feature'])
axes[0].set_xlabel('Importance Score', fontsize=12)
axes[0].set_title('Top 15 Features - XGBoost', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# LightGBM Feature Importance (Top 15)
top_lgb = lgb_importance.head(15)
axes[1].barh(range(len(top_lgb)), top_lgb['importance'], color='lightcoral', edgecolor='black')
axes[1].set_yticks(range(len(top_lgb)))
axes[1].set_yticklabels(top_lgb['feature'])
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_title('Top 15 Features - LightGBM', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nâœ“ Feature importance visualizations complete!")

# Print top 10 features from each model
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

print("\nXGBoost Top 10:")
print(xgb_importance.head(10).to_string(index=False))

print("\n" + "-"*70)

print("\nLightGBM Top 10:")
print(lgb_importance.head(10).to_string(index=False))

# ============================================================
# 7. VISUALIZE MODEL COMPARISON
# ============================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE VISUALIZATION")
print("="*70)

# Create comprehensive comparison visualization
fig = plt.figure(figsize=(18, 6))

# Subplot 1: AUC Comparison
ax1 = plt.subplot(1, 3, 1)
models_list = list(comparison_df.index)
auc_scores = comparison_df['val_auc'].values
colors_auc = ['#2ecc71' if 'Ensemble' in model else '#3498db' for model in models_list]

bars1 = ax1.barh(models_list, auc_scores, color=colors_auc, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Validation AUC', fontsize=12, fontweight='bold')
ax1.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim([min(auc_scores) - 0.01, max(auc_scores) + 0.01])

for i, (model, score) in enumerate(zip(models_list, auc_scores)):
    ax1.text(score + 0.001, i, f'{score:.4f}', 
             va='center', fontweight='bold', fontsize=10)

# Subplot 2: Log Loss Comparison
ax2 = plt.subplot(1, 3, 2)
loss_scores = comparison_df['val_loss'].values
colors_loss = ['#e74c3c' if 'Ensemble' not in model else '#27ae60' for model in models_list]

bars2 = ax2.barh(models_list, loss_scores, color=colors_loss, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Validation Log Loss (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_title('Model Log Loss Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (model, score) in enumerate(zip(models_list, loss_scores)):
    ax2.text(score + 0.001, i, f'{score:.4f}', 
             va='center', fontweight='bold', fontsize=10)

# Subplot 3: Ensemble Weight Distribution
ax3 = plt.subplot(1, 3, 3)
weight_labels = ['Neural\nNetwork', 'XGBoost', 'LightGBM']
weight_values = [ensemble_weights['nn'], ensemble_weights['xgb'], ensemble_weights['lgb']]
weight_colors = ['#9b59b6', '#3498db', '#e67e22']

wedges, texts, autotexts = ax3.pie(weight_values, 
                                      labels=weight_labels,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      colors=weight_colors,
                                      explode=[0.05, 0.05, 0.05],
                                      shadow=True,
                                      textprops={'fontsize': 11, 'fontweight': 'bold'})

ax3.set_title('Optimal Ensemble Weights', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nâœ“ All visualizations complete!")
print("="*70)


# ============================================================
# 10. EVALUATE ENSEMBLE ON TEST SET
# ============================================================

print("\n" + "="*70)
print("EVALUATING ALL MODELS ON TEST SET (Unseen Data)")
print("="*70)

# ============================================================
# 1. GET PREDICTIONS FROM ALL MODELS
# ============================================================

print("\nGenerating predictions from all models...")

# Neural Network predictions
nn_test_preds = nn_model.predict(X_test_final, verbose=0).flatten()
print(f"âœ“ Neural Network predictions complete")

# XGBoost predictions
xgb_test_preds = xgb_model.predict_proba(X_test_final)[:, 1]
print(f"âœ“ XGBoost predictions complete")

# LightGBM predictions
lgb_test_preds = lgb_model.predict_proba(X_test_final)[:, 1]
print(f"âœ“ LightGBM predictions complete")

# Ensemble predictions (using optimal weights)
ensemble_test_preds = (
    ensemble_weights['nn'] * nn_test_preds +
    ensemble_weights['xgb'] * xgb_test_preds +
    ensemble_weights['lgb'] * lgb_test_preds
)
print(f"âœ“ Ensemble predictions complete")

# ============================================================
# 2. EVALUATE EACH MODEL
# ============================================================

from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score, 
                              confusion_matrix, classification_report, roc_curve)

print("\n" + "="*70)
print("TEST SET PERFORMANCE")
print("="*70)

# Store results
test_results = {}

# Evaluate each model
for name, preds in [('Neural Network', nn_test_preds),
                     ('XGBoost', xgb_test_preds),
                     ('LightGBM', lgb_test_preds),
                     ('Ensemble', ensemble_test_preds)]:
    
    # Convert probabilities to classes for accuracy
    pred_classes = (preds > 0.5).astype(int)
    
    test_results[name] = {
        'Log Loss': log_loss(y_test, preds),
        'AUC': roc_auc_score(y_test, preds),
        'Accuracy': accuracy_score(y_test, pred_classes)
    }

# Display results
test_results_df = pd.DataFrame(test_results).T
test_results_df = test_results_df.sort_values('AUC', ascending=False)

print("\n" + test_results_df.to_string())

# Highlight the best model
best_model_name = test_results_df.index[0]
print(f"\nğŸ�† Best Model on Test Set: {best_model_name}")
print(f"   AUC: {test_results_df.loc[best_model_name, 'AUC']:.4f}")
print(f"   Log Loss: {test_results_df.loc[best_model_name, 'Log Loss']:.4f}")
print(f"   Accuracy: {test_results_df.loc[best_model_name, 'Accuracy']:.4f}")

# ============================================================
# 3. DETAILED ENSEMBLE EVALUATION
# ============================================================

print("\n" + "="*70)
print("DETAILED ENSEMBLE EVALUATION")
print("="*70)

# Get ensemble predictions as classes
ensemble_pred_classes = (ensemble_test_preds > 0.5).astype(int)

# Classification Report
print("\nEnsemble Classification Report:")
print(classification_report(y_test, ensemble_pred_classes, 
                            target_names=['No Diabetes (0)', 'Diabetes (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, ensemble_pred_classes)
print("\nEnsemble Confusion Matrix:")
print(cm)

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nEnsemble Additional Metrics:")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  Negative Predictive Value: {npv:.4f} ({npv*100:.2f}%)")

# ============================================================
# 4. VISUALIZATIONS - MODEL COMPARISON
# ============================================================

print("\n" + "="*70)
print("GENERATING TEST SET VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(20, 12))

# Plot 1: Performance Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
models_list = test_results_df.index.tolist()
auc_scores = test_results_df['AUC'].values
colors = ['#27ae60' if model == 'Ensemble' else '#3498db' for model in models_list]

bars = ax1.barh(models_list, auc_scores, color=colors, edgecolor='black', linewidth=2)
ax1.set_xlabel('Test AUC', fontsize=12, fontweight='bold')
ax1.set_title('Model AUC Comparison (Test Set)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (model, score) in enumerate(zip(models_list, auc_scores)):
    ax1.text(score + 0.002, i, f'{score:.4f}', va='center', fontweight='bold')

# Plot 2: Log Loss Comparison
ax2 = plt.subplot(2, 3, 2)
loss_scores = test_results_df['Log Loss'].values
colors_loss = ['#e74c3c' if model != 'Ensemble' else '#27ae60' for model in models_list]

ax2.barh(models_list, loss_scores, color=colors_loss, edgecolor='black', linewidth=2)
ax2.set_xlabel('Test Log Loss (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_title('Model Log Loss Comparison (Test Set)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (model, score) in enumerate(zip(models_list, loss_scores)):
    ax2.text(score + 0.002, i, f'{score:.4f}', va='center', fontweight='bold')

# Plot 3: Accuracy Comparison
ax3 = plt.subplot(2, 3, 3)
acc_scores = test_results_df['Accuracy'].values
colors_acc = ['#9b59b6' if model != 'Ensemble' else '#27ae60' for model in models_list]

ax3.barh(models_list, acc_scores, color=colors_acc, edgecolor='black', linewidth=2)
ax3.set_xlabel('Test Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Model Accuracy Comparison (Test Set)', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for i, (model, score) in enumerate(zip(models_list, acc_scores)):
    ax3.text(score + 0.002, i, f'{score:.4f}', va='center', fontweight='bold')

# Plot 4: Ensemble Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_xlabel('Predicted', fontsize=12)

# Plot 5: ROC Curves for All Models
ax5 = plt.subplot(2, 3, 5)

colors_roc = {'Neural Network': '#e74c3c', 'XGBoost': '#3498db', 
              'LightGBM': '#f39c12', 'Ensemble': '#27ae60'}

for name, preds in [('Neural Network', nn_test_preds),
                     ('XGBoost', xgb_test_preds),
                     ('LightGBM', lgb_test_preds),
                     ('Ensemble', ensemble_test_preds)]:
    fpr, tpr, _ = roc_curve(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    linewidth = 3 if name == 'Ensemble' else 2
    ax5.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', 
             linewidth=linewidth, color=colors_roc[name])

ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax5.set_xlabel('False Positive Rate', fontsize=12)
ax5.set_ylabel('True Positive Rate', fontsize=12)
ax5.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3)

# Plot 6: Ensemble Prediction Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(ensemble_test_preds[y_test == 0], bins=50, alpha=0.6, 
         label='No Diabetes (Actual)', color='blue', edgecolor='black')
ax6.hist(ensemble_test_preds[y_test == 1], bins=50, alpha=0.6, 
         label='Diabetes (Actual)', color='red', edgecolor='black')
ax6.axvline(x=0.5, color='green', linestyle='--', linewidth=2, 
            label='Threshold (0.5)')
ax6.set_xlabel('Predicted Probability', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.set_title('Ensemble Prediction Distribution', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\nâœ“ All visualizations complete!")

# ============================================================
# 5. SAMPLE PREDICTIONS COMPARISON
# ============================================================

print("\n" + "="*70)
print("SAMPLE PREDICTIONS - ALL MODELS (First 10 from Test Set)")
print("="*70)

sample_comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'NN_Prob': nn_test_preds[:10],
    'XGB_Prob': xgb_test_preds[:10],
    'LGB_Prob': lgb_test_preds[:10],
    'Ensemble_Prob': ensemble_test_preds[:10],
    'Ensemble_Class': ensemble_pred_classes[:10]
})

sample_comparison['Correct'] = (
    sample_comparison['Actual'] == sample_comparison['Ensemble_Class']
).astype(str)

print(sample_comparison.to_string(index=False))

print("\nâœ“ Model evaluation complete!")
print("="*70)


# ============================================================
# 11. PREDICT ON KAGGLE TEST DATA & CREATE SUBMISSION
# ============================================================

print("\n" + "="*70)
print("MAKING PREDICTIONS ON KAGGLE TEST DATA")
print("="*70)

# ============================================================
# 1. LOAD TEST DATA
# ============================================================

test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"\nâœ“ Test data loaded")
print(f"  Shape: {test_data.shape}")
print(f"  Columns: {len(test_data.columns)}")

print("\nFirst few rows of test data:")
print(test_data.head())

# ============================================================
# 2. PREPROCESS TEST DATA (Same as training!)
# ============================================================

print("\n" + "="*70)
print("Preprocessing test data...")
print("="*70)

# Make a copy
test_processed = test_data.copy()

# Save IDs for submission
test_ids = test_processed['id'].copy()

# Remove ID column
test_processed = test_processed.drop('id', axis=1)

# Apply same preprocessing as training data
print("\n1. Capping physical activity outliers...")
test_processed['physical_activity_minutes_per_week'] = test_processed['physical_activity_minutes_per_week'].clip(upper=percentile_99)

print("2. Encoding categorical variables...")

# Label encoding for ordinal variables
test_processed['income_level'] = test_processed['income_level'].map(income_mapping)
test_processed['education_level'] = test_processed['education_level'].map(education_mapping)

# One-hot encoding for nominal variables
gender_dummies_test = pd.get_dummies(test_processed['gender'], prefix='gender', drop_first=True)
test_processed = pd.concat([test_processed, gender_dummies_test], axis=1)
test_processed = test_processed.drop('gender', axis=1)

ethnicity_dummies_test = pd.get_dummies(test_processed['ethnicity'], prefix='ethnicity', drop_first=True)
test_processed = pd.concat([test_processed, ethnicity_dummies_test], axis=1)
test_processed = test_processed.drop('ethnicity', axis=1)

smoking_dummies_test = pd.get_dummies(test_processed['smoking_status'], prefix='smoking', drop_first=True)
test_processed = pd.concat([test_processed, smoking_dummies_test], axis=1)
test_processed = test_processed.drop('smoking_status', axis=1)

employment_dummies_test = pd.get_dummies(test_processed['employment_status'], prefix='employment', drop_first=True)
test_processed = pd.concat([test_processed, employment_dummies_test], axis=1)
test_processed = test_processed.drop('employment_status', axis=1)

print(f"\nâœ“ Encoding complete. Features: {test_processed.shape[1]}")

# ============================================================
# 3. SCALE NUMERICAL FEATURES
# ============================================================

print("\n3. Scaling numerical features...")

# Convert boolean columns to int
boolean_cols_test = test_processed.select_dtypes(include=['bool']).columns.tolist()

for col in boolean_cols_test:
    test_processed[col] = test_processed[col].astype(int)

# Scale using the original numerical_cols
test_processed[numerical_cols] = scaler.transform(test_processed[numerical_cols])

print("âœ“ Scaling complete")

# ============================================================
# 4. DROP CHOLESTEROL_TOTAL
# ============================================================

print("\n4. Removing cholesterol_total...")
test_processed = test_processed.drop('cholesterol_total', axis=1)

print(f"âœ“ Final features: {test_processed.shape[1]}")

# ============================================================
# 5. VERIFY COLUMNS MATCH TRAINING DATA
# ============================================================

print("\n" + "="*70)
print("Verifying column alignment...")
print("="*70)

train_columns = X_train_final.columns.tolist()
test_columns = test_processed.columns.tolist()

print(f"Training columns: {len(train_columns)}")
print(f"Test columns: {len(test_columns)}")

if set(train_columns) == set(test_columns):
    print("âœ“ All columns match!")
    test_processed = test_processed[train_columns]
else:
    missing_in_test = set(train_columns) - set(test_columns)
    extra_in_test = set(test_columns) - set(train_columns)
    
    if missing_in_test:
        print(f"âš ï¸� Missing in test: {missing_in_test}")
        for col in missing_in_test:
            test_processed[col] = 0
            print(f"   Added {col} with zeros")
    
    if extra_in_test:
        print(f"âš ï¸� Extra in test: {extra_in_test}")
        test_processed = test_processed.drop(list(extra_in_test), axis=1)
    
    test_processed = test_processed[train_columns]
    print("âœ“ Columns aligned!")

print(f"\nFinal test data shape: {test_processed.shape}")

# ============================================================
# 6. MAKE ENSEMBLE PREDICTIONS
# ============================================================

print("\n" + "="*70)
print("Making predictions with ALL models...")
print("="*70)

# Get predictions from each model
print("\n1. Neural Network predictions...")
nn_kaggle_preds = nn_model.predict(test_processed, verbose=1).flatten()

print("\n2. XGBoost predictions...")
xgb_kaggle_preds = xgb_model.predict_proba(test_processed)[:, 1]

print("\n3. LightGBM predictions...")
lgb_kaggle_preds = lgb_model.predict_proba(test_processed)[:, 1]

# Create ensemble predictions using optimal weights
print("\n4. Creating ensemble predictions...")
ensemble_kaggle_preds = (
    ensemble_weights['nn'] * nn_kaggle_preds +
    ensemble_weights['xgb'] * xgb_kaggle_preds +
    ensemble_weights['lgb'] * lgb_kaggle_preds
)

print(f"\nâœ“ All predictions complete!")

# Print statistics for each model
print("\n" + "="*70)
print("PREDICTION STATISTICS")
print("="*70)

for name, preds in [('Neural Network', nn_kaggle_preds),
                     ('XGBoost', xgb_kaggle_preds),
                     ('LightGBM', lgb_kaggle_preds),
                     ('Ensemble', ensemble_kaggle_preds)]:
    print(f"\n{name}:")
    print(f"  Min: {preds.min():.4f}")
    print(f"  Max: {preds.max():.4f}")
    print(f"  Mean: {preds.mean():.4f}")
    print(f"  Median: {np.median(preds):.4f}")

# ============================================================
# 7. CREATE SUBMISSION FILE
# ============================================================

print("\n" + "="*70)
print("Creating submission file...")
print("="*70)

# Create submission dataframe with ENSEMBLE predictions
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': ensemble_kaggle_preds
})

# Save to CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("âœ“ Submission file created: submission.csv")
print(f"  Number of rows: {len(submission)}")
print(f"  Using: Weighted Ensemble (NN={ensemble_weights['nn']:.2f}, "
      f"XGB={ensemble_weights['xgb']:.2f}, LGB={ensemble_weights['lgb']:.2f})")

print("\nFirst 10 rows of submission:")
print(submission.head(10))

print("\nLast 10 rows of submission:")
print(submission.tail(10))

# ============================================================
# 8. OPTIONAL: CREATE INDIVIDUAL MODEL SUBMISSIONS
# ============================================================

print("\n" + "="*70)
print("Creating individual model submissions (optional)...")
print("="*70)

# Save individual model predictions for comparison
pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': nn_kaggle_preds
}).to_csv('/kaggle/working/submission_nn.csv', index=False)

pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': xgb_kaggle_preds
}).to_csv('/kaggle/working/submission_xgb.csv', index=False)

pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': lgb_kaggle_preds
}).to_csv('/kaggle/working/submission_lgb.csv', index=False)

print("âœ“ Individual model submissions created:")
print("  - submission_nn.csv (Neural Network)")
print("  - submission_xgb.csv (XGBoost)")
print("  - submission_lgb.csv (LightGBM)")

print("\n" + "="*70)
print("âœ“ ALL SUBMISSIONS READY!")
print("="*70)
print("\nMain submission: submission.csv (Ensemble)")
print("Alternative submissions: submission_nn.csv, submission_xgb.csv, submission_lgb.csv")
print("\nYou can now download and submit to Kaggle!")
print("Recommended: Start with 'submission.csv' (ensemble)")
print("="*70)






















