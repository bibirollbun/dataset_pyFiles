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
print(f"\n✓ Income Level: Label encoded (0=Low to 4=High)")

# Education Level (has clear order)
education_mapping = {
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}
df_processed['education_level'] = df_processed['education_level'].map(education_mapping)
print(f"✓ Education Level: Label encoded (0=No formal to 3=Postgraduate)")

# 2.2 One-Hot Encoding for NOMINAL variables (no natural order)
print("\nApplying One-Hot Encoding for nominal variables...")

# Gender
gender_dummies = pd.get_dummies(df_processed['gender'], prefix='gender', drop_first=True)
df_processed = pd.concat([df_processed, gender_dummies], axis=1)
df_processed = df_processed.drop('gender', axis=1)
print(f"✓ Gender: One-hot encoded ({len(gender_dummies.columns)} columns)")

# Ethnicity
ethnicity_dummies = pd.get_dummies(df_processed['ethnicity'], prefix='ethnicity', drop_first=True)
df_processed = pd.concat([df_processed, ethnicity_dummies], axis=1)
df_processed = df_processed.drop('ethnicity', axis=1)
print(f"✓ Ethnicity: One-hot encoded ({len(ethnicity_dummies.columns)} columns)")

# Smoking Status
smoking_dummies = pd.get_dummies(df_processed['smoking_status'], prefix='smoking', drop_first=True)
df_processed = pd.concat([df_processed, smoking_dummies], axis=1)
df_processed = df_processed.drop('smoking_status', axis=1)
print(f"✓ Smoking Status: One-hot encoded ({len(smoking_dummies.columns)} columns)")

# Employment Status
employment_dummies = pd.get_dummies(df_processed['employment_status'], prefix='employment', drop_first=True)
df_processed = pd.concat([df_processed, employment_dummies], axis=1)
df_processed = df_processed.drop('employment_status', axis=1)
print(f"✓ Employment Status: One-hot encoded ({len(employment_dummies.columns)} columns)")

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

print(f"\n✓ Data Split Complete!")
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

print(f"\n✓ Feature Types Identified:")
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

print(f"\n✓ Scaling Complete!")
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

print(f"\n✓ All means should be close to 0")
print(f"✓ All standard deviations should be close to 1")

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

print("\n✓ Data is ready for correlation analysis and model training!")


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

print(f"\n✓ Correlation matrix computed")
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

print("\n✓ Correlation analysis complete!")


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

print(f"\n✓ Dropped features: {features_to_drop}")
print(f"\nFinal feature count: {X_train_final.shape[1]}")
print(f"\nFinal Dataset Shapes:")
print(f"  X_train_final: {X_train_final.shape}")
print(f"  X_val_final: {X_val_final.shape}")
print(f"  X_test_final: {X_test_final.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val: {y_val.shape}")
print(f"  y_test: {y_test.shape}")

print(f"\n✓ Data is ready for Neural Network training!")
print("="*50)

# Display final feature list
print("\nFinal Features (30 total):")
print("-"*50)
for i, col in enumerate(X_train_final.columns, 1):
    print(f"{i:2d}. {col}")


# ============================================================
# NEURAL NETWORK ARCHITECTURE & TRAINING
# ============================================================

print("Building Deep Neural Network...")
print("="*50)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. BUILD THE MODEL
# ============================================================

model = Sequential([
    # Input layer (implicit - defined by first Dense layer)
    
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

# Display model architecture
print("\n✓ Model Architecture:")
model.summary()

# ============================================================
# 2. COMPILE THE MODEL
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

print("\n✓ Model compiled successfully!")
print(f"  Optimizer: Adam")
print(f"  Loss: Binary Crossentropy")
print(f"  Metrics: Accuracy, AUC, Precision, Recall")

# ============================================================
# 3. SETUP CALLBACKS
# ============================================================

# Early Stopping - stop if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Model Checkpoint - save best model
checkpoint = ModelCheckpoint(
    'best_diabetes_model.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

print("\n✓ Callbacks configured:")
print(f"  Early Stopping: patience=10, monitor=val_loss")
print(f"  Model Checkpoint: saves best model based on val_auc")

# ============================================================
# 4. TRAIN THE MODEL
# ============================================================

print("\n" + "="*50)
print("STARTING TRAINING...")
print("="*50)

history = model.fit(
    X_train_final, 
    y_train,
    validation_data=(X_val_final, y_val),
    epochs=100,
    batch_size=512,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

print("\n✓ Training completed!")
print("="*50)


# ============================================================
# 5. VISUALIZE TRAINING HISTORY
# ============================================================

print("\nGenerating training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: AUC
axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Precision & Recall
axes[1, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[1, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualizations complete!")


# ============================================================
# 6. EVALUATE ON TEST SET
# ============================================================

print("\n" + "="*50)
print("EVALUATING ON TEST SET (Unseen Data)")
print("="*50)

# Load the best model
best_model = keras.models.load_model('best_diabetes_model.keras')

# Evaluate on test set
test_results = best_model.evaluate(X_test_final, y_test, verbose=1)

print("\n✓ Test Set Results:")
print(f"  Test Loss: {test_results[0]:.4f}")
print(f"  Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"  Test AUC: {test_results[2]:.4f}")
print(f"  Test Precision: {test_results[3]:.4f}")
print(f"  Test Recall: {test_results[4]:.4f}")

# Get predictions
y_pred_probs = best_model.predict(X_test_final)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

print("\n" + "="*50)
print("Prediction Statistics:")
print("="*50)
print(f"Predicted probabilities - Min: {y_pred_probs.min():.4f}, Max: {y_pred_probs.max():.4f}")
print(f"Predicted probabilities - Mean: {y_pred_probs.mean():.4f}, Median: {np.median(y_pred_probs):.4f}")


# ============================================================
# 7. CONFUSION MATRIX & ROC CURVE
# ============================================================

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

print("\n" + "="*50)
print("DETAILED EVALUATION METRICS")
print("="*50)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=['No Diabetes (0)', 'Diabetes (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)  # Negative Predictive Value

print(f"\nAdditional Metrics:")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  Negative Predictive Value: {npv:.4f} ({npv*100:.2f}%)")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = roc_auc_score(y_test, y_pred_probs)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

# Plot 3: Prediction Distribution
axes[2].hist(y_pred_probs[y_test == 0], bins=50, alpha=0.6, 
             label='No Diabetes (Actual)', color='blue', edgecolor='black')
axes[2].hist(y_pred_probs[y_test == 1], bins=50, alpha=0.6, 
             label='Diabetes (Actual)', color='red', edgecolor='black')
axes[2].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[2].set_xlabel('Predicted Probability', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=15)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n✓ Visualizations complete!")

# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "="*50)
print("SAMPLE PREDICTIONS (First 10 from Test Set)")
print("="*50)

sample_predictions = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted_Probability': y_pred_probs[:10].flatten(),
    'Predicted_Class': y_pred_classes[:10].flatten()
})

sample_predictions['Correct'] = (sample_predictions['Actual'] == sample_predictions['Predicted_Class']).astype(str)

print(sample_predictions.to_string(index=False))

print("\n✓ Model evaluation complete!")
print("="*50)


# ============================================================
# PREDICT ON TEST DATA & CREATE SUBMISSION FILE
# ============================================================

print("Loading test data and making predictions...")
print("="*50)

# ============================================================
# 1. LOAD TEST DATA
# ============================================================

test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"\n✓ Test data loaded")
print(f"  Shape: {test_data.shape}")
print(f"  Columns: {len(test_data.columns)}")

print("\nFirst few rows of test data:")
print(test_data.head())

# ============================================================
# 2. PREPROCESS TEST DATA (Same as training!)
# ============================================================

print("\n" + "="*50)
print("Preprocessing test data...")
print("="*50)

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

print(f"\n✓ Encoding complete. Features: {test_processed.shape[1]}")

# ============================================================
# 3. SCALE NUMERICAL FEATURES (BEFORE dropping cholesterol_total!)
# ============================================================

print("\n3. Scaling numerical features...")

# Get boolean columns and convert to int
boolean_cols_test = test_processed.select_dtypes(include=['bool']).columns.tolist()

for col in boolean_cols_test:
    test_processed[col] = test_processed[col].astype(int)

# Scale using the original numerical_cols (which includes cholesterol_total)
# The scaler expects these columns
test_processed[numerical_cols] = scaler.transform(test_processed[numerical_cols])

print("✓ Scaling complete")

# ============================================================
# 4. NOW DROP CHOLESTEROL_TOTAL (After scaling!)
# ============================================================

print("\n4. Removing cholesterol_total...")
test_processed = test_processed.drop('cholesterol_total', axis=1)

print(f"✓ Final features: {test_processed.shape[1]}")

# ============================================================
# 5. VERIFY COLUMNS MATCH TRAINING DATA
# ============================================================

print("\n" + "="*50)
print("Verifying column alignment...")
print("="*50)

# Check if columns match
train_columns = X_train_final.columns.tolist()
test_columns = test_processed.columns.tolist()

print(f"Training columns: {len(train_columns)}")
print(f"Test columns: {len(test_columns)}")

if set(train_columns) == set(test_columns):
    print("✓ All columns match!")
    # Reorder test columns to match training order
    test_processed = test_processed[train_columns]
else:
    missing_in_test = set(train_columns) - set(test_columns)
    extra_in_test = set(test_columns) - set(train_columns)
    
    if missing_in_test:
        print(f"⚠️ Missing in test: {missing_in_test}")
        # Add missing columns with zeros
        for col in missing_in_test:
            test_processed[col] = 0
            print(f"   Added {col} with zeros")
    
    if extra_in_test:
        print(f"⚠️ Extra in test: {extra_in_test}")
        # Remove extra columns
        test_processed = test_processed.drop(list(extra_in_test), axis=1)
    
    # Reorder to match training
    test_processed = test_processed[train_columns]
    print("✓ Columns aligned!")

print(f"\nFinal test data shape: {test_processed.shape}")
print(f"Expected shape: ({len(test_ids)}, {len(train_columns)})")

# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

print("\n" + "="*50)
print("Making predictions on test data...")
print("="*50)

# Predict probabilities
test_predictions = best_model.predict(test_processed, verbose=1)

print(f"\n✓ Predictions complete!")
print(f"  Number of predictions: {len(test_predictions)}")
print(f"  Min probability: {test_predictions.min():.4f}")
print(f"  Max probability: {test_predictions.max():.4f}")
print(f"  Mean probability: {test_predictions.mean():.4f}")
print(f"  Median probability: {np.median(test_predictions):.4f}")

# ============================================================
# 7. CREATE SUBMISSION FILE
# ============================================================

print("\n" + "="*50)
print("Creating submission file...")
print("="*50)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_predictions.flatten()
})

# Save to CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("✓ Submission file created: submission.csv")
print(f"  Number of rows: {len(submission)}")

print("\nFirst 10 rows of submission:")
print(submission.head(10))

print("\nLast 10 rows of submission:")
print(submission.tail(10))

print("\n" + "="*50)
print("✓ SUBMISSION READY!")
print("="*50)
print("File: submission.csv")
print("Format: id, diagnosed_diabetes (probability)")
print("\nYou can now download 'submission.csv' and submit to Kaggle!")
















