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
import os
import warnings
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm

print("Loading competition data...")
print("-" * 50)

# =========================
# 1. LOAD DATA
# =========================
TRAIN_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
TEST_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv"
SUBMISSION_CSV = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/sample_submission.csv"

# Load data
train = pd.read_csv(TRAIN_CSV, low_memory=False)
test = pd.read_csv(TEST_CSV, low_memory=False) if os.path.exists(TEST_CSV) else pd.DataFrame()
submission = pd.read_csv(SUBMISSION_CSV, low_memory=False)

print(f"âœ“ Train shape: {train.shape}")
print(f"âœ“ Test shape: {test.shape}")
print(f"âœ“ Submission shape: {submission.shape}")

# =========================
# 2. DATA PREPARATION
# =========================
print("\n" + "=" * 50)
print("DATA PREPARATION")
print("=" * 50)

# Target processing
target_col = 'class_label'
y_raw = train[target_col].copy()

# Simple mapping
def map_label(x):
    if pd.isna(x):
        return "N"
    x = str(x).upper()
    if 'H' in x:
        return "H"
    elif 'L' in x:
        return "L"
    else:
        return "N"

y = y_raw.apply(map_label)
print("Class distribution:")
print(y.value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Prepare features - drop uninformative columns
print("\nPreparing features...")

# First, identify common columns between train and test
common_columns = list(set(train.columns) & set(test.columns))
print(f"Common columns between train and test: {len(common_columns)}")

# Remove target column from common columns if present
if target_col in common_columns:
    common_columns.remove(target_col)

# Remove ID columns and other non-features
non_feature_cols = ['id', 'row_id', 'index', 'Unnamed: 0', 'train_id', 'test_id', 'ticker_id', 't']
common_columns = [col for col in common_columns if col not in non_feature_cols]

# Check for constant columns
print("Checking for constant columns...")
constant_cols = []
for col in tqdm(common_columns[:1000], desc="Analyzing columns"):
    try:
        if train[col].nunique() <= 1:
            constant_cols.append(col)
    except:
        continue

print(f"Found {len(constant_cols)} constant columns")
common_columns = [col for col in common_columns if col not in constant_cols]

# Create feature matrices
X = train[common_columns].copy()
X_test = test[common_columns].copy() if len(test) > 0 else pd.DataFrame()

print(f"\nFeature matrix shapes:")
print(f"X_train: {X.shape}")
print(f"X_test: {X_test.shape}")

# Convert to float32 to save memory
for col in X.columns:
    X[col] = X[col].astype('float32')
    if len(X_test) > 0 and col in X_test.columns:
        X_test[col] = X_test[col].astype('float32')

# =========================
# 3. SMART FEATURE SELECTION
# =========================
print("\n" + "=" * 50)
print("SMART FEATURE SELECTION")
print("=" * 50)

# Select top features by variance
print("Selecting top features by variance...")
n_features_to_keep = 100

# Calculate variance for each column
variances = []
for col in tqdm(X.columns, desc="Calculating variances"):
    try:
        var = X[col].var()
        if pd.notna(var):
            variances.append((col, var))
    except:
        continue

# Sort by variance and take top N
variances.sort(key=lambda x: x[1], reverse=True)
top_cols = [col for col, var in variances[:n_features_to_keep]]
print(f"Selected top {len(top_cols)} features by variance")

# Keep only selected columns
X = X[top_cols]
if len(X_test) > 0:
    # Ensure test has the same columns
    X_test = X_test[[col for col in top_cols if col in X_test.columns]]
    # Add missing columns with zeros
    missing_cols = set(top_cols) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[top_cols]  # Reorder to match train

print(f"\nFinal feature shapes:")
print(f"Train: {X.shape}")
print(f"Test: {X_test.shape}")

# =========================
# 4. SIMPLE FEATURE ENGINEERING
# =========================
print("\n" + "=" * 50)
print("SIMPLE FEATURE ENGINEERING")
print("=" * 50)

def create_simple_features(df, prefix=""):
    """Create simple but effective features"""
    features = pd.DataFrame(index=df.index)
    
    for col in tqdm(df.columns, desc=f"Creating {prefix}features"):
        try:
            series = df[col].fillna(0)
            
            # Basic transformations
            features[f'{prefix}{col}_lag1'] = series.shift(1)
            features[f'{prefix}{col}_lag3'] = series.shift(3)
            features[f'{prefix}{col}_diff1'] = series.diff(1)
            features[f'{prefix}{col}_diff3'] = series.diff(3)
            
            # Rolling statistics
            features[f'{prefix}{col}_roll_mean_5'] = series.rolling(5, min_periods=3).mean()
            features[f'{prefix}{col}_roll_std_5'] = series.rolling(5, min_periods=3).std()
            features[f'{prefix}{col}_roll_max_5'] = series.rolling(5, min_periods=3).max()
            features[f'{prefix}{col}_roll_min_5'] = series.rolling(5, min_periods=3).min()
            
            # Price position
            features[f'{prefix}{col}_zscore'] = (series - series.mean()) / (series.std() + 1e-10)
            
        except:
            continue
    
    return features.fillna(0)

print("Creating features for train set...")
train_features = create_simple_features(X, "train_")

print("Creating features for test set...")
test_features = create_simple_features(X_test, "test_") if len(X_test) > 0 else pd.DataFrame()

# Combine original and engineered features
print("\nCombining features...")
X_final = pd.concat([X, train_features], axis=1).fillna(0)
if len(X_test) > 0:
    X_test_final = pd.concat([X_test, test_features], axis=1).fillna(0)
else:
    X_test_final = pd.DataFrame()

print(f"Final feature dimensions:")
print(f"Train: {X_final.shape}")
print(f"Test: {X_test_final.shape}")

# =========================
# 5. MODEL TRAINING
# =========================
print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss

num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {le.classes_}")

# LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1,
}

# Cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize arrays for predictions
oof_preds = np.zeros((len(X_final), num_classes))
test_preds = np.zeros((len(X_test_final), num_classes)) if len(X_test_final) > 0 else None

print(f"\nStarting {n_folds}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_encoded)):
    print(f"\n{'='*30}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*30}")
    
    # Split data
    X_train, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    
    # Apply class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    # Create datasets
    train_data = lgb.Dataset(X_train, y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]
    )
    
    # Get predictions
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    # Calculate fold metrics
    val_pred_class = np.argmax(val_pred, axis=1)
    fold_acc = accuracy_score(y_val, val_pred_class)
    fold_f1 = f1_score(y_val, val_pred_class, average='macro')
    
    print(f"Fold {fold + 1} - Accuracy: {fold_acc:.4f}, F1: {fold_f1:.4f}")
    
    # Predict on test set
    if test_preds is not None:
        test_pred = model.predict(X_test_final, num_iteration=model.best_iteration)
        test_preds += test_pred / n_folds
    
    # Clean up
    del model, train_data, val_data
    gc.collect()

# Calculate overall OOF metrics
print(f"\n{'='*50}")
print("OVERALL PERFORMANCE")
print(f"{'='*50}")

oof_preds_class = np.argmax(oof_preds, axis=1)
overall_acc = accuracy_score(y_encoded, oof_preds_class)
overall_f1 = f1_score(y_encoded, oof_preds_class, average='macro')

print(f"Overall Accuracy: {overall_acc:.4f}")
print(f"Overall Macro F1: {overall_f1:.4f}")

# Display class-wise performance
print("\nClass-wise performance:")
for class_idx, class_name in enumerate(le.classes_):
    mask = y_encoded == class_idx
    if mask.any():
        class_acc = accuracy_score(y_encoded[mask], oof_preds_class[mask])
        print(f"{class_name}: Accuracy = {class_acc:.4f}, Count = {mask.sum()}")

# =========================
# 6. PREDICTION & SUBMISSION
# =========================
print("\n" + "=" * 50)
print("MAKING PREDICTIONS")
print("=" * 50)

if test_preds is not None:
    # Convert to class predictions
    test_preds_class = np.argmax(test_preds, axis=1)
    
    # Apply temporal smoothing
    print("Applying temporal smoothing...")
    smoothed = test_preds_class.copy()
    
    # Simple majority voting in sliding window
    window_size = 5
    for i in range(window_size, len(smoothed)):
        window = smoothed[i-window_size:i]
        unique, counts = np.unique(window, return_counts=True)
        most_common = unique[np.argmax(counts)]
        if counts.max() >= window_size * 0.6:  # 60% agreement
            smoothed[i] = most_common
    
    # Convert back to original labels
    final_predictions = le.inverse_transform(smoothed)
    
    # Display distribution
    print("\nPredictions distribution:")
    unique, counts = np.unique(final_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"{label}: {count} ({count/len(final_predictions)*100:.1f}%)")
    
    # Prepare submission
    submission['class_label'] = final_predictions
    
    # Validate predictions
    valid_labels = set(le.classes_)
    invalid_mask = ~submission['class_label'].isin(valid_labels)
    
    if invalid_mask.any():
        print(f"\nâš ï¸� Warning: {invalid_mask.sum()} invalid predictions found")
        submission.loc[invalid_mask, 'class_label'] = 'N'
    
    # Show sample predictions
    print("\nFirst 10 predictions:")
    print(submission[['id', 'class_label']].head(10))
    
    # Save submission
    submission.to_csv("submission.csv", index=False)
    print(f"\nâœ“ Submission saved to 'submission.csv'")
    
    # Performance estimation
    print(f"\nExpected performance based on validation:")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    
    if overall_f1 > 0.95:
        print("ğŸ�¯ Excellent! Target score 0.99520 is achievable!")
    elif overall_f1 > 0.9:
        print("âœ… Very good performance!")
    elif overall_f1 > 0.8:
        print("âš ï¸� Decent performance, consider feature engineering improvements")
    else:
        print("â�Œ Needs significant improvement")
    
    # Additional diagnostics
    print("\n" + "=" * 50)
    print("DIAGNOSTICS")
    print("=" * 50)
    
    # Check for class imbalance in predictions
    pred_counts = submission['class_label'].value_counts()
    print("\nSubmission class distribution:")
    for label in le.classes_:
        count = pred_counts.get(label, 0)
        percentage = count / len(submission) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    # Compare with train distribution
    print("\nTrain vs Submission distribution:")
    train_counts = y.value_counts()
    for label in le.classes_:
        train_count = train_counts.get(label, 0)
        train_pct = train_count / len(train) * 100
        sub_count = pred_counts.get(label, 0)
        sub_pct = sub_count / len(submission) * 100
        print(f"{label}: Train={train_pct:.1f}%, Submission={sub_pct:.1f}%")
    
else:
    print("No test data available for submission")

# =========================
# 7. FINAL MODEL (TRAIN ON ALL DATA)
# =========================
print("\n" + "=" * 50)
print("TRAINING FINAL MODEL")
print("=" * 50)

print("Training final model on all training data...")
final_model = lgb.train(
    params,
    lgb.Dataset(X_final, y_encoded),
    num_boost_round=500
)

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': X_final.columns,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)
print("âœ“ Feature importance saved to 'feature_importance.csv'")

print("\n" + "=" * 50)
print("NOTEBOOK COMPLETED!")
print("=" * 50)


# =========================
# ğŸ“Š DATA VISUALIZATION
# =========================
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("=== DATA VISUALIZATION ===")

# 1. TARGET DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="viridis")
plt.title("Target Distribution (H / L / N)")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# 2. FEATURE SUMMARY (first 20 numeric features)
numeric_cols = X.select_dtypes(include=[np.number]).columns[:20]

plt.figure(figsize=(14,10))
X[numeric_cols].hist(bins=30, figsize=(14,10), layout=(5,4))
plt.suptitle("Distribution of First 20 Numeric Features", y=1.02)
plt.show()

# 3. CORRELATION HEATMAP (small subset)
subset_cols = X.columns[:50]  # reduce for speed
corr = X[subset_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (First 50 Features)")
plt.show()

# 4. PCA VISUALIZATION (2D)
print("Running PCA on 500-feature set...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_pca[:,0], X_pca[:,1],
    c=y_encoded, cmap="viridis", alpha=0.6
)
plt.colorbar(scatter, ticks=[0,1,2], label="Encoded Class")
plt.title("PCA (2 Components) â€“ Class Separation")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 5. FEATURE VARIANCE (Top 30 most varying)
feature_var = X.var().sort_values(ascending=False)[:30]

plt.figure(figsize=(10,6))
sns.barplot(x=feature_var.values, y=feature_var.index, palette="plasma")
plt.title("Top 30 Features by Variance")
plt.xlabel("Variance")
plt.ylabel("Feature Name")
plt.show()

