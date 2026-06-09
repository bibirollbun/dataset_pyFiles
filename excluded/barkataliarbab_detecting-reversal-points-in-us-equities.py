# ====================================================
# REVERSAL POINTS DETECTION - FIXED NAN ISSUE
# ====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================================================
# 1. DATA LOADING AND EXPLORATION
# ====================================================

print("ğŸ“‚ Loading datasets...")

# Load new competition data
train_path = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv"
test_path = "/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv"
train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)

print(f"\nğŸ“Š Training data shape: {train.shape}")
print(f"ğŸ“Š Test data shape: {test.shape}")

print("\nğŸ�¯ Target distribution in training:")
target_counts = train['class_label'].value_counts()
print(target_counts)
print(train['class_label'].value_counts(normalize=True) * 100)

# Check for unexpected values in target
print("\nğŸ”� Checking for unexpected target values:")
print(f"Unique target values: {train['class_label'].unique()}")
print(f"Target data type: {train['class_label'].dtype}")

# ====================================================
# 2. DATA CLEANING AND PREPARATION
# ====================================================

print("\nğŸ§¹ Cleaning and preparing data...")

# Identify columns
id_cols = ['train_id'] if 'train_id' in train.columns else []
target_col = 'class_label'

# Check for any missing or unexpected values in target
print(f"\nğŸ”� Target value analysis:")
print(f"Null values in target: {train[target_col].isnull().sum()}")
print(f"Unexpected values: {set(train[target_col].unique()) - {'H', 'L'}}")

# Clean target - remove any rows with unexpected values
train_clean = train.copy()
original_size = len(train_clean)

# Keep only rows with valid target values
valid_targets = {'H', 'L'}
train_clean = train_clean[train_clean[target_col].isin(valid_targets)]

if len(train_clean) < original_size:
    print(f"âš ï¸�  Removed {original_size - len(train_clean)} rows with invalid target values")

# Separate features and target from clean data
feature_cols = [col for col in train_clean.columns if col not in id_cols + [target_col]]

print(f"ğŸ“Œ ID columns: {id_cols}")
print(f"ğŸ“Œ Target column: {target_col}")
print(f"ğŸ“Œ Feature columns: {len(feature_cols)}")

X = train_clean[feature_cols].copy()
y = train_clean[target_col].copy()

# Prepare test data
test_ids = test['id'].copy() if 'id' in test.columns else test.index
X_test = test[feature_cols].copy()

# ====================================================
# 3. FEATURE PROCESSING
# ====================================================

print("\nğŸ”„ Processing features...")

# Convert boolean columns to integer (0/1)
bool_cols = [col for col in X.columns if X[col].dtype == 'bool']
if bool_cols:
    print(f"Converting {len(bool_cols)} boolean columns...")
    X[bool_cols] = X[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

# Handle the 't' column (date) if it exists
if 't' in X.columns:
    print("\nğŸ“… Processing date column 't'...")
    
    # Convert to datetime
    X['t'] = pd.to_datetime(X['t'])
    X_test['t'] = pd.to_datetime(X_test['t'])
    
    # Extract useful date features
    for df in [X, X_test]:
        df['t_year'] = df['t'].dt.year
        df['t_month'] = df['t'].dt.month
        df['t_day'] = df['t'].dt.day
        df['t_dayofweek'] = df['t'].dt.dayofweek
        df['t_dayofyear'] = df['t'].dt.dayofyear
        df['t_week'] = df['t'].dt.isocalendar().week
        df['t_quarter'] = df['t'].dt.quarter
    
    # Drop the original date column
    X = X.drop(columns=['t'])
    X_test = X_test.drop(columns=['t'])
    print("âœ… Extracted date features")

# Handle ticker_id if it exists
if 'ticker_id' in X.columns:
    print("\nğŸ“Š Processing ticker_id...")
    # Get unique tickers from both train and test
    all_tickers = pd.concat([X['ticker_id'], X_test['ticker_id']]).unique()
    
    # Create a mapping from ticker to index
    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}
    
    # Map to integers
    X['ticker_id'] = X['ticker_id'].map(ticker_to_idx)
    X_test['ticker_id'] = X_test['ticker_id'].map(ticker_to_idx)
    
    print(f"âœ… Encoded {len(all_tickers)} unique tickers")

# Fill any NaN values
X = X.fillna(0)
X_test = X_test.fillna(0)

print(f"\nâœ… Final feature shapes:")
print(f"X shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")

# ====================================================
# 4. FEATURE SELECTION
# ====================================================

print("\nğŸ�¯ Performing feature selection...")

# Remove constant features
constant_features = []
for col in X.columns:
    if X[col].nunique() <= 1:
        constant_features.append(col)

if constant_features:
    print(f"Removing {len(constant_features)} constant features...")
    X = X.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

print(f"âœ… Features after removing constants: {X.shape[1]}")

# Since we have many features, let's select top features based on variance
if X.shape[1] > 1000:
    print("\nğŸ”� Selecting top 1000 features by variance...")
    variances = X.var()
    top_features = variances.nlargest(1000).index
    X = X[top_features]
    X_test = X_test[top_features]
    print(f"âœ… Selected top 1000 features by variance")

# ====================================================
# 5. PREPARE TARGET VARIABLE
# ====================================================

print("\nğŸ�¯ Preparing target variable...")

# Convert target to numeric - SAFE VERSION
print(f"Target values before conversion: {set(y.unique())}")

# Create mapping
label_mapping = {'H': 0, 'L': 1}
y_numeric = y.map(label_mapping)

# Check for any issues
print(f"Null values in y_numeric: {y_numeric.isnull().sum()}")
print(f"Unique values in y_numeric: {sorted(y_numeric.unique())}")

if y_numeric.isnull().any():
    print("âš ï¸�  Warning: Some target values couldn't be mapped!")
    # Drop rows with NaN in target
    valid_idx = y_numeric.notna()
    X = X[valid_idx]
    y_numeric = y_numeric[valid_idx]
    print(f"Removed {sum(~valid_idx)} rows with unmapped target values")

print(f"\nâœ… Final target distribution:")
print(f"Class 0 (H): {sum(y_numeric == 0)} samples")
print(f"Class 1 (L): {sum(y_numeric == 1)} samples")
print(f"Total: {len(y_numeric)} samples")

# ====================================================
# 6. MODEL TRAINING
# ====================================================

print("\nğŸ¤– Training models...")

# Split data
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )
    
    print(f"ğŸ“ˆ Training set: {X_train.shape}")
    print(f"ğŸ“ˆ Validation set: {X_val.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Train Gradient Boosting
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # ====================================================
    # 7. MODEL EVALUATION
    # ====================================================
    
    print("\nğŸ“Š Model Evaluation")
    print("=" * 50)
    
    for name, model in [("Random Forest", rf_model), ("Gradient Boosting", gb_model)]:
        y_pred = model.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"\n{name}:")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['H', 'L']))
    
    # ====================================================
    # 8. ENSEMBLE PREDICTION
    # ====================================================
    
    print("\nğŸ¤� Creating ensemble predictions...")
    
    # Get probabilities from both models
    rf_proba = rf_model.predict_proba(X_test_scaled)
    gb_proba = gb_model.predict_proba(X_test_scaled)
    
    # Weighted average
    ensemble_proba = 0.7 * rf_proba + 0.3 * gb_proba
    
    # Get predictions
    y_test_pred = np.argmax(ensemble_proba, axis=1)
    
    # Convert back to original labels
    reverse_mapping = {0: 'H', 1: 'L'}
    predictions = [reverse_mapping[pred] for pred in y_test_pred]
    
except Exception as e:
    print(f"â�Œ Error during model training: {e}")
    print("\nâš ï¸�  Falling back to simple baseline model...")
    
    # Simple baseline: predict based on training distribution
    h_ratio = (y_numeric == 0).mean()
    predictions = ['H' if np.random.random() < h_ratio else 'L' for _ in range(len(X_test))]
    
    print(f"Using baseline with H probability: {h_ratio:.2f}")

# ====================================================
# 9. CREATE SUBMISSION
# ====================================================

print("\nğŸ“� Creating submission file...")

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'class_label': predictions
})

# Distribution analysis
dist = submission['class_label'].value_counts(normalize=True) * 100
print("\nğŸ“Š Predicted distribution (%):")
print(dist.round(3))

# Compare with training distribution
train_dist = train_clean['class_label'].value_counts(normalize=True) * 100
print("\nğŸ“Š Training distribution (%):")
print(train_dist.round(3))

# Save submission file
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\nâœ… Submission saved to: {submission_path}")

# ====================================================
# 10. ANALYZE BOOLEAN FEATURES
# ====================================================

print("\nğŸ”� Analyzing boolean features...")

# Count threshold crossing features
cross_features = [col for col in X.columns if 'cross_threshold' in col]
print(f"\nğŸ“Š Threshold crossing features: {len(cross_features)}")

if cross_features:
    # Analyze a sample
    sample_features = sorted(cross_features)[:10]
    print(f"\nSample threshold features and their means:")
    for feat in sample_features:
        feat_mean = X[feat].mean() if feat in X.columns else 0
        print(f"  {feat}: {feat_mean:.3f}")

# ====================================================
# 11. VISUALIZATION
# ====================================================

try:
    print("\nğŸ“ˆ Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Class distribution comparison
    ax1 = axes[0, 0]
    comparison_df = pd.DataFrame({
        'Training': train_dist,
        'Predictions': dist
    }).fillna(0)
    comparison_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Class Distribution: Training vs Predictions')
    ax1.set_ylabel('Percentage')
    ax1.legend()
    
    # 2. Feature importance (if model trained successfully)
    ax2 = axes[0, 1]
    if 'rf_model' in locals() and hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        feature_importance.plot(kind='barh', x='feature', y='importance', ax=ax2)
        ax2.set_title('Top 20 Feature Importances (Random Forest)')
        ax2.set_xlabel('Importance')
    else:
        ax2.text(0.5, 0.5, 'Feature importance not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Feature Importance')
    
    # 3. Confusion matrix (if available)
    ax3 = axes[1, 0]
    if 'y_val' in locals() and 'rf_model' in locals():
        y_val_pred = rf_model.predict(X_val_scaled)
        cm = confusion_matrix(y_val, y_val_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['H', 'L'], 
                    yticklabels=['H', 'L'], ax=ax3)
        ax3.set_title('Confusion Matrix (Random Forest)')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
    else:
        ax3.text(0.5, 0.5, 'Confusion matrix not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Confusion Matrix')
    
    # 4. Prediction confidence
    ax4 = axes[1, 1]
    if 'ensemble_proba' in locals():
        max_proba = np.max(ensemble_proba, axis=1)
        ax4.hist(max_proba, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_title('Distribution of Prediction Confidence')
        ax4.set_xlabel('Maximum Probability')
        ax4.set_ylabel('Count')
        ax4.axvline(0.7, color='red', linestyle='--', label='70% Confidence')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Confidence scores not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Prediction Confidence')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/analysis_plots.png', dpi=100, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"âš ï¸�  Could not create visualizations: {e}")

# ====================================================
# 12. FINAL SUMMARY
# ====================================================

print("\n" + "="*50)
print("ğŸš€ NOTEBOOK EXECUTION COMPLETE!")
print("="*50)

print(f"\nğŸ“‹ Final statistics:")
print(f"Training samples (clean): {len(train_clean)}")
print(f"Test samples: {len(test)}")
print(f"Features used: {X.shape[1]}")
print(f"\nPredictions:")
print(f"Total: {len(submission)}")
print(f"H: {(submission['class_label'] == 'H').sum()} ({dist.get('H', 0):.1f}%)")
print(f"L: {(submission['class_label'] == 'L').sum()} ({dist.get('L', 0):.1f}%)")

print("\nğŸ§¾ First 10 predictions:")
print(submission.head(10))

print("\nğŸ“� Output files created:")
print("1. submission.csv - Your competition submission")
print("2. analysis_plots.png - Visualization of results")

