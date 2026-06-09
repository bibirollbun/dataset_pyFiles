import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# Gradient boosting libraries
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

print("âœ… All libraries imported successfully!")
print("ğŸ�¯ Target: Achieve 97.6% accuracy using the proven approach")


# Configuration class to store all paths and parameters
class CFG:
    """Central configuration for the entire pipeline"""
    
    # File paths
    train_path = '/kaggle/input/playground-series-s5e7/train.csv'
    test_path = '/kaggle/input/playground-series-s5e7/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e7/sample_submission.csv'
    original_path = "/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv"
    
    # Model parameters
    n_folds = 5
    seed = 42
    
    # Feature columns for merging (critical for success!)
    merge_cols = [
        'Time_spent_Alone', 
        'Stage_fear', 
        'Social_event_attendance',
        'Going_outside', 
        'Drained_after_socializing', 
        'Friends_circle_size', 
        'Post_frequency'
    ]
    
    # Categorical columns that need encoding
    cat_cols = ['match_p', 'Stage_fear', 'Drained_after_socializing']

print("âš™ï¸� Configuration loaded!")
print(f"   Cross-validation folds: {CFG.n_folds}")
print(f"   Random seed: {CFG.seed}")
print(f"   Number of features for merging: {len(CFG.merge_cols)}")


# Load all datasets
print("ğŸ“‚ Loading datasets...")

train = pd.read_csv(CFG.train_path)
test_df = pd.read_csv(CFG.test_path)
orig_df = pd.read_csv(CFG.original_path)

print(f"âœ… Training data shape: {train.shape}")
print(f"âœ… Test data shape: {test_df.shape}")
print(f"âœ… Original data shape: {orig_df.shape}")

# Display first few rows
print("\nğŸ“Š Training data preview:")
train.head()


# CRITICAL STEP: Prepare original dataset with deduplication
# This creates the 'match_p' feature that significantly boosts accuracy

# Rename target column in original dataset
df = orig_df.rename(columns={'Personality': 'match_p'})

# Remove duplicates based on feature columns
# This ensures unique mapping for each combination
df = df.drop_duplicates(subset=CFG.merge_cols)

print(f"ğŸ“Š Original dataset after deduplication: {df.shape}")
print(f"   Unique personality mappings: {len(df)}")
print("\nğŸ”� Sample of deduplicated data:")
df.head()


# Define merge function that creates the match_p_null indicator
def merge_with_match_p(df, ref_df, merge_cols):
    """
    Merge dataframe with reference data and create null indicator.
    
    The match_p_null indicator is crucial - it tells the model
    whether we found a matching personality in the original data.
    """
    merged_df = df.merge(ref_df, how='left', on=merge_cols)
    merged_df['match_p_null'] = merged_df['match_p'].isna().astype(int)
    return merged_df

# Apply merging to both train and test
print("ğŸ”— Merging with original dataset...")

train = merge_with_match_p(train, df, CFG.merge_cols)
test_df = merge_with_match_p(test_df, df, CFG.merge_cols)

# Check merge success
train_matches = train['match_p'].notna().sum()
test_matches = test_df['match_p'].notna().sum()

print(f"\nğŸ“Š Merge Results:")
print(f"   Training matches: {train_matches}/{len(train)} ({train_matches/len(train)*100:.1f}%)")
print(f"   Test matches: {test_matches}/{len(test_df)} ({test_matches/len(test_df)*100:.1f}%)")
print(f"   New features added: match_p, match_p_null")


# Remove any duplicate IDs (keeping first occurrence)
print("ğŸ§¹ Cleaning data...")

train = train.drop_duplicates(subset=['id'], keep='first')
test_df = test_df.drop_duplicates(subset=['id'], keep='first')

# Fill missing match_p values with 'unknown'
# This is different from typical approaches and helps the model
train['match_p'] = train['match_p'].fillna('unknown')
test_df['match_p'] = test_df['match_p'].fillna('unknown')

print(f"âœ… Duplicates removed")
print(f"âœ… Missing match_p values filled with 'unknown'")

# Check match_p distribution
print("\nğŸ“Š match_p value distribution in training:")
print(train['match_p'].value_counts())


# CRITICAL: Map target variable
# Note: This mapping is OPPOSITE of many approaches!
# Extrovert = 1, Introvert = 0

print("ğŸ�¯ Encoding target variable...")

# Save original labels for reference
original_labels = train['Personality'].value_counts()
print("Original distribution:")
print(original_labels)

# Apply mapping
train['Personality'] = train['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Verify encoding
print("\nEncoded distribution:")
print(train['Personality'].value_counts())
print("\nMapping: Extrovert â†’ 1, Introvert â†’ 0")


# Define categorical encoding function
def cat_encode(df, cat_cols):
    """
    Encode categorical features with specific mappings.
    
    Key insight: match_p gets 3 levels (0,1,2) which helps
    the model distinguish between unknown and known personalities.
    """
    for col in cat_cols:
        if col == 'match_p':
            # 3-level encoding for match_p
            df[col] = df[col].map({'Extrovert': 2, 'Introvert': 1, 'unknown': 0})
        else:
            # Binary encoding for Yes/No features
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

# Apply encoding
print("ğŸ�·ï¸� Encoding categorical features...")

train = cat_encode(train, CFG.cat_cols)
test_df = cat_encode(test_df, CFG.cat_cols)

print("âœ… Categorical encoding complete")
print("\nEncoding mappings:")
print("   match_p: unknownâ†’0, Introvertâ†’1, Extrovertâ†’2")
print("   Stage_fear: Noâ†’0, Yesâ†’1")
print("   Drained_after_socializing: Noâ†’0, Yesâ†’1")

# Display encoded features
train[CFG.cat_cols].head()


# Fill missing values with column means
print("ğŸ”§ Handling missing values...")

# Check missing values before
print("Missing values before imputation:")
print(train.isnull().sum()[train.isnull().sum() > 0])

# Fill missing values in training data
for col in train.drop(columns=['id', 'Personality']).columns:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mean())

# Fill missing values in test data
for col in test_df.drop(columns=['id']).columns:
    if test_df[col].isnull().any():
        test_df[col] = test_df[col].fillna(test_df[col].mean())

print("\nâœ… All missing values filled with column means")

# Verify no missing values remain
print(f"Missing values in train: {train.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")


# Prepare features and target for modeling
print("ğŸ“¦ Preparing final features and target...")

# Features (exclude id and target)
x = train.drop(['Personality', 'id'], axis=1)
test = test_df.drop('id', axis=1)

# Target
y = train['Personality']

print(f"\nğŸ“Š Final data shapes:")
print(f"   Training features: {x.shape}")
print(f"   Training target: {y.shape}")
print(f"   Test features: {test.shape}")

# Display feature names
print(f"\nğŸ“‹ Features ({len(x.columns)} total):")
for i, col in enumerate(x.columns, 1):
    print(f"   {i}. {col}")


# Calculate scale_pos_weight for handling class imbalance
# This is crucial for model performance

counter = Counter(y)
neg = counter[0]  # Introvert count
pos = counter[1]  # Extrovert count
scale_pos_weight = neg / pos

print("âš–ï¸� Class Balance Analysis:")
print(f"   Introverts (0): {neg:,} samples")
print(f"   Extroverts (1): {pos:,} samples")
print(f"   Imbalance ratio: {neg/pos:.4f}:1")
print(f"   scale_pos_weight: {scale_pos_weight:.4f}")
print("\nğŸ’¡ This weight will be used in all models to handle imbalance")

# Visualize class distribution
plt.figure(figsize=(8, 5))
y.value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Target Distribution (After Encoding)', fontsize=14, fontweight='bold')
plt.xlabel('Personality (0=Introvert, 1=Extrovert)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.show()





# CatBoost parameters - optimized for this specific problem
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.001,  # Very low learning rate for stability
    'depth': 8,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'bagging_temperature': 1,
    'random_strength': 1,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
    'verbose': 0,
    'random_state': 42
}

print("ğŸ�± CatBoost parameters defined")
print(f"   Key settings: iterations={catboost_params['iterations']}, lr={catboost_params['learning_rate']}")


# LightGBM parameters - fine-tuned for optimal performance
lgb_params = {
    'objective': 'binary',
    'n_estimators': 300,
    'max_depth': 10,
    'min_child_samples': 10,
    'num_leaves': 20,
    'learning_rate': 0.010919705161662964,  # Precisely tuned
    'colsample_bytree': 0.881928717897877,
    'subsample': 0.7015184751538656,
    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
    'metric': 'AUC',
    'random_state': 42,
    'verbosity': 0
}

print("ğŸ’¡ LightGBM parameters defined")
print(f"   Key settings: n_estimators={lgb_params['n_estimators']}, max_depth={lgb_params['max_depth']}")


# XGBoost parameters - extensive tuning applied
xgb_params = {
    'n_estimators': 642,
    'learning_rate': 0.13368501085667397,
    'max_depth': 8,
    'subsample': 0.5003247477566168,
    'colsample_bytree': 0.5300137588788056,
    'gamma': 2.3324933690884206,
    'reg_lambda': 2.4718829122342267,
    'reg_alpha': 0.16710802570626998,
    'min_child_weight': 8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
    'random_state': 42
}

print("ğŸš€ XGBoost parameters defined")
print(f"   Key settings: n_estimators={xgb_params['n_estimators']}, regularization applied")


# Random Forest parameters
rf_params = {
    'n_estimators': 608,
    'max_depth': 9,
    'min_samples_split': 17,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'bootstrap': True,
    'class_weight': 'balanced',  # Different approach for RF
    'random_state': 42,
    'n_jobs': -1  # Use all cores
}

print("ğŸŒ² Random Forest parameters defined")
print(f"   Key settings: n_estimators={rf_params['n_estimators']}, class_weight='balanced'")


# Create individual estimators
print("\nğŸ�—ï¸� Building Stacking Ensemble...")

estimators = [
    ('lgb', LGBMClassifier(**lgb_params)),
    ('rf', RandomForestClassifier(**rf_params)),
    ('xgb', XGBClassifier(**xgb_params)),
    ('cat', CatBoostClassifier(**catboost_params))
]

# Meta model - simple logistic regression
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Create stacking classifier
# CRITICAL: passthrough=True means original features are included!
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    passthrough=True,  # This is KEY for 97.6% accuracy!
    cv=5,  # 5-fold CV for creating meta features
    n_jobs=-1
)

print("âœ… Stacking ensemble created with:")
print("   - 4 base models (LightGBM, RandomForest, XGBoost, CatBoost)")
print("   - Logistic Regression meta-model")
print("   - passthrough=True (includes original features)")
print("   - 5-fold CV for meta-feature generation")


# Training with stratified k-fold cross-validation
print("ğŸ�‹ï¸� Starting model training with 5-fold CV...")
print("=" * 60)

# Initialize cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Storage for metrics
auc_scores = []
accuracy_lst = []
test_predict = np.zeros(len(test))  # For averaging predictions

# Training loop
for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
    print(f"\n{'='*50}")
    print(f"ğŸ“� FOLD {fold + 1}/5")
    print(f"{'='*50}")
    
    # Split data
    X_train, X_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"   Train size: {len(X_train):,}")
    print(f"   Valid size: {len(X_val):,}")
    
    # Fit stacking classifier
    print(f"   Training ensemble...")
    stacking_clf.fit(X_train, y_train)
    
    # Validate
    y_pred_proba = stacking_clf.predict_proba(X_val)[:, 1]
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_val, y_pred_proba)
    acc_score = accuracy_score(y_val, y_pred_label)
    
    print(f"\n   ğŸ“Š Fold {fold+1} Results:")
    print(f"      AUC Score: {auc:.6f}")
    print(f"      Accuracy:  {acc_score:.6f}")
    
    auc_scores.append(auc)
    accuracy_lst.append(acc_score)
    
    # Accumulate test predictions (averaging)
    test_predict += stacking_clf.predict_proba(test)[:, 1] / skf.n_splits
    print(f"   âœ… Test predictions accumulated")

# Display final results
print(f"\n{'='*60}")
print(f"ğŸ�¯ CROSS-VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"ğŸ“Š Average AUC across folds: {np.mean(auc_scores):.6f} Â± {np.std(auc_scores):.6f}")
print(f"ğŸ“Š Average Accuracy across folds: {np.mean(accuracy_lst):.6f} Â± {np.std(accuracy_lst):.6f}")

# Visualize fold performance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 6), auc_scores, 'o-', label='AUC', linewidth=2, markersize=8)
plt.plot(range(1, 6), accuracy_lst, 's-', label='Accuracy', linewidth=2, markersize=8)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Performance Across Folds', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.95, 1.0)

plt.subplot(1, 2, 2)
metrics_df = pd.DataFrame({
    'AUC': auc_scores,
    'Accuracy': accuracy_lst
})
metrics_df.boxplot()
plt.title('Score Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()





# Create final predictions
print("ğŸ“� Creating submission file...")

# Apply threshold to get binary predictions
final_predict = (test_predict >= 0.5).astype(int)

# Map back to original labels
label_map = {1: 'Extrovert', 0: 'Introvert'}
final_labeled = pd.Series(final_predict).map(label_map)

print(f"\nğŸ“Š Prediction distribution:")
print(final_labeled.value_counts())
print(f"\nPercentages:")
print(final_labeled.value_counts(normalize=True) * 100)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'], 
    'Personality': final_labeled
})

# Save submission
submission.to_csv('submission_976.csv', index=False)
print(f"\nâœ… Submission saved as 'submission_976.csv'")

# Display preview
print("\nğŸ“‹ Submission preview:")
submission.head(10)


# Display final summary
print("\n" + "="*60)
print("ğŸ�† PIPELINE COMPLETE!")
print("="*60)

print(f"\nğŸ“Š Model Performance:")
print(f"   Average CV Accuracy: {np.mean(accuracy_lst):.6f}")
print(f"   Average CV AUC: {np.mean(auc_scores):.6f}")

print(f"\nğŸ”‘ Key Success Factors:")
print("   1. merge_p feature from original dataset")
print("   2. match_p_null indicator variable")
print("   3. Strategic encoding (unknown=0, Introvert=1, Extrovert=2)")
print("   4. scale_pos_weight for all models")
print("   5. StackingClassifier with passthrough=True")
print("   6. Optimized hyperparameters")

print(f"\nğŸ“ˆ Expected Leaderboard Score: ~97.6%")

if np.mean(accuracy_lst) >= 0.975:
    print("\nğŸ�Š Congratulations! You've achieved the target accuracy! ğŸ�Š")
else:
    print(f"\nğŸ’¡ Your accuracy ({np.mean(accuracy_lst):.4f}) is slightly below 97.6%")
    print("   Double-check all preprocessing steps match exactly")


### Acknowledgement: 
##### Solomon Ajaero 

