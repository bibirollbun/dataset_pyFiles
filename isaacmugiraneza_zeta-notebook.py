import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# File paths
COMPETITION_PATH = '/kaggle/input/mercor-cheating-detection'

# Correct file paths
train_path = f'{COMPETITION_PATH}/train.csv'
test_path = f'{COMPETITION_PATH}/test.csv'
sample_submission_path = f'{COMPETITION_PATH}/sample_submission.csv'
social_graph_path = f'{COMPETITION_PATH}/social_graph.csv'
feature_metadata_path = f'{COMPETITION_PATH}/feature_metadata.json'

# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)
social_graph = pd.read_csv(social_graph_path)

# Load feature metadata
with open(feature_metadata_path, 'r') as f:
    feature_metadata = json.load(f)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)
print("Social graph shape:", social_graph.shape)


# First 5 rows
print(train.head())

# Check missing values
print(train.isnull().sum())

# Check class distribution
print(train['is_cheating'].value_counts(dropna=False))



# Separate feature columns
feature_cols = [f'feature_{i:03}' for i in range(1, 19)]

# Imputer for numeric and binary features
numeric_features = [f for f in feature_cols if feature_metadata[f]['type'] == 'numeric']
binary_features = [f for f in feature_cols if feature_metadata[f]['type'] == 'binary']

# Numeric imputer
num_imputer = SimpleImputer(strategy='median')
train[numeric_features] = num_imputer.fit_transform(train[numeric_features])
test[numeric_features] = num_imputer.transform(test[numeric_features])

# Binary imputer
bin_imputer = SimpleImputer(strategy='most_frequent')
train[binary_features] = bin_imputer.fit_transform(train[binary_features])
test[binary_features] = bin_imputer.transform(test[binary_features])

print("Missing values handled!")



# Based on your data: Train columns: ['user_hash', 'feature_001', 'feature_002', ... 'feature_018', 'high_conf_clean', 'is_cheating']

print("="*60)
print("EXACT DATA SPLIT FOR YOUR DATASET")
print("="*60)

# Your exact columns
target_col = 'is_cheating'
id_col = 'user_hash'

# All feature columns (18 features + high_conf_clean)
feature_cols = ['feature_001', 'feature_002', 'feature_003', 'feature_004', 'feature_005', 
                'feature_006', 'feature_007', 'feature_008', 'feature_009', 'feature_010',
                'feature_011', 'feature_012', 'feature_013', 'feature_014', 'feature_015',
                'feature_016', 'feature_017', 'feature_018', 'high_conf_clean']

# Split the data EXACTLY as you wanted
labeled = train[train['is_cheating'].notnull()].copy()
unlabeled = train[train['is_cheating'].isnull()].copy()

X_labeled = labeled[feature_cols]
y_labeled = labeled['is_cheating']
X_unlabeled = unlabeled[feature_cols]

# Print EXACTLY what you asked for
print(f"Labeled data shape: {labeled.shape}")
print(f"Unlabeled data shape: {unlabeled.shape}")
print(f"X_labeled shape: {X_labeled.shape}")
print(f"y_labeled shape: {y_labeled.shape}")
print(f"X_unlabeled shape: {X_unlabeled.shape}")

print("\n" + "="*60)
print("ADDITIONAL VERIFICATION")
print("="*60)

# Verify counts match what you saw earlier
print(f"Total train: {len(train)}")
print(f"Labeled (non-null is_cheating): {len(labeled)}")
print(f"  - Class 0: {(y_labeled == 0).sum()}")
print(f"  - Class 1: {(y_labeled == 1).sum()}")
print(f"Unlabeled (null is_cheating): {len(unlabeled)}")

# Show first few rows to confirm
print("\nFirst 3 labeled samples:")
print(labeled[['user_hash', 'is_cheating'] + feature_cols[:3]].head(3))

print("\nFirst 3 unlabeled samples:")
print(unlabeled[['user_hash', 'is_cheating'] + feature_cols[:3]].head(3))


from sklearn.model_selection import train_test_split

# Split the labeled data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_labeled, 
    y_labeled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_labeled
)

print("="*50)
print("TRAIN/VALIDATION SPLIT RESULTS")
print("="*50)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")

print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)
print("Training set class distribution:")
print(f"  Class 0 (Not cheating): {(y_train == 0).sum()} samples ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"  Class 1 (Cheating): {(y_train == 1).sum()} samples ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print("\nValidation set class distribution:")
print(f"  Class 0 (Not cheating): {(y_val == 0).sum()} samples ({(y_val == 0).sum()/len(y_val)*100:.1f}%)")
print(f"  Class 1 (Cheating): {(y_val == 1).sum()} samples ({(y_val == 1).sum()/len(y_val)*100:.1f}%)")

print("\n" + "="*50)
print("PERCENTAGE SPLIT")
print("="*50)
print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X_labeled)*100:.1f}% of labeled data)")
print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X_labeled)*100:.1f}% of labeled data)")
print(f"Total labeled data: {len(X_labeled):,} samples")

print("\n" + "="*50)
print("DATA OVERVIEW")
print("="*50)
print(f"Original data: {len(train):,} total samples")
print(f"Labeled data: {len(X_labeled):,} samples ({len(X_labeled)/len(train)*100:.1f}% of total)")
print(f"Unlabeled data: {len(X_unlabeled):,} samples ({len(X_unlabeled)/len(train)*100:.1f}% of total)")


# Run this EXACT code - it's complete and ready
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

print("="*60)
print("COMPLETE FIXED PIPELINE WITH COMMON FEATURES")
print("="*60)

# Step 1: Identify common features between train and test
print("1. Identifying common features...")

# Original feature columns from train
train_feature_cols = ['feature_001', 'feature_002', 'feature_003', 'feature_004', 'feature_005', 
                     'feature_006', 'feature_007', 'feature_008', 'feature_009', 'feature_010',
                     'feature_011', 'feature_012', 'feature_013', 'feature_014', 'feature_015',
                     'feature_016', 'feature_017', 'feature_018', 'high_conf_clean']

# Find which features actually exist in test
common_features = [col for col in train_feature_cols if col in test.columns]

print(f"   Train has: {len(train_feature_cols)} features")
print(f"   Test has: {len(test.columns)} features")
print(f"   Common features: {len(common_features)}")
print(f"   Common features: {common_features}")

# Step 2: Use only common features
X_train_common = X_train[common_features]
X_val_common = X_val[common_features]
X_test_common = test[common_features]

print(f"\n2. Data shapes after selecting common features:")
print(f"   X_train_common: {X_train_common.shape}")
print(f"   X_val_common: {X_val_common.shape}")
print(f"   X_test_common: {X_test_common.shape}")

# Step 3: Handle missing values
print("\n3. Handling missing values...")

imputer = SimpleImputer(strategy='median')

X_train_imputed = imputer.fit_transform(X_train_common)
X_val_imputed = imputer.transform(X_val_common)
X_test_imputed = imputer.transform(X_test_common)

print(f"   ✓ Missing values imputed with median")

# Step 4: Train Random Forest
print("\n4. Training Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf.fit(X_train_imputed, y_train)
print("   ✓ Model training completed")

# Step 5: Validate
print("\n5. Validating model...")
val_preds_proba = rf.predict_proba(X_val_imputed)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds_proba)

print(f"\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Step 6: Make test predictions
print("\n6. Making test predictions...")
test_preds_proba = rf.predict_proba(X_test_imputed)[:, 1]

# Step 7: Create submission
print("\n7. Creating submission file...")

# Check sample_submission format
print(f"Sample submission columns: {sample_submission.columns.tolist()}")

# Create submission matching expected format
submission = sample_submission.copy()
if 'is_cheating' in submission.columns:
    submission['is_cheating'] = test_preds_proba
else:
    # Use the last column for predictions
    submission.iloc[:, -1] = test_preds_proba

# Save submission
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n" + "="*60)
print("SUBMISSION CREATED SUCCESSFULLY")
print("="*60)
print(f"✓ Submission saved to: {submission_path}")
print(f"✓ Submission shape: {submission.shape}")
print(f"\nSubmission preview:")
print(submission.head())
print(f"\nPrediction statistics:")
print(f"   Min: {test_preds_proba.min():.4f}")
print(f"   Max: {test_preds_proba.max():.4f}")
print(f"   Mean: {test_preds_proba.mean():.4f}")
print(f"   Std: {test_preds_proba.std():.4f}")

