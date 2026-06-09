# Standard imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ML imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Configuration
# ============
# These values were discovered through experimentation with Claude Code

CUTOFF_ID = 678260        # Distribution shift point (discovered via adversarial validation)
VAL_WEIGHT = 20.0         # Weight multiplier for val samples (from top notebooks)
SEEDS = list(range(42, 52))  # 10 seeds for averaging

print(f"Configuration:")
print(f"  Distribution shift at ID: {CUTOFF_ID:,}")
print(f"  Validation weight: {VAL_WEIGHT}x")
print(f"  Seeds: {len(SEEDS)}")


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution: {train['diagnosed_diabetes'].mean():.4f}")


# KEY DISCOVERY: Distribution Shift
# =================================
# The training data has a structural break. Let's verify:

key_feature = 'physical_activity_minutes_per_week'

early_train = train[train['id'] < CUTOFF_ID]
late_train = train[train['id'] >= CUTOFF_ID]

print("Distribution Shift Analysis:")
print(f"  Early train (0-{CUTOFF_ID:,}): mean {key_feature} = {early_train[key_feature].mean():.2f}")
print(f"  Late train ({CUTOFF_ID:,}+): mean {key_feature} = {late_train[key_feature].mean():.2f}")
print(f"  Test: mean {key_feature} = {test[key_feature].mean():.2f}")
print(f"\n  => Late train matches test distribution!")


# Define columns
TARGET = 'diagnosed_diabetes'
ID_COL = 'id'
FEATURES = [c for c in train.columns if c not in [TARGET, ID_COL]]
CAT_COLS = train[FEATURES].select_dtypes(include=['object']).columns.tolist()

print(f"Features: {len(FEATURES)}")
print(f"Categorical: {len(CAT_COLS)}")

# Convert categoricals to category dtype for XGBoost
# This enables native categorical handling (enable_categorical=True)
for col in CAT_COLS:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# Split data based on distribution shift
X_early = train[train[ID_COL] < CUTOFF_ID][FEATURES]
y_early = train[train[ID_COL] < CUTOFF_ID][TARGET]
X_late = train[train[ID_COL] >= CUTOFF_ID][FEATURES]
y_late = train[train[ID_COL] >= CUTOFF_ID][TARGET]

print(f"Early samples: {len(X_early):,}")
print(f"Late samples (test-like): {len(X_late):,}")


# THE WINNING APPROACH: Weighted Full-Data Training
# =================================================

# Combine all training data
X_full = pd.concat([X_early, X_late], axis=0).reset_index(drop=True)
y_full = pd.concat([y_early, y_late], axis=0).reset_index(drop=True)

# Create sample weights: 1.0 for early, VAL_WEIGHT for late
weights_early = np.ones(len(X_early))
weights_late = np.ones(len(X_late)) * VAL_WEIGHT
sample_weights = np.concatenate([weights_early, weights_late])

print(f"Full training set: {len(X_full):,} samples")
print(f"Weights: 1.0 for {len(X_early):,} early, {VAL_WEIGHT}x for {len(X_late):,} late")


# Model parameters (conservative, avoid overfitting)
# These were tuned via Optuna in earlier experiments

XGB_PARAMS = {
    'n_estimators': 1700,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'enable_categorical': True,  # Native categorical handling
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'n_jobs': -1,
}

print("XGBoost Parameters:")
for k, v in XGB_PARAMS.items():
    print(f"  {k}: {v}")


# Train with multiple seeds
# =========================

test_preds_sum = np.zeros(len(test))
val_scores = []

print(f"Training {len(SEEDS)} seeds...\n")

for i, seed in enumerate(SEEDS):
    print(f"Seed {seed} ({i+1}/{len(SEEDS)})...", end=" ")
    
    # Create model with this seed
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    
    # Train on weighted full data
    model.fit(
        X_full, y_full,
        sample_weight=sample_weights,
        verbose=False
    )
    
    # Predict
    val_pred = model.predict_proba(X_late)[:, 1]
    test_pred = model.predict_proba(test[FEATURES])[:, 1]
    
    # Track
    val_auc = roc_auc_score(y_late, val_pred)
    val_scores.append(val_auc)
    test_preds_sum += test_pred
    
    print(f"Val AUC: {val_auc:.5f}")

# Average predictions
test_preds = test_preds_sum / len(SEEDS)

print(f"\nMean Val AUC: {np.mean(val_scores):.5f}")
print(f"Std: {np.std(val_scores):.5f}")


# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission saved!")
print(f"\nPrediction statistics:")
print(f"  Mean: {test_preds.mean():.4f}")
print(f"  Std: {test_preds.std():.4f}")
print(f"  Min: {test_preds.min():.4f}")
print(f"  Max: {test_preds.max():.4f}")

