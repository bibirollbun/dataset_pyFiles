import pandas as pd
import numpy as np
import os
import gc
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# Config
DATA_PATH = '/kaggle/input/solana-skill-sprint-memcoin-graduation'
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')
SUBMISSION_FILE = 'submission.csv'

TARGET = 'has_graduated'
MINT_ID = 'mint'
N_SPLITS = 7
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 100

print("âœ… Setup completed!")


def reduce_mem_usage(df, verbose=True):
    """Optimize memory usage by downcasting numeric columns"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'ğŸ”� Memory reduced from {start_mem:.2f} MB â†’ {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


print("ğŸ“‚ Loading data...")

# Load datasets
train_df = reduce_mem_usage(pd.read_csv(TRAIN_FILE))
test_df = reduce_mem_usage(pd.read_csv(TEST_FILE))

# Combine for feature engineering
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df['is_train'] = combined_df[TARGET].notna().astype(int)

print("\nğŸ“Š Data shapes:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Combined: {combined_df.shape}")


print("\nğŸ”§ Creating features...")

# Basic features
if 'slot_min' in combined_df.columns:
    combined_df['slot_diff'] = combined_df['slot_graduated'] - combined_df['slot_min']
    
# Add more feature engineering here as needed

# Select final features
features = [col for col in combined_df.columns if col not in 
           [MINT_ID, TARGET, 'is_train', 'Unnamed: 0', 'slot_graduated', 'is_valid']]

# Split back to train/test
train = combined_df[combined_df['is_train'] == 1].copy()
test = combined_df[combined_df['is_train'] == 0].copy()

# Save mint IDs before deletion
test_ids = test[MINT_ID].copy()  # <-- Critical fix: Preserve IDs for submission

X = train[features]
y = train[TARGET].astype(int)
X_test = test[features]  # Feature matrix for test set

# Clean up (keeping test_ids)
del combined_df, train
gc.collect()

print(f"\nâœ… Final features: {len(features)}")
print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Test IDs preserved: {len(test_ids)}")  # Verification


print("\nğŸ¤– Training models...")

# Initialize
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# K-Fold Training
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”„ Fold {fold+1}/{N_SPLITS}")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    # CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        random_state=RANDOM_SEED + fold,
        verbose=0
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=100)
    
    # Ensemble predictions
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    cb_pred = cb_model.predict_proba(X_val)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    
    ensemble_pred = 0.4*lgb_pred + 0.4*cb_pred + 0.2*xgb_pred
    oof_preds[val_idx] = ensemble_pred
    
    # Test predictions
    test_preds += (0.4*lgb_model.predict_proba(X_test)[:, 1] + 
                 0.4*cb_model.predict_proba(X_test)[:, 1] + 
                 0.2*xgb_model.predict_proba(X_test)[:, 1]) / N_SPLITS
    
    # Store models
    models.append((lgb_model, cb_model, xgb_model))
    
    # Fold evaluation
    fold_score = log_loss(y_val, ensemble_pred)
    print(f"ğŸ“Š Fold {fold+1} LogLoss: {fold_score:.5f}")

# Overall evaluation
overall_score = log_loss(y, oof_preds)

print(f"\nğŸ�† Overall OOF LogLoss: {overall_score:.5f}")


print("\nğŸ“¤ Generating submission...")

# Create submission using the preserved mint IDs
submission = pd.DataFrame({
    'mint': test_ids.values,  # Use the preserved mint IDs
    'has_graduated': test_preds  # Your model predictions
})

# Clip probabilities as required by competition rules
submission['has_graduated'] = np.clip(submission['has_graduated'], 0.0001, 0.9999)

# Save submission file
submission.to_csv(SUBMISSION_FILE, index=False)

print(f"âœ… Submission saved to {SUBMISSION_FILE}")
print("\nğŸ�‰ Pipeline completed successfully! ğŸš€")

