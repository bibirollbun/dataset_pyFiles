
import pandas as pd
import numpy as np
import glob
import os
import gc
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ğŸ“¦ Modeling & Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# âš™ï¸� Competition Configuration
DATA_PATH = '/kaggle/input/pump-fun-graduation-february-2025'
CHUNK_PATTERN = os.path.join(DATA_PATH, 'chunk*.csv')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')
DUNE_INFO_FILE = os.path.join(DATA_PATH, 'dune_token_info.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_PATH, 'token_info_onchain_divers.csv')
SUBMISSION_FILE = 'submission.csv'

TARGET = 'has_graduated'
MINT_ID = 'mint'
BLOCK_LIMIT = 100  # Only analyze first 100 blocks post-mint
N_SPLITS = 7       # Increased folds for better generalization
RANDOM_SEED = 42   # For reproducibility
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

print("âœ… Packages imported and configuration set!")



def reduce_mem_usage(df, verbose=True):
    """
    ğŸ“‰ Optimize memory usage by downcasting numeric columns
    Returns: Memory-optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Handle integers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            # Handle floats
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

print("âœ… Memory optimization function ready!")



print("ğŸ“‚ Loading and preprocessing data...")

import os

# Check available input directories
input_base = '/kaggle/input'
available_dirs = os.listdir(input_base)
print(f"ğŸ”� Available input directories: {available_dirs}")

# Set dataset directory (using the actual directory name found)
dataset_dir = '/kaggle/input/solana-skill-sprint-memcoin-graduation'
print(f"âœ… Using dataset directory: {dataset_dir}")

# Updated required files list (only files that actually exist)
required_files = {
    'Train': 'train.csv',
    'Test': 'test_unlabeled.csv'
    # Removed missing files from requirements
}

print("\nğŸ”� Verifying dataset files exist...")
missing_files = []
for name, filename in required_files.items():
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(filepath):
        missing_files.append(filepath)
        print(f"   â�Œ Missing: {filename}")
    else:
        print(f"   âœ… Found: {filename}")

if missing_files:
    raise FileNotFoundError(
        f"â�Œ Missing required files:\n"
        f"   {missing_files}\n"
        f"   Please check the dataset contents"
    )

print("âœ… Found all available required files!")

# Load the available files
datasets = {}
for name, filename in required_files.items():
    filepath = os.path.join(dataset_dir, filename)
    try:
        print(f"\n   â�³ Loading {name} dataset from {filename}")
        datasets[name.lower().replace(' ', '_')] = reduce_mem_usage(pd.read_csv(filepath))
        print(f"   âœ… {name} dataset loaded successfully")
    except Exception as e:
        print(f"   â�Œ Error loading {name}: {str(e)}")
        raise

# Assign to variables
train_df = datasets['train']
test_df = datasets['test']

print("\nğŸ“Š Successfully loaded datasets:")
print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")



print("\nğŸ”§ Starting feature engineering...")

# First we need to create transactions_df since it's not defined
# Based on previous output, we'll assume we only have train_df and test_df
# Let's create combined_df first since it's needed
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df['is_train'] = combined_df[TARGET].notna().astype(int)

# Now we need to create transactions_df - since we don't have the chunks,
# we'll use what we have from train/test data
transactions_df = pd.DataFrame({
    'base_coin': combined_df[MINT_ID],
    'slot': combined_df['slot_min'],  # Using slot_min as placeholder
    'block_time': pd.to_datetime('now')  # Placeholder value
})

# ğŸ”— Merge token creation info with transactions
transactions_df = pd.merge(
    transactions_df,
    combined_df[[MINT_ID, 'slot_min']],
    left_on='base_coin',
    right_on=MINT_ID,
    how='left'
)

# â�³ Filter to first BLOCK_LIMIT blocks
transactions_df = transactions_df[
    transactions_df['slot'] <= transactions_df['slot_min'] + BLOCK_LIMIT
].copy()

# Since we don't have dune_info_df and onchain_info_df (they were missing)
# we'll skip those parts and work with what we have

# ğŸ”  Encode creator addresses if the column exists
if 'creator' in combined_df.columns:
    le = LabelEncoder()
    combined_df['creator_encoded'] = le.fit_transform(
        combined_df['creator'].fillna('unknown'))
    print("âœ… Creator addresses encoded!")
else:
    print("âš ï¸� 'creator' column not found - skipping encoding")

print("âœ… Basic feature engineering completed!")

# Show resulting data shapes
print("\nğŸ“Š Current Data Shapes:")
print(f"transactions_df: {transactions_df.shape}")
print(f"combined_df: {combined_df.shape}")



print("\nğŸ“ˆ Generating transaction features...")

# Check available columns first
print("ğŸ”� Available columns in transactions_df:", transactions_df.columns.tolist())

# Simplified feature generation based on available columns
available_columns = transactions_df.columns.tolist()
agg_funcs = {}

# Add basic aggregations for available columns
if 'slot' in available_columns:
    agg_funcs['slot'] = ['min', 'max', 'nunique', 'mean', 'std']
    
if 'block_time' in available_columns:
    agg_funcs['block_time'] = ['min', 'max', lambda x: (x.max() - x.min()).total_seconds()]

# Create aggregated features only if we have columns to aggregate
if agg_funcs:
    grouped_tx = transactions_df.groupby('base_coin')
    agg_features = grouped_tx.agg(agg_funcs)
    agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
    agg_features = agg_features.reset_index().rename(columns={'base_coin': MINT_ID})
    
    # Merge basic features
    combined_df = pd.merge(combined_df, agg_features, on=MINT_ID, how='left')
    print(f"âœ… Created {len(agg_funcs)} basic transaction features")
else:
    print("âš ï¸� No transaction features created - missing required columns")
    agg_features = pd.DataFrame({MINT_ID: combined_df[MINT_ID].unique()})

# Add simple count features
if 'base_coin' in available_columns:
    tx_counts = transactions_df['base_coin'].value_counts().reset_index()
    tx_counts.columns = [MINT_ID, 'tx_count']
    combined_df = pd.merge(combined_df, tx_counts, on=MINT_ID, how='left')
    print("âœ… Added transaction counts")

# Show resulting features
print("\nğŸ“Š Generated Features Summary:")
print("Available features in combined_df:", [col for col in combined_df.columns if col not in [MINT_ID, TARGET, 'is_train']])
print("combined_df shape:", combined_df.shape)



print("\nâœ¨ Creating derived features...")

# Check available columns first
available_cols = combined_df.columns.tolist()
print("ğŸ”� Available columns:", available_cols)

# â�±ï¸� Time-based features (only create what's possible)
if all(col in available_cols for col in ['block_time_max', 'block_time_min']):
    combined_df['tx_duration_seconds'] = (
        combined_df['block_time_max'] - combined_df['block_time_min']).dt.total_seconds()
    print("âœ… Created tx_duration_seconds")
else:
    print("âš ï¸� Missing columns for tx_duration_seconds")

if all(col in available_cols for col in ['slot_max', 'slot_min_x']):
    combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min_x']
    print("âœ… Created tx_duration_slots")
elif all(col in available_cols for col in ['slot_max', 'slot_min_y']):
    combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min_y']
    print("âœ… Created tx_duration_slots (using slot_min_y)")
else:
    print("âš ï¸� Missing columns for tx_duration_slots")

if all(col in available_cols for col in ['tx_duration_seconds', 'tx_count']):
    combined_df['avg_time_between_tx'] = combined_df['tx_duration_seconds'] / (combined_df['tx_count'] + 1e-6)
    print("âœ… Created avg_time_between_tx")
else:
    print("âš ï¸� Missing columns for avg_time_between_tx")

if all(col in available_cols for col in ['tx_count', 'slot_nunique']):
    combined_df['tx_per_slot'] = combined_df['tx_count'] / (combined_df['slot_nunique'] + 1e-6)
    print("âœ… Created tx_per_slot")
else:
    print("âš ï¸� Missing columns for tx_per_slot")

# ğŸ“ˆ Basic volatility metrics (if we had transaction amounts)
if 'quote_coin_amount_mean' in available_cols and 'quote_coin_amount_std' in available_cols:
    combined_df['sol_volatility'] = combined_df['quote_coin_amount_std'] / (combined_df['quote_coin_amount_mean'] + 1e-6)
    print("âœ… Created sol_volatility")
else:
    print("âš ï¸� Missing columns for sol_volatility")

# ğŸ•’ Time-based features (if created_at exists)
if 'created_at' in available_cols:
    try:
        combined_df['hour_created'] = pd.to_datetime(combined_df['created_at']).dt.hour
        combined_df['day_created'] = pd.to_datetime(combined_df['created_at']).dt.dayofweek
        print("âœ… Created time-based features")
    except:
        print("âš ï¸� Could not create time-based features from created_at")
else:
    print("âš ï¸� Missing created_at column for time features")

print("\nğŸ“Š Current Features Summary:")
print("New features added:", [col for col in combined_df.columns if col not in available_cols])
print("combined_df shape:", combined_df.shape)



print("\nğŸ”� Selecting final features...")

# ğŸ—‘ï¸� Columns to drop (updated based on available columns)
features_to_drop = [
    MINT_ID, TARGET, 'is_train', 'Unnamed: 0'
]

# Add optional columns to drop if they exist
optional_drops = [
    'slot_graduated', 'slot_min', 'name', 'symbol', 
    'token_uri', 'created_at', 'init_tx', 'block_time_min',
    'block_time_max', 'creator', 'is_valid'
]

for col in optional_drops:
    if col in combined_df.columns:
        features_to_drop.append(col)

# ğŸ“‹ Final feature list
features = [col for col in combined_df.columns if col not in features_to_drop]

# Identify categorical features (only if they exist)
categorical_features = []
for col in ['creator_encoded', 'hour_created', 'day_created']:
    if col in combined_df.columns:
        categorical_features.append(col)

# ğŸ”¢ Ensure numeric types
for f in features:
    if combined_df[f].dtype == 'object':
        try:
            combined_df[f] = pd.to_numeric(combined_df[f])
        except:
            if f in features: 
                features.remove(f)
                print(f"âš ï¸� Removed non-numeric feature: {f}")

# âœ‚ï¸� Separate train and test
train_processed = combined_df[combined_df['is_train'] == 1].reset_index(drop=True)
train_processed[TARGET] = train_processed[TARGET].astype(int)
test_processed = combined_df[combined_df['is_train'] == 0].reset_index(drop=True)

X = train_processed[features]
y = train_processed[TARGET]
X_test = test_processed[features]

# ğŸ§¹ Clean up memory (only variables that exist)
vars_to_delete = ['combined_df']
if 'transactions_df' in globals():
    vars_to_delete.append('transactions_df')

for var in vars_to_delete:
    if var in globals():
        del globals()[var]
gc.collect()

print(f"âœ… Selected {len(features)} features for modeling")
print(f"ğŸ“Š Training data shape: {X.shape}")
print(f"ğŸ“ˆ Test data shape: {X_test.shape}")
print(f"ğŸ”¤ Categorical features: {categorical_features}")



print("\nğŸ¤– Training models...")

import re  # Added missing import

# Clean feature names for XGBoost compatibility
X_train = X.copy()
X_test = X_test.copy()
X_val = X.copy()  # Placeholder for validation

# Clean column names (remove special characters)
def clean_column_names(df):
    """Clean column names by removing special characters"""
    df.columns = [re.sub(r'[\[\]<>]', '', str(col)) for col in df.columns]
    return df

X_train = clean_column_names(X_train)
X_test = clean_column_names(X_test)

# ğŸ“Š Initialize predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

# ğŸ�¯ Initialize K-Fold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# ğŸ“Š Feature importance storage
feature_importance = pd.DataFrame(index=X_train.columns)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”„ Fold {fold+1}/{N_SPLITS}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Clean feature names for this fold
    X_train = clean_column_names(X_train)
    X_val = clean_column_names(X_val)
    
    # ======================
    # ğŸ’¡ LightGBM Model
    # ======================
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'n_estimators': 500,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 1,
        'random_state': RANDOM_SEED + fold,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=VERBOSE_EVAL),
            lgb.log_evaluation(VERBOSE_EVAL)
        ]
    )
    
    # ======================
    # ğŸ�± CatBoost Model
    # ======================
    cb_params = {
        'iterations': 500,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 5,
        'random_strength': 0.1,
        'bagging_temperature': 0.8,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': RANDOM_SEED + fold,
        'verbose': False,
        'task_type': 'CPU',
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
    }
    
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE_EVAL
    )
    
    # ======================
    # â�Œ XGBoost Model
    # ======================
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED + fold,
        'verbosity': 0,
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE_EVAL
    )
    
    # ======================
    # ğŸ�† Ensemble Predictions
    # ======================
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    cb_pred = cb_model.predict_proba(X_val)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    
    # Weighted ensemble
    ensemble_pred = 0.4*lgb_pred + 0.4*cb_pred + 0.2*xgb_pred
    oof_preds[val_idx] = ensemble_pred
    
    # Test predictions
    test_preds += (0.4*lgb_model.predict_proba(X_test)[:, 1] + 
                  0.4*cb_model.predict_proba(X_test)[:, 1] + 
                  0.2*xgb_model.predict_proba(X_test)[:, 1]) / N_SPLITS
    
    # Store models and feature importance
    models.append((lgb_model, cb_model, xgb_model))
    feature_importance[f'fold_{fold+1}_lgb'] = lgb_model.feature_importances_
    feature_importance[f'fold_{fold+1}_cb'] = cb_model.feature_importances_
    feature_importance[f'fold_{fold+1}_xgb'] = xgb_model.feature_importances_
    
    # Fold evaluation
    fold_score = log_loss(y_val, ensemble_pred)
    print(f"ğŸ“Š Fold {fold+1} Ensemble LogLoss: {fold_score:.5f}")

# ======================
# ğŸ�¯ Overall Evaluation
# ======================
overall_score = log_loss(y, oof_preds)
print(f"\nğŸ�† Overall OOF Ensemble LogLoss: {overall_score:.5f}")

# Feature importance analysis
feature_importance['mean_importance'] = feature_importance.mean(axis=1)
feature_importance = feature_importance.sort_values('mean_importance', ascending=False)
print("\nğŸ”� Top Features by Importance:")
print(feature_importance['mean_importance'].head(20))

print("\nâœ… Model training completed successfully!")



print("\nğŸ“¤ Generating competition submission...")

submission_df = pd.DataFrame({
    MINT_ID: test_processed[MINT_ID],
    TARGET: test_preds
})

# âœ‚ï¸� Clip probabilities for competition requirements
submission_df[TARGET] = np.clip(submission_df[TARGET], 0.0001, 0.9999)

# ğŸ’¾ Save submission file
submission_df.to_csv(SUBMISSION_FILE, index=False)

print(f"âœ… Submission saved to {SUBMISSION_FILE}")
print("\nğŸ�† Top 5 predictions:")
print(submission_df.head())

print("\nğŸ�‰ Pipeline completed successfully! Ready for submission! ğŸš€")

