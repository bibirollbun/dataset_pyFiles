import pandas as pd
import numpy as np
import glob
import os
import gc  # Garbage collection
from tqdm.auto import tqdm  # Progress bars

# Modeling & Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import log_loss


# Configuration

DATA_PATH = '/kaggle/input/pump-fun-graduation-february-2025' # Adjust if your data is elsewhere
CHUNK_PATTERN = os.path.join(DATA_PATH, 'chunk*.csv')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')
DUNE_INFO_FILE = os.path.join(DATA_PATH, 'dune_token_info.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_PATH, 'token_info_onchain_divers.csv')
SUBMISSION_FILE = 'submission.csv'

TARGET = 'has_graduated'
MINT_ID = 'mint'
BLOCK_LIMIT = 100 # Only use data from first 100 blocks post-mint
N_SPLITS = 5 # Number of folds for cross-validation
RANDOM_SEED = 42


# --- 1. Load Data ---
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
dune_info_df = pd.read_csv(DUNE_INFO_FILE)
onchain_info_df = pd.read_csv(ONCHAIN_INFO_FILE)

# Combine train and test for easier processing
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Load and combine chunk files
all_chunk_files = glob.glob(CHUNK_PATTERN)
print(f"Found {len(all_chunk_files)} chunk files.")

chunk_list = []
for f in tqdm(all_chunk_files, desc="Loading chunks"):
    try:
        chunk_list.append(pd.read_csv(f))
    except Exception as e:
        print(f"Error loading {f}: {e}")
if not chunk_list:
    raise ValueError("No chunk files loaded. Check CHUNK_PATTERN and file existence.")

transactions_df = pd.concat(chunk_list, ignore_index=True)


transactions_df.head()


# Convert timestamps/slots for filtering
transactions_df['block_time'] = pd.to_datetime(transactions_df['block_time'], errors='coerce')
# Ensure slot is numeric
transactions_df['slot'] = pd.to_numeric(transactions_df['slot'], errors='coerce')
combined_df['slot_min'] = pd.to_numeric(combined_df['slot_min'], errors='coerce')

# --- 2. Data Merging and Preprocessing ---
print("Merging data...")

# Merge token creation info (slot_min) with transactions
transactions_df = pd.merge(
    transactions_df,
    combined_df[[MINT_ID, 'slot_min']],
    left_on='base_coin', # Assuming base_coin is the token mint address
    right_on=MINT_ID,
    how='left'
)

# !!! Crucial Filter: Only keep transactions within the first 100 blocks !!!
transactions_df = transactions_df[
    transactions_df['slot'] <= transactions_df['slot_min'] + BLOCK_LIMIT
    ]
transactions_df.columns


# Rename columns for clarity before merging metadata
dune_info_df = dune_info_df.rename(columns={'token_mint_address': MINT_ID})
# Select relevant columns and handle potential duplicates (keep first)
dune_info_df = dune_info_df[[MINT_ID, 'decimals', 'name', 'symbol', 'token_uri', 'created_at', 'init_tx']].drop_duplicates(subset=[MINT_ID], keep='first')
dune_info_df['created_at'] = pd.to_datetime(dune_info_df['created_at'], errors='coerce')

onchain_info_df = onchain_info_df.rename(columns={'mint': MINT_ID})
# Select relevant columns and handle potential duplicates (keep first)
onchain_info_df = onchain_info_df[[MINT_ID, 'creator', 'bundle_size', 'gas_used']].drop_duplicates(subset=[MINT_ID], keep='first')
# Ensure numeric types
onchain_info_df['bundle_size'] = pd.to_numeric(onchain_info_df['bundle_size'], errors='coerce').fillna(0) # Assume NaN means not bundled (or size 1?) - check data desc
onchain_info_df['gas_used'] = pd.to_numeric(onchain_info_df['gas_used'], errors='coerce')

dune_info_df.columns, onchain_info_df.columns


# Merge metadata into the combined train/test dataframe
combined_df = pd.merge(combined_df, dune_info_df, on=MINT_ID, how='left')
combined_df = pd.merge(combined_df, onchain_info_df, on=MINT_ID, how='left')
combined_df.columns


# --- 3. Exploratory Data Analysis (Conceptual) ---
print("Basic EDA (Conceptual):")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Transactions shape (first 100 blocks): {transactions_df.shape}")
print(f"Combined shape before features: {combined_df.shape}")

# Check missing values in combined metadata
print("\nMissing values in combined metadata:")
combined_df.isnull().sum() / len(combined_df)


# Check target distribution
print("\nTarget Distribution:")
print(combined_df[TARGET].value_counts(normalize=True))


# Basic checks on transaction data
print("\nTransaction Data Info:")
transactions_df.info()


print("\nSample Transactions:")
transactions_df.head()


# --- 4. Feature Engineering ---
print("Starting Feature Engineering...")

# Group transactions by token mint
grouped_tx = transactions_df.groupby('base_coin') # Group by the token's mint address

# Aggregation dictionary
agg_funcs = {
    'tx_idx': ['count'], # Total transactions
    'block_time': ['min', 'max'], # First and last transaction time
    'slot': ['min', 'max', 'nunique'], # First, last, and number of unique blocks with activity
    'signing_wallet': ['nunique'], # Number of unique traders
    'quote_coin_amount': ['sum', 'mean', 'std', 'max'], # SOL volume stats
    'base_coin_amount': ['sum', 'mean', 'std', 'max'], # Token volume stats
    'virtual_sol_balance_after': ['last', 'max', 'min', 'mean', 'std'], # SOL balance proxy
    'virtual_token_balance_after': ['last', 'max', 'min', 'mean', 'std'] # Token balance proxy
}

# Perform aggregation
agg_features = grouped_tx.agg(agg_funcs)
agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values] # Flatten multi-index
agg_features = agg_features.reset_index().rename(columns={'base_coin': MINT_ID})


# --- Buy/Sell specific features ---
buy_tx = transactions_df[transactions_df['direction'] == 'buy']
sell_tx = transactions_df[transactions_df['direction'] == 'sell']

grouped_buy = buy_tx.groupby('base_coin')
grouped_sell = sell_tx.groupby('base_coin')

buy_agg = grouped_buy.agg({
    'tx_idx': ['count'],
    'signing_wallet': ['nunique'],
    'quote_coin_amount': ['sum', 'mean', 'max'],
    'base_coin_amount': ['sum', 'mean', 'max'],
}).reset_index()
buy_agg.columns = [MINT_ID] + ['buy_' + '_'.join(col).strip() for col in buy_agg.columns[1:]]

sell_agg = grouped_sell.agg({
    'tx_idx': ['count'],
    'signing_wallet': ['nunique'],
    'quote_coin_amount': ['sum', 'mean', 'max'],
    'base_coin_amount': ['sum', 'mean', 'max'],
}).reset_index()
sell_agg.columns = [MINT_ID] + ['sell_' + '_'.join(col).strip() for col in sell_agg.columns[1:]]

buy_agg.columns, sell_agg.columns,


print("Merging aggregated features...")

combined_df = pd.merge(combined_df, agg_features[[c for c in agg_features.columns if c != 'slot_min']], on=MINT_ID, how='left')

# --- Merge Buy/Sell specific features --- # Add prints here too if needed
combined_df = pd.merge(combined_df, buy_agg, on=MINT_ID, how='left')

combined_df = pd.merge(combined_df, sell_agg, on=MINT_ID, how='left')


# --- Derived Features ---
print("Calculating derived features...")

# --- Check if the required columns exist before calculation ---
required_cols_for_duration = ['block_time_max', 'block_time_min', 'slot_max', 'slot_min', 'tx_idx_count', 'slot_nunique']

# Time-based features
combined_df['tx_duration_seconds'] = (combined_df['block_time_max'] - combined_df['block_time_min']).dt.total_seconds()
# Use the slot_min and slot_max derived from the transaction aggregation
combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min']
combined_df['avg_time_between_tx'] = combined_df['tx_duration_seconds'] / (combined_df['tx_idx_count'] + 1e-6) # Avoid division by zero for tokens with 0 tx
combined_df['tx_per_slot'] = combined_df['tx_idx_count'] / (combined_df['slot_nunique'] + 1e-6) # Avoid division by zero


# Ratio features (ensure denominators are checked)
required_cols_for_ratios = [
    'buy_tx_idx_count', 'sell_tx_idx_count',
    'buy_quote_coin_amount_sum', 'sell_quote_coin_amount_sum',
    'buy_signing_wallet_nunique', 'sell_signing_wallet_nunique', 'signing_wallet_nunique'
]

combined_df['buy_sell_count_ratio'] = combined_df['buy_tx_idx_count'] / (combined_df['sell_tx_idx_count'] + 1e-6)
combined_df['buy_sell_vol_ratio'] = combined_df['buy_quote_coin_amount_sum'] / (combined_df['sell_quote_coin_amount_sum'] + 1e-6)
combined_df['unique_buyer_ratio'] = combined_df['buy_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)
combined_df['unique_seller_ratio'] = combined_df['sell_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)


# Creator interaction (Check if creator traded)
creator_trades = transactions_df.groupby(['base_coin', 'signing_wallet']).size().reset_index(name='trade_count')
# Need onchain_info_df merged *before* this step if not already done

creator_trades = pd.merge(creator_trades, onchain_info_df[[MINT_ID, 'creator']], left_on='base_coin', right_on=MINT_ID, how='inner')
creator_trades = creator_trades[creator_trades['signing_wallet'] == creator_trades['creator']]
creator_trades = creator_trades[['base_coin', 'trade_count']].rename(columns={'base_coin': MINT_ID, 'trade_count': 'creator_trade_count'})
creator_trades = creator_trades.drop_duplicates(subset=[MINT_ID], keep='first')



combined_df = pd.merge(combined_df, creator_trades, on=MINT_ID, how='left')

combined_df['creator_traded'] = combined_df['creator_trade_count'].notna().astype(int)
combined_df['creator_trade_count'] = combined_df['creator_trade_count'].fillna(0)


# --- Final Feature Selection ---
print("Selecting final features...")
features_to_drop = [
    MINT_ID, TARGET, 'slot_graduated', 'is_train', 'slot_min', # IDs, target, helpers
    'name', 'symbol', 'token_uri', 'created_at', 'init_tx', # Original text/metadata unless parsed
    'block_time_min', 'block_time_max', # Used to create duration, maybe drop
    'creator', # Use encoded version
    'is_valid', 'Unnamed: 0' # different from origin data
    # Potentially drop columns with very high correlation or low variance if needed
]

features = [col for col in combined_df.columns if col not in features_to_drop]
categorical_features = ['creator_encoded'] # Add more if you encode others

print(f"Using {len(features)} features: {features}")
# Ensure all features are numeric or suitable for the model (CatBoost handles categoricals)
for f in features:
    if combined_df[f].dtype == 'object':
        print(f"Warning: Feature '{f}' is object type. Ensure proper handling.")
        # Attempt conversion or drop
        try:
            combined_df[f] = pd.to_numeric(combined_df[f])
        except:
            print(f"Could not convert {f} to numeric. Consider encoding or dropping.")
            if f in features: features.remove(f)


# Separate train and test again
train_processed = combined_df[combined_df['is_train'] == 1].reset_index(drop=True)
train_processed[TARGET] = train_processed[TARGET].astype(int)
test_processed = combined_df[combined_df['is_train'] == 0].reset_index(drop=True)

X = train_processed[features]
y = train_processed[TARGET]
X_test = test_processed[features]

# Clean up memory
del combined_df, transactions_df, agg_features, buy_agg, sell_agg, creator_trades, chunk_list
gc.collect()


# --- 5. Model Training ---
# --- Option A: LightGBM ---
print("\nTraining LightGBM model...")
lgb_oof_preds = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))
lgb_models = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgb_params = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'n_estimators': 1700, # High number, use early stopping
        'learning_rate': 0.01,
        'num_leaves': 31, # Adjust based on data complexity
        'max_depth': 5, # No limit
        'seed': RANDOM_SEED + fold,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)], # Stop if validation logloss doesn't improve for 100 rounds
              categorical_feature=[f for f in categorical_features if f in X.columns] # Pass categorical feature names
              )

    val_preds = model.predict_proba(X_val)[:, 1]
    lgb_oof_preds[val_idx] = val_preds
    lgb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    lgb_models.append(model)
    print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds)}")

from sklearn.metrics import log_loss
overall_oof_logloss = log_loss(y, lgb_oof_preds)
print(f"\nLGBM Overall OOF LogLoss: {overall_oof_logloss}")


# --- Option B: CatBoost ---
print("\nTraining CatBoost model...")
cb_oof_preds = np.zeros(len(X))
cb_test_preds = np.zeros(len(X_test))
cb_models = []

# Identify categorical features for CatBoost (by name or index)
cat_features_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Fill NaNs before CatBoost if not handled earlier or use CatBoost's internal handling
    # X_train = X_train.fillna(-999) # Example simple fill
    # X_val = X_val.fillna(-999)

    model = cb.CatBoostClassifier(
        iterations=1700, # High number, use early stopping
        learning_rate=0.02,
        loss_function='Logloss',
        eval_metric='Logloss',
        task_type='GPU',
        depth=7, # Adjust as needed
        l2_leaf_reg=3, # Regularization
        random_seed=RANDOM_SEED + fold,
        verbose=0, # Suppress verbose output during training
        early_stopping_rounds=100,
        cat_features=cat_features_indices, # Pass indices or names
        # task_type="GPU", # Uncomment if you have a suitable GPU and installed CatBoost with GPU support
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              use_best_model=True)

    val_preds = model.predict_proba(X_val)[:, 1]
    cb_oof_preds[val_idx] = val_preds
    cb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS # Fill test NaNs same way if needed
    cb_models.append(model)
    print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds)}")


from sklearn.metrics import log_loss
overall_oof_logloss_cb = log_loss(y, cb_oof_preds)
print(f"\nCatBoost Overall OOF LogLoss: {overall_oof_logloss_cb}")


# --- Model C: XGBoost with GPU ---
print("\nTraining XGBoost model with GPU acceleration...")


xgb_oof_preds = np.zeros(len(X))
xgb_test_preds = np.zeros(len(X_test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- XGB Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # GPU ID - alternate between GPUs for different folds
    gpu_id = fold % 2  # This will alternate between GPU 0 and GPU 1

    # XGBoost parameters with GPU configuration
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.01,
        'n_estimators': 1700,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'alpha': 0.1,  # L1 regularization
        'lambda': 0.1,  # L2 regularization
        'random_state': RANDOM_SEED + fold,
        'verbosity': 0,
        'tree_method': 'gpu_hist',  # Use GPU for tree construction
        'gpu_id': gpu_id,  # Specify which GPU to use
        'predictor': 'gpu_predictor',  # Use GPU for prediction
    }

    # Create XGBoost classifier with GPU support
    model = xgb.XGBClassifier(**xgb_params)

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Make predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    xgb_oof_preds[val_idx] = val_preds
    
    xgb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds)}")

overall_oof_logloss_xgb = log_loss(y, xgb_oof_preds)
print(f"\nXGBoost Overall OOF LogLoss: {overall_oof_logloss_xgb}")
gc.collect()


final_test_preds = (lgb_test_preds + cb_test_preds + xgb_test_preds) / 3 # Simple Averaging Ensemble Example


submission_df = pd.DataFrame({
    MINT_ID: test_processed[MINT_ID],
    TARGET: final_test_preds
})

# Ensure probabilities are within [0, 1] range (optional clipping)
submission_df[TARGET] = np.clip(submission_df[TARGET], 0.0001, 0.9999)

submission_df.to_csv(SUBMISSION_FILE, index=False)

print(f"Submission file saved to: {SUBMISSION_FILE}")
print("\nSubmission file head:")
print(submission_df.head())

print("\nScript finished successfully!")

