# This code scored 0.0291 on the public leaderboard. It takes about 90 minutes to run the data preparation.
# The prepared datasets are then exported as pkl files.

# Cell 1: Imports and Setup
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from collections import defaultdict
import glob
import os
import gc
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Configuration ---\n",
DATA_DIR = '/kaggle/input/pump-fun-graduation-february-2025'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_unlabeled.csv')
DUNE_INFO_FILE = os.path.join(DATA_DIR, 'dune_token_info.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_DIR, 'token_info_onchain_divers.csv')
SUBMISSION_FILE = 'submission_ensemble.csv'
# Output paths for intermediate features
FEATURES_PKL = 'features_df.pkl'
X_TRAIN_PKL = 'X_train_processed.pkl'
Y_TRAIN_PKL = 'y_train.pkl'
X_TEST_PKL = 'X_test_processed.pkl'
TEST_MINTS_PKL = 'test_mints.pkl'


# --- CV and Model Settings ---
N_SPLITS = 5 # Number of folds for StratifiedKFold
RANDOM_SEED = 42
RUN_FEATURE_ENGINEERING = True # Set to False to load features from pickle

# Early stopping rounds for models during CV
EARLY_STOPPING_ROUNDS = 100

# --- Feature Engineering Settings ---
CHUNKSIZE = 1_000_000
BLOCK_WINDOW = 100

# --- Predefined LGBM Best Params (from Optuna in private notebook) ---
# Using the provided tuned parameters
LGBM_BEST_PARAMS = {
    'objective': 'binary',
    'metric': 'logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 2500,
    'learning_rate': 0.013094862430042712,
    'num_leaves': 30,
    'max_depth': 7,
    'lambda_l1': 0.47723465286137357, # reg_alpha
    'lambda_l2': 0.07951439672408593, # reg_lambda
    'colsample_bytree': 0.6640835024351832,
    'subsample': 0.5933658927947468,
    'min_child_samples': 90,
    'seed': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1,
}

# --- Basic XGBoost Params ---
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05, # learning_rate
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'lambda': 1, # L2 reg
    'alpha': 0, # L1 reg
    'seed': RANDOM_SEED,
    'nthread': -1,
    'tree_method': 'hist'
}
XGB_N_ESTIMATORS = 2000 

# --- Basic CatBoost Params ---
CAT_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'random_seed': RANDOM_SEED,
    'thread_count': -1,
    'verbose': 0,
    'early_stopping_rounds': EARLY_STOPPING_ROUNDS 
}
CAT_ITERATIONS = 2000 


# Cell 2: Data Loading 
print("Cell 2: Loading base data...")
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    try: dune_df = pd.read_csv(DUNE_INFO_FILE)
    except FileNotFoundError: print(f"Warning: {DUNE_INFO_FILE} not found."); dune_df = None
    try: onchain_df = pd.read_csv(ONCHAIN_INFO_FILE)
    except FileNotFoundError: print(f"Warning: {ONCHAIN_INFO_FILE} not found."); onchain_df = None
except FileNotFoundError as e: print(f"Error loading train/test files: {e}"); exit()
print(f"Loaded {len(train_df)} training examples and {len(test_df)} test examples.")
gc.collect()


# Cell 3: Feature Engineering Function Definition
print("Cell 3: Defining memory-optimized feature engineering function...")
# (Use the memory-optimized generate_chunk_features_optimized function from the previous response)
def generate_chunk_features_optimized(chunk_files, train_data, test_data, dune_info_file, onchain_info_file, block_window, chunksize):
    # Combine mints and slot_min for efficient processing
    print("  Combining mint info...")
    train_mints_info = train_data[['mint', 'slot_min']].copy()
    test_mints_info = test_data[['mint', 'slot_min']].copy()
    all_mints_df = pd.concat([train_mints_info, test_mints_info], ignore_index=True)
    all_mints_map = pd.Series(all_mints_df.slot_min.values, index=all_mints_df.mint).to_dict()
    print(f"  Total unique mints to process: {len(all_mints_map)}")
    del train_mints_info, test_mints_info, all_mints_df
    gc.collect()

    # --- Start Chunk Processing ---
    print("  Starting feature aggregation from chunks...")
    feature_agg = defaultdict(lambda: {
        'tx_count': 0, 'buy_count': 0, 'sell_count': 0,
        'total_sol_volume': 0.0, 'buy_sol_volume': 0.0, 'sell_sol_volume': 0.0,
        'total_token_volume': 0.0, 'buy_token_volume': 0.0, 'sell_token_volume': 0.0,
        'max_sol_balance': -1.0, 'last_sol_balance': -1.0, 'last_slot': -1,
        'first_slot': float('inf'), 'total_fees': 0.0, 'total_gas_used': 0.0,
        'sum_sq_sol_buy': 0.0, 'sum_sq_sol_sell': 0.0,
        'creator_buys': 0, 'creator_sells': 0, 'creator_buy_vol': 0.0, 'creator_sell_vol': 0.0,
        'first_50_block_tx': 0, 'last_50_block_tx': 0,
        '_unique_wallets_set': set()
    })

    creator_map = {}
    onchain_metadata_loaded = False
    try:
        onchain_meta_temp = pd.read_csv(onchain_info_file, usecols=['mint', 'creator'])
        creator_map = pd.Series(onchain_meta_temp.creator.values, index=onchain_meta_temp.mint).drop_duplicates().to_dict()
        onchain_metadata_loaded = True
        print(f"  Creator map created with {len(creator_map)} entries.")
        del onchain_meta_temp
        gc.collect()
    except Exception as e:
        print(f"  Warning: Cannot create creator map: {e}. Creator features will be zero.")


    for chunk_file in tqdm(chunk_files, desc="  Processing Chunks"):
        try:
            chunk_iter = pd.read_csv(
                chunk_file, chunksize=chunksize,
                dtype={
                    'slot': np.uint32, 'tx_idx': np.uint16, 'signing_wallet': 'category',
                    'direction': 'category', 'base_coin': 'category',
                    'base_coin_amount': np.float32, 'quote_coin_amount': np.float32,
                    'virtual_token_balance_after': np.float64,
                    'virtual_sol_balance_after': np.float64,
                    'fee': np.float32, 'consumed_gas': np.float32
                },
                usecols=[
                    'slot', 'tx_idx', 'signing_wallet', 'direction', 'base_coin',
                    'base_coin_amount', 'quote_coin_amount', 'virtual_sol_balance_after',
                    'fee', 'consumed_gas'
                ], low_memory=True
            )

            for chunk_df in chunk_iter:
                relevant_mints_in_chunk = set(chunk_df['base_coin'].unique()) & set(all_mints_map.keys())
                if not relevant_mints_in_chunk: continue
                chunk_df = chunk_df[chunk_df['base_coin'].isin(relevant_mints_in_chunk)]
                if chunk_df.empty: continue

                chunk_df['slot_min'] = chunk_df['base_coin'].map(all_mints_map)
                chunk_df['slot_max_limit'] = chunk_df['slot_min'] + block_window
                chunk_df = chunk_df[chunk_df['slot'] <= chunk_df['slot_max_limit']].copy()
                if chunk_df.empty: continue

                if creator_map:
                    chunk_df['creator'] = chunk_df['base_coin'].map(creator_map)
                    chunk_df['is_creator_tx'] = chunk_df['signing_wallet'] == chunk_df['creator']
                else:
                    chunk_df['is_creator_tx'] = False

                chunk_df['relative_slot'] = chunk_df['slot'] - chunk_df['slot_min']

                if isinstance(chunk_df['base_coin'].dtype, pd.CategoricalDtype):
                    grouped = chunk_df.groupby(chunk_df['base_coin'].cat.codes)
                    mint_map = dict(enumerate(chunk_df['base_coin'].cat.categories))
                else:
                    grouped = chunk_df.groupby('base_coin')
                    mint_map = {mint: mint for mint in chunk_df['base_coin'].unique()}

                for group_key, group in grouped:
                    mint = mint_map[group_key]
                    token_features = feature_agg[mint]
                    group_creator = creator_map.get(mint)

                    token_features['tx_count'] += len(group)
                    buys = group[group['direction'] == 'buy']
                    sells = group[group['direction'] == 'sell']
                    token_features['buy_count'] += len(buys)
                    token_features['sell_count'] += len(sells)
                    token_features['_unique_wallets_set'].update(group['signing_wallet'].unique())
                    token_features['total_sol_volume'] += group['quote_coin_amount'].sum()
                    buy_vol = buys['quote_coin_amount'].sum()
                    sell_vol = sells['quote_coin_amount'].sum()
                    token_features['buy_sol_volume'] += buy_vol
                    token_features['sell_sol_volume'] += sell_vol
                    token_features['total_token_volume'] += group['base_coin_amount'].sum()
                    token_features['buy_token_volume'] += buys['base_coin_amount'].sum()
                    token_features['sell_token_volume'] += sells['base_coin_amount'].sum()
                    token_features['sum_sq_sol_buy'] += (buys['quote_coin_amount']**2).sum()
                    token_features['sum_sq_sol_sell'] += (sells['quote_coin_amount']**2).sum()

                    if not group.empty:
                        current_max_sol = group['virtual_sol_balance_after'].max()
                        if current_max_sol > token_features['max_sol_balance']:
                            token_features['max_sol_balance'] = current_max_sol
                        latest_tx_in_group = group.loc[group.sort_values(['slot', 'tx_idx']).index[-1]]
                        if latest_tx_in_group['slot'] >= token_features['last_slot']:
                            token_features['last_sol_balance'] = latest_tx_in_group['virtual_sol_balance_after']
                            token_features['last_slot'] = latest_tx_in_group['slot']
                    current_min_slot = group['slot'].min()
                    if current_min_slot < token_features['first_slot']:
                        token_features['first_slot'] = current_min_slot
                    token_features['total_fees'] += group['fee'].sum(skipna=True)
                    token_features['total_gas_used'] += group['consumed_gas'].sum(skipna=True)

                    if group_creator:
                        creator_txs = group[group['is_creator_tx']]
                        creator_buys_group = creator_txs[creator_txs['direction'] == 'buy']
                        creator_sells_group = creator_txs[creator_txs['direction'] == 'sell']
                        token_features['creator_buys'] += len(creator_buys_group)
                        token_features['creator_sells'] += len(creator_sells_group)
                        token_features['creator_buy_vol'] += creator_buys_group['quote_coin_amount'].sum()
                        token_features['creator_sell_vol'] += creator_sells_group['quote_coin_amount'].sum()
                        del creator_txs, creator_buys_group, creator_sells_group

                    token_features['first_50_block_tx'] += (group['relative_slot'] < 50).sum()
                    token_features['last_50_block_tx'] += (group['relative_slot'] >= 50).sum()

                del chunk_df, grouped, mint_map, group, buys, sells
                gc.collect()
        except Exception as e:
            print(f"Error processing chunk {chunk_file}: {e}")
            continue

    # --- Load Metadata ---
    print("  Loading metadata for final merge...")
    dune_metadata, onchain_metadata = None, None
    try: dune_metadata = pd.read_csv(dune_info_file)
    except Exception as e: print(f"    Warning: Cannot load dune metadata: {e}")
    try: onchain_metadata = pd.read_csv(onchain_info_file)
    except Exception as e: print(f"    Warning: Cannot load onchain metadata: {e}")

    # --- Finalize Features ---
    print("  Finalizing features...")
    feature_list = []
    for mint, data in tqdm(list(feature_agg.items()), desc="  Finalizing Features"):
        features = {'mint': mint}
        features['tx_count'] = data['tx_count']
        features['buy_count'] = data['buy_count']
        features['sell_count'] = data['sell_count']
        features['unique_wallets'] = len(data.pop('_unique_wallets_set', set()))
        features['buy_ratio'] = features['buy_count'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['sell_ratio'] = features['sell_count'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['total_sol_volume'] = data['total_sol_volume']
        features['buy_sol_volume'] = data['buy_sol_volume']
        features['sell_sol_volume'] = data['sell_sol_volume']
        features['avg_buy_sol_volume'] = features['buy_sol_volume'] / features['buy_count'] if features['buy_count'] > 0 else 0
        features['avg_sell_sol_volume'] = features['sell_sol_volume'] / features['sell_count'] if features['sell_count'] > 0 else 0
        features['net_sol_volume'] = features['buy_sol_volume'] - features['sell_sol_volume']
        features['sol_volume_per_wallet'] = features['total_sol_volume'] / features['unique_wallets'] if features['unique_wallets'] > 0 else 0
        features['max_sol_balance'] = data['max_sol_balance'] if data['max_sol_balance'] != -1.0 else 0.0
        features['last_sol_balance'] = data['last_sol_balance'] if data['last_sol_balance'] != -1.0 else 0.0
        if features['buy_count'] > 1:
            mean_buy = features['avg_buy_sol_volume']
            variance_buy = (np.float64(data['sum_sq_sol_buy']) / features['buy_count']) - (np.float64(mean_buy)**2)
            features['std_buy_sol_volume'] = np.sqrt(variance_buy) if variance_buy >= 0 else 0
        else: features['std_buy_sol_volume'] = 0.0
        if features['sell_count'] > 1:
            mean_sell = features['avg_sell_sol_volume']
            variance_sell = (np.float64(data['sum_sq_sol_sell']) / features['sell_count']) - (np.float64(mean_sell)**2)
            features['std_sell_sol_volume'] = np.sqrt(variance_sell) if variance_sell >= 0 else 0
        else: features['std_sell_sol_volume'] = 0.0
        slot_min_actual = all_mints_map.get(mint, None)
        if slot_min_actual is not None and data['first_slot'] != float('inf'):
            features['first_tx_slot_diff'] = data['first_slot'] - slot_min_actual
        else: features['first_tx_slot_diff'] = -1
        if slot_min_actual is not None and data['last_slot'] != -1:
             features['last_tx_slot_diff'] = data['last_slot'] - slot_min_actual
             if data['first_slot'] != float('inf'): features['tx_time_range'] = data['last_slot'] - data['first_slot']
             else: features['tx_time_range'] = 0
        else:
             features['last_tx_slot_diff'] = -1
             features['tx_time_range'] = -1
        features['avg_fee'] = data['total_fees'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['avg_gas_used'] = data['total_gas_used'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['creator_buys'] = data.get('creator_buys', 0)
        features['creator_sells'] = data.get('creator_sells', 0)
        features['creator_buy_vol'] = data.get('creator_buy_vol', 0.0)
        features['creator_sell_vol'] = data.get('creator_sell_vol', 0.0)
        features['creator_net_vol'] = features['creator_buy_vol'] - features['creator_sell_vol']
        features['creator_buy_ratio_tx'] = features['creator_buys'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['creator_sell_ratio_tx'] = features['creator_sells'] / features['tx_count'] if features['tx_count'] > 0 else 0
        features['creator_did_sell'] = 1 if features['creator_sells'] > 0 else 0
        features['first_50_block_tx'] = data.get('first_50_block_tx', 0)
        features['last_50_block_tx'] = data.get('last_50_block_tx', 0)
        features['first_last_50_ratio'] = features['first_50_block_tx'] / features['last_50_block_tx'] if features['last_50_block_tx'] > 0 else 0

        # Type casting
        for col, dtype in {
            'tx_count': np.uint32, 'buy_count': np.uint32, 'sell_count': np.uint32,
            'unique_wallets': np.uint32, 'creator_buys': np.uint16, 'creator_sells': np.uint16,
            'creator_did_sell': np.uint8, 'first_50_block_tx': np.uint32, 'last_50_block_tx': np.uint32,
            'first_tx_slot_diff': np.int32, 'last_tx_slot_diff': np.int32, 'tx_time_range': np.int32
            }.items():
            if col in features: features[col] = np.nan_to_num(features[col], nan=-1).astype(dtype)
        for col, dtype in {
            'buy_ratio': np.float32, 'sell_ratio': np.float32, 'total_sol_volume': np.float32,
            'buy_sol_volume': np.float32, 'sell_sol_volume': np.float32, 'avg_buy_sol_volume': np.float32,
            'avg_sell_sol_volume': np.float32, 'net_sol_volume': np.float32, 'sol_volume_per_wallet': np.float32,
            'max_sol_balance': np.float64, 'last_sol_balance': np.float64,
            'std_buy_sol_volume': np.float32, 'std_sell_sol_volume': np.float32,
            'avg_fee': np.float32, 'avg_gas_used': np.float32, 'creator_buy_vol': np.float32,
            'creator_sell_vol': np.float32, 'creator_net_vol': np.float32,
            'creator_buy_ratio_tx': np.float32, 'creator_sell_ratio_tx': np.float32,
            'first_last_50_ratio': np.float32
            }.items():
             if col in features: features[col] = np.nan_to_num(features[col], nan=0.0).astype(dtype)
        feature_list.append(features)

    features_out_df = pd.DataFrame(feature_list)
    del feature_list, feature_agg
    gc.collect()

    # --- Merge Metadata ---
    print("  Merging metadata features...")
    if dune_metadata is not None:
        dune_meta = dune_metadata[['token_mint_address', 'decimals', 'name', 'symbol', 'token_uri']].rename(columns={'token_mint_address': 'mint'})
        dune_meta['has_dune_meta'] = 1; dune_meta['has_token_uri'] = dune_meta['token_uri'].notna().astype(np.uint8)
        dune_meta['name_len'] = dune_meta['name'].str.len().fillna(0).astype(np.uint16); dune_meta['symbol_len'] = dune_meta['symbol'].str.len().fillna(0).astype(np.uint8)
        features_out_df = features_out_df.merge(dune_meta[['mint', 'decimals', 'has_dune_meta', 'has_token_uri', 'name_len', 'symbol_len']], on='mint', how='left')
        features_out_df['has_dune_meta'].fillna(0, inplace=True); features_out_df['has_token_uri'].fillna(0, inplace=True); features_out_df['decimals'].fillna(-1, inplace=True)
        features_out_df['name_len'].fillna(0, inplace=True); features_out_df['symbol_len'].fillna(0, inplace=True)
        features_out_df = features_out_df.astype({ 'has_dune_meta': np.uint8, 'has_token_uri': np.uint8, 'decimals': np.int8, 'name_len': np.uint16, 'symbol_len': np.uint8 })
        del dune_meta; gc.collect(); print(f"    Merged Dune metadata. Shape: {features_out_df.shape}")
    else:
        features_out_df = features_out_df.assign(has_dune_meta=np.uint8(0), has_token_uri=np.uint8(0), decimals=np.int8(-1), name_len=np.uint16(0), symbol_len=np.uint8(0))

    if onchain_metadata is not None:
         onchain_meta = onchain_metadata[['mint', 'bundle_size', 'gas_used']].copy()
         onchain_meta.rename(columns={'gas_used': 'creation_gas_used'}, inplace=True)
         onchain_meta['has_onchain_meta'] = 1
         features_out_df = features_out_df.merge(onchain_meta[['mint', 'bundle_size', 'creation_gas_used', 'has_onchain_meta']], on='mint', how='left')
         features_out_df['has_onchain_meta'].fillna(0, inplace=True); features_out_df['bundle_size'].fillna(0, inplace=True)
         median_gas = features_out_df['creation_gas_used'].median()
         features_out_df['creation_gas_used'].fillna(median_gas if pd.notna(median_gas) else 0, inplace=True)
         features_out_df = features_out_df.astype({ 'has_onchain_meta': np.uint8, 'bundle_size': np.uint16, 'creation_gas_used': np.float32 })
         del onchain_meta; gc.collect(); print(f"    Merged Onchain metadata. Shape: {features_out_df.shape}")
    else:
        features_out_df = features_out_df.assign(has_onchain_meta=np.uint8(0), bundle_size=np.uint16(0), creation_gas_used=np.float32(0))

    print(f"  Final feature set shape: {features_out_df.shape}")
    return features_out_df


# Cell 4: Feature Engineering Execution & Saving 
print("Cell 4: Running memory-optimized feature engineering...")
if RUN_FEATURE_ENGINEERING or not os.path.exists(FEATURES_PKL):
    CHUNK_FILES = sorted(glob.glob(os.path.join(DATA_DIR, 'chunk_*.csv')))
    if not CHUNK_FILES: print(f"Error: No chunk*.csv files found in {DATA_DIR}."); exit()
    else: print(f"Found {len(CHUNK_FILES)} chunk files in {DATA_DIR}.")

    features_df = generate_chunk_features_optimized(
        CHUNK_FILES, train_df, test_df, DUNE_INFO_FILE, ONCHAIN_INFO_FILE, BLOCK_WINDOW, CHUNKSIZE
    )
    if features_df.empty or features_df.shape[0] < 1: print("Error: Feature generation resulted in empty DF."); exit()
    print("Saving features to pickle file...")
    features_df.to_pickle(FEATURES_PKL); print(f"Features saved to {FEATURES_PKL}")
else:
    print(f"Loading features from {FEATURES_PKL}..."); features_df = pd.read_pickle(FEATURES_PKL); print("Features loaded.")

print("\n--- Final Features Info ---")
print(f"Shape: {features_df.shape}")
# print(features_df.head()) 
features_df.info(memory_usage='deep')
# print("\nFeature columns:", features_df.columns.tolist()) # Keep commented out unless needed
print(f"Number of features generated: {features_df.shape[1] - 1}")
gc.collect()


# Cell 5: Data Preparation for Model (Same as before)
print("Cell 5: Preparing data for modeling...")
train_merged = train_df.merge(features_df, on='mint', how='left')
test_merged = test_df.merge(features_df, on='mint', how='left')
test_mints_final = test_merged['mint'].copy()
test_mints_final.to_pickle(TEST_MINTS_PKL) # Save test mints

feature_cols = [col for col in features_df.columns if col not in ['mint']]
X = train_merged[feature_cols].copy() # Use copy to avoid SettingWithCopy issues later
y = train_merged['has_graduated'].astype(int).copy()
X_test = test_merged[feature_cols].copy()

print(f"Training data shape (X): {X.shape}"); print(f"Target data shape (y): {y.shape}"); print(f"Test data shape (X_test): {X_test.shape}")
print(f"Number of features: {len(feature_cols)}")
del train_df, test_df, train_merged, test_merged, features_df; gc.collect()


# Cell 6: Preprocessing (Imputation) & Saving Processed Data
print("Cell 6: Preprocessing data (Imputation)...")
print(f"NaNs in X before imputation: {X.isna().sum().sum()}"); print(f"NaNs in X_test before imputation: {X_test.isna().sum().sum()}")
X.replace([np.inf, -np.inf], np.nan, inplace=True); X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"NaNs in X after replacing inf: {X.isna().sum().sum()}"); print(f"NaNs in X_test after replacing inf: {X_test.isna().sum().sum()}")

imputer = SimpleImputer(strategy='median')
try:
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)
    print("Imputation successful.")
    X = pd.DataFrame(X_imputed, columns=feature_cols); X_test = pd.DataFrame(X_test_imputed, columns=feature_cols)
    print(f"NaNs in X after imputation: {X.isna().sum().sum()}"); print(f"NaNs in X_test after imputation: {X_test.isna().sum().sum()}")
    print("Saving processed data..."); X.to_pickle(X_TRAIN_PKL); y.to_pickle(Y_TRAIN_PKL); X_test.to_pickle(X_TEST_PKL); print("Processed data saved.")
except Exception as e: print(f"Error during imputation: {e}"); exit()
del X_imputed, X_test_imputed; gc.collect()

# Cell 7: Ensemble Model Training & Prediction (Modified)
print("Cell 7: Training Ensemble Models with CV...")

# Load processed data if not already in memory
if 'X' not in globals() or 'y' not in globals() or 'X_test' not in globals():
    print("Reloading processed data for final training...")
    X = pd.read_pickle(X_TRAIN_PKL)
    y = pd.read_pickle(Y_TRAIN_PKL)
    X_test = pd.read_pickle(X_TEST_PKL)

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# Initialize OOF and Test prediction arrays for each model
oof_lgbm = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
test_lgbm_agg = np.zeros(len(X_test))
test_xgb_agg = np.zeros(len(X_test))
test_cat_agg = np.zeros(len(X_test))

# Store scores and importances per model
scores_lgbm, scores_xgb, scores_cat = [], [], []
importances_lgbm = pd.DataFrame(index=feature_cols)
importances_xgb = pd.DataFrame(index=feature_cols)
importances_cat = pd.DataFrame(index=feature_cols)


# --- Final Cross-validation loop for Ensemble---
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # --- LightGBM ---
    print("Training LightGBM...")
    lgbm_model = lgb.LGBMClassifier(**LGBM_BEST_PARAMS)
    lgbm_model.fit(X_tr, y_tr,
                   eval_set=[(X_va, y_va)],
                   eval_metric='logloss',
                   callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=100)])
    best_iter_lgbm = lgbm_model.best_iteration_ if lgbm_model.best_iteration_ else LGBM_BEST_PARAMS['n_estimators']
    val_preds_lgbm = lgbm_model.predict_proba(X_va)[:, 1]
    test_preds_lgbm = lgbm_model.predict_proba(X_test, num_iteration=best_iter_lgbm)[:, 1]
    # Clip predictions
    val_preds_lgbm = np.clip(val_preds_lgbm, 1e-15, 1 - 1e-15)
    test_preds_lgbm = np.clip(test_preds_lgbm, 1e-15, 1 - 1e-15)
    # Store predictions and score
    oof_lgbm[val_idx] = val_preds_lgbm
    test_lgbm_agg += test_preds_lgbm / N_SPLITS
    fold_score_lgbm = log_loss(y_va, val_preds_lgbm)
    scores_lgbm.append(fold_score_lgbm)
    importances_lgbm[f'Fold_{fold+1}'] = lgbm_model.feature_importances_
    print(f"LGBM Fold {fold+1} LogLoss: {fold_score_lgbm:.6f} (Best iter: {best_iter_lgbm})")
    del lgbm_model; gc.collect() # Cleanup model


    # --- XGBoost ---
    print("Training XGBoost...")
    # Note: XGBoost early stopping uses eval_set parameter directly in fit
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=XGB_N_ESTIMATORS, use_label_encoder=False)
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  verbose=False) # verbose=False keeps output clean
    best_iter_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else XGB_N_ESTIMATORS # Get best iteration if early stopping triggered
    val_preds_xgb = xgb_model.predict_proba(X_va)[:, 1]
    test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1] # XGB uses best iteration by default if stopped early
    # Clip predictions
    val_preds_xgb = np.clip(val_preds_xgb, 1e-15, 1 - 1e-15)
    test_preds_xgb = np.clip(test_preds_xgb, 1e-15, 1 - 1e-15)
    # Store predictions and score
    oof_xgb[val_idx] = val_preds_xgb
    test_xgb_agg += test_preds_xgb / N_SPLITS
    fold_score_xgb = log_loss(y_va, val_preds_xgb)
    scores_xgb.append(fold_score_xgb)
    importances_xgb[f'Fold_{fold+1}'] = xgb_model.feature_importances_
    print(f"XGB Fold {fold+1} LogLoss: {fold_score_xgb:.6f} (Best iter: {best_iter_xgb})")
    del xgb_model; gc.collect()


    # --- CatBoost ---
    print("Training CatBoost...")
    cat_model = cb.CatBoostClassifier(**CAT_PARAMS, iterations=CAT_ITERATIONS)
    cat_model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  # early_stopping_rounds handled by CAT_PARAMS
                  verbose=0) # verbose=0 keeps output clean
    best_iter_cat = cat_model.best_iteration_ if hasattr(cat_model, 'best_iteration_') and cat_model.best_iteration_ is not None else CAT_ITERATIONS
    val_preds_cat = cat_model.predict_proba(X_va)[:, 1]
    test_preds_cat = cat_model.predict_proba(X_test)[:, 1] # Predicts using best iteration by default
    # Clip predictions
    val_preds_cat = np.clip(val_preds_cat, 1e-15, 1 - 1e-15)
    test_preds_cat = np.clip(test_preds_cat, 1e-15, 1 - 1e-15)
    # Store predictions and score
    oof_cat[val_idx] = val_preds_cat
    test_cat_agg += test_preds_cat / N_SPLITS
    fold_score_cat = log_loss(y_va, val_preds_cat)
    scores_cat.append(fold_score_cat)
    importances_cat[f'Fold_{fold+1}'] = cat_model.get_feature_importance()
    print(f"CAT Fold {fold+1} LogLoss: {fold_score_cat:.6f} (Best iter: {best_iter_cat})")
    del cat_model; gc.collect()

    del X_tr, y_tr, X_va, y_va # Clean up fold data
    gc.collect()

# --- Calculate Final Scores and Display Info ---
print(f"\n--- Cross-Validation Summary ---")
print(f"LGBM Mean CV LogLoss: {np.mean(scores_lgbm):.6f} +/- {np.std(scores_lgbm):.6f}")
print(f"XGB  Mean CV LogLoss: {np.mean(scores_xgb):.6f} +/- {np.std(scores_xgb):.6f}")
print(f"CAT  Mean CV LogLoss: {np.mean(scores_cat):.6f} +/- {np.std(scores_cat):.6f}")

# Calculate OOF score for each model
oof_score_lgbm = log_loss(y, oof_lgbm)
oof_score_xgb = log_loss(y, oof_xgb)
oof_score_cat = log_loss(y, oof_cat)
print(f"\nOverall OOF LGBM LogLoss: {oof_score_lgbm:.6f}")
print(f"Overall OOF XGB  LogLoss: {oof_score_xgb:.6f}")
print(f"Overall OOF CAT  LogLoss: {oof_score_cat:.6f}")

# Calculate Simple Average Ensemble OOF Score
oof_ensemble = (oof_lgbm + oof_xgb + oof_cat) / 3.0
oof_score_ensemble = log_loss(y, oof_ensemble)
print(f"\nOverall OOF ENSEMBLE (Avg) LogLoss: {oof_score_ensemble:.6f}")


# Cell 8: Submission File Creation (Using Ensemble Predictions)
print("Cell 8: Creating ensemble submission file...")
# --- Create Submission File ---
print("\nCreating submission file...")
# Load test mints if needed
if 'test_mints_final' not in globals():
    test_mints_final = pd.read_pickle(TEST_MINTS_PKL)

# Simple Average Ensemble for test predictions
ensemble_test_predictions = (test_lgbm_agg + test_xgb_agg + test_cat_agg) / 3.0

submission_df = pd.DataFrame({
    'mint': test_mints_final,
    'has_graduated': ensemble_test_predictions # Use ensembled predictions
})

print(f"Test data shape (X_test): {X_test.shape}")
print(f"Submission DataFrame shape: {submission_df.shape}")
assert submission_df.shape[0] == X_test.shape[0], \
    f"Submission rows ({submission_df.shape[0]}) != Test rows ({X_test.shape[0]})"

submission_df.to_csv(SUBMISSION_FILE, index=False, float_format='%.8f')

print(f"Submission file created successfully at: {SUBMISSION_FILE}")
print("Sample submission:")
print(submission_df.head())


# Cell 9: Feature Importance Display (Per Model)
print("Cell 9: Displaying feature importance for each model...")

def display_feature_importance(importances_df, model_name):
    try:
        importances_df['mean_importance'] = importances_df.mean(axis=1)
        importances_df = importances_df.sort_values('mean_importance', ascending=False)
        print(f"\n--- Top 15 Feature Importances ({model_name}) ---")
        with pd.option_context('display.max_rows', 15):
            print(importances_df[['mean_importance']].head(15))
    except Exception as e:
        print(f"\nCould not display averaged feature importance for {model_name}: {e}")

# Display for each model
display_feature_importance(importances_lgbm, "LightGBM")
display_feature_importance(importances_xgb, "XGBoost")
display_feature_importance(importances_cat, "CatBoost")


print("\nEnsemble script finished successfully!")

