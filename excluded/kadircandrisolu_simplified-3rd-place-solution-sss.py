import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import Pool, EFeaturesSelectionAlgorithm, EShapCalcType
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import gc 
import warnings
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('fivethirtyeight')


DATA_PATH = Path('/kaggle/input/pump-fun-graduation-february-2025')
PUMP_FUN_API_PATH = Path('/kaggle/input/pump-fun-api-solana-tokens-info/pump_fun_api_info.parquet')
N_SPLITS = 3
N_REPEATS = 2
RANDOM_STATE = 42
CV_MODEL_TYPE = 'ctb'
LOAD_MAX_CHUNKS = None
PERFORM_FEATURE_SELECTION = False


def load_and_merge_all_data(data_path: Path, 
                           pump_fun_api_path: Path = None, 
                           max_chunks: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("\n1. Loading main datasets...")
    train = pd.read_csv(data_path / 'train.csv')
    test = pd.read_csv(data_path / 'test_unlabeled.csv')
    
    print(f"  Train set loaded: {train.shape}")
    print(f"  Test set loaded: {test.shape}")
    
    print("\n2. Loading Dune token info...")
    dune_info = pd.read_csv(data_path / 'dune_token_info_v2.csv')
    dune_info.rename(columns={'token_mint_address': 'mint'}, inplace=True)
    dune_info = dune_info.drop_duplicates(subset=['mint'], keep='first')
    print(f"  Dune info loaded: {dune_info.shape}")
    
    print("\n3. Loading Onchain token info...")
    onchain_info = pd.read_csv(data_path / 'token_info_onchain_divers_v2.csv')
    onchain_info = onchain_info.drop_duplicates(subset=['mint'], keep='first')
    print(f"  Onchain info loaded: {onchain_info.shape}")
    
    pump_fun_data = None
    if pump_fun_api_path is not None and pump_fun_api_path.exists():
        print("\n4. Loading Pump Fun API data...")
        try:
            pump_fun_data = pd.read_parquet(pump_fun_api_path)
            pump_fun_data = pump_fun_data.drop_duplicates(subset=['mint'], keep='first')
            print(f"  Pump Fun API data loaded: {pump_fun_data.shape}")
        except Exception as e:
            print(f"  Error loading Pump Fun API data: {e}")
    else:
        print("\n4. Pump Fun API data not provided. Skipping.")
    
    print("\n5. Loading transaction data...")
    transaction_df = load_transaction_data(data_path, max_chunks)
    transaction_df = transaction_df.rename(columns={'base_coin': 'mint'})
    
    print("\n6. Merging all datasets...")
    
    orig_train_rows = len(train)
    orig_test_rows = len(test)
    
    train_merged = pd.merge(train, dune_info, on='mint', how='left')
    test_merged = pd.merge(test, dune_info, on='mint', how='left')
    
    train_merged = pd.merge(train_merged, onchain_info, on='mint', how='left', suffixes=('', '_onchain'))
    test_merged = pd.merge(test_merged, onchain_info, on='mint', how='left', suffixes=('', '_onchain'))
    
    if pump_fun_data is not None:
        train_merged = pd.merge(train_merged, pump_fun_data, on='mint', how='left', suffixes=('', '_pf'))
        test_merged = pd.merge(test_merged, pump_fun_data, on='mint', how='left', suffixes=('', '_pf'))
    
    print(f"\nMerging complete.")
    print(f"  Final train shape: {train_merged.shape}")
    print(f"  Final test shape: {test_merged.shape}")
    
    return train_merged, test_merged, transaction_df


def load_transaction_data(transaction_path: Path, max_chunks: int = None) -> pd.DataFrame:
    """Loads transaction data in chunks."""
    print(f"  Loading transaction data from: {transaction_path}")
    chunk_files = sorted(list(transaction_path.glob('chunk_*.csv')))

    if max_chunks is not None and max_chunks > 0:
         print(f"  Loading a maximum of {max_chunks} transaction chunks.")
         chunk_files = chunk_files[:max_chunks]
    else:
        print(f"  Loading all {len(chunk_files)} transaction chunks.")

    all_transactions = []
    required_cols = ['base_coin', 'quote_coin_amount', 'slot', 'signing_wallet', 'fee', 'consumed_gas']
    optional_cols = ['direction', 'block_time'] 

    for i, file in enumerate(chunk_files):
        chunk = pd.read_csv(file)

        cols_to_load = required_cols + [col for col in optional_cols if col in chunk.columns]
        chunk_subset = chunk[cols_to_load].copy() 

        if 'block_time' in chunk_subset.columns:
            chunk_subset['block_time'] = pd.to_datetime(chunk_subset['block_time'], errors='coerce')

        all_transactions.append(chunk_subset)
        del chunk, chunk_subset 
        if (i+1) % 20 == 0: gc.collect() 

    print("  Concatenating transaction chunks...")
    transactions_df = pd.concat(all_transactions, ignore_index=True)
    del all_transactions
    gc.collect()
    print(f"  Total transactions loaded: {len(transactions_df):,}")

    return transactions_df


def preprocess_merged_data(train_merged: pd.DataFrame, test_merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("\nHandling missing values and data types...")
    
    train_processed = train_merged.copy()
    test_processed = test_merged.copy()
    
    # 1. Handle numeric columns
    numeric_cols_train = train_processed.select_dtypes(include=np.number).columns
    train_processed[numeric_cols_train] = train_processed[numeric_cols_train].fillna(-1)
    numeric_cols_test = test_processed.select_dtypes(include=np.number).columns
    test_processed[numeric_cols_test] = test_processed[numeric_cols_test].fillna(-1)
    
    # 2. Handle categorical columns
    cat_cols_train = train_processed.select_dtypes(include=['object', 'category']).columns
    train_processed[cat_cols_train] = train_processed[cat_cols_train].fillna("unknown")
    cat_cols_test = test_processed.select_dtypes(include=['object', 'category']).columns
    test_processed[cat_cols_test] = test_processed[cat_cols_test].fillna("unknown")
    
    # 3. Handle boolean columns
    bool_cols_train = train_processed.select_dtypes(include=bool).columns
    train_processed[bool_cols_train] = train_processed[bool_cols_train].fillna(False)
    bool_cols_test = test_processed.select_dtypes(include=bool).columns
    test_processed[bool_cols_test] = test_processed[bool_cols_test].fillna(False)
    
    date_cols = ['created_at', 'pf_created_timestamp']
    for col in date_cols:
        if col in train_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(train_processed[col]):
                train_processed[col] = pd.to_datetime(train_processed[col], errors='coerce')
            if not pd.api.types.is_datetime64_any_dtype(test_processed[col]):
                test_processed[col] = pd.to_datetime(test_processed[col], errors='coerce')
    
    return train_processed, test_processed


def create_dune_token_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['dune_has_name'] = df['name'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['dune_name_length'] = df['name'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else len(str(x)))
    df['dune_has_symbol'] = df['symbol'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['dune_symbol_length'] = df['symbol'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else len(str(x)))
    df['dune_has_uri'] = df['token_uri'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['dune_is_ipfs'] = df['token_uri'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else ('ipfs' in str(x).lower()))
    df['dune_has_init_tx'] = df['init_tx'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['dune_decimals_is_standard'] = df['decimals'].apply(lambda x: False if pd.isna(x) or x == "unknown" else (x == 9))
    df = df.drop(columns=['name', 'symbol', 'token_uri', 'init_tx'], errors='ignore')
    
    return df
def create_onchain_features(df: pd.DataFrame) -> pd.DataFrame:

    df['onchain_gas_used'] = df['gas_used']
    df['onchain_amount_of_instructions'] = df['amount_of_instructions']
    df['onchain_amount_of_lookup_reads'] = df['amount_of_lookup_reads']
    df['onchain_amount_of_lookup_writes'] = df['amount_of_lookup_writes']
    df['onchain_bundled_buys_count'] = df['bundled_buys_count']
    df['onchain_bundle_size'] = df['bundle_size']
    df['onchain_is_bundled'] = df['bundle_size'].apply(lambda x: False if pd.isna(x) or x == "unknown" else (x > 1))
    df['onchain_dev_balance'] = df['dev_balance']
    df['onchain_dev_balance_log'] = df['dev_balance'].apply(
        lambda x: 0 if pd.isna(x) or x == "unknown" else np.log1p(max(0, x))
    )
    df['onchain_direct_pf_invocation'] = df['direct_pf_invocation'].apply(
        lambda x: 0 if pd.isna(x) or x == "unknown" else (1 if x else 0)
    )
    df['onchain_has_url'] = df['url'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['onchain_url_length'] = df['url'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else len(str(x)))
    df = df.drop(columns=[
        'gas_used', 'amount_of_instructions', 'amount_of_lookup_reads', 
        'amount_of_lookup_writes', 'bundled_buys_count', 'bundle_size',
        'dev_balance', 'direct_pf_invocation', 'url'
    ])
    
    return df
    
def create_pumpfun_api_features(df: pd.DataFrame) -> pd.DataFrame:

    df['pf_has_description'] = df['description'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_description_length'] = df['description'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else len(str(x)))
    df['pf_has_twitter'] = df['twitter'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_has_telegram'] = df['telegram'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_has_website'] = df['website'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    social_features = ['pf_has_twitter', 'pf_has_telegram', 'pf_has_website']
    df['pf_social_count'] = df[social_features].sum(axis=1)
    df['pf_has_image_uri'] = df['image_uri'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_has_metadata_uri'] = df['metadata_uri'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_has_video_uri'] = df['video_uri'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    media_features = ['pf_has_image_uri', 'pf_has_metadata_uri', 'pf_has_video_uri']
    df['pf_media_count'] = df[media_features].sum(axis=1)
    df['pf_nsfw'] = df['nsfw'].apply(
        lambda x: 0 if pd.isna(x) or x == "unknown" or x == "false" or x == "False" or x == 0 or x is False 
        else 1
    )
    df['pf_show_name'] = df['show_name'].apply(
        lambda x: 0 if pd.isna(x) or x == "unknown" or x == "false" or x == "False" or x == 0 or x is False 
        else 1
    )
    df['pf_initialized'] = df['initialized'].apply(
        lambda x: 0 if pd.isna(x) or x == "unknown" or x == "false" or x == "False" or x == 0 or x is False 
        else 1
    )
    df['pf_has_bonding_curve'] = df['bonding_curve'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df['pf_has_associated_bonding_curve'] = df['associated_bonding_curve'].apply(lambda x: 0 if pd.isna(x) or x == "unknown" else 1)
    df = df.drop(columns=[
        'description', 'twitter', 'telegram', 'website', 
        'image_uri', 'metadata_uri', 'video_uri',
        'nsfw', 'show_name', 'initialized',
        'bonding_curve', 'associated_bonding_curve'
    ])
    
    return df
    
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    if 'created_timestamp' in df.columns:
        df['created_timestamp'] = pd.to_numeric(df['created_timestamp'], errors='coerce')
        if df['created_timestamp'].mean() > 1e11:
            df['created_timestamp'] = df['created_timestamp'] / 1000

    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        df['created_at'] = df['created_at'].dt.tz_localize(None)

    if 'block_time' in df.columns:
        df['block_time'] = pd.to_datetime(df['block_time'], errors='coerce')
    
    if 'created_at' in df.columns and df['created_at'].notna().any():
        df['creation_time'] = df['created_at']
    elif 'created_timestamp' in df.columns and df['created_timestamp'].notna().any():
        df['creation_time'] = pd.to_datetime(df['created_timestamp'], unit='s', errors='coerce')
    else:
        df['creation_time'] = pd.NaT

    if 'creation_time' in df.columns:
        valid_times = df['creation_time'].notna()
        df.loc[valid_times, 'hour_of_day'] = df.loc[valid_times, 'creation_time'].dt.hour
        df.loc[valid_times, 'day_of_week'] = df.loc[valid_times, 'creation_time'].dt.dayofweek
        df.loc[valid_times, 'day_of_month'] = df.loc[valid_times, 'creation_time'].dt.day
        df.loc[valid_times, 'month'] = df.loc[valid_times, 'creation_time'].dt.month
        df.loc[valid_times, 'is_weekend'] = df.loc[valid_times, 'day_of_week'].isin([5, 6]).astype(int)
        df.loc[valid_times, 'is_us_active'] = df.loc[valid_times, 'hour_of_day'].between(13, 21).astype(int)  # 8AM-4PM EST
        df.loc[valid_times, 'is_asia_active'] = df.loc[valid_times, 'hour_of_day'].between(0, 8).astype(int)  # 8AM-4PM Asian markets
        df.loc[valid_times, 'is_eu_active'] = df.loc[valid_times, 'hour_of_day'].between(7, 15).astype(int)  # 8AM-4PM European markets
        df.loc[valid_times, 'is_late_night'] = df.loc[valid_times, 'hour_of_day'].between(0, 5).astype(int)
        df.loc[valid_times, 'is_early_morning'] = df.loc[valid_times, 'hour_of_day'].between(5, 8).astype(int)
        df.loc[valid_times, 'is_degen_hours'] = ((df.loc[valid_times, 'hour_of_day'] >= 22) | 
                                                (df.loc[valid_times, 'hour_of_day'] <= 4)).astype(int)
        
        time_features = ['hour_of_day', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
                        'is_us_active', 'is_asia_active', 'is_eu_active', 'is_late_night', 
                        'is_early_morning', 'is_degen_hours']
        for feature in time_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(-1).astype(int)
    
    if 'block_time' in df.columns and 'creation_time' in df.columns:
        valid_both = df['block_time'].notna() & df['creation_time'].notna()
        
        if valid_both.any():

            df.loc[valid_both, 'creation_to_block_delay'] = (
                df.loc[valid_both, 'block_time'] - df.loc[valid_both, 'creation_time']
            ).dt.total_seconds()
            
            # Flag very quick blocks (potential automation)
            df['is_quick_block'] = 0
            valid_delay = df['creation_to_block_delay'].notna()
            df.loc[valid_delay, 'is_quick_block'] = (df.loc[valid_delay, 'creation_to_block_delay'] < 10).astype(int)
            
            # Flag delayed blocks
            df.loc[valid_delay, 'is_delayed_block'] = (df.loc[valid_delay, 'creation_to_block_delay'] > 300).astype(int)
            
            df['creation_to_block_delay'] = df['creation_to_block_delay'].fillna(-1)
        else:
            df['creation_to_block_delay'] = -1
            df['is_quick_block'] = 0
            df['is_delayed_block'] = 0

    for col in ['creation_time', 'block_time', 'created_at']:
        if col in df.columns:
            df[f'{col}_epoch'] = -1
            valid_dates = df[col].notna()
            if valid_dates.any():
                df.loc[valid_dates, f'{col}_epoch'] = df.loc[valid_dates, col].astype(np.int64) // 10**9
            df = df.drop(columns=[col])
    
    return df
    
def create_transaction_features(df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:

    basic_aggregations = {
        'slot': ['count', 'min', 'max', 'nunique'],
        'quote_coin_amount': ['sum', 'mean', 'std', 'min', 'max'],
        'signing_wallet': ['nunique'],
        'fee': ['mean', 'sum', 'std'],
        'consumed_gas': ['mean', 'sum', 'std']
    }
    
    basic_features = transaction_df.groupby('mint').agg(basic_aggregations)
    basic_features.columns = ['tx_' + '_'.join(col).strip() for col in basic_features.columns.values]
    basic_features = basic_features.reset_index()

    time_aggs = transaction_df.dropna(subset=['block_time']).groupby('mint')['block_time'].agg(['min', 'max'])
    time_aggs['tx_activity_duration_hours'] = (time_aggs['max'] - time_aggs['min']).dt.total_seconds() / 3600
    
    time_features = pd.DataFrame(index=time_aggs.index)
    time_features['tx_block_time_first_activity'] = time_aggs['min']
    time_features['tx_block_time_last_activity'] = time_aggs['max']
    time_features['tx_block_time_activity_duration_hours'] = time_aggs['tx_activity_duration_hours']
    
    slot_quantiles = transaction_df.groupby('mint')['slot'].transform(lambda x: x.quantile(0.2))
    early_mask = transaction_df['slot'] <= slot_quantiles
    buy_mask = transaction_df['direction'] == 'buy'
    
    # Calculate early buy ratio
    early_buy_counts = transaction_df[early_mask & buy_mask].groupby('mint').size()
    total_tx_counts = transaction_df.groupby('mint').size()
    early_buy_ratio = (early_buy_counts / total_tx_counts).fillna(0)
    
    time_features['tx_early_buy_ratio'] = early_buy_ratio
    time_features = time_features.reset_index()
    
    direction_features = pd.DataFrame()
    tx_counts = transaction_df.pivot_table(
        index='mint', 
        columns='direction', 
        values='slot', 
        aggfunc='count',
        fill_value=0
    )
    
    tx_volumes = transaction_df.pivot_table(
        index='mint', 
        columns='direction', 
        values='quote_coin_amount', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Add to direction features
    direction_features['tx_buy_count'] = tx_counts.get('buy', 0)
    direction_features['tx_sell_count'] = tx_counts.get('sell', 0)
    direction_features['tx_buy_volume'] = tx_volumes.get('buy', 0)
    direction_features['tx_sell_volume'] = tx_volumes.get('sell', 0)
    
    # Derived Features
    direction_features['tx_buy_sell_ratio_count'] = direction_features['tx_buy_count'] / (direction_features['tx_sell_count'] + 1e-6)
    direction_features['tx_buy_sell_ratio_volume'] = direction_features['tx_buy_volume'] / (direction_features['tx_sell_volume'] + 1e-6)
    
    direction_features = direction_features.reset_index()
    
    # Wallet behavior features
    buy_txs = transaction_df[transaction_df['direction'] == 'buy']
    wallet_buy_counts = buy_txs.groupby(['mint', 'signing_wallet']).size().reset_index(name='buy_count')
    repeat_buyers = wallet_buy_counts[wallet_buy_counts['buy_count'] > 1].groupby('mint').size()
    unique_buyers = wallet_buy_counts.groupby('mint')['signing_wallet'].nunique()
    
    wallet_features = pd.DataFrame(index=transaction_df['mint'].unique())
    wallet_features['tx_repeat_buyers'] = repeat_buyers
    wallet_features['tx_repeat_buyer_ratio'] = (repeat_buyers / unique_buyers).fillna(0)
    
    # Whale detection 
    whale_thresholds = buy_txs.groupby('mint')['quote_coin_amount'].transform(lambda x: x.quantile(0.95))
    whale_mask = buy_txs['quote_coin_amount'] > whale_thresholds
    whale_wallets = buy_txs[whale_mask].groupby('mint')['signing_wallet'].nunique()
    
    wallet_features['tx_whale_count'] = whale_wallets
    wallet_features['tx_whale_ratio'] = (whale_wallets / unique_buyers).fillna(0)
    
    wallet_features = wallet_features.fillna(0)
    wallet_features = wallet_features.reset_index()
    wallet_features.rename(columns={'index': 'mint'}, inplace=True)
    
    # Combine all transaction features
    feature_sets = [basic_features, time_features, direction_features, wallet_features]
    
    from functools import reduce
    all_tx_features = reduce(lambda left, right: pd.merge(left, right, on='mint', how='outer'), feature_sets)

    all_tx_features = all_tx_features.fillna(-1)
    
    # Merge transaction features with dataframe
    df_with_tx = pd.merge(df, all_tx_features, on='mint', how='left')

    tx_feature_cols = [col for col in all_tx_features.columns if col != 'mint']
    df_with_tx[tx_feature_cols] = df_with_tx[tx_feature_cols].fillna(-1)
    
    for col in df_with_tx.select_dtypes(include=['datetime64']).columns:
        df_with_tx[f'{col}_epoch'] = df_with_tx[col].astype(np.int64) // 10**9
        df_with_tx = df_with_tx.drop(columns=[col])
    
    return df_with_tx
    
def create_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates combined features that span across different data sources."""
    print("\nCreating combined features...")
    
    # 1. Social media presence vs token metrics
    if 'pf_social_count' in df.columns and 'tx_slot_count' in df.columns:
        df['social_to_tx_ratio'] = df['pf_social_count'] / (df['tx_slot_count'] + 1e-6)
    
    # 2. Combine name/symbol information across sources
    name_cols = [
        'dune_name_length', 
        'dune_symbol_length',
        'onchain_url_length'
    ]
    if all(col in df.columns for col in name_cols):
        df['max_text_length'] = df[name_cols].max(axis=1)
        df['mean_text_length'] = df[name_cols].mean(axis=1)
    
    # 3. Create overall quality score
    quality_cols = []
    
    # Add social media presence if available
    if 'pf_social_count' in df.columns:
        quality_cols.append('pf_social_count')
    
    # Add media presence if available
    if 'pf_media_count' in df.columns:
        quality_cols.append('pf_media_count')
    
    # Add description length if available
    if 'pf_description_length' in df.columns:
        quality_cols.append('pf_description_length')
        
        # Normalize description length (longer is better, but with diminishing returns)
        df['pf_description_length_norm'] = np.log1p(df['pf_description_length'])
        quality_cols.append('pf_description_length_norm')
    
    # Subtract NSFW penalty if available
    nsfw_penalty = 0
    if 'pf_nsfw' in df.columns:
        nsfw_penalty = df['pf_nsfw'] * 2  # Penalty of 2 points for NSFW content
    
    # Calculate quality score if we have components
    if quality_cols:
        df['token_quality_score'] = df[quality_cols].sum(axis=1) - nsfw_penalty
    
    print(f"  Created combined features.")
    
    return df


def feature_engineering_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, transaction_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    start_train_cols = train_df.shape[1]
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    feature_steps = [
        (create_dune_token_features, "Dune Token Features"),
        (create_onchain_features, "On-chain Features"),
        (create_pumpfun_api_features, "PumpFun API Features"),
        (create_temporal_features, "Temporal Features"),
        (create_combined_features, "Combined Features"),
        (create_transaction_features, "Transaction Features")
    ]

    print("\nFeature Extraction...")
    for func, desc in tqdm(feature_steps, desc="Train Features"):
        if func.__name__ == "create_transaction_features":
            train_processed = func(train_processed, transaction_df)
        else:
            train_processed = func(train_processed)
    
    print("Processing test data...")
    for func, desc in tqdm(feature_steps, desc="Test Features"):
        if func.__name__ == "create_transaction_features":
            test_processed = func(test_processed, transaction_df)
        else:
            test_processed = func(test_processed)
    
    print("\nFeature Engineering Summary:")
    print(f"  Starting feature count: {start_train_cols}")
    print(f"  Final feature count: {train_processed.shape[1]}")
    print(f"  Added {train_processed.shape[1] - start_train_cols} new features")
    
    return train_processed, test_processed



def perform_repeated_stratified_kfold_cv(X: pd.DataFrame, y: pd.Series, model_type: str = CV_MODEL_TYPE, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> Tuple[float, float, List[float]]:
    """Performs repeated stratified k-fold cross-validation."""
    print(f"\nPerforming {n_repeats} repeats of {n_splits}-fold stratified CV ({model_type.upper()})...")
    X_cv, y_cv = X.copy(), y.copy() 

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    
    lgb_params, cat_params, xgb_params = get_model_params()
    all_fold_scores = []
    all_fold_iterations = []
    oof_preds = np.zeros(len(X_cv)) 
    oof_count = np.zeros(len(X_cv))

    categorical_features = X_cv.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features_indices = [X_cv.columns.get_loc(col) for col in categorical_features]

    if categorical_features:
        print(f"  Categorical features identified: {len(categorical_features)}")
        if model_type == 'ctb':
            X_cv[categorical_features] = X_cv[categorical_features].astype(str) # Make them string for Catboost
        elif model_type == 'lgb':
            for col in categorical_features:
                X_cv[col] = X_cv[col].astype('category')
        elif model_type == 'xgb':
            for col in categorical_features:
                X_cv[col] = pd.factorize(X_cv[col])[0]
    
    fold_counter = 0
    total_folds = n_splits * n_repeats
    
    for train_idx, val_idx in rskf.split(X_cv, y_cv):
        current_repeat = fold_counter // n_splits + 1
        current_fold = fold_counter % n_splits + 1
        fold_counter += 1
        
        print(f"\n--- Repeat {current_repeat}/{n_repeats}, Fold {current_fold}/{n_splits} (Overall: {fold_counter}/{total_folds}) ---")
        X_train, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_train, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
        print(f"  Train size: {len(X_train)}, Validation size: {len(X_val)}")
        print(f"  Train positive ratio: {y_train.mean():.4f}, Validation positive ratio: {y_val.mean():.4f}")

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_data)
            callbacks = [lgb.log_evaluation(period=lgb_params.get('verbose_eval', 100))]
            num_iterations = lgb_params.get('num_iterations', 1500)
            model = lgb.train(lgb_params, train_data, valid_sets=[val_data], num_boost_round=num_iterations, callbacks=callbacks)
            val_preds = model.predict(X_val)
            all_fold_iterations.append(num_iterations)

        elif model_type == 'xgb':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            num_iterations = xgb_params.get('iterations', 1500)
            model = xgb.train(xgb_params, dtrain, 
                            num_boost_round=num_iterations,
                            evals=watchlist,
                            verbose_eval=xgb_params.get('verbose_eval', 100))
            
            val_preds = model.predict(dval)
            all_fold_iterations.append(num_iterations)

        elif model_type == 'ctb':
            model = cb.CatBoostClassifier(**cat_params) 
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                      cat_features=categorical_features_indices, 
                      verbose=cat_params.get('verbose', 200))
            val_preds = model.predict_proba(X_val)[:, 1]
            all_fold_iterations.append(cat_params.get('iterations', 1500))

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        oof_preds[val_idx] += val_preds
        oof_count[val_idx] += 1
        
        fold_score = log_loss(y_val, val_preds)
        all_fold_scores.append(fold_score)
        print(f"  Fold Log Loss: {fold_score:.6f}")
        print(f"  Total Iterations: {all_fold_iterations[-1]}")
        del model, X_train, X_val, y_train, y_val; gc.collect()

    oof_preds = np.divide(oof_preds, oof_count, out=np.zeros_like(oof_preds), where=oof_count > 0)

    mean_score = np.mean(all_fold_scores)
    std_score = np.std(all_fold_scores)
    mean_iterations = int(np.mean(all_fold_iterations))
    std_iterations = int(np.std(all_fold_iterations))
    
    print("\n--- Repeated Stratified CV Summary ---")
    print(f"Mean Log Loss: {mean_score:.6f} ± {std_score:.6f}")
    print(f"Mean Iterations: {mean_iterations} ± {std_iterations}")

    repeat_scores = []
    for r in range(n_repeats):
        repeat_start = r * n_splits
        repeat_end = (r + 1) * n_splits
        repeat_mean = np.mean(all_fold_scores[repeat_start:repeat_end])
        repeat_scores.append(repeat_mean)
    
    print(f"Scores by repeat: {[f'{s:.6f}' for s in repeat_scores]}")
    print(f"Iterations by fold: {all_fold_iterations}")

    valid_indices = oof_count > 0
    if np.any(valid_indices):
        oof_score = log_loss(y_cv[valid_indices], oof_preds[valid_indices])
        print(f"Overall OOF Log Loss: {oof_score:.6f}")

    return mean_score, std_score, all_fold_scores


def perform_feature_selection(X, y, n_splits=5, n_features_to_select=32, feature_selection_algorithm='RecursiveByShapValues'):
    """Performs feature selection using CatBoost's select_features method across multiple CV folds"""
    
    print(f"\nPerforming feature selection using {feature_selection_algorithm} with {n_splits}-fold CV...")
    
    X_fs = X.copy()
    y_fs = y.copy()
    
    cat_cols = X_fs.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        print(f"  Categorical features identified: {len(cat_cols)}")
        for cat_col in cat_cols:
            X_fs[cat_col] = X_fs[cat_col].fillna("nan").astype(str).astype("category")

    _, cat_params = get_model_params()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    summaries = []
    fold_counter = 0
    
    for train_idx, val_idx in skf.split(X_fs, y_fs):
        fold_counter += 1
        print(f"\n--- Feature Selection Fold {fold_counter}/{n_splits} ---")
        
        X_train, X_val = X_fs.iloc[train_idx], X_fs.iloc[val_idx]
        y_train, y_val = y_fs.iloc[train_idx], y_fs.iloc[val_idx]
        
        print(f"  Train size: {len(X_train)}, Validation size: {len(X_val)}")
        print(f"  Train positive ratio: {y_train.mean():.4f}, Validation positive ratio: {y_val.mean():.4f}")
        
        fs_params = {
            'iterations': 2300,
            'learning_rate': 0.05,
            'objective': 'Logloss',  
            'eval_metric': 'Logloss',
            'verbose': 100,
            'task_type': 'CPU', 
            'use_best_model': False,
            'cat_features': cat_cols,
        }
        
        model = cb.CatBoostClassifier(**fs_params)
        
        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_val, y_val, cat_features=cat_cols)

        if feature_selection_algorithm == 'RecursiveByShapValues':
            algorithm = EFeaturesSelectionAlgorithm.RecursiveByShapValues
        elif feature_selection_algorithm == 'RecursiveByLossFunctionChange':
            algorithm = EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange
        else:
            raise ValueError(f"Unsupported feature selection algorithm: {feature_selection_algorithm}")

        n_total_features = X_train.shape[1]
        features_for_select = f'0-{n_total_features-1}'

        summary = model.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=features_for_select,
            num_features_to_select=n_features_to_select,
            steps=5,
            algorithm=algorithm,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False,
            logging_level='Info',
            plot=True
        )
        
        print(f"  Selected {len(summary['selected_features_names'])} features in fold {fold_counter}")
        summaries.append(summary)

    all_selected_features = list(set(np.concatenate([summary["selected_features_names"] for summary in summaries])))
    print(f"\nTotal unique features selected across all folds: {len(all_selected_features)}")
    print(f"Selected features: {all_selected_features}")
    
    return all_selected_features


def create_submission(test_mint_ids: pd.Series, predictions: np.ndarray, filename: str = 'submission.csv') -> pd.DataFrame:

    print(f"\nCreating submission file: {filename}...")
    submission = pd.DataFrame({'mint': test_mint_ids, 'has_graduated': predictions})
    submission['has_graduated'] = submission['has_graduated'].clip(0, 1)
    submission.to_csv(filename, index=False)
    print(f"Submission file saved successfully: {filename}")
    
    return submission


# 1. Load and merge all data sources
train_raw, test_raw, transactions = load_and_merge_all_data(
    DATA_PATH, 
    PUMP_FUN_API_PATH if PUMP_FUN_API_PATH.exists() else None,
    LOAD_MAX_CHUNKS
)


# 2. Preprocess merged data
train_preprocessed, test_preprocessed = preprocess_merged_data(train_raw, test_raw)


# 3. Feature engineering
train_featured, test_featured = feature_engineering_pipeline(
    train_preprocessed, test_preprocessed, transactions
)


target_col = 'has_graduated'
feature_cols = ['pf_nsfw',
                'tx_consumed_gas_sum',
                'tx_slot_min',
                'tx_sell_volume',
                'tx_consumed_gas_mean',
                'tx_fee_std',
                'creation_ix_index',
                'tx_slot_max',
                'tx_buy_count',
                'pf_program_index',
                'tx_buy_sell_ratio_volume',
                'tx_block_time_activity_duration_hours',
                'tx_slot_nunique',
                'dune_symbol_length',
                'tx_block_time_first_activity_epoch',
                'tx_quote_coin_amount_max',
                'tx_quote_coin_amount_mean',
                'onchain_dev_balance_log',
                'pf_description_length',
                'tx_fee_mean',
                'pf_has_twitter',
                'tx_block_time_last_activity_epoch',
                'pf_has_website',
                'tx_buy_volume',
                'bundled_buys',
                'tx_consumed_gas_std',
                'onchain_amount_of_instructions',
                'token_quality_score',
                'pf_has_telegram',
                'onchain_dev_balance',
                'onchain_url_length',
                'pf_social_count',
                'tx_quote_coin_amount_sum',
                'tx_quote_coin_amount_std',
                'dune_name_length',
                'tx_signing_wallet_nunique',
                'tx_slot_count',
                'tx_fee_sum',
                'tx_quote_coin_amount_min',
                'tx_buy_sell_ratio_count',
                'onchain_bundled_buys_count',
                'tx_early_buy_ratio',
                'tx_repeat_buyers',
                'tx_repeat_buyer_ratio',
                'tx_whale_count',
                'tx_whale_ratio',
                'hour_of_day',
                'day_of_week',
                'day_of_month',
                'is_weekend',
                'is_us_active',
                'is_asia_active',
                'is_eu_active',
                'is_late_night',
                'is_early_morning',
                'is_degen_hours',
                'creation_to_block_delay',
                'is_quick_block',
]


X = train_featured[feature_cols]
y = train_featured[target_col]
X_test = test_featured[feature_cols]


if PERFORM_FEATURE_SELECTION:
    
    selected_features = perform_feature_selection(
        X=X, 
        y=y, 
        n_splits=5, 
        n_features_to_select=30, 
        feature_selection_algorithm='RecursiveByShapValues'
    )

    X = X[selected_features]
    X_test = X_test[selected_features]
    
    print(f"\nFeatures after selection: {len(selected_features)}")
    print(f"Selected feature list: {selected_features}")
else:
    print("\nUsing selected feature_cols")


lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'colsample_bytree': 1.0,      
    'max_depth': -1,             
    'learning_rate': 0.02,         
    'lambda_l1': 0.0,            
    'lambda_l2': 0.0,            
    'num_leaves': 31, 
    'boosting_type': 'gbdt',
    'boost_from_average': False,
    'num_iterations': 1200,
    'scale_pos_weight': 1.00035,
    'seed': RANDOM_STATE
}

cat_params = { 
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'learning_rate': 0.039155722248330437,
    'max_depth': 7,
    'colsample_bylevel': 0.37422372358022926,
    'scale_pos_weight': 1.0008169315461717,
    'random_seed': RANDOM_STATE,
    'verbose': 0,
    'iterations': 1900,
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'learning_rate': 0.020076070674451454,
    'lambda': 1.7100783925088332,
    'alpha': 0.851547132318672,
    'max_depth': 7,
    'n_estimators': 1200,
    'min_child_weight': 3,
    'colsample_bytree': 0.6040323554685212,
    'colsample_bylevel': 0.5129688144333449,
    'colsample_bynode': 0.7267458622028685,
    'scale_pos_weight': 1.0002746598078605,
    'seed': RANDOM_STATE
}


xgb_model = xgb.XGBClassifier(**xgb_params)
cat_model = cb.CatBoostClassifier(**cat_params)
lgb_model = lgb.LGBMClassifier(**lgb_params)

estimators = [
    ('xgb', xgb_model),
    ('cat', cat_model),
    ('lgb', lgb_model)
]
model = VotingClassifier(estimators=estimators, voting='soft')
model.fit(X, y)
predictions = model.predict_proba(X_test)[:, 1]


submission = create_submission(test_featured['mint'], predictions, filename='submission.csv')

