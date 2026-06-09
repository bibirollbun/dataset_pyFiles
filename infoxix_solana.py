import pandas as pd
import numpy as np
import glob
import os
import gc
import matplotlib.pyplot as plt
import catboost as cb

from sklearn.inspection import permutation_importance
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from catboost import Pool
from sklearn.metrics import log_loss


DATA_PATH = './data'
CHUNK_PATTERN = os.path.join(DATA_PATH, 'chunk*.csv')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')
DUNE_INFO_FILE = os.path.join(DATA_PATH, 'dune_token_info.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_PATH, 'token_info_onchain_divers.csv')

TARGET = 'has_graduated'
MINT_ID = 'mint'
BLOCK_LIMIT = 100
N_SPLITS = 5
RANDOM_SEED = 42


print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
dune_info_df = pd.read_csv(DUNE_INFO_FILE)
onchain_info_df = pd.read_csv(ONCHAIN_INFO_FILE)

train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], ignore_index=True)

all_chunk_files = glob.glob(CHUNK_PATTERN)
print(f"Found {len(all_chunk_files)} chunk files.")

transactions_df = pd.concat([pd.read_csv(f) for f in tqdm(all_chunk_files)])


transactions_df.head()


transactions_df['block_time'] = pd.to_datetime(transactions_df['block_time'], errors='coerce')
onchain_info_df['block_time'] = pd.to_datetime(onchain_info_df['block_time'], errors='coerce')

transactions_df['slot'] = pd.to_numeric(transactions_df['slot'], errors='coerce')
combined_df['slot_min'] = pd.to_numeric(combined_df['slot_min'], errors='coerce')

# Merge token creation info (slot_min) with transactions
transactions_df = pd.merge(
    transactions_df,
    combined_df[[MINT_ID, 'slot_min']],
    left_on='base_coin',
    right_on=MINT_ID,
    how='left'
)

# Only keep transactions within the first 100 blocks !!!
transactions_df = transactions_df[
    transactions_df['slot'] <= transactions_df['slot_min'] + BLOCK_LIMIT
    ]
transactions_df.columns


# Rename columns for clarity before merging metadata
dune_info_df = dune_info_df.rename(columns={'token_mint_address': MINT_ID})
dune_info_df = dune_info_df[
    [MINT_ID,
     'decimals',
     'name',
     'symbol',
     'token_uri',
     'created_at',
     'init_tx']
    ].drop_duplicates(subset=[MINT_ID], keep='first')
dune_info_df['created_at'] = pd.to_datetime(dune_info_df['created_at'], errors='coerce')

onchain_info_df = onchain_info_df.rename(columns={'mint': MINT_ID})
onchain_info_df = onchain_info_df[[MINT_ID, 'creator', 'bundle_size', 'gas_used', 'block_time']].drop_duplicates(subset=[MINT_ID], keep='first')
onchain_info_df['bundle_size'] = pd.to_numeric(onchain_info_df['bundle_size'], errors='coerce').fillna(0) 
onchain_info_df['gas_used'] = pd.to_numeric(onchain_info_df['gas_used'], errors='coerce')

dune_info_df.columns, onchain_info_df.columns



# Merge metadata into the combined train/test dataframe
combined_df = pd.merge(combined_df, dune_info_df, on=MINT_ID, how='left')
combined_df = pd.merge(combined_df, onchain_info_df, on=MINT_ID, how='left')
combined_df.columns


print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Transactions shape (first 100 blocks): {transactions_df.shape}")
print(f"Combined shape before features: {combined_df.shape}")

# Check missing values in combined metadata
print("\nMissing values in combined metadata:")
combined_df.isnull().sum() / len(combined_df)



print("Target Distribution:")
print(combined_df[TARGET].value_counts(normalize=False))


print("Transaction Data Info:")
transactions_df.info()



transactions_df.describe()


transactions_df.head()


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



def make_advanced_features(chunk_df):
    # 1. Liquidity Ramp Speed
    chunk_df_sorted = chunk_df.sort_values(['mint', 'block_time'])
    mint_groups = chunk_df_sorted.groupby('mint')
    def liquidity_ramp(df):
        if len(df) < 2:
            return np.nan
        max_row = df.loc[df['virtual_sol_balance_after'].idxmax()]
        return (max_row['block_time'] - df['block_time'].min()).total_seconds()

    ramp_speed = mint_groups.apply(liquidity_ramp).reset_index(name='time_to_peak_liquidity')
    print("Liquidity ramp speed calculation completed")

    # 2. Behavioral Burstiness
    def burstiness(df):
        times = df['block_time'].values.astype(np.int64) // 1_000_000_000
        if len(times) < 3:
            return np.nan
        diffs = np.diff(times)
        return np.std(np.log1p(diffs))

    burst = mint_groups.apply(burstiness).reset_index(name='log_std_tx_interval')
    print("Behavioral burstiness calculation completed")

  # 3. Buy Wall Detection
    def buy_wall(df):
        if len(df) < 5:
            return 0
        return np.log1p(df['quote_coin_amount'].rolling(5).sum().max())

    buy_tx = chunk_df[chunk_df['direction'] == 'buy']
    grouped_buy = buy_tx.groupby('mint')
    buy_wall_peak = grouped_buy.apply(buy_wall).reset_index(name='buy_wall_peak')
    print("Buy wall detection completed")

    features = ramp_speed.merge(burst, on='mint', how='outer') \
                         .merge(buy_wall_peak, on='mint', how='outer')
    print("All advanced features merged successfully")

    return features


advanced_features = make_advanced_features(transactions_df)


advanced_features


# Buy/Sell specific features 
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


combined_df = pd.merge(combined_df, agg_features[[c for c in agg_features.columns if c != 'slot_min']], on=MINT_ID, how='left')

# Merge specific features
combined_df = pd.merge(combined_df, buy_agg, on=MINT_ID, how='left')
combined_df = pd.merge(combined_df, sell_agg, on=MINT_ID, how='left')
combined_df = pd.merge(combined_df, advanced_features, on=MINT_ID, how='left')


# Time-based features
combined_df['tx_duration_seconds'] = (combined_df['block_time_max'] - combined_df['block_time_min']).dt.total_seconds()
# Use the slot_min and slot_max derived from the transaction aggregation
combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min']
combined_df['avg_time_between_tx'] = combined_df['tx_duration_seconds'] / (combined_df['tx_idx_count'] + 1e-6) 
combined_df['tx_per_slot'] = combined_df['tx_idx_count'] / (combined_df['slot_nunique'] + 1e-6) 



combined_df['buy_sell_count_ratio'] = combined_df['buy_tx_idx_count'] / (combined_df['sell_tx_idx_count'] + 1e-6)
combined_df['buy_sell_vol_ratio'] = combined_df['buy_quote_coin_amount_sum'] / (combined_df['sell_quote_coin_amount_sum'] + 1e-6)
combined_df['unique_buyer_ratio'] = combined_df['buy_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)
combined_df['unique_seller_ratio'] = combined_df['sell_signing_wallet_nunique'] / (combined_df['signing_wallet_nunique'] + 1e-6)


# Creator interaction
creator_trades = transactions_df.groupby(['base_coin', 'signing_wallet']).size().reset_index(name='trade_count')
creator_trades = pd.merge(creator_trades, onchain_info_df[[MINT_ID, 'creator']], left_on='base_coin', right_on=MINT_ID, how='inner')
creator_trades = creator_trades[creator_trades['signing_wallet'] == creator_trades['creator']]
creator_trades = creator_trades[['base_coin', 'trade_count']].rename(columns={'base_coin': MINT_ID, 'trade_count': 'creator_trade_count'})
creator_trades = creator_trades.drop_duplicates(subset=[MINT_ID], keep='first')


# Merge all and fills NaNs
combined_df = pd.merge(combined_df, creator_trades, on=MINT_ID, how='left')

combined_df['creator_traded'] = combined_df['creator_trade_count'].notna().astype(int)
combined_df['creator_trade_count'] = combined_df['creator_trade_count'].fillna(0)


def engineer_solana_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    epsilon = 1e-9 

    # Ratios
    # Transaction Count Ratios
    df_eng['buy_tx_ratio'] = df_eng['buy_tx_idx_count'] / (df_eng['tx_idx_count'] + epsilon)
    df_eng['sell_tx_ratio'] = df_eng['sell_tx_idx_count'] / (df_eng['tx_idx_count'] + epsilon)
    df_eng['alt_buy_sell_count_ratio'] = df_eng['buy_tx_idx_count'] / (df_eng['sell_tx_idx_count'] + epsilon)

    # Wallet Ratios
    df_eng['active_buyer_ratio'] = df_eng['buy_signing_wallet_nunique'] / (df_eng['signing_wallet_nunique'] + epsilon)
    df_eng['active_seller_ratio'] = df_eng['sell_signing_wallet_nunique'] / (df_eng['signing_wallet_nunique'] + epsilon)
    df_eng['buyer_seller_wallet_ratio'] = df_eng['buy_signing_wallet_nunique'] / (df_eng['sell_signing_wallet_nunique'] + epsilon)
    df_eng['avg_tx_per_wallet'] = df_eng['tx_idx_count'] / (df_eng['signing_wallet_nunique'] + epsilon)
    df_eng['avg_buy_tx_per_buyer'] = df_eng['buy_tx_idx_count'] / (df_eng['buy_signing_wallet_nunique'] + epsilon)
    df_eng['avg_sell_tx_per_seller'] = df_eng['sell_tx_idx_count'] / (df_eng['sell_signing_wallet_nunique'] + epsilon)
    
    # Volume Ratios (Quote/SOL)
    df_eng['buy_quote_volume_ratio'] = df_eng['buy_quote_coin_amount_sum'] / (df_eng['quote_coin_amount_sum'] + epsilon)
    df_eng['sell_quote_volume_ratio'] = df_eng['sell_quote_coin_amount_sum'] / (df_eng['quote_coin_amount_sum'] + epsilon)
    df_eng['avg_quote_per_wallet'] = df_eng['quote_coin_amount_sum'] / (df_eng['signing_wallet_nunique'] + epsilon)
    df_eng['avg_buy_quote_per_buyer'] = df_eng['buy_quote_coin_amount_sum'] / (df_eng['buy_signing_wallet_nunique'] + epsilon)
    df_eng['avg_sell_quote_per_seller'] = df_eng['sell_quote_coin_amount_sum'] / (df_eng['sell_signing_wallet_nunique'] + epsilon)
    df_eng['mean_buy_sell_quote_ratio'] = df_eng['buy_quote_coin_amount_mean'] / (df_eng['sell_quote_coin_amount_mean'] + epsilon)


    # Volume Ratios (Base/Token)
    df_eng['buy_base_volume_ratio'] = df_eng['buy_base_coin_amount_sum'] / (df_eng['base_coin_amount_sum'] + epsilon)
    df_eng['sell_base_volume_ratio'] = df_eng['sell_base_coin_amount_sum'] / (df_eng['base_coin_amount_sum'] + epsilon)
    df_eng['avg_base_per_wallet'] = df_eng['base_coin_amount_sum'] / (df_eng['signing_wallet_nunique'] + epsilon)
    df_eng['avg_buy_base_per_buyer'] = df_eng['buy_base_coin_amount_sum'] / (df_eng['buy_signing_wallet_nunique'] + epsilon)
    df_eng['avg_sell_base_per_seller'] = df_eng['sell_base_coin_amount_sum'] / (df_eng['sell_signing_wallet_nunique'] + epsilon)
    df_eng['mean_buy_sell_base_ratio'] = df_eng['buy_base_coin_amount_mean'] / (df_eng['sell_base_coin_amount_mean'] + epsilon)

    # Creator Ratios
    df_eng['creator_trade_ratio'] = df_eng['creator_trade_count'] / (df_eng['tx_idx_count'] + epsilon)

    # Differences and Spreads
    df_eng['quote_amount_range'] = df_eng['quote_coin_amount_max'] - df_eng['quote_coin_amount_mean']
    df_eng['base_amount_range'] = df_eng['base_coin_amount_max'] - df_eng['base_coin_amount_mean']
    df_eng['virtual_sol_balance_range'] = df_eng['virtual_sol_balance_after_max'] - df_eng['virtual_sol_balance_after_min']
    df_eng['virtual_token_balance_range'] = df_eng['virtual_token_balance_after_max'] - df_eng['virtual_token_balance_after_min']
    
    # Volatility Measurements (Coefficient of Variation)
    df_eng['quote_amount_cv'] = df_eng['quote_coin_amount_std'] / (df_eng['quote_coin_amount_mean'] + epsilon)
    df_eng['base_amount_cv'] = df_eng['base_coin_amount_std'] / (df_eng['base_coin_amount_mean'] + epsilon)
    df_eng['virtual_sol_balance_cv'] = df_eng['virtual_sol_balance_after_std'] / (df_eng['virtual_sol_balance_after_mean'] + epsilon)
    df_eng['virtual_token_balance_cv'] = df_eng['virtual_token_balance_after_std'] / (df_eng['virtual_token_balance_after_mean'] + epsilon)

    # Timing and Intensity
    df_eng['wallets_per_slot'] = df_eng['signing_wallet_nunique'] / (df_eng['slot_nunique'] + epsilon)
    df_eng['sol_volume_per_slot'] = df_eng['quote_coin_amount_sum'] / (df_eng['slot_nunique'] + epsilon)
    df_eng['token_volume_per_slot'] = df_eng['base_coin_amount_sum'] / (df_eng['slot_nunique'] + epsilon)
    df_eng['avg_slot_per_tx'] = df_eng['tx_duration_slots'] / (df_eng['tx_idx_count'] + epsilon)
    
    # Interactions (caution)
    # Interaction of total volume with unique wallet count
    df_eng['volume_wallet_interaction'] = df_eng['quote_coin_amount_sum'] * df_eng['signing_wallet_nunique']
    # Interaction of buy volume with buyer count
    df_eng['buy_volume_buyer_interaction'] = df_eng['buy_quote_coin_amount_sum'] * df_eng['buy_signing_wallet_nunique']


    # Replace infinite values with NaN
    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)


    return df_eng



new_combined_df = engineer_solana_features(combined_df)


new_combined_df.to_csv('new_combined_df.csv', index=False)


# Final Feature Selection
features_to_drop = [
    MINT_ID, TARGET, 'slot_graduated', 'is_train', 'slot_min',
    'name', 'symbol', 'token_uri', 'created_at', 'init_tx',
    'block_time_min', 'block_time_max', "block_time",
    'creator',
    'is_valid', 'Unnamed: 0'
]

features = [col for col in new_combined_df.columns if col not in features_to_drop]


print(f"Using {len(features)} features: {features}")
for f in features:
    if new_combined_df[f].dtype == 'object':
        print(f"Warning: Feature '{f}' is object type. Ensure proper handling.")
        try:
            new_combined_df[f] = pd.to_numeric(new_combined_df[f])
        except:
            print(f"Could not convert {f} to numeric. Consider encoding or dropping.")
            if f in features: features.remove(f)


# Separate train and test again
train_processed = new_combined_df[new_combined_df['is_train'] == 1].reset_index(drop=True)
train_processed[TARGET] = train_processed[TARGET].astype(int)
test_processed = new_combined_df[new_combined_df['is_train'] == 0].reset_index(drop=True)

X = train_processed[features]
y = train_processed[TARGET]
X_test = test_processed[features]

# Clean up memory
del new_combined_df, transactions_df, agg_features, buy_agg, sell_agg, creator_trades
gc.collect()


print(y.value_counts(normalize=False))


cb_oof_preds = np.zeros(len(X))
cb_test_preds = np.zeros(len(X_test))
cb_models = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = cb.CatBoostClassifier(
        iterations=2700,
        learning_rate=0.012,
        loss_function='Logloss',
        eval_metric='Logloss',
        depth=7, # !
        l2_leaf_reg=3,
        random_seed=RANDOM_SEED + fold,
        verbose=0,
        early_stopping_rounds=100,
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              use_best_model=True)

    val_preds = model.predict_proba(X_val)[:, 1]
    cb_oof_preds[val_idx] = val_preds
    cb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_models.append(model)
    print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds)}")


overall_oof_logloss_cb = log_loss(y, cb_oof_preds)
print(f"\nCatBoost Overall OOF LogLoss: {overall_oof_logloss_cb}")


# Save predictions before feature selection
final_test_preds = cb_test_preds

submission_df = pd.DataFrame({
    MINT_ID: test_processed[MINT_ID],
    TARGET: final_test_preds
})
submission_df[TARGET] = np.clip(submission_df[TARGET], 0.0001, 0.9999)
submission_df.to_csv("submission_pre.csv", index=False) 


fi = model.get_feature_importance(Pool(X_val, y_val), type='FeatureImportance')
fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': fi})
fi_df = fi_df.sort_values(by='importance', ascending=False)

fi_df.plot(kind='barh', x='feature', y='importance', figsize=(10, 8))
plt.title("Feature Importance (CatBoost)")
plt.show()


def model_predict(X): return model.predict(X)

result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, scoring='neg_log_loss')

pi_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

pi_df.plot(kind='barh', x='feature', y='importance', figsize=(10, 8))
plt.title("Permutation Importance")
plt.show()


pi_df[pi_df["importance"] <= 0]["feature"]


merged = fi_df.merge(pi_df, on='feature', suffixes=('_model', '_perm'))

model_thresh = merged['importance_model'].mean() * 0.5
perm_thresh = merged['importance_perm'].mean() * 0.5

selected_features = merged[
    (merged['importance_model'] > model_thresh) |
    (merged['importance_perm'] > perm_thresh)
]['feature'].tolist()


X = train_processed[selected_features]
X_test = test_processed[selected_features]


cb_oof_preds_final = np.zeros(len(X))
cb_test_preds_final = np.zeros(len(X_test))
cb_models_final = []

skf_final = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (train_idx, val_idx) in enumerate(skf_final.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    new_model = cb.CatBoostClassifier(
        iterations=2700,
        learning_rate=0.012,
        loss_function='Logloss',
        eval_metric='Logloss',
        depth=7, # !
        l2_leaf_reg=3,
        random_seed=RANDOM_SEED + fold,
        verbose=0,
        early_stopping_rounds=100,
    )

    new_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              use_best_model=True)

    val_preds = new_model.predict_proba(X_val)[:, 1]
    cb_oof_preds_final[val_idx] = val_preds
    cb_test_preds_final += new_model.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_models_final.append(new_model)
    print(f"Fold {fold+1} OOF LogLoss: {log_loss(y_val, val_preds)}")


overall_oof_logloss_cb = log_loss(y, cb_oof_preds_final)
print(f"\nCatBoost Overall OOF LogLoss: {overall_oof_logloss_cb}")


final_test_preds = cb_test_preds_final

submission_df = pd.DataFrame({
    MINT_ID: test_processed[MINT_ID],
    TARGET: final_test_preds
})

submission_df[TARGET] = np.clip(submission_df[TARGET], 0.0001, 0.9999)
submission_df.to_csv("submission_final.csv", index=False)

