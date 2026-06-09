import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


DIR = '/kaggle/input/pump-fun-graduation-february-2025/'
chunk_names = []
for _, _, filenames in os.walk(DIR):
    for filename in filenames:
        if 'chunk' in filename:
            chunk_names.append(filename)
print(chunk_names)


def merge_csv_chunks(chunk_files, folder_path='.', verbose=False):
    df_list = []
    for file in chunk_files:
        full_path = os.path.join(folder_path, file)
        if verbose:
            print(f"Reading {full_path}...")
        df = pd.read_csv(full_path)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df



chunk_data = merge_csv_chunks(chunk_names, folder_path=DIR, verbose=True)


chunk_data.head()


chunk_data.columns


BLOCK_LIMIT = 100

# Ensure datetime
chunk_data['block_time'] = pd.to_datetime(chunk_data['block_time'])

# --- Filter first 100 blocks after mint ---
slot_min_map = chunk_data.groupby('base_coin')['slot'].min().rename('slot_min')
chunk_data = chunk_data.merge(slot_min_map, on='base_coin', how='left')
chunk_data = chunk_data[chunk_data['slot'] <= chunk_data['slot_min'] + BLOCK_LIMIT]

# --- Time-based features ---
time_features = chunk_data.groupby('base_coin')['block_time'].agg([
    ('first_tx_time', 'min'),
    ('last_tx_time', 'max'),
    ('activity_duration_sec', lambda x: (x.max() - x.min()).total_seconds())
])

# --- Numeric features ---
numeric_features = chunk_data.groupby('base_coin').agg({
    'base_coin_amount': ['sum', 'mean', 'max'],
    'quote_coin_amount': ['sum', 'mean'],
    'fee': ['mean', 'sum'],
    'consumed_gas': ['mean', 'sum'],
    'signature': 'count',
    'signing_wallet': ['nunique'],
    'virtual_sol_balance_after': ['last', 'max', 'min', 'mean', 'std'],
    'virtual_token_balance_after': ['last', 'max', 'min', 'mean', 'std']
})
numeric_features.columns = ['_'.join(col) for col in numeric_features.columns]

# --- Directional transaction counts (buy/sell) ---
direction_counts = chunk_data.pivot_table(index='base_coin', columns='direction',
                                          values='signature', aggfunc='count', fill_value=0)
direction_counts.columns = [f'tx_count_{col}' for col in direction_counts.columns]


# --- First hour features ---
first_tx_time = chunk_data.groupby('base_coin')['block_time'].min().rename('first_tx_time')
chunk_data = chunk_data.merge(first_tx_time, on='base_coin', how='left')
chunk_data['since_first_tx_sec'] = (chunk_data['block_time'] - chunk_data['first_tx_time']).dt.total_seconds()
txs_in_first_hour = chunk_data[chunk_data['since_first_tx_sec'] <= 3600].groupby('base_coin').size().rename('tx_count_first_hour')

# --- Gas in first 10 txs ---
chunk_data_sorted = chunk_data.sort_values(['base_coin', 'block_time'])
first_10 = chunk_data_sorted.groupby('base_coin').head(10)
gas_first_10 = first_10.groupby('base_coin')['consumed_gas'].sum().rename('gas_sum_first_10_tx')

# --- Derived transactional features ---
agg_all = chunk_data.groupby('base_coin').agg({
    'block_time': ['min', 'max'],
    'slot': ['min', 'max', 'nunique'],
    'signature': 'count',
})
agg_all.columns = ['_'.join(col) for col in agg_all.columns]
agg_all = agg_all.rename(columns={
    'block_time_min': 'block_time_min',
    'block_time_max': 'block_time_max',
    'slot_min': 'slot_min',
    'slot_max': 'slot_max',
    'slot_nunique': 'slot_nunique',
    'signature_count': 'tx_idx_count'
})

# Directional aggregates
buy_tx = chunk_data[chunk_data['direction'] == 'buy']
sell_tx = chunk_data[chunk_data['direction'] == 'sell']

buy_agg = buy_tx.groupby('base_coin').agg({
    'signature': 'count',
    'quote_coin_amount': 'sum',
    'signing_wallet': 'nunique'
}).rename(columns={
    'signature': 'buy_tx_idx_count',
    'quote_coin_amount': 'buy_quote_coin_amount_sum',
    'signing_wallet': 'buy_signing_wallet_nunique'
})

sell_agg = sell_tx.groupby('base_coin').agg({
    'signature': 'count',
    'quote_coin_amount': 'sum',
    'signing_wallet': 'nunique'
}).rename(columns={
    'signature': 'sell_tx_idx_count',
    'quote_coin_amount': 'sell_quote_coin_amount_sum',
    'signing_wallet': 'sell_signing_wallet_nunique'
})



combined_df = agg_all.join([buy_agg, sell_agg], how='left').fillna(0)


train=pd.read_csv('train.csv')
test_unlabled=pd.read_csv('test_unlabeled.csv')
tk_info_v2=pd.read_csv('dune_token_info_v2.csv')
tk_info_onchain_v2=pd.read_csv('token_info_onchain_divers_v2.csv')


train['is_train']=1
test_unlabled['is_train']=0
train = pd.concat([train, test_unlabled], ignore_index=True)


train['is_valid'].value_counts()


# --- Derived features ---
combined_df['tx_duration_seconds'] = (combined_df['block_time_max'] - combined_df['block_time_min']).dt.total_seconds()
combined_df['tx_duration_slots'] = combined_df['slot_max'] - combined_df['slot_min']
combined_df['avg_time_between_tx'] = combined_df['tx_duration_seconds'] / (combined_df['tx_idx_count'] + 1e-6)
combined_df['tx_per_slot'] = combined_df['tx_idx_count'] / (combined_df['slot_nunique'] + 1e-6)
combined_df['buy_sell_count_ratio'] = combined_df['buy_tx_idx_count'] / (combined_df['sell_tx_idx_count'] + 1e-6)
combined_df['buy_sell_vol_ratio'] = combined_df['buy_quote_coin_amount_sum'] / (combined_df['sell_quote_coin_amount_sum'] + 1e-6)
combined_df['unique_buyer_ratio'] = combined_df['buy_signing_wallet_nunique'] / (numeric_features['signing_wallet_nunique'] + 1e-6)
combined_df['unique_seller_ratio'] = combined_df['sell_signing_wallet_nunique'] / (numeric_features['signing_wallet_nunique'] + 1e-6)

# --- Creator interaction ---
creator_trades = chunk_data.groupby(['base_coin', 'signing_wallet']).size().reset_index(name='trade_count')
creator_trades = pd.merge(creator_trades, tk_info_onchain_v2[['mint', 'creator']], left_on='base_coin', right_on='mint', how='inner')
creator_trades = creator_trades[creator_trades['signing_wallet'] == creator_trades['creator']]
creator_trades = creator_trades[['base_coin', 'trade_count']].drop_duplicates(subset=['base_coin'])
creator_trades = creator_trades.rename(columns={'base_coin': 'mint', 'trade_count': 'creator_trade_count'})

# --- Final aggregation ---
aggregated_features = pd.concat([
    time_features,
    numeric_features,
    direction_counts,
    txs_in_first_hour,
    gas_first_10,
    combined_df,
], axis=1)


aggregated_features = aggregated_features.reset_index().rename(columns={'base_coin': 'mint'})
aggregated_features = pd.merge(aggregated_features, creator_trades, on='mint', how='left').fillna({'creator_trade_count': 0})


aggregated_features.columns


print('train/',train.columns)
print('-------------------------------------------------------------------------------------------------------------------------------')
print('chunk_data/',aggregated_features.columns)
print('-------------------------------------------------------------------------------------------------------------------------------')
print('tk_info_v2/',tk_info_v2.columns)
print('-------------------------------------------------------------------------------------------------------------------------------')
print('tk_info_onchain_v2/',tk_info_onchain_v2.columns)


# Step 1: Merge chunk_data with tk_info_v2
merged_chunk_data = aggregated_features.merge(
    tk_info_v2,
    left_on='mint',
    right_on='token_mint_address',
     how='inner'
)

# Step 2: Merge the result with tk_info_onchain_v2
merged_chunk_data = merged_chunk_data.merge(
    tk_info_onchain_v2,
    left_on='mint',
    right_on='mint',
    how='inner'
)


merged_chunk_data.head()


merged_chunk_data.shape


merged_chunk_data['block_time'] = pd.to_datetime(merged_chunk_data['block_time'])



train['is_valid'].value_counts()


train_final = train.merge(merged_chunk_data, on='mint', how='left')


print(train.shape)
print(train_final.shape)


train_final['has_graduated'].value_counts()


train_final.columns


class_distribution = train_final['has_graduated'].value_counts()
print("Class Counts:")
print(class_distribution)
print("\nClass Percentages:")
print(class_distribution / len(train_final) * 100)
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution of has_graduated')
plt.xlabel('Class (0 = Not Graduated, 1 = Graduated)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


data= train_final
del train_final


obj_lst=[]
for obj in data.columns:
    if data[obj].dtype=='O':
        obj_lst.append(obj)
    


data = data.drop(columns=['first_tx_time',
 'last_tx_time',
 'block_time_min',
 'block_time_max',
 'token_uri',
 'creator',
 'name_y',
 'symbol_y'])


data=data.drop(columns=['Unnamed: 0','slot_graduated','is_valid','curve_address','created_at','init_tx','block_time','tx_idx','bundle_size','url','amount_of_lookup_reads','amount_of_lookup_writes','bundle_structure', 'bundled_buys_count','creation_ix_index','curve_address','pf_program_index','direct_pf_invocation', 'version'],axis=1)


data.duplicated().sum()


data=data.drop_duplicates()


data.shape


famous_terms = [
    # People
    'elon', 'musk', 'trump', 'vitalik', 'satoshi', 'cz', 'zhao', 
    'buterin', 'saylor', 'woods', 'novogratz', 'armstrong', 'schiff',
    'wood', 'ackman', 'dalio', 'dorian', 'nakamoto', 'fried', 'bankman',
    
    # Companies/Projects
    'binance', 'coinbase', 'ftx', 'solana', 'ethereum', 'bitcoin',
    'cardano', 'ripple', 'polkadot', 'terra', 'luna', 'avax', 'chainlink',
    'uniswap', 'aave', 'compound', 'maker', 'yearn', 'curve', 'balancer',
    
    # Memes/Animals
    'doge', 'shiba', 'floki', 'kishu', 'hoge', 'safemoon', 'wojak',
    'pepe', 'dogelon', 'husky', 'akita', 'kiba', 'babydoge', 'mona',
    
    # Tech Terms
    'web3', 'metaverse', 'nft', 'dao', 'defi', 'gamefi', 'l2', 'rollup',
    'zksync', 'optimism', 'arbitrum', 'polygon', 'sidechain', 'oracle',
    
    # Financial Terms
    'moon', 'lambo', 'rocket', 'bull', 'bear', 'whale', 'diamond', 'hands',
    'hodl', 'fomo', 'fud', 'rekt', 'ape', 'wagmi', 'ngmi', 'dyor', 'tvl',
    
    # Numbers/Dates
    '10x','50x', '100x', '1000x', '1mil', '1billion','1m','1b'
    '69', '420', '777', '1337', '10k', '100k', '1m', '1b',
    
    # Geographic
    'dubai', 'singapore', 'miami', 'zurich', 'malta', 'hk', 'hkong',
    'korea', 'japan', 'china', 'america', 'europe', 'africa', 'asia',
    
    # Pop Culture
    'tesla', 'spacex', 'twitter', 'facebook', 'google', 'amazon',
    'apple', 'microsoft', 'netflix', 'disney', 'marvel', 'dc',
    'starwars', 'startrek', 'matrix', 'avatar', 'spiderman',
    
    # Luxury Brands
    'rolex', 'patek', 'audemars', 'gucci', 'prada', 'versace',
    'ferrari', 'lambo', 'bugatti', 'yacht', 'jet', 'private'
]

common_symbols = [
    'BTC', 'ETH', 'SOL', 'USD', 'XRP',  # majors
    'DOGE', 'SHIBA', 'PEPE', 'FLOKI', 'BABYDOGE',  # classic memecoins
    'TRUMP', 'ELON', 'MUSK', 'BIDEN',  # politics/influencers
    'MEME', 'MEMEFI', 'DEGEN', 'REKT', 'MOON', 'PUMP', 'AI',  # buzzwords
    'POL', 'ARB', 'BONK', 'WIF', 'JUP', 'PYTH',  # recent airdrop/trending
    'GME', 'AMC', 'STONK', 'ROCKET',  # retail trader memes
    'CHAT', 'GPT', 'LLM', 'SOLANA', 'LAMBO',  # tech/culture
    'TOKEN', 'COIN', 'CASH', 'INU'
    "TRUMP2024","TRUMP2025","TRUMP2030","DOGE100","DOGE1000",'1000DOGE'
]


from collections import Counter


data['name_x'] = data['name_x'].fillna('') 
data['symbol_x'] = data['symbol_x'].fillna('')

# Features on name
data['name_length'] = data['name_x'].str.len()

data['name_has_pump'] = data['name_x'].str.contains(
    'pump|moon|rocket|100x|50x|10x', case=False, regex=True).astype(int)

data['name_word_count'] = data['name_x'].str.split().str.len()

data['name_starts_with_symbol'] = data['name_x'].str.match(r'^[^\w]').fillna(False).astype(int)

data['name_has_trendy_name'] = data['name_x'].str.lower().apply(
    lambda x: int(any(term in x for term in famous_terms)) if pd.notnull(x) else 0
)

# Features on symbol
data['symbol_length'] = data['symbol_x'].str.len()

data['symbol_has_digits'] = data['symbol_x'].str.contains(r'\d', regex=True).fillna(False).astype(int)

data['symbol_repeated_chars'] = data['symbol_x'].apply(
    lambda x: max(Counter(str(x)).values()) if pd.notnull(x) and str(x).strip() != '' else 0
)

data['symbol_is_common'] = data['symbol_x'].str.upper().apply(
    lambda x: int(any(sym in x for sym in common_symbols)) if pd.notnull(x) else 0
)

# Combined Features
data['name_symbol_match'] = (data['name_x'].str.lower() == data['symbol_x'].str.lower()).astype(int)

data['name_symbol_length_ratio'] = data['name_length'] / (data['symbol_length'] + 1e-6)



data[['name_length', 'symbol_length', 'symbol_repeated_chars', 'name_symbol_length_ratio']].describe().T


num_cols = ['name_length', 'symbol_length', 'symbol_repeated_chars', 'name_symbol_length_ratio']
for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

cat_cols = ['name_has_pump', 'name_starts_with_symbol', 'symbol_has_digits', 'symbol_is_common']
for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])


data.describe()


data['dev_balance_ratio'] = data['dev_balance'] / (data['base_coin_amount_sum'] + 1e-6)


train= data[data['is_train'] == 1].drop(columns=['is_train'])
test_unlabeled= data[data['is_train'] == 0].drop(columns=['is_train'])


test_unlabeled.shape


train.to_csv('train_v3.csv',index=False)
test_unlabeled.to_csv('test_v3.csv',index=False)


def detect_outliers_iqr(df, iqr_multiplier=1.5):
    outlier_summary = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = is_outlier.sum()
        total_count = df[col].count()

        outlier_summary.append({
            'column': col,
            'outlier_count': outlier_count,
            'total_count': total_count,
            'outlier_percent': round(100 * outlier_count / total_count, 2),
            'lower_threshold': lower_bound,
            'upper_threshold': upper_bound
        })

    return pd.DataFrame(outlier_summary).sort_values(by='outlier_percent', ascending=False).reset_index(drop=True)



outlier_report = detect_outliers_iqr(data)
outlier_report.head(50)


outlier_report.tail(15)


from sklearn.preprocessing import RobustScaler
def scale_with_robust_scaler(df):
    """
    Automatically detects numeric columns and applies RobustScaler to them.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Scaled DataFrame with same column names
    """
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    scaler = RobustScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_scaled



scaled_df = scale_with_robust_scaler(data)

