import pandas as pd
import numpy as np
import glob
import os
import gc
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStoppings
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



DATA_PATH = '/kaggle/input/pump-fun-graduation-february-2025'
CHUNK_PATTERN = os.path.join(DATA_PATH, 'chunk*.csv')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test_unlabeled.csv')
DUNE_INFO_FILE = os.path.join(DATA_PATH, 'dune_token_info_v2.csv')
ONCHAIN_INFO_FILE = os.path.join(DATA_PATH, 'token_info_onchain_divers_v2.csv')
SUBMISSION_FILE = 'submission.csv'

TARGET = 'has_graduated'
MINT_ID = 'mint'
BLOCK_LIMIT = 100
RANDOM_SEED = 42
N_SPLITS = 5
SEQUENCE_LENGTH = 100
BATCH_SIZE = 50
EPOCHS = 50


# 1. Load Data
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
dune_info_df = pd.read_csv(DUNE_INFO_FILE)
onchain_info_df = pd.read_csv(ONCHAIN_INFO_FILE)

# Combine train and test for processing
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Load chunk files
all_chunk_files = glob.glob(CHUNK_PATTERN)
print(f"Found {len(all_chunk_files)} chunk files.")
chunk_list = []
for f in tqdm(all_chunk_files, desc="Loading chunks"):
    chunk_list.append(pd.read_csv(f))
transactions_df = pd.concat(chunk_list, ignore_index=True)

# Convert and ensure numeric types
transactions_df['block_time'] = pd.to_datetime(transactions_df['block_time'], errors='coerce')
transactions_df['slot'] = pd.to_numeric(transactions_df['slot'], errors='coerce')
combined_df['slot_min'] = pd.to_numeric(combined_df['slot_min'], errors='coerce')


# 2. Data Merging and Preprocessing
print("Merging data...")
# Merge slot_min for time-aware filtering
transactions_df = pd.merge(
    transactions_df,
    combined_df[[MINT_ID, 'slot_min']],
    left_on='base_coin',
    right_on=MINT_ID,
    how='left'
)

# Filter transactions to first 100 blocks
transactions_df = transactions_df[
    (transactions_df['slot'] >= transactions_df['slot_min']) &
    (transactions_df['slot'] <= transactions_df['slot_min'] + BLOCK_LIMIT - 1)
]

# Rename and clean metadata
dune_info_df = dune_info_df.rename(columns={'token_mint_address': MINT_ID})
dune_info_df = dune_info_df[[MINT_ID, 'decimals', 'name', 'symbol', 'token_uri', 'created_at']].drop_duplicates(subset=[MINT_ID])
dune_info_df['created_at'] = pd.to_datetime(dune_info_df['created_at'], errors='coerce')

onchain_info_df = onchain_info_df.rename(columns={'mint': MINT_ID})
onchain_info_df = onchain_info_df[[MINT_ID, 'creator', 'bundle_size', 'gas_used']].drop_duplicates(subset=[MINT_ID])
onchain_info_df['bundle_size'] = pd.to_numeric(onchain_info_df['bundle_size'], errors='coerce').fillna(0)
onchain_info_df['gas_used'] = pd.to_numeric(onchain_info_df['gas_used'], errors='coerce').fillna(0)

# Merge metadata
combined_df = pd.merge(combined_df, dune_info_df, on=MINT_ID, how='left')
combined_df = pd.merge(combined_df, onchain_info_df, on=MINT_ID, how='left')



# 3. Exploratory Data Analysis
print("Basic EDA (Conceptual):")
print(f"Train shape: {train_df.shape}")
train_df.head()


print(f"Test shape: {test_df.shape}")
test_df.head()


print(f"Transactions shape (first 100 blocks): {transactions_df.shape}")
transactions_df.head()


print(f"Combined shape before features: {combined_df.shape}")
combined_df.head()


# 3. Sequence Feature Engineering
print("Creating sequence features...")
# Create block-level sequences for each token
def create_sequences(df, mint_ids, slot_mins):
    sequence_features = ['tx_count', 'total_quote', 'total_base', 'buy_count', 'sell_count', 'unique_wallets']
    sequences = []
    for mint, slot_min in tqdm(zip(mint_ids, slot_mins), total=len(mint_ids), desc="Building sequences"):
        token_tx = df[(df['base_coin'] == mint) & (df['slot'] >= slot_min) & (df['slot'] < slot_min + SEQUENCE_LENGTH)]
        block_range = pd.DataFrame({'slot': range(int(slot_min), int(slot_min) + SEQUENCE_LENGTH)})
        
        # Aggregate per block
        agg = token_tx.groupby('slot').agg({
            'tx_idx': 'count',
            'quote_coin_amount': 'sum',
            'base_coin_amount': 'sum',
            'signing_wallet': 'nunique'
        }).rename(columns={
            'tx_idx': 'tx_count',
            'quote_coin_amount': 'total_quote',
            'base_coin_amount': 'total_base',
            'signing_wallet': 'unique_wallets'
        })
        
        # Buy and sell counts
        buy_count = token_tx[token_tx['direction'] == 'buy'].groupby('slot')['tx_idx'].count().rename('buy_count')
        sell_count = token_tx[token_tx['direction'] == 'sell'].groupby('slot')['tx_idx'].count().rename('sell_count')
        agg = agg.join(buy_count).join(sell_count).fillna(0)
        
        # Merge with block range and fill missing blocks
        agg = block_range.merge(agg, on='slot', how='left').fillna(0)
        seq = agg[sequence_features].values
        sequences.append(seq)
    
    return np.array(sequences)

# Generate sequences
mint_ids = combined_df[MINT_ID].values
slot_mins = combined_df['slot_min'].values
sequences = create_sequences(transactions_df, mint_ids, slot_mins)



#4. Static Feature Engineering
print("Creating static features...")
# Static features from metadata
combined_df['name_length'] = combined_df['name'].str.len().fillna(0)
combined_df['symbol_length'] = combined_df['symbol'].str.len().fillna(0)
combined_df['has_token_uri'] = combined_df['token_uri'].notna().astype(int)
combined_df['created_at_days'] = (combined_df['created_at'] - combined_df['created_at'].min()).dt.total_seconds() / (24 * 3600)
combined_df['created_at_days'] = combined_df['created_at_days'].fillna(combined_df['created_at_days'].median())

# Encode creator
le = LabelEncoder()
combined_df['creator_encoded'] = le.fit_transform(combined_df['creator'].fillna('unknown'))

# Creator trading features
creator_trades = transactions_df.groupby(['base_coin', 'signing_wallet']).size().reset_index(name='trade_count')
creator_trades = pd.merge(creator_trades, onchain_info_df[[MINT_ID, 'creator']], left_on='base_coin', right_on=MINT_ID, how='inner')
creator_trades = creator_trades[creator_trades['signing_wallet'] == creator_trades['creator']]
creator_trades = creator_trades[['base_coin', 'trade_count']].rename(columns={'base_coin': MINT_ID, 'trade_count': 'creator_trade_count'})
creator_trades = creator_trades.drop_duplicates(subset=[MINT_ID])
combined_df = pd.merge(combined_df, creator_trades, on=MINT_ID, how='left')
combined_df['creator_traded'] = combined_df['creator_trade_count'].notna().astype(int)
combined_df['creator_trade_count'] = combined_df['creator_trade_count'].fillna(0)

# Select static features
static_features = [
    'decimals', 'created_at_days', 'name_length', 'symbol_length', 'has_token_uri',
    'bundle_size', 'gas_used', 'creator_encoded', 'creator_traded', 'creator_trade_count'
]
static_data = combined_df[static_features].copy()

# Handle missing values
static_data = static_data.fillna(static_data.median())


# --- 5. Data Preparation for LSTM ---
print("Preparing data for LSTM...")
# Split train and test
train_mask = combined_df['is_train'] == 1
train_sequences = sequences[train_mask]
test_sequences = sequences[~train_mask]
train_static = static_data[train_mask]
test_static = static_data[~train_mask]
y = combined_df[train_mask][TARGET].astype(int)

# Scale features
sequence_scaler = StandardScaler()
train_sequences_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
train_sequences_scaled = sequence_scaler.fit_transform(train_sequences_flat).reshape(train_sequences.shape)
test_sequences_scaled = sequence_scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)

static_scaler = StandardScaler()
train_static_scaled = static_scaler.fit_transform(train_static)
test_static_scaled = static_scaler.transform(test_static)

# Train-validation split
X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
    train_sequences_scaled, train_static_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)



#  6. LSTM Model 
print("Building LSTM model...")
# Sequence input
seq_input = Input(shape=(SEQUENCE_LENGTH, len(['tx_count', 'total_quote', 'total_base', 'buy_count', 'sell_count', 'unique_wallets'])), name='seq_input')
lstm_out = LSTM(32, return_sequences=False)(seq_input)

# Static input
static_input = Input(shape=(len(static_features),), name='static_input')
static_dense = Dense(16, activation='relu')(static_input)

# Combine
combined = Concatenate()([lstm_out, static_dense])
combined = Dense(64, activation='relu')(combined)
combined = Dropout(0.3)(combined)
output = Dense(1, activation='sigmoid')(combined)

# Model
model = Model(inputs=[seq_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train
print("Training LSTM model...")
history = model.fit(
    [X_seq_train, X_static_train], y_train,
    validation_data=([X_seq_val, X_static_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
val_preds = model.predict([X_seq_val, X_static_val]).flatten()
val_logloss = log_loss(y_val, val_preds)
print(f"Validation LogLoss: {val_logloss}")


# 7. Prediction and Submission 
print("Generating predictions...")
test_preds = model.predict([test_sequences_scaled, test_static_scaled]).flatten()

# Create submission
submission_df = pd.DataFrame({
    MINT_ID: test_df[MINT_ID],
    TARGET: np.clip(test_preds, 0.0001, 0.9999)
})
submission_df.to_csv(SUBMISSION_FILE, index=False)

print(f"Submission saved to {SUBMISSION_FILE}")
print(submission_df.head())

# Clean up
del sequences, train_sequences, test_sequences, static_data
gc.collect()
















