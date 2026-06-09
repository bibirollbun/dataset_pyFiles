import pandas as pd
import numpy as np
import glob
from datetime import datetime, timezone

# === Step 1: Load train and all chunks ===
train = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/train.csv")

chunk_files = sorted(glob.glob("/kaggle/input/pump-fun-graduation-february-2025/chunk_*.csv"))
chunks = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)

# Filter only tokens that exist in train
chunks = chunks[chunks["base_coin"].isin(train["mint"])]

# Sort each token's transactions by slot
chunks = chunks.sort_values(by=["base_coin", "slot"]).reset_index(drop=True)

# Merge label
chunks = chunks.merge(train[["mint", "has_graduated"]], left_on="base_coin", right_on="mint", how="left")

# === Step 2: Feature Aggregation (Transaction-based) ===
agg = chunks.groupby("base_coin").agg(
    tx_count=('tx_idx', 'count'),
    total_sol=('quote_coin_amount', 'sum'),
    unique_wallets=('signing_wallet', pd.Series.nunique),
    buy_count=('direction', lambda x: (x == "buy").sum()),
    sell_count=('direction', lambda x: (x == "sell").sum()),
    first_slot=('slot', 'min'),
    last_slot=('slot', 'max')
).reset_index()

agg['duration_slots'] = agg['last_slot'] - agg['first_slot']

# Merge graduation label
agg = agg.merge(train[['mint', 'has_graduated']], left_on='base_coin', right_on='mint', how='left')

# === Step 3: Load Metadata v2 ===
dune_v2 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/dune_token_info_v2.csv")
divers_v2 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/token_info_onchain_divers_v2.csv")

# === Step 4: Prepare Metadata Features ===

# Fix created_at datetime
dune_v2['created_at'] = pd.to_datetime(dune_v2['created_at'], errors='coerce')

# Engineer features safely (timezone-aware now)
now_utc = datetime.now(timezone.utc)
dune_v2['token_age_days'] = (now_utc - dune_v2['created_at']).dt.total_seconds() / (3600 * 24)
dune_v2['token_name_length'] = dune_v2['name'].fillna("").apply(len)
dune_v2['token_symbol_length'] = dune_v2['symbol'].fillna("").apply(len)

# Select needed dune columns
dune_v2_selected = dune_v2[['token_mint_address', 'decimals', 'token_age_days', 'token_name_length', 'token_symbol_length']]

# Engineer Divers features
divers_v2['instructions_density'] = np.where(divers_v2['gas_used'] > 0,
                                              divers_v2['amount_of_instructions'] / divers_v2['gas_used'], 0)
divers_v2['lookup_density'] = np.where(divers_v2['gas_used'] > 0,
                                        (divers_v2['amount_of_lookup_reads'] + divers_v2['amount_of_lookup_writes']) / divers_v2['gas_used'], 0)

# Select needed divers columns
divers_v2_selected = divers_v2[['mint', 'bundle_size', 'gas_used', 'dev_balance',
                                'instructions_density', 'lookup_density', 'pf_program_index', 'direct_pf_invocation']]

# === Step 5: Merge Metadata into agg ===

# Merge dune v2
agg = agg.merge(dune_v2_selected, left_on='base_coin', right_on='token_mint_address', how='left')
if 'token_mint_address' in agg.columns:
    agg.drop(columns=['token_mint_address'], inplace=True)

# Merge divers v2
agg = agg.merge(divers_v2_selected, left_on='base_coin', right_on='mint', how='left')
if 'mint' in agg.columns:
    agg.drop(columns=['mint'], inplace=True)

# === Step 6: Handle Missing Values ===
agg.fillna(0, inplace=True)

# === Step 7: Final Available Features ===
features_to_plot = [
    'tx_count', 'total_sol', 'unique_wallets', 'buy_count', 'sell_count', 'duration_slots',
    'decimals', 'token_age_days', 'token_name_length', 'token_symbol_length',
    'bundle_size', 'gas_used', 'dev_balance', 'instructions_density', 'lookup_density',
    'pf_program_index', 'direct_pf_invocation'
]

print("✅ Metadata integration completed! agg shape:", agg.shape)
print("✅ Final features:", features_to_plot)


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Step 1: Config ===
FEATURES = [
    'tx_count', 'total_sol', 'unique_wallets', 'buy_count', 'sell_count', 'duration_slots',
    'decimals', 'token_age_days', 'token_name_length', 'token_symbol_length',
    'bundle_size', 'gas_used', 'dev_balance', 'instructions_density', 'lookup_density',
    'pf_program_index', 'direct_pf_invocation'
]
SEED = 42

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
DENSE_UNITS = 128
DROPOUT = 0.0
EMBED_DIM = 32
NUM_BLOCKS = 6
NUM_HEADS = 10
FF_DIM = 256
CONV_KERNEL_SIZE = 5

# === Step 2: Prepare Data ===
X = agg[FEATURES].values
y = agg['has_graduated'].astype(int).values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled[..., np.newaxis]  # (samples, n_features, 1)
y = y.reshape(-1, 1)

# Split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=SEED, stratify=y)

print(f"✅ Training set: {X_train.shape}, Validation set: {X_val.shape}")

# === Step 3: Conformer Block ===
def conformer_block(inputs, num_heads, ff_dim, conv_kernel_size, dropout):
    x = tf.keras.layers.LayerNormalization()(inputs)

    # Feedforward Module (Pre)
    ff1 = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
    ff1 = tf.keras.layers.Dropout(dropout)(ff1)
    ff1 = tf.keras.layers.Dense(inputs.shape[-1])(ff1)
    x = tf.keras.layers.Add()([inputs, ff1])

    # Multi-Head Attention
    attn = tf.keras.layers.LayerNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(attn, attn)
    x = tf.keras.layers.Add()([x, attn])

    # Convolution Module
    conv = tf.keras.layers.LayerNormalization()(x)
    conv = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(conv)
    conv = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=conv_kernel_size, padding='same', activation='relu')(conv)
    conv = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(conv)
    x = tf.keras.layers.Add()([x, conv])

    # Feedforward Module (Post)
    ff2 = tf.keras.layers.LayerNormalization()(x)
    ff2 = tf.keras.layers.Dense(ff_dim, activation='relu')(ff2)
    ff2 = tf.keras.layers.Dropout(dropout)(ff2)
    ff2 = tf.keras.layers.Dense(inputs.shape[-1])(ff2)
    x = tf.keras.layers.Add()([x, ff2])

    return tf.keras.layers.LayerNormalization()(x)

# === Step 4: Custom Loss ===
def custom_binary_crossentropy_with_punishment(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    punish_mask = tf.where(tf.abs(y_true - y_pred) > 0.25, 3.0, 1.0)
    punished_loss = bce * punish_mask
    return punished_loss

# === Step 5: F1 Metric ===
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# === Step 6: Build Conformer Model ===
def build_conformer_model():
    input_layer = tf.keras.Input(shape=(X_train.shape[1], 1))

    x = tf.keras.layers.Conv1D(filters=EMBED_DIM, kernel_size=1, activation='relu')(input_layer)

    for _ in range(NUM_BLOCKS):
        x = conformer_block(x, num_heads=NUM_HEADS, ff_dim=FF_DIM, conv_kernel_size=CONV_KERNEL_SIZE, dropout=DROPOUT)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=custom_binary_crossentropy_with_punishment,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            F1Score(name='f1_score'),
            tf.keras.metrics.BinaryCrossentropy(name='log_loss')
        ]
    )
    return model

# === Step 7: Train Model with EarlyStopping ===
model = build_conformer_model()
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)


import pandas as pd
import numpy as np

# === Step 1: Load Test Set ===
test = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/test_unlabeled.csv")
chunk_files = sorted(glob.glob("/kaggle/input/pump-fun-graduation-february-2025/chunk_*.csv"))
chunks = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)

# Filter only test tokens
test_chunks = chunks[chunks["base_coin"].isin(test["mint"])]

# Sort chronologically
test_chunks = test_chunks.sort_values(by=["base_coin", "slot"]).reset_index(drop=True)

# === Step 2: Aggregate Features for Test Set ===
test_agg = test_chunks.groupby("base_coin").agg(
    tx_count=('tx_idx', 'count'),
    total_sol=('quote_coin_amount', 'sum'),
    unique_wallets=('signing_wallet', pd.Series.nunique),
    buy_count=('direction', lambda x: (x == "buy").sum()),
    sell_count=('direction', lambda x: (x == "sell").sum()),
    first_slot=('slot', 'min'),
    last_slot=('slot', 'max')
).reset_index()

test_agg['duration_slots'] = test_agg['last_slot'] - test_agg['first_slot']

# Ensure all test mints are included
test_agg = test[['mint']].merge(test_agg, left_on='mint', right_on='base_coin', how='left')

# === Step 3: Merge Metadata (Dune V2 + Divers V2) ===
# Load metadata
dune_v2 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/dune_token_info_v2.csv")
divers_v2 = pd.read_csv("/kaggle/input/pump-fun-graduation-february-2025/token_info_onchain_divers_v2.csv")

# Prepare Dune metadata
dune_v2['created_at'] = pd.to_datetime(dune_v2['created_at'], errors='coerce')
now_utc = datetime.now(timezone.utc)
dune_v2['token_age_days'] = (now_utc - dune_v2['created_at']).dt.total_seconds() / (3600 * 24)
dune_v2['token_name_length'] = dune_v2['name'].fillna("").apply(len)
dune_v2['token_symbol_length'] = dune_v2['symbol'].fillna("").apply(len)
dune_v2_selected = dune_v2[['token_mint_address', 'decimals', 'token_age_days', 'token_name_length', 'token_symbol_length']]

# Prepare Divers metadata
divers_v2['instructions_density'] = np.where(divers_v2['gas_used'] > 0,
                                              divers_v2['amount_of_instructions'] / divers_v2['gas_used'], 0)
divers_v2['lookup_density'] = np.where(divers_v2['gas_used'] > 0,
                                        (divers_v2['amount_of_lookup_reads'] + divers_v2['amount_of_lookup_writes']) / divers_v2['gas_used'], 0)
divers_v2_selected = divers_v2[['mint', 'bundle_size', 'gas_used', 'dev_balance',
                                'instructions_density', 'lookup_density', 'pf_program_index', 'direct_pf_invocation']]

# Merge metadata into test_agg
test_agg = test_agg.merge(dune_v2_selected, left_on='mint', right_on='token_mint_address', how='left')
if 'token_mint_address' in test_agg.columns:
    test_agg.drop(columns=['token_mint_address'], inplace=True)

test_agg = test_agg.merge(divers_v2_selected, left_on='mint', right_on='mint', how='left')
if 'mint_y' in test_agg.columns:
    test_agg.rename(columns={'mint_x': 'mint'}, inplace=True)

# Fill missing values
test_agg.fillna(0, inplace=True)

# === Step 4: Prepare Test Features ===
test_features = [
    'tx_count', 'total_sol', 'unique_wallets', 'buy_count', 'sell_count', 'duration_slots',
    'decimals', 'token_age_days', 'token_name_length', 'token_symbol_length',
    'bundle_size', 'gas_used', 'dev_balance', 'instructions_density', 'lookup_density',
    'pf_program_index', 'direct_pf_invocation'
]

X_test = test_agg[test_features].values
X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled[..., np.newaxis]

# === Step 5: Predict ===
test_preds = model.predict(X_test_scaled, batch_size=BATCH_SIZE).flatten()

# === Step 6: Build Submission ===
submission = pd.DataFrame({
    "mint": test_agg['mint'],
    "has_graduated": test_preds
})

# Drop duplicate mint entries
submission = submission.drop_duplicates(subset='mint', keep='first')

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved successfully (no duplicates)!")

