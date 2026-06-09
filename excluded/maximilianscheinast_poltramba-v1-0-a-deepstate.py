import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D, Conv1D, MultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import re

# --- 1. Configuration ---
# Hyperparameters for the model and training
MAX_LEN = 384
RANDOM_STATE = 42

# --- Architectural Improvements ---
EMBEDDING_DIM = 512 # Reduced slightly to accommodate a deeper model
FF_DIM = EMBEDDING_DIM * 4 # Standard practice: FFN is 4x embedding dim
NUM_HEADS = 6
DROPOUT_RATE = 0.15

# Deeper Architecture: We will alternate Mamba and Transformer blocks
NUM_BLOCKS = 8 # Total number of core blocks (4 Mamba, 4 Transformer)

# Mamba specific hyperparameters (inspired by the reference script)
MAMBA_STATE_DIM = 16
MAMBA_CONV_WIDTH = 4
MAMBA_EXPAND_FACTOR = 2

# CNN specific hyperparameters for local feature extraction
CONV_FILTERS = EMBEDDING_DIM # Match CONV output dim with core block input dim
CONV_KERNEL_SIZE = 5

# General model hyperparameters
DENSE_UNITS = 512
EPOCHS = 200 # Increased epochs for a deeper model, EarlyStopping will manage it
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.1

# Define target columns
TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
NUM_TARGETS = len(TARGET_COLS)

# --- 2. Load and Combine Data ---
print("Loading data...")
# Set to the real Kaggle path. The dummy data section below will be skipped if this path exists.
data_path = '/kaggle/input/neurips-open-polymer-prediction-2025/'
if not os.path.exists(data_path):
    print("Kaggle path not found. Creating dummy data for local testing.")
    data_path = './dummy_data/'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, 'train_supplement'), exist_ok=True)
    pd.DataFrame({
        'id': range(100), 'SMILES': ['C'*i for i in range(1, 101)], 'Tg': np.random.rand(100) * 100,
        'FFV': np.random.rand(100), 'Tc': np.random.rand(100) * 50, 'Density': np.random.rand(100) + 1,
        'Rg': np.random.rand(100) * 10
    }).to_csv(os.path.join(data_path, 'train.csv'), index=False)
    pd.DataFrame({'id': range(100, 200), 'SMILES': ['N'*i for i in range(1, 101)]}).to_csv(os.path.join(data_path, 'test.csv'), index=False)
    pd.DataFrame({'SMILES': ['O'*i for i in range(1, 51)], 'Tc': np.random.rand(50) * 50}).to_csv(os.path.join(data_path, 'train_supplement/dataset1.csv'), index=False)
    pd.DataFrame({'SMILES': ['P'*i for i in range(1, 51)], 'Tg': np.random.rand(50) * 100}).to_csv(os.path.join(data_path, 'train_supplement/dataset2.csv'), index=False)
    pd.DataFrame({'SMILES': ['Q'*i for i in range(1, 51)], 'FFV': np.random.rand(50)}).to_csv(os.path.join(data_path, 'train_supplement/dataset3.csv'), index=False)
    pd.DataFrame({'SMILES': ['R'*i for i in range(1, 51)], 'Density': np.random.rand(50) + 1}).to_csv(os.path.join(data_path, 'train_supplement/dataset4.csv'), index=False)

train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
print(f"Original train.csv shape: {train_df.shape}")

print("\nLoading and combining supplemental data...")
supplement_path = os.path.join(data_path, 'train_supplement')
supp_dfs = []
if os.path.exists(supplement_path):
    for file in os.listdir(supplement_path):
        if file.endswith('.csv'):
            file_path = os.path.join(supplement_path, file)
            print(f"Found and loading: {file_path}")
            df_supp = pd.read_csv(file_path)
            for col in TARGET_COLS:
                if col not in df_supp.columns:
                    df_supp[col] = np.nan
            supp_dfs.append(df_supp)

all_train_dfs = [train_df] + supp_dfs
combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
combined_train_df.drop_duplicates(subset=['SMILES'], inplace=True, keep='first')
combined_train_df.reset_index(drop=True, inplace=True)
print(f"\nCombined train data shape (after dropping duplicates): {combined_train_df.shape}")

# --- 3. Refined Preprocessing and Tokenization ---
print("\nRefining SMILES preprocessing...")
def tokenize_smiles(smiles_string):
    pattern = "(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|C|H|.)"
    regex = re.compile(pattern)
    tokens = regex.findall(smiles_string)
    return tokens

VOCAB = [
    '<pad>', '<unk>', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
    'c', 'n', 'o', 's', 'p', '[B]', '[Si]', '[H]', '[H-]', '[Li]', '[Na]', '[K]',
    '[C-]', '[N+]', '[O-]', '[S-]', '[S+]', '[n+]', '[n-]',
    '(', ')', '[', ']', '=', '#', '.', '+', '-', '%',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\', '/'
]
VOCAB_SIZE = len(VOCAB)
char_to_int = {token: i for i, token in enumerate(VOCAB)}

def smiles_to_int_sequence(smiles, tokenizer):
    tokens = tokenize_smiles(smiles)
    sequence = [tokenizer.get(token, tokenizer['<unk>']) for token in tokens]
    return sequence

print("Converting SMILES to padded integer sequences...")
X_combined_tokens = [smiles_to_int_sequence(s, char_to_int) for s in combined_train_df['SMILES']]
X_test_tokens = [smiles_to_int_sequence(s, char_to_int) for s in test_df['SMILES']]

X_combined_pad = pad_sequences(X_combined_tokens, maxlen=MAX_LEN, padding='post', truncating='post', value=char_to_int['<pad>'])
X_test_pad = pad_sequences(X_test_tokens, maxlen=MAX_LEN, padding='post', truncating='post', value=char_to_int['<pad>'])

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Padded training data shape: {X_combined_pad.shape}")
print(f"Padded test data shape: {X_test_pad.shape}")


# --- 4. Custom Architectural Layers ---
@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return self.token_embeddings.compute_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="gelu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False):
        attn_output = self.att(query=inputs, value=inputs, key=inputs, training=training, use_causal_mask=False)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

@tf.keras.utils.register_keras_serializable()
class MambaBlock(tf.keras.layers.Layer):
    """Mamba State Space Model Block, corrected implementation."""
    def __init__(self, model_dim, state_dim, conv_width, expand_factor, **kwargs):
        super().__init__(**kwargs)
        self.model_dim, self.state_dim, self.conv_width, self.expand_factor = model_dim, state_dim, conv_width, expand_factor
        self.D_inner = self.model_dim * self.expand_factor
        
        self.in_proj = Dense(self.D_inner * 2, name="in_proj")
        self.conv1d = Conv1D(filters=self.D_inner, kernel_size=self.conv_width, groups=self.D_inner, padding='causal', name="conv1d")
        
        self.x_proj = Dense(self.D_inner + self.state_dim * 2, name="x_proj")
        
        self.out_proj = Dense(self.model_dim, name="out_proj")
        
        self._A_log = self.add_weight(shape=(self.D_inner, self.state_dim), initializer=tf.keras.initializers.RandomUniform(minval=-16.0, maxval=0.0), trainable=True, name='A_log')
        self.D_gate = self.add_weight(shape=(self.D_inner,), initializer=tf.keras.initializers.Ones(), trainable=True, name='D_gate')
        self.norm = LayerNormalization(epsilon=1e-5)

    def call(self, inputs, training=False):
        residual, x_norm = inputs, self.norm(inputs)
        x_and_res = self.in_proj(x_norm)
        x_conv_input, res_gate_input = tf.split(x_and_res, 2, axis=-1)
        
        x_conv_output = self.conv1d(x_conv_input)
        x_activated = tf.nn.silu(x_conv_output)
        
        x_proj_output = self.x_proj(x_activated)
        
        delta_raw, B_raw, C_raw = tf.split(x_proj_output, [self.D_inner, self.state_dim, self.state_dim], axis=-1)
        
        delta = tf.nn.softplus(delta_raw)
        A_continuous = -tf.exp(tf.cast(self._A_log, 'float32'))

        delta_T = tf.transpose(delta, perm=[1, 0, 2])
        B_raw_T = tf.transpose(B_raw, perm=[1, 0, 2])
        C_raw_T = tf.transpose(C_raw, perm=[1, 0, 2])
        x_activated_T = tf.transpose(x_activated, perm=[1, 0, 2])
        
        h_initial = tf.zeros((tf.shape(inputs)[0], self.D_inner, self.state_dim), dtype=inputs.dtype)
        # --- FIX: Create an initial y tensor to match the structure of the scan function's output ---
        y_initial = tf.zeros((tf.shape(inputs)[0], self.D_inner), dtype=inputs.dtype)

        # --- FIX: The accumulator 'prev' must be a tuple (h_prev, y_prev) to match the initializer ---
        def ssm_scan_fn(prev, elems):
            h_prev, _ = prev # Unpack previous state tuple
            dt_t, B_t, C_t, x_t = elems
            A_bar = tf.exp(tf.expand_dims(dt_t, -1) * A_continuous)
            B_bar = tf.expand_dims(dt_t, -1) * tf.expand_dims(B_t, 1)
            h_curr = A_bar * h_prev + B_bar * tf.expand_dims(x_t, -1)
            y_t = tf.einsum('bdn,bn->bd', h_curr, C_t)
            return (h_curr, y_t) # Return tuple (new_h, new_y)

        # --- FIX: Pass a tuple as the initializer and unpack the result correctly ---
        _, ys_T = tf.scan(ssm_scan_fn, (delta_T, B_raw_T, C_raw_T, x_activated_T), initializer=(h_initial, y_initial))
        
        ys = tf.transpose(ys_T, perm=[1, 0, 2])
        
        y_gated = (ys + x_activated * self.D_gate) * tf.nn.silu(res_gate_input)
        return self.out_proj(y_gated) + residual

    def get_config(self):
        config = super().get_config()
        config.update({"model_dim": self.model_dim, "state_dim": self.state_dim, "conv_width": self.conv_width, "expand_factor": self.expand_factor})
        return config


# --- 5. Build the Deeper Hybrid Model ---
print("\nBuilding the Hybrid Mamba-Transformer model architecture...")
def build_mamba_transformer_model(
    vocab_size, max_len, embedding_dim, num_heads, ff_dim, num_blocks,
    mamba_state_dim, mamba_conv_width, mamba_expand_factor,
    conv_filters, conv_kernel_size, dense_units, dropout_rate):
    
    inputs = Input(shape=(max_len,), dtype="int32", name="input_smiles")
    x = PositionalEmbedding(sequence_length=max_len, vocab_size=vocab_size, embed_dim=embedding_dim)(inputs)
    
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, padding='same', activation='gelu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    for i in range(num_blocks):
        if i % 2 == 0:
            x = MambaBlock(
                model_dim=embedding_dim,
                state_dim=mamba_state_dim,
                conv_width=mamba_conv_width,
                expand_factor=mamba_expand_factor,
                name=f'mamba_block_{i}'
            )(x)
        else:
            x = TransformerBlock(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                rate=dropout_rate,
                name=f'transformer_block_{i}'
            )(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(dense_units, activation='gelu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear', name="output_property")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mae')
    return model

# --- 6. Train Separate Models for Each Target ---
print("\n--- Preparing to train a separate model for each target property ---")
all_predictions = pd.DataFrame({'id': test_df['id']})

for i, target_col in enumerate(TARGET_COLS):
    print(f"\n--- Training model for: {target_col} (Model {i+1}/{NUM_TARGETS}) ---")
    
    target_data = combined_train_df[['SMILES', target_col]].dropna(subset=[target_col]).copy()
    
    if target_data.empty:
        print(f"WARNING: No non-NaN data found for target '{target_col}'. Skipping.")
        all_predictions[target_col] = 0.0
        continue

    smiles_to_padded_seq = {s: X_combined_pad[j] for j, s in enumerate(combined_train_df['SMILES'])}
    X_train_target = np.array([smiles_to_padded_seq[s] for s in target_data['SMILES']])
    
    y_target = target_data[target_col].values.reshape(-1, 1)
    scaler_target = StandardScaler()
    y_scaled_target = scaler_target.fit_transform(y_target)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_target, y_scaled_target, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )

    print(f"Training samples for {target_col}: {X_train.shape[0]}")
    print(f"Validation samples for {target_col}: {X_val.shape[0]}")

    model = build_mamba_transformer_model(
        vocab_size=VOCAB_SIZE, max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS, ff_dim=FF_DIM, num_blocks=NUM_BLOCKS,
        mamba_state_dim=MAMBA_STATE_DIM, mamba_conv_width=MAMBA_CONV_WIDTH,
        mamba_expand_factor=MAMBA_EXPAND_FACTOR,
        conv_filters=CONV_FILTERS, conv_kernel_size=CONV_KERNEL_SIZE,
        dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE
    )
    if i == 0:
        model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'best_model_{target_col}.keras', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    predictions_scaled_target = model.predict(X_test_pad)
    predictions_target = scaler_target.inverse_transform(predictions_scaled_target)
    all_predictions[target_col] = predictions_target.flatten()

# --- 7. Prediction and Submission ---
print("\nMaking final predictions on the test set...")
submission_df = all_predictions[['id'] + TARGET_COLS]
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())



