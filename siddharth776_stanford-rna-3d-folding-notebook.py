# 1. Imports, Mixed Precision, and XLA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, Dropout, Add,
                                     Activation, LayerNormalization)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Enable XLA for additional performance (if supported)
tf.config.optimizer.set_jit(True)
# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 2. Data Loading and Exploration
TRAIN_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
TRAIN_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'
VALID_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv'
VALID_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_labels.csv'
TEST_SEQ_PATH  = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
SAMPLE_SUB_PATH = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'

train_sequences = pd.read_csv(TRAIN_SEQ_PATH)
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
valid_sequences = pd.read_csv(VALID_SEQ_PATH)
valid_labels = pd.read_csv(VALID_LABELS_PATH)
test_sequences = pd.read_csv(TEST_SEQ_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

# Fill missing label values with 0
train_labels.fillna(0, inplace=True)
valid_labels.fillna(0, inplace=True)

print("Train Sequences Shape:", train_sequences.shape)
print("Train Labels Shape:", train_labels.shape)
print("Validation Sequences Shape:", valid_sequences.shape)
print("Validation Labels Shape:", valid_labels.shape)
print("Test Sequences Shape:", test_sequences.shape)

# 3. Data Preprocessing

## 3.1 Sequence Encoding
# Map nucleotides: A:1, C:2, G:3, U:4; unknown -> 0
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
def encode_sequence(seq):
    return [nucleotide_map.get(ch, 0) for ch in seq]

train_sequences['encoded'] = train_sequences['sequence'].apply(encode_sequence)
valid_sequences['encoded'] = valid_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)

## 3.2 Processing Label Data
def process_labels(labels_df):
    label_dict = {}
    for idx, row in labels_df.iterrows():
        parts = row['ID'].split('_')
        target_id = "_".join(parts[:-1])
        resid = int(parts[-1])
        coord = np.array([row['x_1'], row['y_1'], row['z_1']], dtype=np.float32)
        if target_id not in label_dict:
            label_dict[target_id] = []
        label_dict[target_id].append((resid, coord))
    for key in label_dict:
        sorted_coords = sorted(label_dict[key], key=lambda x: x[0])
        coords = np.stack([c for r, c in sorted_coords])
        label_dict[key] = coords
    return label_dict

train_labels_dict = process_labels(train_labels)
valid_labels_dict = process_labels(valid_labels)

## 3.3 Creating Datasets and Padding
def create_dataset(sequences_df, labels_dict):
    X, y, target_ids = [], [], []
    for idx, row in sequences_df.iterrows():
        tid = row['target_id']
        if tid in labels_dict:
            X.append(row['encoded'])
            y.append(labels_dict[tid])
            target_ids.append(tid)
    return X, y, target_ids

X_train, y_train, train_ids = create_dataset(train_sequences, train_labels_dict)
X_valid, y_valid, valid_ids = create_dataset(valid_sequences, valid_labels_dict)

# Determine maximum sequence length from training set (this is fixed)
max_len = max(len(seq) for seq in X_train)
print("Maximum sequence length (train):", max_len)

X_train_pad = pad_sequences(X_train, maxlen=max_len, padding='post', value=0)
X_valid_pad = pad_sequences(X_valid, maxlen=max_len, padding='post', value=0)

def pad_coordinates(coord_array, max_len):
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))
        return np.pad(coord_array, pad_width, mode='constant', constant_values=0)
    else:
        return coord_array

y_train_pad = np.array([pad_coordinates(arr, max_len) for arr in y_train])
y_valid_pad = np.array([pad_coordinates(arr, max_len) for arr in y_valid])

print("Any NaN in y_train_pad?", np.isnan(y_train_pad).any())
print("X_train_pad shape:", X_train_pad.shape)
print("y_train_pad shape:", y_train_pad.shape)

# Create and cache tf.data datasets
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_pad, y_train_pad)) \
                              .cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_valid_pad, y_valid_pad)) \
                            .cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 4. Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_len, d_model)
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config
    
    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        # Cast the positional encoding to the same dtype as the inputs
        pos_encoding = tf.cast(self.pos_encoding, inputs.dtype)
        return inputs + pos_encoding[:, :seq_len, :]
    
    def compute_output_shape(self, input_shape):
        return input_shape

# 5. Custom Windowed Self-Attention Layer (Optimized)
class WindowedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, key_dim, **kwargs):
        super(WindowedSelfAttention, self).__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    def call(self, inputs):
        # Use static shape information (max_len is fixed)
        input_shape = inputs.shape  # (batch, max_len, channels)
        seq_len_static = input_shape[1]
        channels_static = input_shape[-1]
        seq_len = seq_len_static  # fixed = max_len
        remainder = seq_len % self.window_size
        if remainder != 0:
            pad_len = self.window_size - remainder
            padded_inputs = tf.pad(inputs, [[0,0], [0, pad_len], [0,0]])
            new_seq_len = seq_len + pad_len
        else:
            padded_inputs = inputs
            new_seq_len = seq_len
        num_windows = new_seq_len // self.window_size
        channels = channels_static if channels_static is not None else tf.shape(inputs)[-1]
        windows = tf.reshape(padded_inputs, (-1, num_windows, self.window_size, channels))
        batch_size = tf.shape(windows)[0]
        windows_reshaped = tf.reshape(windows, (batch_size * num_windows, self.window_size, channels))
        attn_output = self.mha(windows_reshaped, windows_reshaped)
        attn_windows = tf.reshape(attn_output, (batch_size, num_windows, self.window_size, channels))
        output = tf.reshape(attn_windows, (batch_size, new_seq_len, channels))
        output = output[:, :seq_len, :]
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

# 6. Advanced Model Building with Positional Encoding & Layer Normalization
vocab_size = max(nucleotide_map.values()) + 1  # +1 for padding token
embedding_dim = 32  # d_model for positional encoding
num_filters = 128
kernel_size = 3
drop_rate = 0.3

def residual_block(x, filters, kernel_size, dropout_rate, block_name, dilation_rate=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu',
               dilation_rate=dilation_rate, name=f'{block_name}_conv1')(x)
    x = LayerNormalization(name=f'{block_name}_ln1')(x)
    x = Dropout(dropout_rate, name=f'{block_name}_drop1')(x)
    x = Conv1D(filters, kernel_size, padding='same', activation='linear',
               dilation_rate=dilation_rate, name=f'{block_name}_conv2')(x)
    x = LayerNormalization(name=f'{block_name}_ln2')(x)
    x = Add(name=f'{block_name}_add')([shortcut, x])
    x = Activation('relu', name=f'{block_name}_out')(x)
    return x

input_seq = Input(shape=(max_len,), name='input_seq')
# Embedding + Positional Encoding
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='embedding')(input_seq)
x = PositionalEncoding(max_len=max_len, d_model=embedding_dim, name='pos_encoding')(x)
# Projection to higher dimension
x = Conv1D(num_filters, 1, padding='same', activation='relu', name='proj_conv')(x)
# Residual blocks
x = residual_block(x, num_filters, kernel_size, drop_rate, block_name='resblock1')
x = residual_block(x, num_filters, kernel_size, drop_rate, block_name='resblock2', dilation_rate=2)
# Windowed Self-Attention
window_size = 256
x = WindowedSelfAttention(window_size=window_size, num_heads=4, key_dim=32, name='window_attn')(x)
x = residual_block(x, num_filters, kernel_size, drop_rate, block_name='resblock3')
# Additional convolution for further feature extraction
x = Conv1D(num_filters, kernel_size, padding='same', activation='relu', name='conv_final')(x)
x = LayerNormalization(name='ln_final')(x)
x = Dropout(drop_rate, name='drop_final')(x)
# Output: Predict 3 coordinates per residue (x, y, z)
output_coords = Conv1D(3, 1, padding='same', activation='linear', name='predicted_coords')(x)

# Cosine decay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=1000)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = Model(inputs=input_seq, outputs=output_coords)
model.compile(optimizer=optimizer, loss='mse')
model.summary()

# 7. Model Training with Optimized Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=100,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Optimized Model Training vs. Validation Loss")
plt.legend()
plt.show()

# 8. Generating Predictions with Monte Carlo Dropout
X_test = test_sequences['encoded'].tolist()
X_test_pad = pad_sequences(X_test, maxlen=max_len, padding='post', value=0)

num_MC = 5
mc_predictions = []
for i in range(num_MC):
    preds = model(X_test_pad, training=True).numpy()
    mc_predictions.append(preds)

# 9. Building the Submission File
submission_rows = []
for idx, row in test_sequences.iterrows():
    target_id = row['target_id']
    seq_encoded = row['encoded']
    seq_length = len(seq_encoded)
    drop_preds = [mc_predictions[m][idx, :seq_length, :] for m in range(num_MC)]
    for i in range(seq_length):
        coords = [drop_preds[m][i] for m in range(num_MC)]
        row_dict = {
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1
        }
        for j in range(num_MC):
            row_dict[f"x_{j+1}"] = coords[j][0]
            row_dict[f"y_{j+1}"] = coords[j][1]
            row_dict[f"z_{j+1}"] = coords[j][2]
        submission_rows.append(row_dict)

submission_df = pd.DataFrame(submission_rows)
print("Submission DataFrame shape:", submission_df.shape)
print(submission_df.head(10))

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


