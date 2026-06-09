import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

np.random.seed(42)
tf.random.set_seed(42)

# Define file paths
SEQ_TRAIN_PATH = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
LABELS_TRAIN_PATH = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'
SEQ_VALID_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv'
LABELS_VALID_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_labels.csv'
SEQ_TEST_PATH = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'

# Load CSV files
train_seq = pd.read_csv(SEQ_TRAIN_PATH)
train_lbls = pd.read_csv(LABELS_TRAIN_PATH).fillna(0)
valid_seq = pd.read_csv(SEQ_VALID_PATH)
valid_lbls = pd.read_csv(LABELS_VALID_PATH).fillna(0)
test_seq = pd.read_csv(SEQ_TEST_PATH)

# Nucleotide encoding
nucleotide_dict = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
train_seq['encoded'] = train_seq['sequence'].apply(lambda seq: [nucleotide_dict.get(ch, 0) for ch in seq])
valid_seq['encoded'] = valid_seq['sequence'].apply(lambda seq: [nucleotide_dict.get(ch, 0) for ch in seq])
test_seq['encoded'] = test_seq['sequence'].apply(lambda seq: [nucleotide_dict.get(ch, 0) for ch in seq])

# Process labels
def process_labels_data(labels_data):
    label_dict = {}
    for idx, row in labels_data.iterrows():
        target_id, resid = "_".join(row['ID'].split('_')[:-1]), int(row['ID'].split('_')[-1])
        coord = np.array([row['x_1'], row['y_1'], row['z_1']], dtype=np.float32)
        label_dict.setdefault(target_id, []).append((resid, coord))
    
    for key in label_dict:
        label_dict[key] = np.stack([c for _, c in sorted(label_dict[key])])
    return label_dict

train_labels_dict = process_labels_data(train_lbls)
valid_labels_dict = process_labels_data(valid_lbls)

# Create datasets
def create_data(sequences_data, labels_mapping):
    X, y = [], []
    for _, row in sequences_data.iterrows():
        if row['target_id'] in labels_mapping:
            X.append(row['encoded'])
            y.append(labels_mapping[row['target_id']])
    return X, y

X_train_data, y_train_data = create_data(train_seq, train_labels_dict)
X_valid_data, y_valid_data = create_data(valid_seq, valid_labels_dict)

# Padding sequences and labels
max_seq_len = max(len(seq) for seq in X_train_data)
X_train_padded = pad_sequences(X_train_data, maxlen=max_seq_len, padding='post')
X_valid_padded = pad_sequences(X_valid_data, maxlen=max_seq_len, padding='post')

def pad_labels(coord_array, max_len):
    return np.pad(coord_array, ((0, max_len - coord_array.shape[0]), (0, 0)), mode='constant')

y_train_padded = np.array([pad_labels(arr, max_seq_len) for arr in y_train_data])
y_valid_padded = np.array([pad_labels(arr, max_seq_len) for arr in y_valid_data])

# Model Architecture
vocab_size = len(nucleotide_dict) + 1
embedding_dim = 32
num_filters = 128
kernel_size = 3
drop_rate = 0.3

input_seq = Input(shape=(max_seq_len,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(input_seq)

# First Conv Block
x1 = Conv1D(num_filters, kernel_size, padding='same')(x)
x1 = BatchNormalization()(x1)
x1 = LeakyReLU()(x1)
x1 = Dropout(drop_rate)(x1)

# Second Conv Block with Residual Connection
x2 = Conv1D(num_filters, kernel_size, padding='same')(x1)
x2 = BatchNormalization()(x2)
x2 = LeakyReLU()(x2)
x2 = Dropout(drop_rate)(x2)
x2 = Add()([x1, x2])  # Residual Connection

# Third Conv Block
x3 = Conv1D(num_filters, kernel_size, padding='same')(x2)
x3 = BatchNormalization()(x3)
x3 = LeakyReLU()(x3)
x3 = Dropout(drop_rate)(x3)

# Output Layer
output_coords = Conv1D(3, kernel_size=1, padding='same', activation='linear')(x3)

model = Model(inputs=input_seq, outputs=output_coords)
model.compile(optimizer='adam', loss='mse')

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

]

# Train Model
history = model.fit(X_train_padded, y_train_padded,
                    validation_data=(X_valid_padded, y_valid_padded),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks)

# Test Data Preparation
X_test_data = test_seq['encoded'].tolist()
X_test_padded = pad_sequences(X_test_data, maxlen=max_seq_len, padding='post')

# Predictions
predictions = model.predict(X_test_padded)

# Create Submission File
submission_rows = []
for idx, row in test_seq.iterrows():
    target_id, encoded_seq = row['target_id'], row['encoded']
    pred_coords = predictions[idx][:len(encoded_seq)]

    for i, coords in enumerate(pred_coords):
        submission_rows.append({
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1,
            **{f"x_{j+1}": coords[0] for j in range(5)},
            **{f"y_{j+1}": coords[1] for j in range(5)},
            **{f"z_{j+1}": coords[2] for j in range(5)}
        })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file created successfully.")


