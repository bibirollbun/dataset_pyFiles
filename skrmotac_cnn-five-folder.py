"""
This script builds an ensemble of very small CNN models to predict 3D coordinates
for RNA nucleotide residues based on training data. It:
  - Loads training labels (with per-residue coordinates) and sequences.
  - For each target, uses the number of coordinate rows (residues) as the effective length and slices
    the full sequence accordingly.
  - Creates per-residue features: one-hot encoding of nucleotide and normalized position.
  - Pads all sequences to a global maximum length.
  - Defines a small CNN model using a Masking layer, a 1D convolution, and TimeDistributed dense layers.
  - Trains five separate CNN models (an ensemble) on the training set.
  - Loads test sequences, creates and pads features, and uses each model to predict coordinates.
  - For each residue in each test sequence, collects the five coordinate predictions and writes a submission CSV.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Conv1D, TimeDistributed, Dense
import tensorflow.keras.backend as K
import gc

# --- Helper functions ---

def one_hot_encode(seq):
    """
    Given an RNA sequence string, return a numpy array of shape (L, 4)
    with one-hot encoding for nucleotides: A, C, G, U.
    Unrecognized characters are encoded as zeros.
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1]
    }
    seq = seq.upper().strip()
    return np.array([mapping.get(nuc, [0, 0, 0, 0]) for nuc in seq], dtype=np.float32)

def prepare_train_data():
    """
    Loads training labels and sequences.
    Merges them using target_id (extracted from the ID column in train_labels.csv).
    For each target, uses the number of label rows as the effective length,
    and slices the full sequence to that length.
    For each sequence, creates features for each residue:
      - One-hot encoding (4 dims)
      - Normalized residue index (1 dim)
    Returns:
      - X_list: list of (L, 5) arrays (features for each target)
      - y_list: list of (L, 3) arrays (coordinates per residue)
      - lengths: list of effective lengths L (number of residues)
    """
    labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
    seqs = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
    
    # Create target_id from the ID column (e.g. "1SCL_A_1" -> "1SCL_A")
    labels['target_id'] = labels['ID'].apply(lambda x: "_".join(x.split('_')[:-1]))
    
    # Merge labels with sequences on target_id.
    train = pd.merge(labels, seqs[['target_id', 'sequence']], on='target_id', how='left')
    # Drop rows missing coordinates or sequence.
    train = train.dropna(subset=['x_1', 'y_1', 'z_1', 'sequence'])
    
    X_list, y_list, lengths = [], [], []
    # Group by target_id. Each group corresponds to one RNA target.
    for target_id, group in train.groupby('target_id'):
        group = group.sort_values('resid')
        # Use the number of rows as the effective length.
        L = group.shape[0]
        # Slice the sequence to the effective length.
        seq = group['sequence'].iloc[0].strip()[:L]
        lengths.append(L)
        # Create features: one-hot encoding (4 dims) and normalized position (1 dim).
        onehot = one_hot_encode(seq)             # shape (L, 4)
        norm_pos = np.array([[ (i+1)/L ] for i in range(L)], dtype=np.float32)  # shape (L, 1)
        features = np.hstack([onehot, norm_pos])   # shape (L, 5)
        X_list.append(features)
        # Get coordinate targets as (L, 3) array (from columns x_1, y_1, z_1).
        coords = group[['x_1','y_1','z_1']].to_numpy(dtype=np.float32)
        y_list.append(coords)
    return X_list, y_list, lengths

def prepare_test_data():
    """
    Loads test sequences and builds per-residue features.
    Returns:
      - X_test_list: list of (L,5) feature arrays for each test sequence
      - test_ids: list of target_id strings
      - test_sequences: dict mapping target_id to the raw sequence string
      - lengths: dict mapping target_id to sequence length
    """
    test_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
    X_test_list = []
    test_ids = []
    test_sequences = {}
    lengths = {}
    for _, row in test_df.iterrows():
        target_id = row['target_id']
        seq = row['sequence'].strip()
        L = len(seq)
        lengths[target_id] = L
        test_ids.append(target_id)
        test_sequences[target_id] = seq
        onehot = one_hot_encode(seq)  # shape (L,4)
        norm_pos = np.array([[ (i+1)/L ] for i in range(L)], dtype=np.float32)  # shape (L,1)
        features = np.hstack([onehot, norm_pos])  # shape (L,5)
        X_test_list.append(features)
    return X_test_list, test_ids, test_sequences, lengths

def pad_data(X_list, y_list, max_len):
    """
    Pads each array in X_list and y_list (which are arrays of shape (L, d))
    along the time (first) dimension to max_len.
    For X, pad with zeros; for y, pad with zeros as well.
    Returns numpy arrays of shape (n_samples, max_len, d).
    """
    X_padded = []
    y_padded = []
    for i in range(len(X_list)):
        X = X_list[i]
        y = y_list[i]
        L = X.shape[0]
        pad_width = max_len - L
        if pad_width > 0:
            X_pad = np.pad(X, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            y_pad = np.pad(y, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        else:
            X_pad = X
            y_pad = y
        X_padded.append(X_pad)
        y_padded.append(y_pad)
    return np.array(X_padded, dtype=np.float32), np.array(y_padded, dtype=np.float32)

def pad_test_data(X_list, max_len):
    """
    Pads each array in X_list (for test data) to max_len.
    Returns a numpy array of shape (n_samples, max_len, d).
    """
    X_padded = []
    for X in X_list:
        L = X.shape[0]
        pad_width = max_len - L
        if pad_width > 0:
            X_pad = np.pad(X, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
        else:
            X_pad = X
        X_padded.append(X_pad)
    return np.array(X_padded, dtype=np.float32)

def build_model(max_len, feature_dim=5):
    """
    Builds a very small CNN model.
    Input shape is (max_len, feature_dim) and output is (max_len, 3) for the coordinates.
    A Masking layer is applied to ignore padded timesteps.
    """
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(max_len, feature_dim)),
        Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(3))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Main training and prediction workflow ---

def main():
    # Prepare training data.
    X_train_list, y_train_list, train_lengths = prepare_train_data()
    # Prepare test data.
    X_test_list, test_ids, test_sequences, test_lengths_dict = prepare_test_data()
    
    # Determine global maximum sequence length from both training and test sets.
    max_train = max(train_lengths) if train_lengths else 0
    max_test = max(test_lengths_dict.values()) if test_lengths_dict else 0
    global_max_len = max(max_train, max_test)
    print("Global max sequence length:", global_max_len)
    
    # Pad training and test data to global_max_len.
    X_train, y_train = pad_data(X_train_list, y_train_list, global_max_len)
    X_test = pad_test_data(X_test_list, global_max_len)
    
    # Ensemble: train 5 separate CNN models one at a time to save resources.
    n_models = 5
    predictions_ensemble = []  # to store predictions from each model on test data
    epochs = 10  # adjust epochs as needed
    batch_size = 2
    
    for m in range(n_models):
        print(f"\nTraining CNN model {m+1}/{n_models}")
        # Build and train the model.
        model = build_model(global_max_len, feature_dim=5)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Predict on test data; result shape: (n_test, global_max_len, 3)
        pred = model.predict(X_test)
        predictions_ensemble.append(pred)
        
        # Clear model from memory to free resources.
        del model
        K.clear_session()
        gc.collect()
    
    # Build submission rows for each test sequence, using only non-padded positions.
    submission_rows = []
    n_test = len(X_test_list)
    # The order of test_ids and test_sequences corresponds to X_test_list.
    for i in range(n_test):
        target_id = test_ids[i]
        seq = test_sequences[target_id]
        L = len(seq)
        for j in range(L):
            row = {}
            row["ID"] = f"{target_id}_{j+1}"
            row["resname"] = seq[j]
            row["resid"] = j+1
            # For each model in the ensemble, record the prediction for residue j.
            for m in range(n_models):
                coord = predictions_ensemble[m][i, j]  # shape (3,)
                row[f"x_{m+1}"] = coord[0]
                row[f"y_{m+1}"] = coord[1]
                row[f"z_{m+1}"] = coord[2]
            submission_rows.append(row)
    
    submission = pd.DataFrame(submission_rows)
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("\nSubmission file saved as submission.csv")

if __name__ == '__main__':
    main()


