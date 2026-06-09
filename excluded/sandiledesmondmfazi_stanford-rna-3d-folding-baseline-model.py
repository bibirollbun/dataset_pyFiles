# Core Inputs
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM,Embedding, Dropout, concatenate
from tensorflow.keras.optimizers import Adam

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def load_data(sequence_file, label_file):
    """Loads the sequence and label data."""
    seq_df = pd.read_csv(sequence_file)
    label_df = pd.read_csv(label_file)
    return seq_df, label_df



def preprocess_sequences(seq_df, max_seq_len):
    vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq_df['encoded_sequence'] = seq_df['sequence'].apply(lambda seq: [vocab[base] for base in seq])
    seq_df['padded_sequence'] = tf.keras.preprocessing.sequence.pad_sequences(
        seq_df['encoded_sequence'], maxlen=max_seq_len, padding='post'
    ).tolist()
    return seq_df


def preprocess_labels(label_df, max_seq_len):
    """
    Processes label data by grouping by target_id, sorting by resid, 
    and padding the list of coordinates to length max_seq_len.
    """
    # Sort labels for each target by resid
    label_df = label_df.sort_values('resid')
    
    # Group labels by target_id
    grouped = label_df.groupby('target_id')
    padded_coords = {}
    
    for target_id, group in grouped:
        # Extract the coordinates for the first experimental structure (e.g., x_1, y_1, z_1)
        coords = group[['x_1', 'y_1', 'z_1']].values.tolist()
        # Pad the list of coordinates so that each target has max_seq_len residues.
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            [coords],
            maxlen=max_seq_len,
            dtype='float32',
            padding='post',
            value=0.0
        )[0]
        # Flatten the array to get a vector of length max_seq_len*3
        padded_coords[target_id] = padded.flatten()
    return padded_coords


def merge_data(seq_df, padded_coords, max_seq_len):
    """
    Merges the sequence DataFrame with corresponding padded coordinates.
    """
    seq_df['padded_coordinates'] = seq_df['target_id'].apply(
        lambda tid: padded_coords.get(tid, np.zeros(max_seq_len * 3))
    )
    return seq_df


# Define the GPU device
device_name = tf.test.gpu_device_name()


# Use the GPU device for model creation and training
with tf.device(device_name):
    def build_model(max_seq_len, max_coord_len, dropout_rate=0.2):
        """
        Builds an LSTM-based model that predicts per-residue coordinates.
        
        Args:
            max_seq_len: Maximum sequence length.
            max_coord_len: Number of coordinates per residue (typically 3).
            dropout_rate: Dropout rate.
            
        Returns:
            A compiled Keras model.
        """
        # Input layer for the sequence
        sequence_input = Input(shape=(max_seq_len,), name='sequence_input')
        
        # Embedding layer to convert nucleotide indices to dense vectors
        x = Embedding(input_dim=4, output_dim=64)(sequence_input)
        
        # LSTM layer with dropout enabled
        x = LSTM(128, dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)
        
        # Dense layer with additional dropout (active during inference for MC dropout)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x, training=True)
        
        # Output layer: Predict per-residue coordinates (flattened)
        output = Dense(max_seq_len * max_coord_len)(x)
        
        model = Model(inputs=sequence_input, outputs=output)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        return model


def predict_with_uncertainty(model, X_test, num_samples=5):
    """
    Generates multiple predictions per test sample using Monte Carlo dropout.
    
    Args:
        model: The trained Keras model.
        X_test: Test sequences (numpy array of shape (num_test_samples, max_seq_len)).
        num_samples: Number of stochastic forward passes (i.e. predicted structures) per sample.
        
    Returns:
        A numpy array of shape (num_test_samples, num_samples, max_seq_len * 3).
    """
    predictions = []
    for _ in range(num_samples):
        # Use training=True to ensure dropout is active.
        preds = model(X_test, training=True)
        predictions.append(preds.numpy())
    return np.stack(predictions, axis=1)


def train_model(model, X_train, y_train, epochs=10, batch_size=64):
    """Trains the model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)


def create_submission(predictions, test_seq_df, sample_submission_path, output_file='submission.csv'):
    """
    Creates the submission file in the required format.
    
    For each target sequence, each residue is assigned five sets of coordinates (one for each prediction).
    The resulting CSV will follow the format:
      ID,resname,resid,x_1,y_1,z_1,...,x_5,y_5,z_5

    Args:
        predictions: A numpy array with shape (num_test_samples, num_samples, max_seq_len*3).
        test_seq_df: DataFrame for test sequences containing 'target_id' and 'sequence' columns.
        sample_submission_path: Path to the sample_submission.csv (to enforce column order).
        output_file: Name of the output CSV file.
    """
    submission_rows = []
    # Determine padded length based on the flattened size (each sample: max_seq_len*3).
    num_samples = predictions.shape[1]
    padded_length = predictions.shape[2]
    max_seq_len = padded_length // 3

    # Iterate over each test sequence.
    for idx, row in test_seq_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        seq_len = len(sequence)
        # Get predictions for this test sample and reshape to (num_samples, max_seq_len, 3).
        sample_preds = predictions[idx].reshape((num_samples, max_seq_len, 3))
        
        # For each residue in the actual (unpadded) sequence:
        for resid in range(seq_len):
            row_dict = {
                'ID': f"{target_id}_{resid+1}",
                'resname': sequence[resid],
                'resid': resid+1
            }
            # Add the x, y, z for each of the predicted structures.
            for s in range(num_samples):
                coords = sample_preds[s, resid, :]
                row_dict[f'x_{s+1}'] = coords[0]
                row_dict[f'y_{s+1}'] = coords[1]
                row_dict[f'z_{s+1}'] = coords[2]
            submission_rows.append(row_dict)
            
    submission_df = pd.DataFrame(submission_rows)
    # Enforce the column order based on sample_submission.csv.
    sample_sub = pd.read_csv(sample_submission_path)
    submission_df = submission_df[sample_sub.columns]
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file '{output_file}' created successfully!")
    return submission_df


# 1. File paths
train_sequence_file = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
train_label_file = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'
test_sequence_file = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
sample_submission_path = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'


# Example usage:
# Load your sequence and label data
train_seq_df = pd.read_csv(train_sequence_file)
train_label_df = pd.read_csv(train_label_file)

# Create a common target_id in label dataframe (if needed)
train_label_df['target_id'] = train_label_df['ID'].str.extract(r'^(.*)_\d+$')[0]

train_label_df['x_1'].fillna(80.4, inplace=True)
train_label_df['y_1'].fillna(84.0, inplace=True)
train_label_df['z_1'].fillna(98.6, inplace=True)

valid_bases = {'A', 'C', 'G', 'U'}
train_seq_df = train_seq_df[train_seq_df['sequence'].apply(lambda seq: all(base in valid_bases for base in seq))]

# Determine max sequence length (e.g., maximum length among training sequences)
MAX_SEQ_LEN = int(train_seq_df['sequence'].apply(len).max())

# Preprocess sequences and labels
train_seq_df = preprocess_sequences(train_seq_df, MAX_SEQ_LEN)
padded_coords = preprocess_labels(train_label_df, MAX_SEQ_LEN)
train_seq_df = merge_data(train_seq_df, padded_coords, MAX_SEQ_LEN)

# Create training arrays: X for sequences, y for coordinates (flattened)
X = np.stack(train_seq_df['padded_sequence'].values)
y = np.stack(train_seq_df['padded_coordinates'].values)


# Detect NaNs in the dataset
print(train_seq_df.isna().sum())


# Detect NaNs in the dataset
print(train_label_df.isna().sum())


# Check if the GPU device exists
if device_name != '/device:GPU:0':
    print(f"GPU device not found: {device_name}")
else:
    print(f"Using GPU: {device_name}")


# 6. Build model
model = build_model(MAX_SEQ_LEN, 3, dropout_rate=0.3)

# 7. Train model
train_model(model, X, y)


# Load and preprocess test data
# 8. Load test data
test_seq_df = load_data(test_sequence_file, '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')[0]
vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
test_seq_df['encoded_sequence'] = test_seq_df['sequence'].apply(lambda seq: [vocab[base] for base in seq])
test_seq_df['padded_sequence'] = tf.keras.preprocessing.sequence.pad_sequences(test_seq_df['encoded_sequence'], maxlen=MAX_SEQ_LEN, padding='post').tolist()
X_test = np.stack(test_seq_df['padded_sequence'].values)

# 9. Prediction with model
predictions = predict_with_uncertainty(model, X_test, num_samples=5)


# Create submission file with sample_submission.csv
submission_df = create_submission(predictions, test_seq_df, sample_submission_path)


submission_df


# Generate predictions
predictions = predict_with_uncertainty(model, X_test, num_samples=5)

# Check predictions for NaNs
print("Predictions shape:", predictions.shape)
print("Minimum prediction value:", np.nanmin(predictions))
print("Maximum prediction value:", np.nanmax(predictions))
print("Number of NaNs in predictions:", np.isnan(predictions).sum())

# If NaNs are detected, consider retraining or adjusting hyperparameters.

