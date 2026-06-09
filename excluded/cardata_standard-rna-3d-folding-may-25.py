import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras for deep learning model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Define file paths (Kaggle input paths)
TRAIN_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
TRAIN_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'
VALID_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv'
VALID_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_labels.csv'
TEST_SEQ_PATH  = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
SAMPLE_SUB_PATH = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'

# Load CSV files
train_sequences = pd.read_csv(TRAIN_SEQ_PATH)
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
valid_sequences = pd.read_csv(VALID_SEQ_PATH)
valid_labels = pd.read_csv(VALID_LABELS_PATH)
test_sequences = pd.read_csv(TEST_SEQ_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)


# Fill missing values in labels with 0
train_labels.fillna(0, inplace=True)
valid_labels.fillna(0, inplace=True)


# Display basic info
print("Train Sequences Shape:", train_sequences.shape)
print("Train Labels Shape:", train_labels.shape)
print("Validation Sequences Shape:", valid_sequences.shape)
print("Validation Labels Shape:", valid_labels.shape)
print("Test Sequences Shape:", test_sequences.shape)

# Look at a few examples
print("\nTrain Sequences Head:")
print(train_sequences.head())
print("\nTrain Labels Head:")
print(train_labels.head())


# Define nucleotide mapping
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

def encode_sequence(seq):
    """Encodes an RNA sequence into a list of integers based on nucleotide_map."""
    return [nucleotide_map.get(ch, 0) for ch in seq]

# Apply encoding to all sequence files
train_sequences['encoded'] = train_sequences['sequence'].apply(encode_sequence)
valid_sequences['encoded'] = valid_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)

# Determine the maximum sequence length for padding
max_seq_length = max(train_sequences['encoded'].apply(len).max(),
                     valid_sequences['encoded'].apply(len).max(),
                     test_sequences['encoded'].apply(len).max())

# Pad sequences
X_train = pad_sequences(train_sequences['encoded'], maxlen=max_seq_length, padding='post')
X_valid = pad_sequences(valid_sequences['encoded'], maxlen=max_seq_length, padding='post')
X_test = pad_sequences(test_sequences['encoded'], maxlen=max_seq_length, padding='post')

# Prepare labels
# Extract x, y, z coordinates from train_labels
# Assuming train_labels has columns: ID, resname, resid, x_1, y_1, z_1
# Group by target_id to align with sequences
train_labels['target_id'] = train_labels['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
grouped = train_labels.groupby('target_id')
y_train = []
for target_id in train_sequences['target_id']:
    group = grouped.get_group(target_id)
    coords = group[['x_1', 'y_1', 'z_1']].values
    # Pad coordinates if necessary
    if coords.shape[0] < max_seq_length:
        padding = np.zeros((max_seq_length - coords.shape[0], 3))
        coords = np.vstack([coords, padding])
    y_train.append(coords)
y_train = np.array(y_train)


# Build the model
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(input_dim=5, output_dim=64, input_length=max_seq_length)(input_layer)
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(embedding_layer)
bn1 = BatchNormalization()(conv1)
drop1 = Dropout(0.3)(bn1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(drop1)
bn2 = BatchNormalization()(conv2)
drop2 = Dropout(0.3)(bn2)
flatten = Flatten()(drop2)
dense1 = Dense(256, activation='relu')(flatten)
output_layer = Dense(max_seq_length * 3)(dense1)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

# Reshape y_train for training
y_train_reshaped = y_train.reshape(y_train.shape[0], -1)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train_reshaped, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stopping])

# Predict on test data
predictions = model.predict(X_test)
# Reshape predictions to (num_samples, max_seq_length, 3)
predictions = predictions.reshape(predictions.shape[0], max_seq_length, 3)


# Prepare submission
submission = []
for idx, target_id in enumerate(test_sequences['target_id']):
    sequence = test_sequences.loc[test_sequences['target_id'] == target_id, 'sequence'].values[0]
    for resid, nucleotide in enumerate(sequence, start=1):
        coords = predictions[idx][resid - 1]
        row = {
            'ID': f"{target_id}_{resid}",
            'resname': nucleotide,
            'resid': resid,
            'x_1': coords[0],
            'y_1': coords[1],
            'z_1': coords[2],
            # If you have x_2 to z_5, add them here:
            # 'x_2': coords[3], 'y_2': coords[4], 'z_2': coords[5], ...
        }
        submission.append(row)

# Convert to DataFrame and save
submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)

# Confirmation
print("Submission file 'submission.csv' has been saved successfully.")


print(submission_df.head())
print(f"Total rows in submission: {len(submission_df)}")
print(f"Unique target_ids: {submission_df['ID'].apply(lambda x: x.split('_')[0]).nunique()}")


submission = []

# Ensure target_id is unique per sequence
for idx, target_id in enumerate(test_sequences['target_id'].unique()):
    # Extract the corresponding sequence
    sequence_row = test_sequences[test_sequences['target_id'] == target_id]
    
    if sequence_row.empty:
        print(f"Warning: target_id {target_id} not found in test_sequences.")
        continue
    
    sequence = sequence_row.iloc[0]['sequence']
    predicted_coords = predictions[idx]  # shape: (sequence_length, 3)

    for resid, nucleotide in enumerate(sequence, start=1):
        coords = predicted_coords[resid - 1]
        
        row = {
            'ID': f"{target_id}_{resid}",
            'resname': nucleotide,
            'resid': resid,
            'x_1': coords[0],
            'y_1': coords[1],
            'z_1': coords[2],
        }
        submission.append(row)

# Convert to DataFrame and save
submission_df = pd.DataFrame(submission)
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' has been saved successfully.")




