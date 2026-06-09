import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import polars as pl
import kaggle_evaluation.cmi_inference_server

# Set global seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Imports loaded")


# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
print(f"Loaded {len(df)} rows.")


label_encoder = LabelEncoder()
df['gesture'] = label_encoder.fit_transform(df['gesture'].astype(str))

# Save class names for inference
np.save('gesture_classes.npy', label_encoder.classes_)

# Print class label mapping
print("Gesture label mapping:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"  {idx}: {label}")


print("Checking for IMU-only sequences...")

def check_for_imu_only_seqs():
    # Identify thermopile and TOF columns
    thermal_tof_cols = [col for col in df.columns if col.startswith('thm_') or col.startswith('tof_')]
    
    # Group by sequence and check if all thm_/tof_ values are null
    imu_only_flags = df[thermal_tof_cols].isna().groupby(df['sequence_id']).all().all(axis=1)
    
    # Report statistics
    total_sequences = df['sequence_id'].nunique()
    imu_only_count = imu_only_flags.sum()
    imu_only_pct = (imu_only_count / total_sequences) * 100
    
    print(f"Total sequences: {total_sequences}")
    print(f"IMU-only sequences (all thm_/tof_ null): {imu_only_count} ({imu_only_pct:.1f}%)")

check_for_imu_only_seqs()


excluded_cols = {
    'gesture', 'sequence_type', 'behavior', 'orientation',  # train-only
    'row_id', 'subject', 'phase',  # metadata
    'sequence_id', 'sequence_counter'  # identifiers
}

# Setting this true makes model ignore thermal and tof data
drop_thermal_and_tof = True

if drop_thermal_and_tof:
    thermal_tof_cols = [col for col in df.columns if col.startswith('thm_') or col.startswith('tof_')]
    excluded_cols.update(thermal_tof_cols)
    print(f"Ignoring {len(thermal_tof_cols)} thermopile / time-of-flight columns.")

# Select numeric feature columns
feature_cols = [col for col in df.columns if col not in excluded_cols]
print(f"Using {len(feature_cols)} numeric feature columns for training:")
print(feature_cols)


# Check for NaNs in selected feature columns
nan_counts = df[feature_cols].isna().sum()
total_nans = nan_counts.sum()
print(f"\nTotal missing values in feature columns: {total_nans}")
if total_nans > 0:
    print("Columns with missing values:")
    print(nan_counts[nan_counts > 0])
else:
    print("No missing values found in feature columns.")


def preprocess_sequence(df_sequence: pd.DataFrame, feature_cols: list) -> np.ndarray:
    data = df_sequence[feature_cols].copy()
    data = data.ffill().bfill().fillna(0)
    scaled = StandardScaler().fit_transform(data)
    return scaled


# Build sequences
sequence_ids = df['sequence_id'].unique()
sequences = df.groupby('sequence_id')

X = []
seq_lengths = []

print("Building sequences...")
for i, (seq_id, seq) in enumerate(sequences):
    if i % 500 == 0:
        print(f"Processing sequence {i}...")
    processed = preprocess_sequence(seq, feature_cols)
    X.append(processed)
    seq_lengths.append(processed.shape[0])

max_len_perentile = 90

# Report sequence length stats
minlen = min(seq_lengths)
avglen = int(np.mean(seq_lengths))
pad_len_to_use = int(np.percentile(seq_lengths, max_len_perentile))  
print(f"Sequence length stats - Min: {minlen}, Avg: {avglen}, {max_len_perentile}th percentile: {pad_len_to_use}")
print(f"Padding / truncating all sequences to fixed length {pad_len_to_use}...")

np.save("sequence_maxlen.npy", pad_len_to_use)  # Save for inference

# Pad/truncate to fixed length
X = pad_sequences(X, maxlen=pad_len_to_use, dtype='float32', padding='post', truncating='post')


# Use groupby to get the first gesture per sequence (already integer-encoded)
y = df.groupby('sequence_id')['gesture'].first().values

print("Integer labels:", y[:4])

# Convert to one-hot vectors
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

print("After one-hot encoding:", y[:4])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


%%time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

# Build 1D CNN model
model = Sequential([
    # Block 1
    Conv1D(filters=256, kernel_size=7, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Block 2
    Conv1D(filters=384, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Block 3
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    # Flatten and Dense
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),

    # Output
    Dense(num_classes, activation='softmax')
])

# Compile model with categorical crossentropy loss (for one-hot labels)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define early stopping to prevent overfitting and restore best model
early_stopping = EarlyStopping(
    monitor='val_accuracy',      # Watch validation loss
    patience=15,              # Stop if no improvement after 5 epochs
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-8,
    verbose=1
)

# Train model using explicitly split validation set (80/20 held out)
print("Training model...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

model.save("gesture_cnn_model.h5")
print("Training complete.")


from cmi_2025_metric_copy_for_import import CompetitionMetric

# Get predicted labels for the validation set
print("Predicting on validation set...")
y_val_pred_probs = model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)
y_val_true = np.argmax(y_val, axis=1)

# Map integer labels back to gesture strings
gesture_classes = np.load("gesture_classes.npy", allow_pickle=True)
val_pred_labels = pd.Series(y_val_pred).map(lambda i: gesture_classes[i])
val_true_labels = pd.Series(y_val_true).map(lambda i: gesture_classes[i])

# Build DataFrames for the metric
val_submission = pd.DataFrame({'gesture': val_pred_labels})
val_solution = pd.DataFrame({'gesture': val_true_labels})

# Run competition metric
metric = CompetitionMetric()
score = metric.calculate_hierarchical_f1(val_solution, val_submission)
print(f"Estimated leaderboard (val) score: {score:.4f}")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    processed = preprocess_sequence(df_seq, feature_cols)
    maxlen = int(np.load("sequence_maxlen.npy"))  # ensure consistent shape
    padded = pad_sequences([processed], maxlen=maxlen, dtype='float32', padding='post', truncating='post')
    model = load_model("gesture_cnn_model.h5")
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction, axis=1)[0]
    gesture_classes = np.load("gesture_classes.npy", allow_pickle=True)
    return gesture_classes[predicted_index]

# Launch inference server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


# Manual test (only runs outside Kaggle gateway)
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("\nRunning manual test...")
    test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
    sample_seq_id = test_df['sequence_id'].unique()[0]
    test_seq = test_df[test_df['sequence_id'] == sample_seq_id]
    prediction = predict(pl.DataFrame(test_seq), None)
    print(f"Manual prediction result for sequence_id {sample_seq_id}: {prediction}")

