# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization,
    LSTM, Bidirectional, GlobalAveragePooling1D, Input, Concatenate,
    SeparableConv1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import polars as pl
from scipy import signal
from scipy.stats import skew, kurtosis
import json






class Config:
    RANDOM_STATE = 42
    N_FOLDS = 5
    SEQUENCE_PERCENTILE = 95
    EPOCHS = 100
    DROP_THERMAL_TOF = True
    USE_FEATURE_ENGINEERING = True
    USE_ATTENTION = True
    USE_SUBJECT_CV = True
    
    # OPTIMAL PARAMETERS (from hyperparameter search)
    OPTIMAL_PARAMS = {
        'conv1_filters': 64,
        'conv2_filters': 128,
        'conv3_filters': 256,
        'lstm1_units': 128,
        'lstm2_units': 64,
        'conv_dropout': 0.35,
        'lstm_dropout': 0.3,
        'dense_dropout': 0.4,
        'dense_size': 512,
        'learning_rate': 0.0005,
        'batch_size': 64,
        'l2_reg': 0.005,
        'attention_heads': 8
    }

# GPU Configuration
print("Setting up GPU configuration...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        print(f"✅ GPU acceleration enabled! Found {len(gpus)} GPU(s)")
        print(f"✅ Mixed precision enabled for faster training")
        print(f"GPU devices: {[gpu.name for gpu in gpus]}")
        
    except RuntimeError as e:
        print(f"⚠️  GPU setup error: {e}")
else:
    print("⚠️  No GPU found, using CPU")

tf.random.set_seed(Config.RANDOM_STATE)
np.random.seed(Config.RANDOM_STATE)

print("Loading sensor dataset...")
root = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

df = pd.read_csv(f"{root}/train.csv")
print(f"Loaded {len(df):,} rows of sensor frames")

print("Merging demographic attributes...")
demographics = pd.read_csv(f"{root}/train_demographics.csv")
df = df.merge(demographics, on="subject", how="left")


def add_engineered_features(df):
    """Add engineered features to the dataframe"""
    if not Config.USE_FEATURE_ENGINEERING:
        return df
    
    print("Engineering additional features...")
    
    # Acceleration magnitude
    if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # Gyroscope magnitude
    if all(col in df.columns for col in ['gyr_x', 'gyr_y', 'gyr_z']):
        df['gyr_magnitude'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
    
    # Jerk (acceleration derivative)
    for axis in ['x', 'y', 'z']:
        if f'acc_{axis}' in df.columns:
            df[f'jerk_{axis}'] = df.groupby('sequence_id')[f'acc_{axis}'].diff()
    
    # Rolling statistics (window of 5)
    sensor_cols = [c for c in df.columns if c.startswith(('acc_', 'gyr_', 'mag_'))]
    for col in sensor_cols:
        if col in df.columns:
            df[f'{col}_roll_mean'] = df.groupby('sequence_id')[col].rolling(5, center=True).mean().reset_index(0, drop=True)
            df[f'{col}_roll_std'] = df.groupby('sequence_id')[col].rolling(5, center=True).std().reset_index(0, drop=True)
    
    return df

# Apply feature engineering
df = add_engineered_features(df)


label_encoder = LabelEncoder()
df["gesture"] = label_encoder.fit_transform(df["gesture"].astype(str))
np.save("gesture_classes.npy", label_encoder.classes_)

print("Gesture label mapping:")
for idx, lab in enumerate(label_encoder.classes_):
    print(f"  {idx}: {lab}")

# Feature selection
excluded_cols = {
    "gesture", "sequence_type", "behavior", "orientation",
    "row_id", "subject", "phase",
    "sequence_id", "sequence_counter"
}

thermal_tof_cols = [c for c in df.columns if c.startswith(("thm_", "tof_"))]
if Config.DROP_THERMAL_TOF:
    excluded_cols.update(thermal_tof_cols)
    print(f"Excluding {len(thermal_tof_cols)} thermal/TOF channels")

feature_cols = [c for c in df.columns if c not in excluded_cols]
print(f"Using {len(feature_cols)} feature columns")



def preprocess_sequence_enhanced(df_seq: pd.DataFrame, feature_columns: list[str], 
                               scaler=None, fit_scaler=True) -> tuple:
    """Enhanced preprocessing with better scaling strategy"""
    data = df_seq[feature_columns].copy()
    
    # Handle missing values
    data = data.ffill().bfill().fillna(0.0)
    
    # Apply filtering to reduce noise
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].rolling(window=3, center=True, min_periods=1).mean()
    
    # Robust scaling
    if scaler is None:
        scaler = RobustScaler()
    
    if fit_scaler:
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    
    return scaled.astype("float32"), scaler


def create_production_model(input_shape, num_classes):
    """Create production model with optimal parameters"""
    
    params = Config.OPTIMAL_PARAMS
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        if Config.USE_ATTENTION:
            # Model with attention mechanism
            inputs = Input(shape=input_shape, dtype=tf.float16 if tf.config.list_physical_devices('GPU') else tf.float32)
            
            # CNN feature extraction
            x = Conv1D(params['conv1_filters'], 3, activation="relu", padding="same")(inputs)
            x = BatchNormalization()(x)
            x = SeparableConv1D(params['conv1_filters'], 3, activation="relu", padding="same")(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(params['conv_dropout'])(x)
            
            x = Conv1D(params['conv2_filters'], 5, activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = SeparableConv1D(params['conv2_filters'], 5, activation="relu", padding="same")(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(params['conv_dropout'])(x)
            
            x = Conv1D(params['conv3_filters'], 7, activation="relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = SeparableConv1D(params['conv3_filters'], 7, activation="relu", padding="same")(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(params['conv_dropout'])(x)
            
            # BiLSTM layers
            x = Bidirectional(LSTM(params['lstm1_units'], return_sequences=True, dropout=params['lstm_dropout']))(x)
            x = Bidirectional(LSTM(params['lstm2_units'], return_sequences=True, dropout=params['lstm_dropout']))(x)
            
            # Multi-head attention
            attention = MultiHeadAttention(num_heads=params['attention_heads'], key_dim=params['lstm2_units'])(x, x)
            attention = LayerNormalization()(attention + x)
            
            # Global pooling
            global_avg = GlobalAveragePooling1D()(attention)
            global_max = GlobalMaxPooling1D()(attention)
            concat = Concatenate()([global_avg, global_max])
            
            # Dense layers
            x = Dense(params['dense_size'], activation="relu", kernel_regularizer=l2(params['l2_reg']))(concat)
            x = BatchNormalization()(x)
            x = Dropout(params['dense_dropout'])(x)
            x = Dense(params['dense_size'] // 2, activation="relu", kernel_regularizer=l2(params['l2_reg']))(x)
            x = Dropout(params['dense_dropout'] * 0.6)(x)
            
            outputs = Dense(num_classes, activation="softmax", dtype='float32')(x)
            model = Model(inputs, outputs)
            
        else:
            # Sequential model
            model = Sequential([
                Conv1D(params['conv1_filters'], 3, activation="relu", input_shape=input_shape, padding="same"),
                BatchNormalization(),
                SeparableConv1D(params['conv1_filters'], 3, activation="relu", padding="same"),
                MaxPooling1D(2),
                Dropout(params['conv_dropout']),

                Conv1D(params['conv2_filters'], 5, activation="relu", padding="same"),
                BatchNormalization(),
                SeparableConv1D(params['conv2_filters'], 5, activation="relu", padding="same"),
                MaxPooling1D(2),
                Dropout(params['conv_dropout']),

                Conv1D(params['conv3_filters'], 7, activation="relu", padding="same"),
                BatchNormalization(),
                SeparableConv1D(params['conv3_filters'], 7, activation="relu", padding="same"),
                MaxPooling1D(2),
                Dropout(params['conv_dropout']),

                Bidirectional(LSTM(params['lstm1_units'], return_sequences=True, dropout=params['lstm_dropout'])),
                Bidirectional(LSTM(params['lstm2_units'], return_sequences=False, dropout=params['lstm_dropout'])),

                Dense(params['dense_size'], activation="relu", kernel_regularizer=l2(params['l2_reg'])),
                BatchNormalization(),
                Dropout(params['dense_dropout']),
                Dense(params['dense_size'] // 2, activation="relu", kernel_regularizer=l2(params['l2_reg'])),
                Dropout(params['dense_dropout'] * 0.6),

                Dense(num_classes, activation="softmax", dtype='float32'),
            ])
    
    # Compile with optimal parameters
    optimizer = Adam(learning_rate=params['learning_rate'])
    if tf.config.list_physical_devices('GPU') and tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def get_subject_cv_splits(X, y, subjects):
    """Get subject-wise cross-validation splits"""
    print("Setting up subject-wise cross-validation...")
    
    # Encode subjects to integers
    unique_subjects = np.unique(subjects)
    subject_encoder = LabelEncoder()
    subjects_encoded = subject_encoder.fit_transform(subjects)
    
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Subject distribution (sequences per subject): {np.bincount(subjects_encoded)}")
    
    # Use GroupKFold
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    splits = list(gkf.split(X, y, groups=subjects_encoded))
    
    # Verify no subject leakage
    for i, (train_idx, val_idx) in enumerate(splits):
        train_subjects = set(subjects[train_idx])
        val_subjects = set(subjects[val_idx])
        overlap = train_subjects & val_subjects
        if overlap:
            print(f"WARNING: Subject overlap in fold {i}: {overlap}")
        else:
            print(f"Fold {i}: Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
    
    return splits


print("Constructing dataset...")
seq_groups = df.groupby("sequence_id")

# Prepare sequences, labels, and subject information
sequences, labels, scalers = [], [], []
sequence_ids, subjects_list = [], []

for seq_id, seq in seq_groups:
    arr, scaler = preprocess_sequence_enhanced(seq, feature_cols, fit_scaler=True)
    sequences.append(arr)
    labels.append(seq["gesture"].iloc[0])
    scalers.append(scaler)
    sequence_ids.append(seq_id)
    subjects_list.append(seq["subject"].iloc[0])

# Convert to arrays
sequence_ids = np.array(sequence_ids)
subjects_array = np.array(subjects_list)

# Determine padding length
seq_lengths = [seq.shape[0] for seq in sequences]
pad_len = int(np.percentile(seq_lengths, Config.SEQUENCE_PERCENTILE))
print(f"{Config.SEQUENCE_PERCENTILE}th-percentile length = {pad_len}")
np.save("sequence_maxlen.npy", pad_len)

# Pad sequences
X = pad_sequences(sequences, maxlen=pad_len, dtype="float32", padding="post", truncating="post")
y = np.array(labels)

print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
print(f"Number of unique subjects: {len(np.unique(subjects_array))}")


print("\n" + "="*60)
print("TRAINING PRODUCTION MODEL WITH OPTIMAL PARAMETERS")
print("="*60)
print(f"Optimal parameters: {Config.OPTIMAL_PARAMS}")

splits = get_subject_cv_splits(X, y, subjects_array)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n=== FOLD {fold + 1}/{Config.N_FOLDS} ===")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Print subject distribution
    train_subjects = set(subjects_array[train_idx])
    val_subjects = set(subjects_array[val_idx])
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
    
    # Convert to categorical
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
    # Create model
    model = create_production_model((X_train.shape[1], X_train.shape[2]), num_classes)
    print(f"Model created with {model.count_params():,} parameters")
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'best_model_fold_{fold}.h5', save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        print(f"Training fold {fold+1} on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        history = model.fit(
            X_train, y_train_cat,
            epochs=Config.EPOCHS,
            batch_size=Config.OPTIMAL_PARAMS['batch_size'],
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluate fold
    val_pred = model.predict(X_val, verbose=0)
    val_pred_labels = np.argmax(val_pred, axis=1)
    
    # Calculate metric
    try:
        from cmi_2025_metric_copy_for_import import CompetitionMetric
        cls = label_encoder.classes_
        val_pred_df = pd.DataFrame({"gesture": [cls[i] for i in val_pred_labels]})
        val_true_df = pd.DataFrame({"gesture": [cls[i] for i in y_val]})
        
        metric = CompetitionMetric()
        score = metric.calculate_hierarchical_f1(val_true_df, val_pred_df)
        cv_scores.append(score)
        print(f"Fold {fold + 1} Hierarchical F1: {score:.4f}")
    except:
        accuracy = np.mean(val_pred_labels == y_val)
        cv_scores.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    
    models.append(model)

print(f"\nFinal Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Save best model
best_fold = np.argmax(cv_scores)
best_model = models[best_fold]
best_model.save("gesture_cnn_model_enhanced.h5")
print(f"Best model from fold {best_fold + 1} saved")

# Save configuration
with open('production_config.json', 'w') as f:
    json.dump({
        'optimal_parameters': Config.OPTIMAL_PARAMS,
        'cv_scores': [float(score) for score in cv_scores],
        'mean_cv_score': float(np.mean(cv_scores)),
        'std_cv_score': float(np.std(cv_scores)),
        'best_fold': int(best_fold),
        'model_parameters': int(best_model.count_params())
    }, f, indent=2)

