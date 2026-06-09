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


import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import os
import threading



try:
    import kaggle_evaluation.cmi_inference_server
except ImportError:
    # Mock the class if the library is not available locally for testing
    class MockCMIInferenceServer:
        def __init__(self, predict_fn):
            self.predict_fn = predict_fn
        def serve(self):
            print("Running in mock server mode. `serve()` called.")
        def run_local_gateway(self, data_paths):
            print("Running local gateway with mock server.")
            # This mock gateway will not actually loop through sequences.
            # It's just to ensure the server setup code runs without error.
            print("Local gateway mock run complete.")

    kaggle_evaluation = type('KaggleEvaluation', (), {})()
    kaggle_evaluation.cmi_inference_server = type('CMIInferenceServer', (), {'CMIInferenceServer': MockCMIInferenceServer})()



# Define constants
NUM_IMU = 6  # 3 accel + 3 rotation
NUM_THERM = 5
NUM_TOF = 5
NUM_CLASSES = 9  # 8 BFRB-like + 1 non_target
SEQUENCE_LENGTH = 200  # Adjust based on dataset exploration
BATCH_SIZE = 32
EPOCHS = 50


# Gesture mapping (for multiclass)
GESTURE_MAP = {
    'above_ear_pull_hair': 0,
    'forehead_pull_hairline': 1,
    'forehead_scratch': 2,
    'eyebrow_pull_hair': 3,
    'eyelash_pull_hair': 4,
    'neck_pinch_skin': 5,
    'neck_scratch': 6,
    'cheek_pinch_skin': 7,
    'non_target': 8
}
NUM_CLASSES = len(GESTURE_MAP)


# --- Global Variables for Trained Artifacts ---
# These will be populated by the one-time training process.
FULL_MODEL = None
IMU_MODEL = None
SCALER_FULL = None
SCALER_IMU = None
REVERSE_GESTURE_MAP = {}
MODELS_TRAINED_FLAG = threading.Event()


# --- Main One-Time Training and Server Setup ---
def main():
    """
    This function handles the one-time model training. It's called only once.
    All helper functions for training are nested inside to prevent scope issues.
    """
    global FULL_MODEL, IMU_MODEL, SCALER_FULL, SCALER_IMU, REVERSE_GESTURE_MAP
    
    # --- NESTED HELPER FUNCTIONS FOR TRAINING ---
    def normalize_gesture(gesture):
        if not isinstance(gesture, str): return 'non_target'
        normalized = gesture.lower().replace(' - ', '_').replace(' ', '_').replace('/', '_')
        return 'non_target' if normalized not in GESTURE_MAP else normalized

    def preprocess_training_data(df, is_full_sensor=True):
        df = df.copy()
        df['gesture_normalized'] = df['gesture'].apply(normalize_gesture)
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']
        if is_full_sensor:
            sensor_cols += [f'thm_{i}' for i in range(1, 6)]
            for i in range(1, 6):
                tof_cols = [f'tof_{i}_v{j}' for j in range(64) if f'tof_{i}_v{j}' in df.columns]
                if tof_cols: df[f'tof_{i}'] = df[tof_cols].mean(axis=1)
            sensor_cols += [f'tof_{i}' for i in range(1, 6) if f'tof_{i}' in df.columns]
        
        nan_seq_ids = df[df[sensor_cols].isnull().any(axis=1)]['sequence_id'].unique()
        df = df[~df['sequence_id'].isin(nan_seq_ids)]
        
        sequences, labels, valid_seq_ids = [], [], []
        for seq_id, group in df.groupby('sequence_id'):
            X = group[sensor_cols].values
            if len(X) > SEQUENCE_LENGTH: X = X[:SEQUENCE_LENGTH]
            elif len(X) < SEQUENCE_LENGTH: X = np.pad(X, ((0, SEQUENCE_LENGTH - len(X)), (0, 0)), mode='constant')
            sequences.append(X)
            labels.append(GESTURE_MAP[group['gesture_normalized'].iloc[0]])
            valid_seq_ids.append(seq_id)
            
        if not sequences: return np.array([]), np.array([]), StandardScaler(), []
        
        X_out = np.array(sequences)
        y_multi = np.array(labels)
        scaler = StandardScaler()
        X_out = scaler.fit_transform(X_out.reshape(-1, X_out.shape[-1])).reshape(X_out.shape)
        
        return X_out, y_multi, scaler, valid_seq_ids

    def build_cnn_model(input_shape, num_classes=NUM_CLASSES):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='binary')(x)
        multi_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='multi')(x)
        return tf.keras.Model(inputs, [binary_output, multi_output])

    def train_model(X_train, y_train, X_val, y_val, input_shape):
        model = build_cnn_model(input_shape)
        model.compile(optimizer='adam',
                      loss={'binary': 'binary_crossentropy', 'multi': 'sparse_categorical_crossentropy'},
                      metrics={'binary': 'accuracy', 'multi': 'accuracy'})
        y_binary_train = (y_train != GESTURE_MAP['non_target']).astype(int)
        y_binary_val = (y_val != GESTURE_MAP['non_target']).astype(int)
        model.fit(X_train, {'binary': y_binary_train, 'multi': y_train},
                  batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(X_val, {'binary': y_binary_val, 'multi': y_val}),
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        return model

    # --- TRAINING EXECUTION ---
    try:
        df_train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
    except FileNotFoundError:
        print("Training file not found. Skipping training.")
        MODELS_TRAINED_FLAG.set()
        return

    print("--- Starting One-Time Model Training ---")
    
    # Create the reverse map from normalized name to original name
    original_gestures = df_train['gesture'].dropna().unique()
    REVERSE_GESTURE_MAP = {normalize_gesture(g): g for g in original_gestures}
    
    print("Preprocessing full-sensor data...")
    X_full, y_multi_full, scaler_full, valid_seqs = preprocess_training_data(df_train, is_full_sensor=True)
    
    df_train_filtered = df_train[df_train['sequence_id'].isin(valid_seqs)].copy()
    
    print("\nPreprocessing IMU-only data...")
    X_imu, y_multi_imu, scaler_imu, _ = preprocess_training_data(df_train_filtered, is_full_sensor=False)

    assert len(X_full) == len(X_imu) and np.array_equal(y_multi_full, y_multi_imu)

    print(f"\nSplitting {len(X_full)} sequences into training and validation sets...")
    X_full_train, X_full_val, X_imu_train, X_imu_val, y_train, y_val = train_test_split(
        X_full, X_imu, y_multi_full, test_size=0.2, random_state=42, stratify=y_multi_full)

    print("\n--- Training Full Sensor Model ---")
    FULL_MODEL = train_model(X_full_train, y_train, X_full_val, y_val, X_full_train.shape[1:])
    SCALER_FULL = scaler_full

    print("\n--- Training IMU-Only Model ---")
    IMU_MODEL = train_model(X_imu_train, y_train, X_imu_val, y_val, X_imu_train.shape[1:])
    SCALER_IMU = scaler_imu
    
    print("\n--- Models and scalers are trained and ready for inference. ---")
    MODELS_TRAINED_FLAG.set() # Signal that training is complete

# --- Prediction Function for the Kaggle API ---
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predicts the gesture for a single sequence DataFrame.
    This function will be called repeatedly by the evaluation server.
    """
    # This ensures training happens only on the first call and other calls wait.
    if not MODELS_TRAINED_FLAG.is_set():
        MODELS_TRAINED_FLAG.wait()

    df_test = sequence.to_pandas()
    df_columns = df_test.columns.tolist()
    is_full_sensor = any('thm' in col.lower() or 'tof' in col.lower() for col in df_columns)
    
    model = FULL_MODEL if is_full_sensor else IMU_MODEL
    scaler = SCALER_FULL if is_full_sensor else SCALER_IMU

    if model is None or scaler is None:
        return 'non_target'

    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']
    if is_full_sensor:
        sensor_cols += [f'thm_{i}' for i in range(1, 6)]
        for i in range(1, 6):
            tof_cols = [f'tof_{i}_v{j}' for j in range(64) if f'tof_{i}_v{j}' in df_columns]
            if tof_cols: df_test[f'tof_{i}'] = df_test[tof_cols].mean(axis=1)
        sensor_cols += [f'tof_{i}' for i in range(1, 6) if f'tof_{i}' in df_test.columns]

    if not all(col in df_test.columns for col in sensor_cols):
        return 'non_target'

    X = df_test[sensor_cols].values
    if np.any(np.isnan(X)): X = np.nan_to_num(X, nan=0.0)
    
    if len(X) > SEQUENCE_LENGTH: X = X[:SEQUENCE_LENGTH]
    elif len(X) < SEQUENCE_LENGTH: X = np.pad(X, ((0, SEQUENCE_LENGTH - len(X)), (0, 0)), mode='constant')
    
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(1, SEQUENCE_LENGTH, X.shape[-1])
    
    _, multi_pred = model.predict(X_scaled, verbose=0)
    multi_pred = np.nan_to_num(multi_pred, nan=0.0)
    gesture_idx = np.argmax(multi_pred, axis=1)[0]
    
    # Look up the normalized gesture name
    normalized_gesture = [k for k, v in GESTURE_MAP.items() if v == gesture_idx][0]
    
    # Convert back to the original gesture string for submission
    original_gesture = REVERSE_GESTURE_MAP.get(normalized_gesture, 'non_target')
    
    return original_gesture

# Start the training in a separate thread to allow the server to start immediately.
training_thread = threading.Thread(target=main)
training_thread.start()

# Set up the server with the globally-scoped predict function.
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# Start the server to listen for prediction requests.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # This local gateway is for basic testing and may not fully replicate the environment.
    print("\nStarting local gateway for basic testing...")
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

# Clean up the training thread
training_thread.join()


