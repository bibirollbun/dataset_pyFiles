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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import skew, kurtosis, iqr
from sklearn.metrics import classification_report, accuracy_score

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# IMPORTANT: Define your target behaviors (BFRB-like gestures) here.
# Any behavior NOT in this list will be classified as 'non_target' for the submission file.
# Replace these examples with the actual target behaviors from your dataset.
target_behaviors = ['bfrb_type_A', 'bfrb_type_B'] # Example BFRB gesture types
# Assuming other behaviors like 'walking', 'sitting', 'eating' etc. would be non-target

# Define expected sensor columns for the Helios device
# IMPORTANT: Adjust these lists to match the exact column names in your 'your_actual_data.csv'
# If a sensor type is not present in your training data, it will be handled.
IMU_SENSOR_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
THERMOPILE_SENSOR_COLUMNS = ['therm_1', 'therm_2', 'therm_3', 'therm_4'] # Example thermopile columns
TOF_SENSOR_COLUMNS = ['tof_distance', 'tof_signal_strength'] # Example Time-of-Flight columns

ALL_EXPECTED_SENSOR_COLUMNS = IMU_SENSOR_COLUMNS + THERMOPILE_SENSOR_COLUMNS + TOF_SENSOR_COLUMNS


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')



df.columns[1:50]


data_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv' # <--- CHANGE THIS TO YOUR DATA FILE PATH

try:
    df = pd.read_csv(data_path)
    # Check for core columns needed for windowing and labeling
    core_required_columns = ['timestamp', 'behavior'] + [col for col in IMU_SENSOR_COLUMNS if col in df.columns]
    if not all(col in df.columns for col in core_required_columns):
        print(f"Warning: Missing some core IMU columns or 'timestamp'/'behavior'. Please check your CSV header.")
        print(f"Expected: {core_required_columns} + any thermopile/ToF columns")
        # Do not exit immediately, but proceed with available columns.
        # This will lead to empty features for missing sensors, which is desired for inference on partial data.

    print(f"Data loaded successfully from: {data_path}")
    print("Sample Data Head:")
    print(df.head())
    print(f"Total number of samples: {len(df)}")
    print(f"Columns found in data: {df.columns.tolist()}")
    print("\n")
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_path}'. Please check the path.")
    print("Exiting script. Please provide a valid data_path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    exit()



window_size = 50 # Number of data points in each window (e.g., 50 samples)
overlap = 25     # Number of overlapping data points (e.g., 25 samples)

features = []
labels = []
window_indices = []

# Helper function to extract statistical features for a single sensor series
def extract_sensor_features(series):
    if series.empty:
        # Return zeros for all features if series is empty (e.g., sensor not present)
        # This list must match the number of features extracted per sensor type below
        return [0.0] * 12 # 12 features per sensor (mean, std, min, max, median, var, rms, sma, ptp, iqr, skew, kurt)
    return [
        series.mean(), series.std(), series.min(), series.max(),
        series.median(), series.var(), np.sqrt(np.mean(series**2)),
        series.abs().sum(), np.ptp(series), iqr(series),
        skew(series), kurtosis(series)
    ]

# The total number of features we expect from all sensors
# (12 features * number of sensors in ALL_EXPECTED_SENSOR_COLUMNS)
TOTAL_FEATURE_DIMENSION = len(ALL_EXPECTED_SENSOR_COLUMNS) * 12

for i in range(0, len(df) - window_size, window_size - overlap):
    window = df.iloc[i : i + window_size]

    # Ensure the window has enough data points and a consistent label
    if len(window) == window_size and window['behavior'].nunique() == 1:
        current_window_features = []
        for col in ALL_EXPECTED_SENSOR_COLUMNS:
            if col in window.columns:
                current_window_features.extend(extract_sensor_features(window[col]))
            else:
                # If sensor column is not in this window/dataframe, fill with zeros
                current_window_features.extend([0.0] * 12) # 12 features per sensor

        # Ensure the feature vector has the consistent total dimension
        if len(current_window_features) == TOTAL_FEATURE_DIMENSION:
            features.append(current_window_features)
            labels.append(window['behavior'].iloc[0])
            window_indices.append(i) # Store the starting index of the window
        else:
            print(f"Warning: Feature dimension mismatch at window {i}. Expected {TOTAL_FEATURE_DIMENSION}, got {len(current_window_features)}. Skipping.")
if not features:
    print("No features extracted. Please ensure your data is large enough for the chosen window size and overlap.")
    exit()

X = np.array(features)
y = np.array(labels)
window_ids = np.array(window_indices)

print(f"Extracted {X.shape[0]} windows with {X.shape[1]} features each.")
print(f"Labels shape: {y.shape}")
print("\n")



try:
    # Ensure y is a numpy array and not empty
    if not isinstance(y, np.ndarray) or y.size == 0:
        print("Error: 'y' (labels) is not a valid numpy array or is empty when trying to encode. Check data loading and feature engineering steps.")
        exit()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} behavior classes: {label_encoder.classes_}")

    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    if len(X) < 2 or num_classes < 2:
        print("Not enough samples or unique classes to split data. Need at least 2 samples and 2 unique classes.")
        print("Consider adjusting window_size/overlap or providing more data with varied behaviors.")
        exit()

    X_train, X_test, y_train, y_test, window_ids_train, window_ids_test = train_test_split(
        X, y_categorical, window_ids, test_size=0.3, random_state=42, stratify=y_encoded
    )
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    print(f"Testing window IDs shape: {window_ids_test.shape}")
    print("\n")

except NameError:
    print("Critical Error: 'y' is not defined before starting label encoding. This indicates an issue in the preceding 'Feature Engineering' section or data loading, or an unstable environment.")
    print("Please ensure that 'your_actual_data.csv' exists, contains sufficient data, and has a 'behavior' column for labels.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during label encoding or data splitting: {e}")
    exit()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")
print("\n")




timesteps = 1
input_features = X_train_scaled.shape[1] # This will now be TOTAL_FEATURE_DIMENSION

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], timesteps, input_features)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], timesteps, input_features)

print(f"Reshaped training features shape: {X_train_reshaped.shape}")
print(f"Reshaped testing features shape: {X_test_reshaped.shape}")
print("\n")



model = Sequential()
model.add(LSTM(units=100, input_shape=(timesteps, input_features), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

print("\nModel training complete.")
print("\n")


print("Evaluating the LSTM Model on the Test Set...")
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print("\n")

y_pred_probs = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report on Test Set (LSTM):")
target_names_for_report = label_encoder.inverse_transform(np.arange(num_classes))
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names_for_report))
print("\n")



print("Making Predictions on New Data (for a single new window)...")
if len(df) >= window_size:
    # Simulating a new window, potentially with missing sensor data if you manually
    # create a sub-dataframe with only IMU for example.
    new_sensor_data_for_prediction = df.iloc[len(df) - window_size:] # Takes last window

    new_features_for_prediction = []
    if len(new_sensor_data_for_prediction) == window_size:
        current_prediction_features = []
        for col in ALL_EXPECTED_SENSOR_COLUMNS:
            if col in new_sensor_data_for_prediction.columns:
                current_prediction_features.extend(extract_sensor_features(new_sensor_data_for_prediction[col]))
            else:
                current_prediction_features.extend([0.0] * 12)

        if len(current_prediction_features) == TOTAL_FEATURE_DIMENSION:
            new_features_for_prediction.append(current_prediction_features)
        else:
             print(f"Warning: Prediction feature dimension mismatch. Expected {TOTAL_FEATURE_DIMENSION}, got {len(current_prediction_features)}. Skipping.")
    if new_features_for_prediction:
        X_new_prediction_scaled = scaler.transform(np.array(new_features_for_prediction))
        X_new_prediction_reshaped = X_new_prediction_scaled.reshape(1, timesteps, input_features)

        predicted_probs = model.predict(X_new_prediction_reshaped)
        predicted_class_index = np.argmax(predicted_probs, axis=1)
        predicted_behavior = label_encoder.inverse_transform(predicted_class_index)

        # Apply contest-specific mapping for single prediction
        if predicted_behavior[0] not in target_behaviors:
            predicted_behavior[0] = 'non_target'

        print(f"Predicted behavior for a new window of data: {predicted_behavior[0]}")
    else:
        print("Could not generate features for prediction (window size mismatch or other issue).")
else:
    print("Not enough total data samples to simulate a 'new' window for prediction.")
print("\n")


print("Generating Submission File (Contest Ready)...")

# Inverse transform predicted classes back to original behavior names
predicted_behavior_names_full = label_encoder.inverse_transform(y_pred_classes)

# Apply contest rule: collapse non-target behaviors into 'non_target' class
final_submission_gestures = []
for behavior_name in predicted_behavior_names_full:
    if behavior_name in target_behaviors:
        final_submission_gestures.append(behavior_name)
    else:
        final_submission_gestures.append('non_target')

# Create a DataFrame for the submission file
# 'Window_ID' maps to 'sequence_id' as per contest requirements
submission_df = pd.DataFrame({
    'sequence_id': window_ids_test, # Using window_ids_test as sequence_id
    'gesture': final_submission_gestures # Column name changed to 'gesture'
})

# Sort by sequence_id for a cleaner submission
submission_df = submission_df.sort_values(by='sequence_id').reset_index(drop=True)

# Define the submission file name
submission_file_name = 'submission_predictions_contest.csv' # Changed filename for clarity
submission_df.to_csv(submission_file_name, index=False)

print(f"Submission file '{submission_file_name}' created successfully.")
print("Sample of the submission file head:")
print(submission_df.head())
print("\n")

