import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import joblib

# --- 1. Load and Merge Data ---
# Load training data and demographics
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
demographics_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')

# Merge dataframes
train_df = train_df.merge(demographics_df, on='subject')

# --- 2. Preprocessing and Feature Engineering ---
print("Starting Feature Engineering...")

# Define feature engineering function
def create_features(df):
    # Group by sequence and extract features
    features = []
    for seq_id, seq_data in df.groupby('sequence_id'):
        row = {'sequence_id': seq_id}

        # Demographics features
        row.update(seq_data[['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']].iloc[0].to_dict())

        # Target and other metadata
        row['gesture'] = seq_data['gesture'].iloc[0]
        row['is_full_data'] = 1 if not seq_data['thm_1'].isnull().all() else 0

        # IMU Features
        acc_cols = [f'acc_{axis}' for axis in 'xyz']
        rot_cols = [f'rot_{axis}' for axis in 'wxyz']

        for col in acc_cols + rot_cols:
            row[f'{col}_mean'] = seq_data[col].mean()
            row[f'{col}_std'] = seq_data[col].std()
            row[f'{col}_min'] = seq_data[col].min()
            row[f'{col}_max'] = seq_data[col].max()
            row[f'{col}_skew'] = seq_data[col].skew()

        # Thermopile (Thm) and Time-of-Flight (ToF) Features
        thm_cols = [f'thm_{i}' for i in range(1, 6)]
        tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

        # We will only create these features if it's a full-data sequence
        if row['is_full_data'] == 1:
            for col in thm_cols:
                row[f'{col}_mean'] = seq_data[col].mean()
                row[f'{col}_std'] = seq_data[col].std()
                row[f'{col}_min'] = seq_data[col].min()
                row[f'{col}_max'] = seq_data[col].max()
            
            # Inter-thermopile features
            for i in range(1, 5):
                row[f'thm_diff_{i}_{i+1}'] = seq_data[f'thm_{i}'].mean() - seq_data[f'thm_{i+1}'].mean()

            # ToF features (treating each sensor as a unit)
            for i in range(1, 6):
                tof_sensor_cols = [f'tof_{i}_v{j}' for j in range(64)]
                # Replace -1 with NaN for statistical calculations
                tof_data = seq_data[tof_sensor_cols].replace(-1, np.nan)
                row[f'tof_{i}_mean'] = tof_data.mean().mean()
                row[f'tof_{i}_std'] = tof_data.std().mean()
                row[f'tof_{i}_pixel_count'] = (~tof_data.isnull()).sum().mean()
        
        features.append(row)
    
    return pd.DataFrame(features)

# Create the final feature dataframe
features_df = create_features(train_df)
print("Feature Engineering Complete.")

# --- 3. Model Training ---
# Label encode the target variable 'gesture'
le = LabelEncoder()
features_df['gesture_encoded'] = le.fit_transform(features_df['gesture'])

# --- Custom Evaluation Metric ---
# This is crucial for matching the competition's scoring.
def macro_f1_score(y_true, y_pred, labels):
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Binary F1
    y_true_binary = (y_true != le.transform(['non_target'])[0]).astype(int)
    y_pred_binary = (y_pred_labels != le.transform(['non_target'])[0]).astype(int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary)

    # Macro F1
    y_pred_macro = y_pred_labels.copy()
    y_true_macro = y_true.copy()
    non_target_idx = le.transform(['non_target'])[0]
    
    y_pred_macro[y_pred_macro == non_target_idx] = -1
    y_true_macro[y_true_macro == non_target_idx] = -1
    
    macro_f1 = f1_score(y_true_macro[y_true_macro != -1], y_pred_macro[y_true_macro != -1], average='macro', zero_division=0)

    return 'macro_f1', (binary_f1 + macro_f1) / 2, True

# --- Train the Full-Data Model ---
print("\nTraining Full-Data Model...")
full_data_df = features_df[features_df['is_full_data'] == 1].drop(columns=['is_full_data', 'gesture'])
X_full = full_data_df.drop(columns=['sequence_id', 'gesture_encoded'])
y_full = full_data_df['gesture_encoded']

lgb_full = lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), random_state=42, n_estimators=500, learning_rate=0.05)
lgb_full.fit(X_full, y_full)
joblib.dump(lgb_full, 'lgb_full_data_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Full-Data Model Trained and Saved.")

# --- Train the IMU-Only Model ---
print("\nTraining IMU-Only Model...")
imu_features = [col for col in X_full.columns if 'acc_' in col or 'rot_' in col or col in demographics_df.columns]
X_imu = features_df[imu_features]
y_imu = features_df['gesture_encoded']

lgb_imu = lgb.LGBMClassifier(objective='multiclass', num_class=len(le.classes_), random_state=42, n_estimators=500, learning_rate=0.05)
lgb_imu.fit(X_imu, y_imu)
joblib.dump(lgb_imu, 'lgb_imu_only_model.pkl')
print("IMU-Only Model Trained and Saved.")


import os
import pandas as pd
import polars as pl
import joblib
import numpy as np

# --- 1. Global Variables and Model Loading ---
# We will load the models and the label encoder into global variables.
# This ensures that they are only loaded once when the notebook starts up,
# and not on every single call to the `predict` function. This is critical for performance.
lgb_full = None
lgb_imu = None
le = None

def load_models():
    """
    Loads the pre-trained LightGBM models and the label encoder.
    This function should be called once at the start of the notebook session.
    """
    global lgb_full, lgb_imu, le
    if lgb_full is None or lgb_imu is None or le is None:
        try:
            lgb_full = joblib.load('/kaggle/working/lgb_full_data_model.pkl')
            lgb_imu = joblib.load('/kaggle/working/lgb_imu_only_model.pkl')
            le = joblib.load('/kaggle/working/label_encoder.pkl')
            demographics_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv') # Use train demographics for simplicity in this example
            print("Models and Label Encoder loaded successfully.")
        except FileNotFoundError:
            print("Error: Model files not found. Please ensure 'lgb_full_data_model.pkl', 'lgb_imu_only_model.pkl', and 'label_encoder.pkl' are in the working directory.")
            # In a real submission, this would likely cause a runtime error, which is desired.
            raise

# Call the model loading function once
load_models()

# --- 2. Feature Engineering Function ---
# This function is designed to work on a single sequence at a time.
# It replicates the feature engineering from the training notebook.
def create_inference_features(sequence: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the same features as the training script for a single sequence.
    This function assumes the input is a Pandas DataFrame.
    """
    features = {}

    # Demographics features
    features.update(demographics.iloc[0][['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']].to_dict())

    # Check if it's a full-data sequence
    is_full_data = 1 if not sequence['thm_1'].isnull().all() else 0
    features['is_full_data'] = is_full_data

    # IMU Features
    acc_cols = [f'acc_{axis}' for axis in 'xyz']
    rot_cols = [f'rot_{axis}' for axis in 'wxyz']

    for col in acc_cols + rot_cols:
        features[f'{col}_mean'] = sequence[col].mean()
        features[f'{col}_std'] = sequence[col].std()
        features[f'{col}_min'] = sequence[col].min()
        features[f'{col}_max'] = sequence[col].max()
        features[f'{col}_skew'] = sequence[col].skew()

    # Thermopile (Thm) and Time-of-Flight (ToF) Features (only for full-data)
    if is_full_data == 1:
        thm_cols = [f'thm_{i}' for i in range(1, 6)]
        for col in thm_cols:
            features[f'{col}_mean'] = sequence[col].mean()
            features[f'{col}_std'] = sequence[col].std()
            features[f'{col}_min'] = sequence[col].min()
            features[f'{col}_max'] = sequence[col].max()
        
        for i in range(1, 5):
            features[f'thm_diff_{i}_{i+1}'] = sequence[f'thm_{i}'].mean() - sequence[f'thm_{i+1}'].mean()

        for i in range(1, 6):
            tof_sensor_cols = [f'tof_{i}_v{j}' for j in range(64)]
            tof_data = sequence[tof_sensor_cols].replace(-1, np.nan)
            features[f'tof_{i}_mean'] = tof_data.mean().mean()
            features[f'tof_{i}_std'] = tof_data.std().mean()
            features[f'tof_{i}_pixel_count'] = (~tof_data.isnull()).sum().mean()
    
    return pd.DataFrame([features])

# --- 3. The Competition's `predict` function ---
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    This function is called by the Kaggle inference server for each test sequence.
    It takes a Polars DataFrame and returns a single gesture string.
    """
    # Convert Polars DataFrames to Pandas for easier feature engineering
    sequence_pd = sequence.to_pandas()
    demographics_pd = demographics.to_pandas()

    # Create the feature vector for this single sequence
    features_df = create_inference_features(sequence_pd, demographics_pd)

    # Use the appropriate model based on data availability
    is_full_data = features_df['is_full_data'].iloc[0]

    if is_full_data == 1:
        # Use the full-data model
        X_full_test = features_df[lgb_full.feature_name_]
        y_pred_encoded = lgb_full.predict(X_full_test)[0]
    else:
        # Use the IMU-only model
        X_imu_test = features_df[lgb_imu.feature_name_]
        y_pred_encoded = lgb_imu.predict(X_imu_test)[0]
    
    # Decode the prediction back to the original gesture string
    predicted_gesture = le.inverse_transform([y_pred_encoded])[0]
    
    return predicted_gesture




import os

import pandas as pd
import polars as pl

import kaggle_evaluation.cmi_inference_server
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

