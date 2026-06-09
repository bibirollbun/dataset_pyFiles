# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

!pip install --no-index --find-links=/kaggle/input/cmi-hexo-agent-run-1-data/wheels/wheels scikit-learn==1.7.1

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
import polars as pl
import pickle
import os
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


def load_models():
    """Load trained models and configuration"""
    with open('/kaggle/input/cmi-hexo-agent-run-1-data/binary_model.pkl', 'rb') as f:
        binary_model = pickle.load(f)
    
    with open('/kaggle/input/cmi-hexo-agent-run-1-data/multiclass_model.pkl', 'rb') as f:
        multiclass_model = pickle.load(f)
    
    with open('/kaggle/input/cmi-hexo-agent-run-1-data/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    with open('/kaggle/input/cmi-hexo-agent-run-1-data/target_gestures.pkl', 'rb') as f:
        target_gestures = pickle.load(f)
    
    return binary_model, multiclass_model, feature_cols, target_gestures


def create_sequence_features(seq_data_pl, demographics_pl):
    """Create features for a single sequence"""
    # Convert to pandas for easier processing
    seq_data = seq_data_pl.to_pandas()
    demographics = demographics_pl.to_pandas() if len(demographics_pl) > 0 else pd.DataFrame()
    
    if len(seq_data) == 0:
        return None
    
    # Basic sequence info
    features = {
        'sequence_length': len(seq_data)
    }
    
    # Get subject info
    if 'subject' in seq_data.columns:
        subject = seq_data['subject'].iloc[0]
        if len(demographics) > 0 and 'subject' in demographics.columns:
            demo_row = demographics[demographics['subject'] == subject]
            if len(demo_row) > 0:
                for col in ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 
                           'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']:
                    if col in demo_row.columns:
                        features[col] = demo_row[col].iloc[0]
    
    # Fill missing demographic features with defaults
    demo_defaults = {
        'adult_child': 1, 'age': 25, 'sex': 1, 'handedness': 1, 
        'height_cm': 170, 'shoulder_to_wrist_cm': 52, 'elbow_to_wrist_cm': 25
    }
    for key, default in demo_defaults.items():
        if key not in features:
            features[key] = default
    
    # IMU features (always available)
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    for col in imu_cols:
        if col in seq_data.columns:
            values = seq_data[col].dropna()
            if len(values) > 0:
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std()
                features[f'{col}_min'] = values.min()
                features[f'{col}_max'] = values.max()
                features[f'{col}_range'] = values.max() - values.min()
            else:
                features[f'{col}_mean'] = 0
                features[f'{col}_std'] = 0
                features[f'{col}_min'] = 0
                features[f'{col}_max'] = 0
                features[f'{col}_range'] = 0
    
    # Thermopile features (may be missing)
    thm_cols = [f'thm_{i}' for i in range(1, 6)]
    has_thm_data = False
    for col in thm_cols:
        if col in seq_data.columns:
            values = seq_data[col].dropna()
            if len(values) > 0:
                has_thm_data = True
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std()
                features[f'{col}_min'] = values.min()
                features[f'{col}_max'] = values.max()
            else:
                features[f'{col}_mean'] = np.nan
                features[f'{col}_std'] = np.nan
                features[f'{col}_min'] = np.nan
                features[f'{col}_max'] = np.nan
        else:
            features[f'{col}_mean'] = np.nan
            features[f'{col}_std'] = np.nan
            features[f'{col}_min'] = np.nan
            features[f'{col}_max'] = np.nan
    
    features['has_thm_data'] = has_thm_data
    
    # Time-of-flight features (may be missing) - sample subset
    tof_sample_cols = [f'tof_1_v{i}' for i in [0, 15, 31, 47, 63]] + [f'tof_3_v{i}' for i in [0, 31, 63]]
    has_tof_data = False
    for col in tof_sample_cols:
        if col in seq_data.columns:
            values = seq_data[col].dropna()
            values = values[values != -1]  # Remove no-response values
            if len(values) > 0:
                has_tof_data = True
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std()
            else:
                features[f'{col}_mean'] = np.nan
                features[f'{col}_std'] = np.nan
        else:
            features[f'{col}_mean'] = np.nan
            features[f'{col}_std'] = np.nan
    
    features['has_tof_data'] = has_tof_data
    
    # Phase-specific features
    for phase in ['Transition', 'Pause', 'Gesture']:
        if 'behavior' in seq_data.columns:
            phase_data = seq_data[seq_data['behavior'].str.contains(phase, na=False)]
        else:
            phase_data = pd.DataFrame()
            
        features[f'{phase.lower()}_length'] = len(phase_data)
        
        if len(phase_data) > 0:
            for col in ['acc_x', 'acc_y', 'acc_z']:
                if col in phase_data.columns:
                    values = phase_data[col].dropna()
                    if len(values) > 0:
                        features[f'{phase.lower()}_{col}_mean'] = values.mean()
                        features[f'{phase.lower()}_{col}_std'] = values.std()
                    else:
                        features[f'{phase.lower()}_{col}_mean'] = 0
                        features[f'{phase.lower()}_{col}_std'] = 0
                else:
                    features[f'{phase.lower()}_{col}_mean'] = 0
                    features[f'{phase.lower()}_{col}_std'] = 0
        else:
            for col in ['acc_x', 'acc_y', 'acc_z']:
                features[f'{phase.lower()}_{col}_mean'] = 0
                features[f'{phase.lower()}_{col}_std'] = 0
    
    return features


# Load models at module level
print("Loading trained models...")
try:
    BINARY_MODEL, MULTICLASS_MODEL, FEATURE_COLS, TARGET_GESTURES = load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    BINARY_MODEL = MULTICLASS_MODEL = FEATURE_COLS = TARGET_GESTURES = None


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predict gesture for a single sequence.
    
    Args:
        sequence: Polars DataFrame with sensor data for one sequence
        demographics: Polars DataFrame with demographics data
    
    Returns:
        str: Predicted gesture name
    """
    
    if BINARY_MODEL is None or MULTICLASS_MODEL is None:
        # Fallback prediction
        return 'Text on phone'
    
    try:
        # Create features
        features_dict = create_sequence_features(sequence, demographics)
        
        if features_dict is None:
            return 'Text on phone'
        
        # Convert to DataFrame with proper feature columns
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all required features are present
        for col in FEATURE_COLS:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Select and order features to match training
        X = features_df[FEATURE_COLS].copy()
        X = X.fillna(0)
        
        # Get predictions from both models
        binary_pred = BINARY_MODEL.predict(X)[0]
        multiclass_pred = MULTICLASS_MODEL.predict(X)[0]
        
        # If binary model says it's non-target, return a non-target gesture
        if binary_pred == 0:
            # Return the multiclass prediction if it's non_target, otherwise a common non-target gesture
            if multiclass_pred == 'non_target':
                return 'Text on phone'  # Most common non-target gesture
            else:
                return 'Text on phone'  # Default to common non-target
        else:
            # Binary model says it's target, use multiclass prediction
            if multiclass_pred in TARGET_GESTURES:
                return multiclass_pred
            else:
                # Fallback to most common target gesture
                return 'Neck - scratch'
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 'Text on phone'  # Safe fallback


! cp -r /kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation .


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







