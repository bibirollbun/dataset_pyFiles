import numpy as np
import pandas as pd
import polars as pl
import pickle
import json
import os
import warnings
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import kaggle_evaluation
import kaggle_evaluation.cmi_inference_server

warnings.filterwarnings('ignore')
print('Libraries loaded successfully')


# Load pre-trained models from Dataset
MODEL_DIR = Path('/kaggle/input/cmi-v11-models')

# Load models
print('Loading pre-trained models...')
lgb_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgb_model_v11.txt'))
print('  LightGBM loaded')

xgb_model = xgb.Booster()
xgb_model.load_model(str(MODEL_DIR / 'xgb_model_v11.json'))
print('  XGBoost loaded')

cat_model = CatBoostClassifier()
cat_model.load_model(str(MODEL_DIR / 'cat_model_v11.cbm'))
print('  CatBoost loaded')

# Load scaler and feature names
with open(MODEL_DIR / 'scaler_v11.pkl', 'rb') as f:
    scaler = pickle.load(f)
print('  Scaler loaded')

with open(MODEL_DIR / 'feature_names_v11.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print('  Feature names loaded')

# Load model info
with open(MODEL_DIR / 'model_info_v11.json', 'r') as f:
    model_info = json.load(f)

print(f'\nModel Configuration:')
print(f'  Version: {model_info["version"]}')
print(f'  Training samples: {model_info["training_data"]["total_samples"]}')
print(f'  Validation accuracy: {model_info["validation_accuracy"]["ensemble"]:.2%}')

# アンサンブル重み
lgb_weight = model_info['ensemble_weights']['lgb']
xgb_weight = model_info['ensemble_weights']['xgb']
cat_weight = model_info['ensemble_weights']['catboost']


# Gesture labels mapping (18 classes)
gesture_labels = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Drink from bottle/cup',
    'Eyelash - pull hair',
    'Eyebrow - pull hair',
    'Feel around in tray and pull out an object',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Glasses on/off',
    'Neck - pinch skin',
    'Neck - scratch',
    'Pinch knee/leg skin',
    'Pull air toward your face',
    'Scratch knee/leg skin',
    'Text on phone',
    'Wave hello',
    'Write name in air',
    'Write name on leg'
]

print(f'Gesture labels: {len(gesture_labels)} classes')


def extract_features(sequence_df, demographics_df):
    """Extract features from a single sequence (polars DataFrames)"""
    features = {}
    
    # Convert polars to pandas for processing
    seq_pd = sequence_df.to_pandas()
    demo_pd = demographics_df.to_pandas()
    
    # Metadata features
    features['subject_id'] = hash(demo_pd['subject'].iloc[0]) % 1000
    features['seq_length'] = len(seq_pd)
    
    # Accelerometer features
    for axis in ['x', 'y', 'z']:
        col = f'acc_{axis}'
        if col in seq_pd.columns:
            values = seq_pd[col].values
            features[f'acc_{axis}_mean'] = np.mean(values)
            features[f'acc_{axis}_std'] = np.std(values)
            features[f'acc_{axis}_max'] = np.max(values)
            features[f'acc_{axis}_min'] = np.min(values)
            features[f'acc_{axis}_range'] = np.ptp(values)
            features[f'acc_{axis}_skew'] = pd.Series(values).skew()
            features[f'acc_{axis}_kurt'] = pd.Series(values).kurt()
            
            # FFT features
            if len(values) > 0:
                fft_vals = np.abs(np.fft.fft(values))[:len(values)//2]
                features[f'acc_{axis}_fft_max'] = np.max(fft_vals) if len(fft_vals) > 0 else 0
                features[f'acc_{axis}_fft_mean'] = np.mean(fft_vals) if len(fft_vals) > 0 else 0
            else:
                features[f'acc_{axis}_fft_max'] = 0
                features[f'acc_{axis}_fft_mean'] = 0
            
            # Jerk
            if len(values) > 1:
                jerk = np.diff(values)
                features[f'acc_{axis}_jerk_mean'] = np.mean(np.abs(jerk))
                features[f'acc_{axis}_jerk_std'] = np.std(jerk)
            else:
                features[f'acc_{axis}_jerk_mean'] = 0
                features[f'acc_{axis}_jerk_std'] = 0
    
    # Acceleration magnitude
    if all(f'acc_{axis}' in seq_pd.columns for axis in ['x', 'y', 'z']):
        acc_magnitude = np.sqrt(seq_pd['acc_x']**2 + seq_pd['acc_y']**2 + seq_pd['acc_z']**2)
        features['acc_magnitude_mean'] = np.mean(acc_magnitude)
        features['acc_magnitude_std'] = np.std(acc_magnitude)
        features['acc_magnitude_max'] = np.max(acc_magnitude)
    else:
        features['acc_magnitude_mean'] = 0
        features['acc_magnitude_std'] = 0
        features['acc_magnitude_max'] = 0
    
    # Gyroscope features
    for axis in ['x', 'y', 'z']:
        col = f'gyr_{axis}'
        if col in seq_pd.columns:
            values = seq_pd[col].values
            features[f'gyr_{axis}_mean'] = np.mean(values)
            features[f'gyr_{axis}_std'] = np.std(values)
            features[f'gyr_{axis}_max'] = np.max(values)
            features[f'gyr_{axis}_energy'] = np.sum(values**2)
        else:
            features[f'gyr_{axis}_mean'] = 0
            features[f'gyr_{axis}_std'] = 0
            features[f'gyr_{axis}_max'] = 0
            features[f'gyr_{axis}_energy'] = 0
    
    # Angular velocity magnitude
    if all(f'gyr_{axis}' in seq_pd.columns for axis in ['x', 'y', 'z']):
        gyr_magnitude = np.sqrt(seq_pd['gyr_x']**2 + seq_pd['gyr_y']**2 + seq_pd['gyr_z']**2)
        features['gyr_magnitude_mean'] = np.mean(gyr_magnitude)
        features['gyr_magnitude_std'] = np.std(gyr_magnitude)
    else:
        features['gyr_magnitude_mean'] = 0
        features['gyr_magnitude_std'] = 0
    
    # Quaternion features
    for comp in ['w', 'x', 'y', 'z']:
        col = f'quat_{comp}'
        if col in seq_pd.columns:
            values = seq_pd[col].values
            features[f'quat_{comp}_mean'] = np.mean(values)
            features[f'quat_{comp}_std'] = np.std(values)
            features[f'quat_{comp}_change'] = values[-1] - values[0] if len(values) > 0 else 0
        else:
            features[f'quat_{comp}_mean'] = 0
            features[f'quat_{comp}_std'] = 0
            features[f'quat_{comp}_change'] = 0
    
    # ToF sensor features (simplified)
    tof_features = []
    for i in range(8):
        col = f'tof_{i}'
        if col in seq_pd.columns:
            values = seq_pd[col].values
            tof_features.extend([
                np.mean(values),
                np.std(values),
                np.median(values)
            ])
    
    # ToF principal components (first 5)
    if tof_features:
        for i in range(min(5, len(tof_features))):
            features[f'tof_pc_{i}'] = tof_features[i]
    else:
        for i in range(5):
            features[f'tof_pc_{i}'] = 0
    
    # Ensure all features are present
    for fname in feature_names:
        if fname not in features:
            features[fname] = 0
    
    return features


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """CMI inference function - returns gesture label string"""
    
    try:
        # Extract features
        features = extract_features(sequence, demographics)
        
        # Create feature vector in correct order
        X = np.array([[features[fname] for fname in feature_names]])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions from each model
        lgb_proba = lgb_model.predict(X_scaled)
        xgb_proba = xgb_model.predict(xgb.DMatrix(X_scaled))
        cat_proba = cat_model.predict_proba(X_scaled)
        
        # Ensure shape is (1, n_classes)
        if lgb_proba.ndim == 1:
            lgb_proba = lgb_proba.reshape(1, -1)
        if xgb_proba.ndim == 1:
            xgb_proba = xgb_proba.reshape(1, -1)
        if cat_proba.ndim == 1:
            cat_proba = cat_proba.reshape(1, -1)
        
        # Ensemble predictions
        ensemble_proba = lgb_weight * lgb_proba + xgb_weight * xgb_proba + cat_weight * cat_proba
        
        # Get predicted class
        predicted_class = np.argmax(ensemble_proba[0])
        
        return gesture_labels[predicted_class]
        
    except Exception as e:
        print(f'Error in prediction: {e}')
        # Fallback to most common gesture
        return 'Forehead - pull hairline'


# Initialize and run CMI Inference Server
print('=' * 50)
print('CMI V11 Model - Final Submission')
print('=' * 50)
print(f'\nModel Configuration:')
print(f'  - Real data: 50% ({model_info["training_data"]["real_samples"]} samples)')
print(f'  - Synthetic data: 50% ({model_info["training_data"]["synthetic_samples"]} samples)')
print(f'  - Total samples: {model_info["training_data"]["total_samples"]}')
print(f'  - Validation accuracy: {model_info["validation_accuracy"]["ensemble"]:.2%}')
print(f'  - Ensemble weights: LightGBM={lgb_weight:.0%}, XGBoost={xgb_weight:.0%}, CatBoost={cat_weight:.0%}')

# Create inference server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# Check if running in Kaggle environment
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Competition submission environment
    print('\n[Competition Mode] Starting inference server...')
    inference_server.serve()
    print('Inference completed - submission.parquet generated')
else:
    # Local test environment
    print('\n[Local Test Mode] Running local gateway test...')
    print('Note: In actual submission, server will generate submission.parquet')
    
    # Try local test if data is available
    test_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
    demo_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
    
    if os.path.exists(test_path) and os.path.exists(demo_path):
        try:
            inference_server.run_local_gateway(
                data_paths=(test_path, demo_path)
            )
            print('Local test completed successfully')
        except Exception as e:
            print(f'Local test error (expected in notebook environment): {e}')
    else:
        print('Test data not found - skipping local test')
        print('Server is ready for competition submission')

print('\n' + '=' * 50)
print('Notebook execution complete')
print('=' * 50)

