import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

warnings.filterwarnings('ignore')
print('Libraries loaded')


# Load pre-trained models from Dataset
MODEL_DIR = Path('/kaggle/input/cmi-v11-models')

# Load models
print('Loading models...')
lgb_model = lgb.Booster(model_file=str(MODEL_DIR / 'lgb_model_v11.txt'))
xgb_model = xgb.Booster()
xgb_model.load_model(str(MODEL_DIR / 'xgb_model_v11.json'))
cat_model = CatBoostClassifier()
cat_model.load_model(str(MODEL_DIR / 'cat_model_v11.cbm'))

# Load scaler and feature names
with open(MODEL_DIR / 'scaler_v11.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(MODEL_DIR / 'feature_names_v11.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load model info
with open(MODEL_DIR / 'model_info_v11.json', 'r') as f:
    model_info = json.load(f)

print(f'Models loaded successfully')
print(f'Model version: {model_info["version"]}')
print(f'Validation accuracy: {model_info["validation_accuracy"]["ensemble"]:.2%}')


# Gesture labels mapping
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
    """Extract features from a single sequence"""
    features = {}
    
    # Metadata features
    features['subject_id'] = hash(demographics_df['subject'].iloc[0]) % 1000
    features['seq_length'] = len(sequence_df)
    
    # Accelerometer features
    for axis in ['x', 'y', 'z']:
        col = f'acc_{axis}'
        if col in sequence_df.columns:
            values = sequence_df[col].values
            features[f'acc_{axis}_mean'] = np.mean(values)
            features[f'acc_{axis}_std'] = np.std(values)
            features[f'acc_{axis}_max'] = np.max(values)
            features[f'acc_{axis}_min'] = np.min(values)
            features[f'acc_{axis}_range'] = np.ptp(values)
            features[f'acc_{axis}_skew'] = pd.Series(values).skew()
            features[f'acc_{axis}_kurt'] = pd.Series(values).kurt()
            
            # FFT features
            fft_vals = np.abs(np.fft.fft(values))[:len(values)//2]
            features[f'acc_{axis}_fft_max'] = np.max(fft_vals) if len(fft_vals) > 0 else 0
            features[f'acc_{axis}_fft_mean'] = np.mean(fft_vals) if len(fft_vals) > 0 else 0
            
            # Jerk
            if len(values) > 1:
                jerk = np.diff(values)
                features[f'acc_{axis}_jerk_mean'] = np.mean(np.abs(jerk))
                features[f'acc_{axis}_jerk_std'] = np.std(jerk)
            else:
                features[f'acc_{axis}_jerk_mean'] = 0
                features[f'acc_{axis}_jerk_std'] = 0
    
    # Acceleration magnitude
    if all(f'acc_{axis}' in sequence_df.columns for axis in ['x', 'y', 'z']):
        acc_magnitude = np.sqrt(sequence_df['acc_x']**2 + sequence_df['acc_y']**2 + sequence_df['acc_z']**2)
        features['acc_magnitude_mean'] = np.mean(acc_magnitude)
        features['acc_magnitude_std'] = np.std(acc_magnitude)
        features['acc_magnitude_max'] = np.max(acc_magnitude)
    
    # Gyroscope features
    for axis in ['x', 'y', 'z']:
        col = f'gyr_{axis}'
        if col in sequence_df.columns:
            values = sequence_df[col].values
            features[f'gyr_{axis}_mean'] = np.mean(values)
            features[f'gyr_{axis}_std'] = np.std(values)
            features[f'gyr_{axis}_max'] = np.max(values)
            features[f'gyr_{axis}_energy'] = np.sum(values**2)
    
    # Angular velocity magnitude
    if all(f'gyr_{axis}' in sequence_df.columns for axis in ['x', 'y', 'z']):
        gyr_magnitude = np.sqrt(sequence_df['gyr_x']**2 + sequence_df['gyr_y']**2 + sequence_df['gyr_z']**2)
        features['gyr_magnitude_mean'] = np.mean(gyr_magnitude)
        features['gyr_magnitude_std'] = np.std(gyr_magnitude)
    
    # Quaternion features
    for comp in ['w', 'x', 'y', 'z']:
        col = f'quat_{comp}'
        if col in sequence_df.columns:
            values = sequence_df[col].values
            features[f'quat_{comp}_mean'] = np.mean(values)
            features[f'quat_{comp}_std'] = np.std(values)
            features[f'quat_{comp}_change'] = values[-1] - values[0] if len(values) > 0 else 0
    
    # ToF sensor features
    tof_features = []
    for i in range(8):
        col = f'tof_{i}'
        if col in sequence_df.columns:
            values = sequence_df[col].values
            tof_features.extend([
                np.mean(values),
                np.std(values),
                np.median(values)
            ])
    
    # ToF principal components (first 5)
    if tof_features:
        for i in range(min(5, len(tof_features))):
            features[f'tof_pc_{i}'] = tof_features[i]
    
    # Ensure all features are present
    for fname in feature_names:
        if fname not in features:
            features[fname] = 0
    
    return features


def predict(sequence, demographics):
    """CMI inference function"""
    
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
    lgb_weight = model_info['ensemble_weights']['lgb']
    xgb_weight = model_info['ensemble_weights']['xgb']
    cat_weight = model_info['ensemble_weights']['catboost']
    
    ensemble_proba = lgb_weight * lgb_proba + xgb_weight * xgb_proba + cat_weight * cat_proba
    
    # Get predicted class
    predicted_class = np.argmax(ensemble_proba[0])
    
    return gesture_labels[predicted_class]


# Initialize and run CMI Inference Server
print('=' * 50)
print('Starting CMI Inference Server with V11 Model')
print('=' * 50)
print(f'\nModel Configuration:')
print(f'  - Real data: 50%')
print(f'  - Synthetic data: 50%')
print(f'  - Validation accuracy: {model_info["validation_accuracy"]["ensemble"]:.2%}')
print(f'\nStarting server...')

server = CMIInferenceServer(predict)
server.serve()

print('\nInference complete')

