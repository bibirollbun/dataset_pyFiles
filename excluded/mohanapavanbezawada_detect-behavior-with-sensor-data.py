import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from scipy import stats
import os
import kaggle_evaluation.cmi_inference_server

# Load data
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

# Filter to Target sequences
train_df = train_df.loc[train_df['sequence_type'] == 'Target'].reset_index(drop=True)

# Define sensor columns
sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
all_sensor_cols = [col for col in train_df.columns if any(s in col for s in ['acc_', 'rot_', 'thm_', 'tof_'])]

# Improved feature engineering function
def create_features(df):
    """Create comprehensive features from sensor data"""
    features_df = df[sensor_cols].copy()
    
    # Fill missing values with forward fill then backward fill, then -1
    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(-1)
    
    # Basic statistics per sequence
    seq_features = []
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id][sensor_cols]
        
        seq_feat = {}
        
        # Basic statistics
        seq_feat.update({f'{col}_mean': seq_data[col].mean() for col in sensor_cols})
        seq_feat.update({f'{col}_std': seq_data[col].std() for col in sensor_cols})
        seq_feat.update({f'{col}_min': seq_data[col].min() for col in sensor_cols})
        seq_feat.update({f'{col}_max': seq_data[col].max() for col in sensor_cols})
        seq_feat.update({f'{col}_median': seq_data[col].median() for col in sensor_cols})
        seq_feat.update({f'{col}_q25': seq_data[col].quantile(0.25) for col in sensor_cols})
        seq_feat.update({f'{col}_q75': seq_data[col].quantile(0.75) for col in sensor_cols})
        seq_feat.update({f'{col}_skew': seq_data[col].skew() for col in sensor_cols})
        seq_feat.update({f'{col}_kurt': seq_data[col].kurtosis() for col in sensor_cols})
        
        # Magnitude features for accelerometer
        seq_data_np = seq_data[['acc_x', 'acc_y', 'acc_z']].values
        magnitude = np.sqrt(np.sum(seq_data_np**2, axis=1))
        seq_feat['acc_magnitude_mean'] = np.mean(magnitude)
        seq_feat['acc_magnitude_std'] = np.std(magnitude)
        seq_feat['acc_magnitude_max'] = np.max(magnitude)
        
        # Rotation magnitude
        rot_data = seq_data[['rot_x', 'rot_y', 'rot_z']].values
        rot_magnitude = np.sqrt(np.sum(rot_data**2, axis=1))
        seq_feat['rot_magnitude_mean'] = np.mean(rot_magnitude)
        seq_feat['rot_magnitude_std'] = np.std(rot_magnitude)
        
        # Derivatives (rate of change)
        for col in sensor_cols:
            values = seq_data[col].values
            if len(values) > 1:
                diff = np.diff(values)
                seq_feat[f'{col}_diff_mean'] = np.mean(diff)
                seq_feat[f'{col}_diff_std'] = np.std(diff)
                seq_feat[f'{col}_diff_max'] = np.max(np.abs(diff))
        
        # Zero crossing rate
        for col in sensor_cols:
            values = seq_data[col].values - seq_data[col].mean()
            # Handle NaN values before sign calculation
            values = values[~np.isnan(values)]  # Remove NaN values
            if len(values) > 1:
                zero_crossings = np.sum(np.diff(np.sign(values)) != 0)
                seq_feat[f'{col}_zero_crossings'] = zero_crossings / len(values)
            else:
                seq_feat[f'{col}_zero_crossings'] = 0
        
        # Energy features
        for col in sensor_cols:
            seq_feat[f'{col}_energy'] = np.sum(seq_data[col].values**2)
            seq_feat[f'{col}_rms'] = np.sqrt(np.mean(seq_data[col].values**2))
        
        # Sequence length
        seq_feat['sequence_length'] = len(seq_data)
        seq_feat['sequence_id'] = seq_id
        
        seq_features.append(seq_feat)
    
    return pd.DataFrame(seq_features)

# Create features for training data
print("Creating features...")
train_features = create_features(train_df)

# Add target variable
gesture_mapping = train_df[['sequence_id', 'gesture']].drop_duplicates()
train_features = train_features.merge(gesture_mapping, on='sequence_id')

# Encode labels
le = LabelEncoder()
train_features['encoded_gesture'] = le.fit_transform(train_features['gesture'])

# Prepare features for training
feature_cols = [col for col in train_features.columns if col not in ['sequence_id', 'gesture', 'encoded_gesture']]
X = train_features[feature_cols]
y = train_features['encoded_gesture']

# Handle any infinite or NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# Improved model with better hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

print("Training model...")
rf_model.fit(X_train, y_train)

# Validation
y_val_pred = rf_model.predict(X_val)

def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    """Competition metric calculation"""
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    y_true_binary = np.ones_like(y_true, dtype=int)
    y_pred_binary = np.ones_like(y_pred, dtype=int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    final_score = (binary_f1 + macro_f1) / 2
    return final_score

all_original_gestures_in_train = train_df['gesture'].unique()
validation_score = competition_metric(y_val, y_val_pred, le, all_original_gestures_in_train)
print(f"Validation Score: {validation_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for inference"""
    sequence_pd = sequence.to_pandas()
    
    # Create features for the single sequence
    seq_features = []
    seq_data = sequence_pd[sensor_cols].fillna(method='ffill').fillna(method='bfill').fillna(-1)
    
    seq_feat = {}
    
    # Apply same feature engineering as training
    for col in sensor_cols:
        seq_feat[f'{col}_mean'] = seq_data[col].mean()
        seq_feat[f'{col}_std'] = seq_data[col].std()
        seq_feat[f'{col}_min'] = seq_data[col].min()
        seq_feat[f'{col}_max'] = seq_data[col].max()
        seq_feat[f'{col}_median'] = seq_data[col].median()
        seq_feat[f'{col}_q25'] = seq_data[col].quantile(0.25)
        seq_feat[f'{col}_q75'] = seq_data[col].quantile(0.75)
        seq_feat[f'{col}_skew'] = seq_data[col].skew()
        seq_feat[f'{col}_kurt'] = seq_data[col].kurtosis()
    
    # Magnitude features
    acc_data = seq_data[['acc_x', 'acc_y', 'acc_z']].values
    magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
    seq_feat['acc_magnitude_mean'] = np.mean(magnitude)
    seq_feat['acc_magnitude_std'] = np.std(magnitude)
    seq_feat['acc_magnitude_max'] = np.max(magnitude)
    
    rot_data = seq_data[['rot_x', 'rot_y', 'rot_z']].values
    rot_magnitude = np.sqrt(np.sum(rot_data**2, axis=1))
    seq_feat['rot_magnitude_mean'] = np.mean(rot_magnitude)
    seq_feat['rot_magnitude_std'] = np.std(rot_magnitude)
    
    # Derivatives
    for col in sensor_cols:
        values = seq_data[col].values
        if len(values) > 1:
            diff = np.diff(values)
            seq_feat[f'{col}_diff_mean'] = np.mean(diff)
            seq_feat[f'{col}_diff_std'] = np.std(diff)
            seq_feat[f'{col}_diff_max'] = np.max(np.abs(diff))
        else:
            seq_feat[f'{col}_diff_mean'] = 0
            seq_feat[f'{col}_diff_std'] = 0
            seq_feat[f'{col}_diff_max'] = 0
    
    # Zero crossing rate
    for col in sensor_cols:
        values = seq_data[col].values - seq_data[col].mean()
        # Handle NaN values before sign calculation
        values = values[~np.isnan(values)]  # Remove NaN values
        if len(values) > 1:
            zero_crossings = np.sum(np.diff(np.sign(values)) != 0)
            seq_feat[f'{col}_zero_crossings'] = zero_crossings / len(values)
        else:
            seq_feat[f'{col}_zero_crossings'] = 0
    
    # Energy features
    for col in sensor_cols:
        seq_feat[f'{col}_energy'] = np.sum(seq_data[col].values**2)
        seq_feat[f'{col}_rms'] = np.sqrt(np.mean(seq_data[col].values**2))
    
    seq_feat['sequence_length'] = len(seq_data)
    
    # Convert to DataFrame and align with training features
    seq_feat_df = pd.DataFrame([seq_feat])
    
    # Ensure all features are present and in correct order
    for col in feature_cols:
        if col not in seq_feat_df.columns:
            seq_feat_df[col] = 0
    
    X_inference = seq_feat_df[feature_cols]
    X_inference = X_inference.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_inference_scaled = scaler.transform(X_inference)
    
    predicted_label_id = rf_model.predict(X_inference_scaled)[0]
    predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
    
    return predicted_gesture_str

# Inference server setup
try:
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
except ValueError as e:
    if "server has already started" in str(e):
        print("Server already running. Please restart the kernel and run again.")
        print("Go to Kernel → Restart in Jupyter notebook")
    else:
        raise e




