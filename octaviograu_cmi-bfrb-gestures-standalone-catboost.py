import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
import catboost as cb
from catboost import CatBoostClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# Import evaluation API
import kaggle_evaluation.cmi_inference_server


# Load data
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
train_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Filter to target sequences only (BFRB gestures)
train_df = train_df.loc[train_df['sequence_type'] == 'Target'].reset_index(drop=True)
print(f"Target sequences shape: {train_df.shape}")

# Analyze gesture distribution
print("\nGesture distribution:")
print(train_df['gesture'].value_counts())

# Define sensor columns
sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
all_sensor_cols = [col for col in train_df.columns if any(s in col for s in ['acc_', 'rot_', 'thm_', 'tof_'])]
print(f"\nTotal sensor columns: {len(all_sensor_cols)}")

# Fill missing values
train_df[all_sensor_cols] = train_df[all_sensor_cols].fillna(-1)

# Encode target
le = LabelEncoder()
train_df['encoded_gesture'] = le.fit_transform(train_df['gesture'])
print(f"\nEncoded gestures: {len(le.classes_)} classes")
print("Classes:", le.classes_)

def create_sequence_features(df, demographics_df=None):
    """
    Create comprehensive features from sensor sequences
    """
    features = []
    
    # Group by sequence_id to create sequence-level features
    for seq_id, group in df.groupby('sequence_id'):
        seq_features = {'sequence_id': seq_id}
        
        # Basic sequence info
        seq_features['sequence_length'] = len(group)
        seq_features['subject'] = group['subject'].iloc[0]
        
        # Add demographics if available
        if demographics_df is not None and not demographics_df.empty:
            subject_demo = demographics_df[demographics_df['subject'] == seq_features['subject']]
            if not subject_demo.empty:
                seq_features['adult_child'] = subject_demo['adult_child'].iloc[0]
                seq_features['age'] = subject_demo['age'].iloc[0]
                seq_features['sex'] = subject_demo['sex'].iloc[0]
                seq_features['handedness'] = subject_demo['handedness'].iloc[0]
                seq_features['height_cm'] = subject_demo['height_cm'].iloc[0]
                seq_features['shoulder_to_wrist_cm'] = subject_demo['shoulder_to_wrist_cm'].iloc[0]
                seq_features['elbow_to_wrist_cm'] = subject_demo['elbow_to_wrist_cm'].iloc[0]
            else:
                # Set default values if demographics not found
                seq_features['adult_child'] = -1
                seq_features['age'] = -1
                seq_features['sex'] = -1
                seq_features['handedness'] = -1
                seq_features['height_cm'] = -1
                seq_features['shoulder_to_wrist_cm'] = -1
                seq_features['elbow_to_wrist_cm'] = -1
        else:
            # Set default values if demographics not available
            seq_features['adult_child'] = -1
            seq_features['age'] = -1
            seq_features['sex'] = -1
            seq_features['handedness'] = -1
            seq_features['height_cm'] = -1
            seq_features['shoulder_to_wrist_cm'] = -1
            seq_features['elbow_to_wrist_cm'] = -1
        
        # Behavior phase encoding (if available)
        if 'behavior' in group.columns:
            behavior_counts = group['behavior'].value_counts()
            for behavior in ['Transition', 'Pause', 'Gesture']:
                seq_features[f'{behavior.lower()}_count'] = behavior_counts.get(behavior, 0)
                seq_features[f'{behavior.lower()}_ratio'] = behavior_counts.get(behavior, 0) / len(group)
        else:
            # Set default values if behavior column is not available
            for behavior in ['Transition', 'Pause', 'Gesture']:
                seq_features[f'{behavior.lower()}_count'] = 0
                seq_features[f'{behavior.lower()}_ratio'] = 0
        
        # Statistical features for each sensor type
        sensor_groups = {
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
            'thm': [col for col in all_sensor_cols if 'thm_' in col],
            'tof': [col for col in all_sensor_cols if 'tof_' in col]
        }
        
        for sensor_type, cols in sensor_groups.items():
            available_cols = [col for col in cols if col in group.columns]
            if available_cols:
                sensor_data = group[available_cols].values
                
                # Basic statistics
                seq_features[f'{sensor_type}_mean'] = np.mean(sensor_data)
                seq_features[f'{sensor_type}_std'] = np.std(sensor_data)
                seq_features[f'{sensor_type}_min'] = np.min(sensor_data)
                seq_features[f'{sensor_type}_max'] = np.max(sensor_data)
                seq_features[f'{sensor_type}_range'] = np.max(sensor_data) - np.min(sensor_data)
                seq_features[f'{sensor_type}_median'] = np.median(sensor_data)
                
                # Percentiles
                seq_features[f'{sensor_type}_q25'] = np.percentile(sensor_data, 25)
                seq_features[f'{sensor_type}_q75'] = np.percentile(sensor_data, 75)
                seq_features[f'{sensor_type}_iqr'] = np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)
                
                # Advanced statistics
                seq_features[f'{sensor_type}_skew'] = pd.Series(sensor_data.flatten()).skew()
                seq_features[f'{sensor_type}_kurtosis'] = pd.Series(sensor_data.flatten()).kurtosis()
                
                # Signal characteristics
                seq_features[f'{sensor_type}_zero_crossings'] = np.sum(np.diff(np.sign(sensor_data.flatten())) != 0)
                seq_features[f'{sensor_type}_energy'] = np.sum(sensor_data**2)
                seq_features[f'{sensor_type}_rms'] = np.sqrt(np.mean(sensor_data**2))
        
        # Specific features for IMU data (acceleration and rotation)
        if all(col in group.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            acc_data = group[['acc_x', 'acc_y', 'acc_z']].values
            # Magnitude of acceleration
            acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
            seq_features['acc_magnitude_mean'] = np.mean(acc_magnitude)
            seq_features['acc_magnitude_std'] = np.std(acc_magnitude)
            seq_features['acc_magnitude_max'] = np.max(acc_magnitude)
            
        if all(col in group.columns for col in ['rot_w', 'rot_x', 'rot_y', 'rot_z']):
            rot_data = group[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
            # Rotation magnitude
            rot_magnitude = np.sqrt(np.sum(rot_data**2, axis=1))
            seq_features['rot_magnitude_mean'] = np.mean(rot_magnitude)
            seq_features['rot_magnitude_std'] = np.std(rot_magnitude)
            seq_features['rot_magnitude_max'] = np.max(rot_magnitude)
        
        # Add target if available
        if 'encoded_gesture' in group.columns:
            seq_features['target'] = group['encoded_gesture'].iloc[0]
            seq_features['gesture'] = group['gesture'].iloc[0]
        
        features.append(seq_features)
    
    return pd.DataFrame(features)

# Create features
print("\nCreating sequence-level features...")
train_features = create_sequence_features(train_df, train_demographics_df)
print(f"Training features shape: {train_features.shape}")

# Prepare features and target
feature_cols = [col for col in train_features.columns 
                if col not in ['sequence_id', 'target', 'gesture', 'subject']]
X = train_features[feature_cols].fillna(-1)
y = train_features['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    """
    Competition metric calculation
    """
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    # Binary F1: All are Target in this filtered dataset
    y_true_binary = np.ones_like(y_true, dtype=int)
    y_pred_binary = np.ones_like(y_pred, dtype=int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)
    
    # Macro F1: specific gesture classification
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    return final_score, binary_f1, macro_f1

# Cross-validation setup
print("\nSetting up cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

all_original_gestures_in_train = train_df['gesture'].unique()

# CatBoost model with cross-validation
print("\nTraining CatBoost models with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold + 1}/5")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost model with GPU acceleration
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42 + fold,
        verbose=100,  # Print every 100 iterations
        eval_metric='MultiClass',
        early_stopping_rounds=100,
        task_type='GPU',  # Enable GPU acceleration
        devices='0'  # Use GPU device 0
    )
    
    # Train model with verbose output
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        verbose=100  # Print every 100 iterations
    )
    
    # Predict
    y_pred_fold = model.predict(X_val_fold)
    
    # Calculate score
    score, binary_f1, macro_f1 = competition_metric(
        y_val_fold, y_pred_fold, le, all_original_gestures_in_train
    )
    
    cv_scores.append(score)
    models.append(model)
    
    print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")

print(f"\nCross-validation results:")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")

# Train final model on all data with GPU acceleration
print("\nTraining final model on all training data...")
final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,  # Print every 100 iterations
    eval_metric='MultiClass',
    task_type='GPU',  # Enable GPU acceleration
    devices='0'  # Use GPU device 0
)

final_model.fit(X, y, verbose=100)

# Feature importance
print("\nTop 20 most important features:")
feature_importance = final_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20))


# Prediction function for submission
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Prediction function for Kaggle evaluation
    """
    try:
        # Convert to pandas
        sequence_pd = sequence.to_pandas()
        demographics_pd = demographics.to_pandas()
        
        # Fill missing values for all sensor columns that exist
        existing_sensor_cols = [col for col in all_sensor_cols if col in sequence_pd.columns]
        if existing_sensor_cols:
            sequence_pd[existing_sensor_cols] = sequence_pd[existing_sensor_cols].fillna(-1)
        
        # Create features for this single sequence
        seq_features = create_sequence_features(sequence_pd, demographics_pd)
        
        # Prepare feature vector - ensure all expected features are present
        X_inference = seq_features[feature_cols].fillna(-1)
        
        # Predict using ensemble of CV models
        predictions = []
        for model in models:
            pred = model.predict(X_inference)
            # Ensure we get a scalar value
            if isinstance(pred, np.ndarray):
                pred = pred[0]
            predictions.append(int(pred))
        
        # Use majority vote or most confident prediction
        predicted_label_id = max(set(predictions), key=predictions.count)
        
        # Convert back to gesture string
        predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
        
        return predicted_gesture_str
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Return a default gesture if prediction fails
        return le.classes_[0]

# Test the prediction function
print("\nTesting prediction function...")
sample_sequence = train_df[train_df['sequence_id'] == train_df['sequence_id'].iloc[0]]
sample_demographics = train_demographics_df[train_demographics_df['subject'] == sample_sequence['subject'].iloc[0]]

sample_seq_pl = pl.from_pandas(sample_sequence)
sample_demo_pl = pl.from_pandas(sample_demographics)

test_prediction = predict(sample_seq_pl, sample_demo_pl)
actual_gesture = sample_sequence['gesture'].iloc[0]
print(f"Test prediction: {test_prediction}")
print(f"Actual gesture: {actual_gesture}")
print(f"Match: {test_prediction == actual_gesture}")


# Setup inference server
print("\nSetting up inference server...")
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

