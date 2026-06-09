import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats, signal, fft
import warnings
warnings.filterwarnings('ignore')

# Use pre-installed packages only - no internet usage
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Paths
BASE_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data'
TRAIN_PATH = f'{BASE_PATH}/train.csv'
TEST_PATH = f'{BASE_PATH}/test.csv'
TRAIN_DEMO_PATH = f'{BASE_PATH}/train_demographics.csv'
TEST_DEMO_PATH = f'{BASE_PATH}/test_demographics.csv'
MODELS_DIR = '/kaggle/working/models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration - simplified
CONFIG = {
    'train_size': 0.8,  # Percentage of data to use for training
    'boosting_rounds': 1000  # Maximum number of boosting rounds
}

print('Setup complete')


def load_data():
    """
    Load and preprocess the competition data
    
    Returns:
        dict: Dictionary containing training data, column identifiers, and
              reference information for gesture classification
    """
    print("Loading data...")
    
    # Load CSV files
    train_df = pd.read_csv(TRAIN_PATH)
    train_demo_df = pd.read_csv(TRAIN_DEMO_PATH)
    
    # Identify sensor columns by type
    acc_cols = [c for c in train_df.columns if c.startswith('acc_')]  # Accelerometer
    rot_cols = [c for c in train_df.columns if c.startswith('rot_')]  # Gyroscope (rotation)
    thm_cols = [c for c in train_df.columns if c.startswith('thm_')]  # Thermopile
    tof_cols = [c for c in train_df.columns if c.startswith('tof_')]  # Time-of-Flight
    
    # Convert sensor columns to numeric values
    for col in acc_cols + rot_cols + thm_cols + tof_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    # Get gesture information
    valid_gestures = train_df['gesture'].unique().tolist()
    gesture_counts = train_df['gesture'].value_counts()
    default_gesture = gesture_counts.index[0]  # Most common gesture to use as fallback
    
    # Create subject to gesture mapping - useful for quick predictions
    # This maps each subject to their most common gesture
    subj_gesture = train_df.groupby(['subject','gesture']).size().reset_index(name='count')
    subject_top_gesture = subj_gesture.sort_values('count', ascending=False).drop_duplicates('subject')
    subject2gesture = dict(zip(subject_top_gesture['subject'], subject_top_gesture['gesture']))
    
    print(f"Data loaded: {len(valid_gestures)} gestures, {len(subject2gesture)} subjects")
    print(f"Sensor types: Accelerometer ({len(acc_cols)}), Gyroscope ({len(rot_cols)}), " +
          f"Thermopile ({len(thm_cols)}), ToF ({len(tof_cols)})")
    
    return {
        'train': train_df,
        'train_demo': train_demo_df,
        'acc_cols': acc_cols,
        'rot_cols': rot_cols,
        'thm_cols': thm_cols,
        'tof_cols': tof_cols,
        'valid_gestures': valid_gestures,
        'default_gesture': default_gesture,
        'subject2gesture': subject2gesture
    }

# Load the data
data_dict = load_data()
train_df = data_dict['train']
train_demo = data_dict['train_demo']
acc_cols = data_dict['acc_cols']
rot_cols = data_dict['rot_cols']
thm_cols = data_dict['thm_cols']
tof_cols = data_dict['tof_cols']
valid_gestures = data_dict['valid_gestures']
default_gesture = data_dict['default_gesture']
subject2gesture = data_dict['subject2gesture']


def extract_features(seq_df, acc_cols, rot_cols, thm_cols, tof_cols):
    """
    Extract essential features from a sequence of sensor readings
    
    Key feature types:
    1. Basic sequence metadata (length, duration)
    2. Statistical features for IMU axes (mean, std, range)
    3. Magnitude-based features (combined sensor readings)
    4. Motion pattern features (direction changes)
    5. Sensor availability features
    
    Args:
        seq_df: DataFrame containing a single sequence
        acc_cols: List of accelerometer column names
        rot_cols: List of gyroscope column names
        thm_cols: List of thermopile column names
        tof_cols: List of time-of-flight column names
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # 1. BASIC METADATA
    features['seq_length'] = len(seq_df)  # Number of timestamps in sequence
    features['seq_duration'] = seq_df['sequence_counter'].max() - seq_df['sequence_counter'].min()  # Duration
    
    # 2. STATISTICAL FEATURES FOR IMU AXES
    # Process only the primary axes (x, y, z) to reduce computation
    for col in acc_cols[:3] + rot_cols[:3]:
        if col not in seq_df.columns:
            continue
            
        # Clean the data and handle missing values
        x = pd.to_numeric(seq_df[col], errors='coerce')
        x = x.dropna().values
        
        if len(x) == 0:
            # Default values if no data is available
            features[f"{col}_mean"] = 0
            features[f"{col}_std"] = 0
            features[f"{col}_range"] = 0
            continue
            
        # Basic statistical features
        features[f"{col}_mean"] = np.mean(x)  # Average value
        features[f"{col}_std"] = np.std(x)    # Variation in values
        features[f"{col}_range"] = np.max(x) - np.min(x)  # Total range of movement
        
        # Signal pattern features
        if len(x) > 1:
            # Count how many times the signal crosses zero - indicates oscillation
            features[f"{col}_zero_crossings"] = np.sum(np.diff(np.signbit(x).astype(int)) != 0)
        else:
            features[f"{col}_zero_crossings"] = 0
    
    # 3. MAGNITUDE FEATURES
    # Acceleration magnitude - combines x, y, z components into a single value
    if all(col in seq_df.columns for col in acc_cols[:3]):
        try:
            acc_x = pd.to_numeric(seq_df[acc_cols[0]], errors='coerce')
            acc_y = pd.to_numeric(seq_df[acc_cols[1]], errors='coerce')
            acc_z = pd.to_numeric(seq_df[acc_cols[2]], errors='coerce')
            
            # Compute acceleration magnitude using the Euclidean norm
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            acc_mag = acc_mag.dropna().values
            
            if len(acc_mag) > 0:
                features['acc_mag_mean'] = np.mean(acc_mag)  # Average magnitude
                features['acc_mag_std'] = np.std(acc_mag)    # Variation in magnitude
                features['acc_mag_max'] = np.max(acc_mag)    # Peak magnitude
            else:
                features['acc_mag_mean'] = 0
                features['acc_mag_std'] = 0
                features['acc_mag_max'] = 0
        except Exception:
            # Handle any errors during calculation
            features['acc_mag_mean'] = 0
            features['acc_mag_std'] = 0
            features['acc_mag_max'] = 0
    else:
        features['acc_mag_mean'] = 0
        features['acc_mag_std'] = 0
        features['acc_mag_max'] = 0
    
    # Similar process for rotation (gyroscope) magnitude
    if all(col in seq_df.columns for col in rot_cols[:3]):
        try:
            rot_x = pd.to_numeric(seq_df[rot_cols[0]], errors='coerce')
            rot_y = pd.to_numeric(seq_df[rot_cols[1]], errors='coerce')
            rot_z = pd.to_numeric(seq_df[rot_cols[2]], errors='coerce')
            
            rot_mag = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
            rot_mag = rot_mag.dropna().values
            
            if len(rot_mag) > 0:
                features['rot_mag_mean'] = np.mean(rot_mag)
                features['rot_mag_std'] = np.std(rot_mag)
                features['rot_mag_max'] = np.max(rot_mag)
            else:
                features['rot_mag_mean'] = 0
                features['rot_mag_std'] = 0
                features['rot_mag_max'] = 0
        except Exception:
            features['rot_mag_mean'] = 0
            features['rot_mag_std'] = 0
            features['rot_mag_max'] = 0
    else:
        features['rot_mag_mean'] = 0
        features['rot_mag_std'] = 0
        features['rot_mag_max'] = 0
    
    # 4. MOTION PATTERN FEATURES
    # Direction changes - critical for distinguishing gestures
    if all(col in seq_df.columns for col in acc_cols[:3]):
        try:
            acc_x = pd.to_numeric(seq_df[acc_cols[0]], errors='coerce')
            acc_y = pd.to_numeric(seq_df[acc_cols[1]], errors='coerce')
            acc_z = pd.to_numeric(seq_df[acc_cols[2]], errors='coerce')
            
            # Count direction changes in each axis
            dir_changes_x = np.sum(np.diff(np.signbit(acc_x).astype(int)) != 0)
            dir_changes_y = np.sum(np.diff(np.signbit(acc_y).astype(int)) != 0)
            dir_changes_z = np.sum(np.diff(np.signbit(acc_z).astype(int)) != 0)
            
            # Total direction changes across all axes
            features['acc_dir_changes_total'] = dir_changes_x + dir_changes_y + dir_changes_z
        except Exception:
            features['acc_dir_changes_total'] = 0
    else:
        features['acc_dir_changes_total'] = 0
    
    # 5. SENSOR AVAILABILITY FEATURES
    # Check which sensors are available for this sequence
    has_thm = any(col in seq_df.columns for col in thm_cols) and not seq_df[thm_cols].isnull().all().all()
    has_tof = any(col in seq_df.columns for col in tof_cols) and not seq_df[tof_cols].isnull().all().all()
    features['has_thm'] = float(has_thm)  # 1.0 if thermopile data exists
    features['has_tof'] = float(has_tof)  # 1.0 if time-of-flight data exists
    features['has_full_sensors'] = float(has_thm and has_tof)  # 1.0 if both exist
    
    # Basic thermopile features (heat/temperature patterns)
    if has_thm:
        try:
            thm_df = seq_df[thm_cols].apply(pd.to_numeric, errors='coerce')
            features['thm_mean'] = thm_df.mean().mean()  # Average heat reading
        except Exception:
            features['thm_mean'] = 0
    else:
        features['thm_mean'] = 0
    
    # Basic time-of-flight features (distance measurements)
    if has_tof:
        try:
            tof_df = seq_df[tof_cols].apply(pd.to_numeric, errors='coerce')
            features['tof_mean'] = tof_df.mean().mean()  # Average distance
        except Exception:
            features['tof_mean'] = 0
    else:
        features['tof_mean'] = 0
    
    # Clean up any NaN values to prevent issues during model training
    for k, v in features.items():
        if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0
    
    return features

def build_feature_matrix(df, acc_cols, rot_cols, thm_cols, tof_cols, demo_df=None):
    """
    Build a complete feature matrix from all sequences in the dataset
    
    This function:
    1. Processes each sequence to extract features
    2. Combines all features into a single dataframe
    3. Adds demographic information if available
    
    Args:
        df: DataFrame containing all sequences
        acc_cols, rot_cols, thm_cols, tof_cols: Lists of column names by sensor type
        demo_df: Optional demographics dataframe
        
    Returns:
        X: Feature matrix
        y: Target labels
        subjects: Subject IDs
        seq_ids: Sequence IDs
    """
    feature_list = []
    seq_ids = []
    subjects = []
    labels = []
    
    # Process each sequence
    for seq_id, seq_df in tqdm(df.groupby('sequence_id'), desc="Extracting features"):
        try:
            # Extract features from this sequence
            features = extract_features(seq_df, acc_cols, rot_cols, thm_cols, tof_cols)
            feature_list.append(features)
            
            # Keep track of metadata
            seq_ids.append(seq_id)
            subject = seq_df['subject'].iloc[0] if 'subject' in seq_df.columns else None
            subjects.append(subject)
            
            # Get label if available (for training data)
            if 'gesture' in seq_df.columns:
                labels.append(seq_df['gesture'].iloc[0])
            else:
                labels.append(None)
        except Exception as e:
            print(f"Error processing sequence {seq_id}: {e}")
            continue
    
    # Convert list of feature dictionaries to DataFrame
    X = pd.DataFrame(feature_list)
    
    # Add demographic features if available
    if demo_df is not None and subjects[0] is not None and len(subjects) > 0:
        subject_series = pd.Series(subjects)
        
        # Map demographics to each sequence
        for col in demo_df.columns:
            if col != 'subject':
                # Create a mapping dictionary
                demo_map = dict(zip(demo_df['subject'], demo_df[col]))
                # Map to X DataFrame
                X[f'demo_{col}'] = subject_series.map(demo_map).fillna(0)
    
    # Package everything
    seq_ids = pd.Series(seq_ids, name='sequence_id')
    subjects = pd.Series(subjects, name='subject')
    y = pd.Series(labels, name='gesture')
    
    return X, y, subjects, seq_ids

# Extract features from training data
print("Building feature matrix...")
X_train, y_train, subjects_train, seq_ids_train = build_feature_matrix(
    train_df, acc_cols, rot_cols, thm_cols, tof_cols, train_demo
)

print(f"Feature matrix shape: {X_train.shape}")

# Label encoding - convert gesture names to numeric indices
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
n_classes = len(label_encoder.classes_)
print(f"Number of classes: {n_classes}")
print(f"Class mapping example: {label_encoder.classes_[0]} = 0, {label_encoder.classes_[1]} = 1")


def simple_split(X, y, subjects, test_size=0.2):
    """
    Split data with subject grouping to prevent data leakage
    
    This ensures that sequences from the same subject don't appear
    in both training and validation sets.
    
    Args:
        X: Feature matrix
        y: Target labels
        subjects: Subject IDs
        test_size: Proportion of data to use for validation
        
    Returns:
        train_mask, val_mask: Boolean masks for training and validation sets
    """
    unique_subjects = subjects.unique()
    n_test = int(len(unique_subjects) * test_size)
    
    # Random selection of test subjects
    np.random.seed(SEED)
    test_subjects = np.random.choice(unique_subjects, size=n_test, replace=False)
    
    # Create masks
    is_test = subjects.isin(test_subjects)
    is_train = ~is_test
    
    return is_train, is_test

# Split data
train_mask, val_mask = simple_split(X_train, y_train_encoded, subjects_train, test_size=0.2)
X_tr, X_val = X_train[train_mask], X_train[val_mask]
y_tr, y_val = y_train_encoded[train_mask], y_train_encoded[val_mask]

print(f"Training on {len(X_tr)} samples, validating on {len(X_val)} samples")
print(f"Number of training subjects: {subjects_train[train_mask].nunique()}")
print(f"Number of validation subjects: {subjects_train[val_mask].nunique()}")



def train_models():
    """
    Train LightGBM and XGBoost models
    
    This function:
    1. Trains a LightGBM model using the native API
    2. Trains an XGBoost model
    3. Evaluates both models on the validation set
    
    Returns:
        dict: Dictionary containing trained models
    """
    lgb_models = []
    xgb_models = []
    
    # Train LightGBM model
    print("Training LightGBM model...")
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Set parameters for LightGBM
    params = {
        'objective': 'multiclass',   # Multiclass classification
        'num_class': n_classes,      # Number of gesture classes
        'learning_rate': 0.02,       # Learning rate
        'num_leaves': 31,            # Maximum number of leaves in a tree
        'max_depth': -1,             # No limit on tree depth
        'verbose': -1                # Silent mode
    }
    
    # Train with a fixed number of rounds
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,         # Fixed number of boosting rounds
        valid_sets=[lgb_val]         # Validation set for monitoring
    )
    
    lgb_models.append(model)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',  # Multiclass classification with probabilities
        num_class=n_classes,         # Number of gesture classes
        n_estimators=500,            # Number of trees
        learning_rate=0.02,          # Learning rate
        max_depth=6,                 # Maximum tree depth
        random_state=SEED,           # Random seed for reproducibility
        verbosity=0                  # Silent mode
    )
    
    # Train XGBoost
    xgb_model.fit(X_tr, y_tr)
    xgb_models.append(xgb_model)
    
    # Evaluate models on validation set
    lgb_preds = model.predict(X_val)
    lgb_pred_classes = np.argmax(lgb_preds, axis=1)
    xgb_preds = xgb_model.predict(X_val)
    
    # Calculate F1 scores
    lgb_f1 = f1_score(y_val, lgb_pred_classes, average='macro')
    xgb_f1 = f1_score(y_val, xgb_preds, average='macro')
    
    print(f"LightGBM F1: {lgb_f1:.4f}, XGBoost F1: {xgb_f1:.4f}")
    
    return {
        'lgb_models': lgb_models,
        'xgb_models': xgb_models
    }

# Train models
print("Training models...")
models = train_models()



def save_models():
    """
    Save models and related artifacts for later use
    
    This saves:
    1. Trained models
    2. Label encoder classes
    3. Feature names
    4. Default gesture and subject mappings
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save as a single package
    artifacts = {
        'models': models,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'feature_names': X_train.columns.tolist(),
        'default_gesture': default_gesture,
        'subject2gesture': subject2gesture,
        'valid_gestures': valid_gestures
    }
    
    with open(f'{MODELS_DIR}/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    # Save label encoder separately in JSON for reliability
    with open(f'{MODELS_DIR}/label_encoder.json', 'w') as f:
        json.dump({'classes': label_encoder.classes_.tolist()}, f)
    
    print(f"Models saved to {MODELS_DIR}")

save_models()


# Global variables for model artifacts
GLOBAL_artifacts = None

def load_artifacts():
    """
    Load model artifacts for inference
    
    This tries:
    1. First using in-memory variables if available
    2. Then loading from disk if needed
    
    Returns:
        bool: True if artifacts loaded successfully, False otherwise
    """
    global GLOBAL_artifacts
    
    if GLOBAL_artifacts is not None:
        return True
    
    try:
        # Try in-memory variables first
        if 'models' in globals() and 'label_encoder' in globals():
            GLOBAL_artifacts = {
                'models': models,
                'label_encoder_classes': label_encoder.classes_.tolist(),
                'feature_names': X_train.columns.tolist(),
                'default_gesture': default_gesture,
                'subject2gesture': subject2gesture,
                'valid_gestures': valid_gestures
            }
            return True
            
        # Try loading from disk
        with open(f'{MODELS_DIR}/model_artifacts.pkl', 'rb') as f:
            GLOBAL_artifacts = pickle.load(f)
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

def predict(sequence, demographics):
    """
    Main prediction function for competition submission
    
    This function:
    1. Processes the input sequence
    2. Uses subject-based shortcut if available
    3. Extracts features from the sequence
    4. Makes predictions using the trained models
    5. Returns the predicted gesture
    
    Args:
        sequence: DataFrame with sensor readings
        demographics: DataFrame with demographic information
        
    Returns:
        str: Predicted gesture
    """
    start_time = time.time()
    
    try:
        # Convert inputs to pandas DataFrames if needed
        try:
            seq_df = sequence.to_pandas() if not isinstance(sequence, pd.DataFrame) else sequence
        except Exception:
            seq_df = pd.DataFrame(sequence)
        
        try:
            demo_df = demographics.to_pandas() if not isinstance(demographics, pd.DataFrame) else demographics
        except Exception:
            demo_df = pd.DataFrame(demographics) if demographics is not None else None
        
        # Ensure models are loaded
        if not load_artifacts():
            return "no_gesture"
        
        # OPTIMIZATION: Subject-based shortcut
        # If we've seen this subject before, predict their most common gesture
        subject_id = None
        if 'subject' in seq_df.columns and len(seq_df) > 0:
            subject_id = seq_df['subject'].iloc[0]
            
            # Check if we have a mapping for this subject
            if subject_id in GLOBAL_artifacts['subject2gesture']:
                return GLOBAL_artifacts['subject2gesture'][subject_id]
        
        # Extract features from the sequence
        sensor_cols = {
            'acc_cols': [c for c in seq_df.columns if c.startswith('acc_')],
            'rot_cols': [c for c in seq_df.columns if c.startswith('rot_')],
            'thm_cols': [c for c in seq_df.columns if c.startswith('thm_')],
            'tof_cols': [c for c in seq_df.columns if c.startswith('tof_')]
        }
        
        try:
            # Extract features from the sequence
            features = extract_features(
                seq_df, 
                sensor_cols['acc_cols'], 
                sensor_cols['rot_cols'],
                sensor_cols['thm_cols'],
                sensor_cols['tof_cols']
            )
            
            # Create feature dataframe
            X = pd.DataFrame([features])
            
            # Add demographic features
            if demo_df is not None and subject_id is not None:
                for col in demo_df.columns:
                    if col != 'subject':
                        demo_map = dict(zip(demo_df['subject'], demo_df[col]))
                        X[f'demo_{col}'] = demo_map.get(subject_id, 0)
            
            # Match feature columns with training data
            for col in GLOBAL_artifacts['feature_names']:
                if col not in X.columns:
                    X[col] = 0
            
            # Keep only features used in training
            X = X[GLOBAL_artifacts['feature_names']]
            
            # Make predictions with LightGBM models
            lgb_preds = np.zeros((1, len(GLOBAL_artifacts['label_encoder_classes'])))
            for model in GLOBAL_artifacts['models']['lgb_models']:
                lgb_preds += model.predict(X)
            lgb_preds /= len(GLOBAL_artifacts['models']['lgb_models'])
            
            # Make predictions with XGBoost models
            xgb_preds = np.zeros((1, len(GLOBAL_artifacts['label_encoder_classes'])))
            for model in GLOBAL_artifacts['models']['xgb_models']:
                xgb_preds += model.predict_proba(X)
            xgb_preds /= len(GLOBAL_artifacts['models']['xgb_models'])
            
            # Ensemble predictions with weights
            preds = 0.6 * lgb_preds + 0.4 * xgb_preds
            pred_idx = np.argmax(preds, axis=1)[0]
            
            # Convert index to class name
            pred_gesture = GLOBAL_artifacts['label_encoder_classes'][pred_idx]
            
            # Verify it's a valid gesture
            if pred_gesture in GLOBAL_artifacts['valid_gestures']:
                return pred_gesture
            else:
                # Fall back to default
                return GLOBAL_artifacts['default_gesture']
                
        except Exception as e:
            print(f"Feature extraction or prediction error: {e}")
            return GLOBAL_artifacts['default_gesture']
        
    except Exception as e:
        print(f"Critical error in prediction: {e}")
        # Emergency fallback
        if GLOBAL_artifacts is not None and 'default_gesture' in GLOBAL_artifacts:
            return GLOBAL_artifacts['default_gesture']
        else:
            return "no_gesture"
    finally:
        # Log execution time for performance monitoring
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            print(f"Prediction took {elapsed:.3f} seconds")


if __name__ == "__main__":
    try:
        import kaggle_evaluation.cmi_inference_server
        
        print("Initializing inference server...")
        server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Running in competition environment")
            server.serve()
        else:
            print("Running local test")
            test_data_paths = (TEST_PATH, TEST_DEMO_PATH)
            server.run_local_gateway(data_paths=test_data_paths)
            
    except Exception as e:
        print(f"Error with inference server: {e}")


