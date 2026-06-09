# CMI BFRB Detection - LightGBM with World Acceleration Feature
# A simple LightGBM implementation for BFRB classification with world coordinate transformation
# 
# Key Innovation: Converting device acceleration to world coordinates using quaternion rotations
# This helps normalize hand orientation differences across subjects and positions
# Thanks https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/discussion/583080 @tatamikenn for your idea!

import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML utilities
from sklearn.model_selection import StratifiedGroupKFold, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import xgboost as xgb

# World coordinate transformation
from scipy.spatial.transform import Rotation as R

# Competition specific
import kaggle_evaluation.cmi_inference_server

print("âœ“ All imports loaded successfully")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config: 
    """Central configuration class for training and data parameters"""
    
    # Paths for Kaggle environment
    TRAIN_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    TEST_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
    TEST_DEMOGRAPHICS_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
    
    # Training parameters
    SEED = 42
    N_FOLDS = 5
    
    # Feature columns
    ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
    ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'n_estimators': 1024,
        'max_depth': 8,
        'learning_rate': 0.025,
        'colsample_bytree': 0.5,
        'n_jobs': -1,
        'num_leaves': 20,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.5,
        'verbosity': -1,
        # 'device': 'gpu',  # Will be set automatically based on availability
    }

    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': SEED,
        'n_jobs': -1
    }

    XGB_PARAMS = {
        'objective': 'multi:softmax',
        'eval_metric': ['mlogloss', 'merror'],
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': SEED,
        'n_jobs': -1
    }

# Set reproducibility
np.random.seed(Config.SEED)


def check_gpu_availability():
    """Check if GPU is available for LightGBM"""
    try:
        # Try to create a simple LightGBM model with GPU
        from lightgbm import LGBMClassifier
        import numpy as np
        
        # Create dummy data
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        
        # Try GPU
        model = LGBMClassifier(n_estimators=1, device='gpu', verbosity=-1)
        model.fit(X_dummy, y_dummy)
        print("âœ“ GPU available for LightGBM")
        return 'gpu'
    except:
        print("âš ï¸�  GPU not available, using CPU for LightGBM")
        return 'cpu'

# Check GPU availability
DEVICE = check_gpu_availability()

print(f"âœ“ Configuration loaded for Kaggle environment (Device: {DEVICE})")


# Gesture mapping (targets 0-7 are BFRB, 8-17 are non-BFRB)
GESTURE_MAPPER = {
    "Above ear - pull hair": 0,
    "Cheek - pinch skin": 1,
    "Eyebrow - pull hair": 2,
    "Eyelash - pull hair": 3, 
    "Forehead - pull hairline": 4,
    "Forehead - scratch": 5,
    "Neck - pinch skin": 6, 
    "Neck - scratch": 7,
    
    "Drink from bottle/cup": 8,
    "Feel around in tray and pull out an object": 9,
    "Glasses on/off": 10,
    "Pinch knee/leg skin": 11, 
    "Pull air toward your face": 12,
    "Scratch knee/leg skin": 13,
    "Text on phone": 14,
    "Wave hello": 15,
    "Write name in air": 16,
    "Write name on leg": 17,
}

REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}


def competition_metric(y_true, y_pred) -> tuple:
    """Calculate the competition metric (Binary F1 + Macro F1) / 2"""
    
    # Binary F1: BFRB vs non-BFRB
    binary_f1 = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: specific gesture classification (only for BFRB gestures)
    macro_f1 = f1_score(
        np.where(y_true <= 7, y_true, 99),  # Map non-BFRB to 99
        np.where(y_pred <= 7, y_pred, 99),  # Map non-BFRB to 99
        average="macro", 
        zero_division=0.0,
    )
    
    # Final competition score
    final_score = 0.5 * (binary_f1 + macro_f1)
    
    return final_score, binary_f1, macro_f1


def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    """
    Handle missing values in quaternion data intelligently
    
    Key insight: Quaternions must have unit length |q| = 1
    If one component is missing, we can reconstruct it from the others
    """
    rot_cleaned = rot_data.copy()
    
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        
        if missing_count == 0:
            # No missing values, normalize to unit quaternion
            norm = np.linalg.norm(row)
            if norm > 1e-8:
                rot_cleaned[i] = row / norm
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
                
        elif missing_count == 1:
            # One missing value, reconstruct using unit quaternion constraint
            # |w|Â² + |x|Â² + |y|Â² + |z|Â² = 1
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0, 1.0 - sum_squares))
                # Choose sign for continuity with previous quaternion
                if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                    if rot_cleaned[i-1, missing_idx] < 0:
                        missing_value = -missing_value
                rot_cleaned[i, missing_idx] = missing_value
                rot_cleaned[i, ~np.isnan(row)] = valid_values
            else:
                rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
        else:
            # More than one missing value, use identity quaternion
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
    
    return rot_cleaned


def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Convert acceleration from device coordinates to world coordinates
    
    This is the key innovation: normalizing for device orientation
    
    Args:
        acc: acceleration in device coordinates, shape (time_steps, 3) [x, y, z]
        rot: rotation quaternion, shape (time_steps, 4) [w, x, y, z] (normalized)
    
    Returns:
        acc_world: acceleration in world coordinates, shape (time_steps, 3)
        
    Why this matters:
    - Device acceleration depends on how the watch is oriented on the wri/st
    - World acceleration is independent of device orientation
    - This helps the model focus on actual hand motion rather than wrist rotation
    """
    try:
        # Convert quaternion format from [w, x, y, z] to [x, y, z, w] for scipy
        rot_scipy = rot[:, [1, 2, 3, 0]]
        
        # Verify quaternions are valid (non-zero norm)
        norms = np.linalg.norm(rot_scipy, axis=1)
        if np.any(norms < 1e-8):
            # Replace problematic quaternions with identity
            mask = norms < 1e-8
            rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion in scipy format
        
        # Create rotation object and apply transformation
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
        
    except Exception:
        # Fallback to original acceleration if transformation fails
        print("Warning: World coordinate transformation failed, using device coordinates")
        acc_world = acc.copy()
    
    return acc_world


def extract_comprehensive_features(sequence: pl.DataFrame, demographics: pl.DataFrame) -> pd.DataFrame:
    """
    Extract features from IMU data with world acceleration transformation
    
    Feature Groups:
    1. Device Acceleration (acc_x, acc_y, acc_z) - raw sensor data
    2. Rotation Quaternion (rot_w, rot_x, rot_y, rot_z) - device orientation  
    3. World Acceleration (NEW) - orientation-normalized acceleration
    4. Demographics - subject characteristics
    5. Sequence metadata - length, etc.
    """
    
    # Convert to pandas for processing
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    
    # Handle missing values in basic sensor data
    acc_data = seq_df[Config.ACC_COLS].copy()
    acc_data = acc_data.ffill().bfill().fillna(0)
    
    rot_data = seq_df[Config.ROT_COLS].copy()
    rot_data = rot_data.ffill().bfill()
    
    # Handle quaternion missing values and normalize
    rot_data_clean = handle_quaternion_missing_values(rot_data.values)
    
    # CORE INNOVATION: Compute world acceleration
    try:
        world_acc_data = compute_world_acceleration(acc_data.values, rot_data_clean)
        # print("âœ“ World acceleration computed successfully")  # Reduced verbosity
    except Exception as e:
        print(f"Warning: World acceleration computation failed: {e}")
        world_acc_data = acc_data.values.copy()  # Fallback to device coordinates
    
    # Initialize feature dictionary
    features = {}
    
    # Add sequence metadata
    features['sequence_length'] = len(seq_df)
    
    # Add demographics features
    if len(demo_df) > 0:
        demo_row = demo_df.iloc[0]
        features['age'] = demo_row.get('age', 0)
        features['adult_child'] = demo_row.get('adult_child', 0)
        features['sex'] = demo_row.get('sex', 0)
        features['handedness'] = demo_row.get('handedness', 0)
        features['height_cm'] = demo_row.get('height_cm', 0)
        features['shoulder_to_wrist_cm'] = demo_row.get('shoulder_to_wrist_cm', 0)
        features['elbow_to_wrist_cm'] = demo_row.get('elbow_to_wrist_cm', 0)
    
    # Define feature arrays for statistical extraction
    feature_arrays = {
        'acc': acc_data.values,           # Device acceleration (3D)
        'rot': rot_data_clean,            # Rotation quaternion (4D) 
        'world_acc': world_acc_data,      # World acceleration (3D) - KEY INNOVATION
    }
    
    # Extract statistical features for each data source
    for source_name, array in feature_arrays.items():
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        
        n_features = array.shape[1]
        
        for feat_idx in range(n_features):
            feat_data = array[:, feat_idx]
            
            # Create feature name
            if source_name == 'acc':
                axis_names = ['x', 'y', 'z']
                prefix = f"acc_{axis_names[feat_idx]}"
            elif source_name == 'rot':
                comp_names = ['w', 'x', 'y', 'z']
                prefix = f"rot_{comp_names[feat_idx]}"
            elif source_name == 'world_acc':
                axis_names = ['x', 'y', 'z']  
                prefix = f"world_acc_{axis_names[feat_idx]}"
            else:
                prefix = f"{source_name}_{feat_idx}" if n_features > 1 else source_name
            
            # Extract comprehensive statistical features
            features.update(extract_statistical_features(feat_data, prefix))
    
    # Compute magnitude features (important for motion intensity)
    acc_magnitude = np.linalg.norm(acc_data.values, axis=1)
    world_acc_magnitude = np.linalg.norm(world_acc_data, axis=1)
    
    features.update(extract_statistical_features(acc_magnitude, 'acc_magnitude'))
    features.update(extract_statistical_features(world_acc_magnitude, 'world_acc_magnitude'))
    
    # Cross-feature: difference between device and world acceleration magnitudes
    # This captures how much device orientation affects motion measurement
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, 'acc_world_diff'))
    
    # Convert to DataFrame
    result_df = pd.DataFrame([features])
    
    # Handle any remaining NaN values
    result_df = result_df.fillna(0)
    
    return result_df


def extract_statistical_features(data: np.ndarray, prefix: str) -> dict:
    """
    Extract comprehensive statistical features from a 1D time series
    
    Returns features that capture:
    - Central tendency: mean, median, mode region
    - Spread: std, variance, range, IQR  
    - Shape: skewness, kurtosis
    - Dynamics: differences, trends, changes
    - Segments: beginning vs middle vs end behavior
    """
    
    features = {}
    
    # Basic statistics
    features[f'{prefix}_mean'] = np.mean(data)
    features[f'{prefix}_std'] = np.std(data)
    features[f'{prefix}_var'] = np.var(data)
    features[f'{prefix}_min'] = np.min(data)
    features[f'{prefix}_max'] = np.max(data)
    features[f'{prefix}_median'] = np.median(data)
    features[f'{prefix}_q25'] = np.percentile(data, 25)
    features[f'{prefix}_q75'] = np.percentile(data, 75)
    features[f'{prefix}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
    
    # Range and boundary features
    features[f'{prefix}_range'] = np.max(data) - np.min(data)
    features[f'{prefix}_first'] = data[0] if len(data) > 0 else 0
    features[f'{prefix}_last'] = data[-1] if len(data) > 0 else 0
    features[f'{prefix}_delta'] = data[-1] - data[0] if len(data) > 0 else 0
    
    # Higher order moments (shape of distribution)
    if len(data) > 1 and np.std(data) > 1e-8:
        features[f'{prefix}_skew'] = pd.Series(data).skew()
        features[f'{prefix}_kurt'] = pd.Series(data).kurtosis()
    else:
        features[f'{prefix}_skew'] = 0
        features[f'{prefix}_kurt'] = 0
    
    # Differential features (capture dynamics)
    if len(data) > 1:
        diff_data = np.diff(data)
        features[f'{prefix}_diff_mean'] = np.mean(diff_data)
        features[f'{prefix}_diff_std'] = np.std(diff_data)
        features[f'{prefix}_n_changes'] = np.sum(np.abs(diff_data) > np.std(data) * 0.1)  # Significant changes
    else:
        features[f'{prefix}_diff_mean'] = 0
        features[f'{prefix}_diff_std'] = 0
        features[f'{prefix}_n_changes'] = 0
    
    # Correlation with time (trend detection)
    if len(data) > 2:
        time_indices = np.arange(len(data))
        try:
            corr_coef = np.corrcoef(time_indices, data)[0, 1]
            features[f'{prefix}_time_corr'] = corr_coef if not np.isnan(corr_coef) else 0
        except:
            features[f'{prefix}_time_corr'] = 0
    else:
        features[f'{prefix}_time_corr'] = 0
    
    # Segment features (beginning, middle, end patterns)
    seq_len = len(data)
    if seq_len >= 9:  # Need sufficient data for meaningful segments
        seg_size = seq_len // 3
        seg1 = data[:seg_size]           # Beginning (Transition phase)
        seg2 = data[seg_size:2*seg_size] # Middle (Pause phase)  
        seg3 = data[2*seg_size:]         # End (Gesture phase)
        
        features[f'{prefix}_seg1_mean'] = np.mean(seg1)
        features[f'{prefix}_seg2_mean'] = np.mean(seg2)
        features[f'{prefix}_seg3_mean'] = np.mean(seg3)
        
        features[f'{prefix}_seg1_std'] = np.std(seg1)
        features[f'{prefix}_seg2_std'] = np.std(seg2)
        features[f'{prefix}_seg3_std'] = np.std(seg3)
        
        # Segment transitions (important for distinguishing gesture types)
        features[f'{prefix}_seg1_to_seg2'] = np.mean(seg2) - np.mean(seg1)
        features[f'{prefix}_seg2_to_seg3'] = np.mean(seg3) - np.mean(seg2)
    else:
        # Not enough data for meaningful segments
        for seg in [1, 2, 3]:
            features[f'{prefix}_seg{seg}_mean'] = features[f'{prefix}_mean']
            features[f'{prefix}_seg{seg}_std'] = features[f'{prefix}_std']
        features[f'{prefix}_seg1_to_seg2'] = 0
        features[f'{prefix}_seg2_to_seg3'] = 0
    
    return features


def prepare_data_thm_tof(train_df: pl.dataframe):
    print("Calculating ToF features with vectorized Polars and NumPy...")

    # 1. 'tof_1_v0', 'tof_1_v1', ... ì™€ ê°™ì�€ ì—´ ì�´ë¦„ì�„ ë�™ì �ìœ¼ë¡œ ìƒ�ì„±
    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    
    # 2. -1 ê°’ì�„ Nullë¡œ ëŒ€ì²´í•˜ê³ , Polarsì�˜ í‘œí˜„ì‹�ì�„ ì‚¬ìš©í•˜ì—¬ NumPy ë°°ì—´ë¡œ ë³€í™˜
    # is_null()ì—� ëŒ€í•œ is_not_null()ì�€ .when().then()ì�„ ì‚¬ìš©í•˜ì—¬ êµ¬í˜„í•  ìˆ˜ ì�ˆìŠµë‹ˆë‹¤.
    # ë¨¼ì € í•„ìš”í•œ ì—´ë§Œ ì„ íƒ�í•œ í›„ -1 ê°’ì�„ nullë¡œ ëŒ€ì²´í•©ë‹ˆë‹¤.
    train_df_selected = train_df.select(pl.col(tof_pixel_cols).replace(-1, None))
    tof_data_np = train_df_selected.to_numpy()
    
    # 3. NumPyë¥¼ ì‚¬ìš©í•˜ì—¬ í†µê³„ê°’ ê³„ì‚° (ê¸°ì¡´ ë¡œì§�ê³¼ ë�™ì�¼)
    reshaped_tof = tof_data_np.reshape(len(train_df), 5, 64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    
    # 4. Polarsì�˜ 'with_columns'ë¥¼ ì‚¬ìš©í•˜ì—¬ ìƒˆë¡œìš´ ì—´ì�„ íš¨ìœ¨ì �ìœ¼ë¡œ ì¶”ê°€
    # 'with_columns'ëŠ” ì—¬ëŸ¬ ì—´ì�„ í•œ ë²ˆì—� ì¶”ê°€í•  ìˆ˜ ì�ˆì–´ íš¨ìœ¨ì �ì�…ë‹ˆë‹¤.
    for i in range(1, 6):
        train_df = train_df.with_columns(
            pl.Series(name=f'tof_{i}_mean', values=mean_vals[:, i-1]),
            pl.Series(name=f'tof_{i}_std', values=std_vals[:, i-1]),
            pl.Series(name=f'tof_{i}_min', values=min_vals[:, i-1]),
            pl.Series(name=f'tof_{i}_max', values=max_vals[:, i-1]),
        )
    
    # 5. ìµœì¢… íŠ¹ì§• ì—´ ëª©ë¡� ìƒ�ì„± (ê¸°ì¡´ ë¡œì§�ê³¼ ë�™ì�¼)
    tof_agg_cols = [f'tof_{i}_{agg}' for i in range(1, 6) for agg in ['mean', 'std', 'min', 'max']]
    imu_cols = [c for c in train_df.columns if c.startswith(('acc_', 'rot_'))]
    thm_cols = [c for c in train_df.columns if c.startswith('thm_')]
    final_feature_cols = imu_cols + thm_cols + tof_agg_cols
    imu_dim_final = len(imu_cols)

    tof_thm_aggregated_dim_final = len(thm_cols) + len(tof_agg_cols)
    np.save(os.path.join("./", "feature_cols.npy"), np.array(final_feature_cols))
    print("  Building, scaling, and padding sequences...")
    # ì˜¤ë¥˜ë‚˜ëŠ” ë¶€
    """X_list, y_list, lens = [], [], []
    for seq_id, seq_df in df.groupby('sequence_id'):
        X_list.append(seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32'))
        y_list.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(seq_df))
    
    print("âœ“ Polars ë�°ì�´í„°í”„ë ˆì�„ì—�ì„œ ToF íŠ¹ì§• ê³„ì‚° ì™„ë£Œ!")
    print(f"ìƒˆë¡œ ì¶”ê°€ë�œ íŠ¹ì§• ì—´: {tof_agg_cols}")
    print(f"ìµœì¢… íŠ¹ì§• ì—´ì�˜ ì´� ê°œìˆ˜: {len(final_feature_cols)}")
    """



def load_and_prepare_data():
    """Load and prepare training data with comprehensive features"""
    
    print("Loading training data...")
    train_df = pl.read_csv(Config.TRAIN_PATH)
    train_demographics = pl.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)
    
    print("Loading test data...")
    test_df = pl.read_csv(Config.TEST_PATH)
    test_demographics = pl.read_csv(Config.TEST_DEMOGRAPHICS_PATH)
    
    # Get common columns between train and test (exclude thermal and ToF sensors)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    
    # Filter to IMU-only columns (remove thermal and ToF sensors)
    imu_cols = [col for col in common_cols if not (col.startswith('thm_') or col.startswith('tof_'))]
    
    print(f"âœ“ Using {len(imu_cols)} common IMU columns")
    print(f"âœ“ Train-only columns: {train_cols - test_cols}")
    print(f"âœ“ Test-only columns: {test_cols - train_cols}")
    
    print("Extracting features for training sequences...")
    train_features_list = []
    train_labels = []
    train_subjects = []
    train_sequence_ids = []
    
    # Group by sequence_id for training data - need to include gesture column for labels
    train_imu_cols = imu_cols + ['gesture'] if 'gesture' not in imu_cols else imu_cols
    train_sequences = train_df.select(pl.col(train_imu_cols)).group_by('sequence_id', maintain_order=True)
    
    for sequence_id, sequence_data in train_sequences:
        # Get sequence features
        sequence_id_val = sequence_id[0] if isinstance(sequence_id, tuple) else sequence_id
        
        # Get demographics for this sequence
        subject_id = sequence_data['subject'][0]
        subject_demographics = train_demographics.filter(pl.col('subject') == subject_id)
        
        # Extract features (only IMU columns for feature extraction)
        imu_only_data = sequence_data.select(pl.col(imu_cols))
        features = extract_comprehensive_features(imu_only_data, subject_demographics)
        features['sequence_id'] = sequence_id_val
        
        train_features_list.append(features)
        
        # Get label (gesture) for this sequence
        gesture = sequence_data['gesture'][0]
        label = GESTURE_MAPPER[gesture]
        train_labels.append(label)
        train_subjects.append(subject_id)
        train_sequence_ids.append(sequence_id_val)
    
    # Combine all training features
    X_train = pd.concat(train_features_list, ignore_index=True)
    y_train = np.array(train_labels)
    subjects = np.array(train_subjects)
    
    print("Extracting features for test sequences...")
    test_features_list = []
    test_sequence_ids = []
    
    # Group by sequence_id for test data  
    test_sequences = test_df.select(pl.col(imu_cols)).group_by('sequence_id', maintain_order=True)
    
    for sequence_id, sequence_data in test_sequences:
        sequence_id_val = sequence_id[0] if isinstance(sequence_id, tuple) else sequence_id
        
        # Get demographics for this sequence
        subject_id = sequence_data['subject'][0]
        subject_demographics = test_demographics.filter(pl.col('subject') == subject_id)
        
        # Extract features using the same function as training
        features = extract_comprehensive_features(sequence_data, subject_demographics)
        features['sequence_id'] = sequence_id_val
        
        test_features_list.append(features)
        test_sequence_ids.append(sequence_id_val)
    
    # Combine all test features
    X_test = pd.concat(test_features_list, ignore_index=True)
    
    print(f"âœ“ Training features shape: {X_train.shape}")
    print(f"âœ“ Training labels shape: {y_train.shape}")
    print(f"âœ“ Test features shape: {X_test.shape}")
    print(f"âœ“ Number of features: {X_train.shape[1] - 1}")  # -1 for sequence_id
    
    return X_train, y_train, subjects, X_test, test_sequence_ids, imu_cols


def train_lightgbm_models(X_train, y_train, subjects):
    """Train LightGBM models using stratified group k-fold cross-validation"""
    
    print(f"Training LightGBM models with {Config.N_FOLDS}-fold cross-validation...")
    
    # Prepare features (remove sequence_id)
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    X_features = X_train[feature_cols]
    
    # Setup cross-validation
    cv = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    models = []
    oof_predictions = np.zeros(len(y_train))
    cv_scores = []
    
    print(f"Feature columns: {len(feature_cols)}")
    print("Starting cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        
        # Split data
        X_fold_train = X_features.iloc[train_idx]
        X_fold_val = X_features.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        
        # Train model with monitoring and device detection
        lgbm_params = Config.LGBM_PARAMS.copy()
        lgbm_params['device'] = DEVICE  # Use detected device
        
        model = LGBMClassifier(**lgbm_params)
        
        print(f"Training fold {fold + 1} with monitoring every 5 rounds (Device: {DEVICE})...")
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
            eval_names=['train', 'valid'],
            eval_metric=['multi_logloss', 'multi_error'],
            callbacks=[
                log_evaluation(period=5),  # Output every 5 rounds
                early_stopping(stopping_rounds=100, verbose=True)
            ]
        )
        
        # Predictions
        val_preds = model.predict(X_fold_val)
        oof_predictions[val_idx] = val_preds
        
        # Calculate metrics
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_preds)
        cv_scores.append(score)
        
        print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
        
        models.append(model)
    
    # Overall CV performance
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, oof_predictions)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Competition Score: {overall_score:.4f} Â± {np.std(cv_scores):.4f}")
    print(f"Overall Binary F1: {overall_binary_f1:.4f}")
    print(f"Overall Macro F1: {overall_macro_f1:.4f}")
    print(f"Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"{'='*60}")
    
    return models, cv_scores, overall_score


def train_randomforest_models(X_train, y_train, subjects):
    """Train RandomForest models using stratified group k-fold cross-validation"""
    
    print(f"Training RandomForest models with {Config.N_FOLDS}-fold cross-validation...")
    
    # Prepare features (remove sequence_id)
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    X_features = X_train[feature_cols]
    
    # Setup cross-validation
    cv = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    models = []
    oof_predictions = np.zeros(len(y_train), dtype=int)
    cv_scores = []
    
    print(f"Feature columns: {len(feature_cols)}")
    print("Starting cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        
        # Split data
        X_fold_train = X_features.iloc[train_idx]
        X_fold_val = X_features.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        
        # Train model
        model = RandomForestClassifier(**Config.RF_PARAMS)
        
        print(f"Training fold {fold + 1}...")
        model.fit(X_fold_train, y_fold_train)
        
        # Predictions
        val_preds = model.predict(X_fold_val)
        oof_predictions[val_idx] = val_preds
        
        # Calculate metrics
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_preds)
        cv_scores.append(score)
        
        print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
        
        models.append(model)
    
    # Overall CV performance
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, oof_predictions)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Competition Score: {overall_score:.4f} Â± {np.std(cv_scores):.4f}")
    print(f"Overall Binary F1: {overall_binary_f1:.4f}")
    print(f"Overall Macro F1: {overall_macro_f1:.4f}")
    print(f"Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"{'='*60}")
    
    return models, cv_scores, overall_score



def train_xgboost_models(X_train, y_train, subjects):
    """Train XGBoost models using stratified group k-fold cross-validation"""
    
    print(f"Training XGBoost models with {Config.N_FOLDS}-fold cross-validation...")
    
    # Prepare features (remove sequence_id)
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    X_features = X_train[feature_cols]
    
    # Setup cross-validation
    cv = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    models = []
    oof_predictions = np.zeros(len(y_train), dtype=int)
    cv_scores = []
    
    print(f"Feature columns: {len(feature_cols)}")
    print("Starting cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_features, y_train, subjects)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        
        # Split data
        X_fold_train = X_features.iloc[train_idx]
        X_fold_val = X_features.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        print(f"Train size: {len(X_fold_train)}, Val size: {len(X_fold_val)}")
        
        # Train model with monitoring and device detection
        xgb_params = Config.XGB_PARAMS.copy()
        
        # Adjust tree_method for GPU if available and desired
        if DEVICE == 'gpu':
            xgb_params['tree_method'] = 'gpu_hist'

        model = xgb.XGBClassifier(**xgb_params)
        
        print(f"Training fold {fold + 1} with early stopping (Device: {DEVICE})...")
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        # Predictions
        val_preds = model.predict(X_fold_val)
        oof_predictions[val_idx] = val_preds
        
        # Calculate metrics
        score, binary_f1, macro_f1 = competition_metric(y_fold_val, val_preds)
        cv_scores.append(score)
        
        print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")
        
        models.append(model)
    
    # Overall CV performance
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, oof_predictions)
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Competition Score: {overall_score:.4f} Â± {np.std(cv_scores):.4f}")
    print(f"Overall Binary F1: {overall_binary_f1:.4f}")
    print(f"Overall Macro F1: {overall_macro_f1:.4f}")
    print(f"Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"{'='*60}")
    
    return models, cv_scores, overall_score


def create_prediction_function(models, feature_cols, imu_cols):
    """Create prediction function for Kaggle evaluation"""
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """
        Prediction function for Kaggle evaluation
        Uses ensemble of trained LightGBM models
        """
        try:
            # Filter sequence to only include IMU columns that we trained with
            available_cols = sequence.columns
            sequence_imu_cols = [col for col in imu_cols if col in available_cols]
            sequence_filtered = sequence.select(pl.col(sequence_imu_cols))
            
            # Extract features using the same method as training
            features = extract_comprehensive_features(sequence_filtered, demographics)
            
            # Ensure we have the same features as training
            missing_features = [col for col in feature_cols if col not in features.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}, filling with zeros")
                for col in missing_features:
                    features[col] = 0
            
            X_pred = features[feature_cols]
            
            # Get predictions from all models
            predictions = []
            for model in models:
                pred_probs = model.predict_proba(X_pred)
                pred_class = np.argmax(pred_probs, axis=1)[0]
                predictions.append(pred_class)
            
            # Ensemble prediction (majority vote)
            final_prediction = max(set(predictions), key=predictions.count)
            
            # Convert back to gesture name
            gesture_name = REVERSE_GESTURE_MAPPER[final_prediction]
            
            print(f"Predicted: {gesture_name} (class {final_prediction})")
            return gesture_name
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 'Text on phone'  # Fallback prediction
    
    return predict


def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("CMI BFRB Detection - LightGBM with World Acceleration Feature")
    print("="*60)
    print("ğŸš€ Key Innovation: World Coordinate Transformation")
    print("   Converting device acceleration to world coordinates")
    print("   This normalizes for different wrist orientations!")
    print("="*60)
    
    # Load and prepare data
    X_train, y_train, subjects, X_test, test_sequence_ids, imu_cols = load_and_prepare_data()
    
    # Train models
    # models, cv_scores, overall_score = train_lightgbm_models(X_train, y_train, subjects)
    # models, cv_scores, overall_score = train_randomforest_models(X_train, y_train, subjects)
    models, cv_scores, overall_score = train_xgboost_models(X_train, y_train, subjects)
    
    # Prepare feature columns for inference
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    
    # Create prediction function
    predict_func = create_prediction_function(models, feature_cols, imu_cols)
    
    print(f"\nâœ“ Training completed successfully!")
    print(f"âœ“ Final CV Score: {overall_score:.4f}")
    print(f"âœ“ Core innovation: World acceleration transformation")
    print(f"âœ“ Models ready for inference")
    
    return predict_func, models, cv_scores

# Execute main pipeline
if __name__ == "__main__":
    predict_function, trained_models, cv_scores = main()
    
    # Setup inference server
    print("\nSetting up inference server...")
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict_function)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # Local testing
        inference_server.run_local_gateway(
            data_paths=(
                Config.TEST_PATH,
                Config.TEST_DEMOGRAPHICS_PATH,
            )
        )

