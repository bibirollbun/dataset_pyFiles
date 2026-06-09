!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.common-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.core-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.features-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.multimodal-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.tabular-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/autogluon-new/wheelhouse/autogluon.timeseries-1.4.0-py3-none-any.whl


import os
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor

# World coordinate transformation
from scipy.spatial.transform import Rotation as R


import kaggle_evaluation.cmi_inference_server

# model
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.model_selection import StratifiedGroupKFold, PredefinedSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

print(" * All imports loaded successfully!")


class Config:
    # path
    TRAIN_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    TEST_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
    TEST_DEMOGRAPHICS_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

    # Training parameters
    SEED = 42
    N_FOLDS = 5
    SEQ_WINDOW = 60

    # Feature columns
    ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
    ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
# Set reproducibility
np.random.seed(Config.SEED)


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
        print("✓ GPU available for LightGBM")
        return 'gpu'
    except:
        print("⚠️  GPU not available, using CPU for LightGBM")
        return 'cpu'

# Check GPU availability
DEVICE = check_gpu_availability()

print(f"✓ Configuration loaded for Kaggle environment (Device: {DEVICE})")


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
            # |w|² + |x|² + |y|² + |z|² = 1
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
    
    # to pandas
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
        # print("✓ World acceleration computed successfully")  # Reduced verbosity
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

# 쿼터니언에서 회전각 추출
    rot_angle = 2 * np.arccos(np.clip(rot_data_clean[:, 0], -1, 1))
    features.update(extract_statistical_features(rot_angle, 'rot_angle'))
    
    # 수평면 가속도 크기
    acc_xy = np.sqrt(acc_data.values[:, 0]**2 + acc_data.values[:, 1]**2)
    features.update(extract_statistical_features(acc_xy, 'acc_xy'))
    
    # 오일러 각 변환 (roll, pitch, yaw)
    rot_w, rot_x, rot_y, rot_z = (
        rot_data_clean[:, 0],
        rot_data_clean[:, 1],
        rot_data_clean[:, 2],
        rot_data_clean[:, 3],
    )
    
    roll = np.arctan2(
        2 * (rot_w * rot_x + rot_y * rot_z),
        1 - 2 * (rot_x**2 + rot_y**2)
    )
    features.update(extract_statistical_features(roll, 'roll'))
    
    pitch = np.arcsin(
        np.clip(2 * (rot_w * rot_y - rot_z * rot_x), -1, 1)
    )
    features.update(extract_statistical_features(pitch, 'pitch'))
    
    yaw = np.arctan2(
        2 * (rot_w * rot_z + rot_x * rot_y),
        1 - 2 * (rot_y**2 + rot_z**2)
    )
    features.update(extract_statistical_features(yaw, 'yaw'))
    
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


def sequence_window(df: pl.DataFrame, needed_cols: list[str]) -> pl.DataFrame:
    """
    주어진 컬럼만 유지하면서, 각 sequence_id 별 row 개수를 100으로 맞춤.
    - row 수 >= 100 → 뒤에서 200개만 유지
    - row 수 < 100 → 앞을 padding
        * 수치형 컬럼 → 0
        * 문자열 컬럼 → 첫 번째 값
    """

    # 필요한 컬럼만 추출 (단, sequence_counter는 나중에 다시 만들 거니까 제외)
    needed_cols_no_counter = [c for c in needed_cols if ((c != "sequence_counter") & (c != 'row_id'))]
    df = df.select(needed_cols_no_counter)

    out = []
    for seq_id, subdf in df.group_by("sequence_id"):
        n = subdf.height

        if n >= Config.SEQ_WINDOW:
            trimmed = subdf.tail(Config.SEQ_WINDOW)
        else:
            pad_len = Config.SEQ_WINDOW - n
            pad_dict = {}

            for col in subdf.columns:
                dtype = subdf[col].dtype
                first_val = subdf[col][0]

                if dtype.is_numeric():
                    pad_dict[col] = [0] * pad_len
                else:  # 문자열 컬럼
                    pad_dict[col] = [first_val] * pad_len

            pad = pl.DataFrame(pad_dict, schema=subdf.schema)
            trimmed = pl.concat([pad, subdf], how="vertical_relaxed")

        # 시퀀스 내 row 번호 (0 ~ 199) 다시 생성
        trimmed = trimmed.with_columns(
            pl.arange(0, Config.SEQ_WINDOW).alias("sequence_counter")
        )

        # 최종 필요한 컬럼만 유지
        trimmed = trimmed.select(needed_cols_no_counter + ["sequence_counter"])
        out.append(trimmed)

    final_df = pl.concat(out, how="vertical_relaxed")
    return final_df


def load_and_prepare_data():
    print("Loading training data...")
    train_df = pl.read_csv(Config.TRAIN_PATH)
    train_demographics = pl.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)
    
    print("Loading test data...")
    test_df = pl.read_csv(Config.TEST_PATH)
    test_demographics = pl.read_csv(Config.TEST_DEMOGRAPHICS_PATH)

    # remove row_id
    train_df = train_df.drop("row_id")
    test_df = test_df.drop("row_id")
    
    # Get common columns between train and test (exclude thermal and ToF sensors)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    
    # Filter to IMU-only columns (remove thermal and ToF sensors)
    imu_cols = [col for col in common_cols if not (col.startswith('thm_') or col.startswith('tof_'))]

     # 각 row를 window 크기로 자르기 (변경 부분)
    imu_train_df = sequence_window(train_df, imu_cols + ['gesture'])
    
    print(f"✓ Using {len(imu_cols)} common IMU columns")
    print(f"✓ Train-only columns: {train_cols - test_cols}")
    print(f"✓ Test-only columns: {test_cols - train_cols}")

    print("Extracting features for training sequences...")
    train_features_list = []
    train_labels = []
    train_subjects = []
    train_sequence_ids = []
    
    # Group by sequence_id for training data - need to include gesture column for labels
    train_imu_cols = imu_cols + ['gesture'] if 'gesture' not in imu_cols else imu_cols
    train_sequences = imu_train_df.select(pl.col(train_imu_cols)).group_by('sequence_id', maintain_order=True)
    
    for sequence_id, sequence_data in train_sequences:
        # Get sequence features
        sequence_id_val = sequence_id[0] if isinstance(sequence_id, tuple) else sequence_id
        
        # Get demographics for this sequence
        subject_id = sequence_data['subject'][0]
        subject_demographics = train_demographics.filter(pl.col('subject') == subject_id)
        
        # Extract features (only IMU columns for feature extraction)
        imu_only_data = sequence_data.select(pl.col(imu_cols))
        
        # features : dataframe
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
    
    return X_train, imu_cols


def create_prediction_function(feature_cols, imu_cols):

    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """
        Prediction function for Kaggle evaluation
        Uses AutoGluon predictor (internally an ensemble of models)
        """
        try:
            # Filter sequence to only include IMU columns that we trained with
            available_cols = sequence.columns
            sequence_imu_cols = [col for col in imu_cols if col in available_cols]
            sequence_filtered = sequence.select(pl.col(sequence_imu_cols))

            #sequence_filtered_df = sequence_window(sequence_filtered, imu_cols)
            
            # Extract features using the same method as training
            features = extract_comprehensive_features(sequence_filtered, demographics)
            
            # Ensure we have the same features as training
            missing_features = [col for col in feature_cols if col not in features.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}, filling with zeros")
                for col in missing_features:
                    features[col] = 0
            
            X_pred = features[feature_cols]
    
            predictor = TabularPredictor.load("/kaggle/input/ag-0902/ag-20250902_082410")
            
            # AutoGluon predictor 직접 사용
            pred_class = predictor.predict(X_pred)[0]
            
            # predictor.predict()는 class label 그대로 반환할 수 있음
            # 만약 숫자로 반환되면 REVERSE_GESTURE_MAPPER 적용
            if isinstance(pred_class, (int, np.integer)):
                gesture_name = REVERSE_GESTURE_MAPPER[pred_class]
            else:
                gesture_name = pred_class
            
            print(f"Predicted: {gesture_name}")
            return gesture_name
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 'Text on phone'  # Fallback prediction
    return predict


def main():
    """Main execution pipeline"""
    
    # Load and prepare data
    X_train, imu_cols = load_and_prepare_data()
    
    # Prepare feature columns for inference
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    
    # Create prediction function
    predict_func = create_prediction_function(feature_cols, imu_cols)
    
    print(f"\n✓ Training completed successfully!")
    #print(f"✓ Final CV Score: {overall_score:.4f}")
    print(f"✓ Core innovation: World acceleration transformation")
    print(f"✓ Models ready for inference")
    
    return predict_func
    
if __name__ == "__main__":
    predict_function = main()
    
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

