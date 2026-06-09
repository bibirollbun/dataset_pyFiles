import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import os
import polars as pl
import kaggle_evaluation.cmi_inference_server
from sklearn.metrics import accuracy_score, f1_score
import joblib
from scipy.spatial.transform import Rotation as R


# Tải tập dữ liệu
#Dữ liệu cảm biến cho tập huấn luyện (Train Sensor Data): 
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")

#Thông tin nhân khẩu học của đối tượng trong tập huấn luyện (giới tính, tuổi, ...): 
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

#Dữ liệu cảm biến cho tập kiểm tra:
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")

# Thông tin nhân khẩu học của đối tượng trong tập kiểm tra:
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


datasets = {
    "Train Data": train_df,
    "Train Demographics": train_dem_df,
    "Test Data": test_df,
    "Test Demographics": test_dem_df,
}

for name, df in datasets.items():
    print(f"Missing values in {name}:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("  ✅ No missing values.")
    print()



#Kiểm tra trùng lặp (Duplicate Rows): .duplicated(): Trả về Series boolean, dòng nào trùng lặp sẽ là True.
#                                     .sum(): Đếm tổng số dòng trùng lặp.
#Kiểm tra xem dữ liệu có bản ghi nào bị trùng không — quan trọng cho việc làm sạch dữ liệu.

# Đếm các hàng trùng lặp trong train_df
train_duplicates = train_df.duplicated().sum()

# Đếm các hàng trùng lặp trong  test_df
test_duplicates = test_df.duplicated().sum()

# Đếm các hàng trùng lặp trong train_dem_df (optional)
train_dem_duplicates = train_dem_df.duplicated().sum()
# Đếm các hàng trùng lặp trong test_dem_df (optional)
test_dem_duplicates = test_dem_df.duplicated().sum()

# In số lượng dòng trùng:
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in test_df: {test_duplicates}")
print(f"Number of duplicate rows in train_dem_df: {train_dem_duplicates}")
print(f"Number of duplicate rows in test_dem_df: {test_dem_duplicates}")


def null_percent(df):
    per=((df.isnull().sum()/len(df))*100).round(2)
    return per[per>0]

print("Nan Values in Train data")
print(null_percent(train_df))


# --- 2.4 Hợp nhất dữ liệu nhân khẩu học và cảm biến ---
merged_train = pd.merge(train_df, train_dem_df, on="subject", how="left")
merged_test = pd.merge(test_df, test_dem_df, on="subject", how="left")

print("Merged Train Shape:", merged_train.shape)
print("Merged Test Shape:", merged_test.shape)
display(merged_train.head(2))



# --- Thống kê mô tả toàn bộ merged_train ---
merged_train.describe().T



#Kiểm tra thiếu giá trị & thống kê cảm biến: 
#Danh sách các cột không thuộc cảm biến (có thể là ID, thông tin khác). 

excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
sensor_cols = [col for col in train_df.columns if not col.startswith(excluded_prefixes)]

# Sensor Data Summary for TRAIN
#isnull().sum(): Đếm số giá trị bị thiếu.
missing_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    '[TRAIN] Missing Count': train_df[sensor_cols].isnull().sum().values,
    '[TRAIN] Missing %': (train_df[sensor_cols].isnull().sum().values / len(train_df)) * 100
})

#nunique(): Đếm số lượng giá trị duy nhất.
unique_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    'Unique Values [TRAIN]': train_df[sensor_cols].nunique().values
})

#dtypes: Lấy kiểu dữ liệu của từng cột.
dtypes_sensor = pd.DataFrame({
    'Feature': sensor_cols,
    'Data Type': train_df[sensor_cols].dtypes.values
})

# Merge all summaries (NO test set)
#merge: Gộp các bảng thống kê thành bảng duy nhất theo Feature.
sensor_summary = missing_sensor_train \
    .merge(unique_sensor_train, on='Feature', how='left') \
    .merge(dtypes_sensor, on='Feature', how='left')

# Display styled DataFrame (mask NaNs just for styling)
#fillna(0): Điền giá trị thiếu bằng 0 (cho đẹp mắt khi hiển thị).
#.style.background_gradient: Tô màu nền theo giá trị giúp dễ nhìn.
styled_df = sensor_summary.fillna(0)
styled_df.style.background_gradient(cmap='viridis')


#Thống kê tương tự cho nhân khẩu học: Tương tự như phần thống kê cảm biến nhưng áp dụng cho dữ liệu nhân khẩu học.

# Cột nhân khẩu học (không loại trừ)
dem_cols = train_dem_df.columns

# Giá trị bị thiếu trong dữ liệu nhân khẩu học của train 
missing_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    '[TRAIN DEMO] Missing Count': train_dem_df[dem_cols].isnull().sum().values,
    '[TRAIN DEMO] Missing %': (train_dem_df[dem_cols].isnull().sum().values / len(train_dem_df)) * 100
})

# Giá trị duy nhất được tính trong dữ liệu nhân khẩu học của train 
unique_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    'Unique Values [TRAIN DEMO]': train_dem_df[dem_cols].nunique().values
})

# Data types
dtypes_demo = pd.DataFrame({
    'Feature': dem_cols,
    'Data Type': train_dem_df[dem_cols].dtypes.values
})

# Tóm tắt hợp nhất (chỉ dành cho đào tạo)
demo_summary = (
    missing_demo_train
    .merge(unique_demo_train, on='Feature', how='left')
    .merge(dtypes_demo, on='Feature', how='left')
)

# Hiển thị tóm tắt theo phong cách
demo_summary.style.background_gradient(cmap='viridis')


import numpy as np
import pandas as pd

# 1) Sao chép đào tạo và kiểm tra để chúng ta không sửa đổi DataFrame gốc
train_temp = train_df.copy()
test_temp  = test_df.copy()

# 2) GIA TỐC KẾ: tính toán độ lớn tại mỗi dấu thời gian
train_temp['acc_mag'] = np.sqrt(
    train_temp['acc_x']**2 + train_temp['acc_y']**2 + train_temp['acc_z']**2
)
test_temp['acc_mag'] = np.sqrt(
    test_temp['acc_x']**2 + test_temp['acc_y']**2 + test_temp['acc_z']**2
)

# 3) ROTATION: tính toán “góc quay” từ thành phần quaternion w
# (Lưu ý: rot_w nằm trong [-1,1], do đó arccos hợp lệ. Chúng tôi bỏ qua NaN nếu có.)
train_temp['rot_angle'] = 2 * np.arccos(train_temp['rot_w'].clip(-1,1))
test_temp['rot_angle']  = 2 * np.arccos(test_temp['rot_w'].clip(-1,1))

# 4) Nhóm theo sequence_id và tổng hợp các tóm tắt gia tốc kế
acc_agg_funcs = {
    'acc_mag': ['mean', 'std', 'max']
}
train_acc_summary = train_temp.groupby('sequence_id').agg(acc_agg_funcs)
test_acc_summary  = test_temp.groupby('sequence_id').agg(acc_agg_funcs)

# Làm phẳng cột MultiIndex
train_acc_summary.columns = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]
test_acc_summary.columns  = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]

# 5) Nhóm theo sequence_id và tổng hợp tóm tắt vòng quay
rot_agg_funcs = {
    'rot_angle': ['mean', 'std', 'max']
}
train_rot_summary = train_temp.groupby('sequence_id').agg(rot_agg_funcs)
test_rot_summary  = test_temp.groupby('sequence_id').agg(rot_agg_funcs)

train_rot_summary.columns = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]
test_rot_summary.columns  = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]

# 6) NHIỆT ĐỘ: năm cảm biến thm_1 … thm_5
thm_cols = [f"thm_{i}" for i in range(1, 6)]

# Xác định hàm tổng hợp: trung bình + độ lệch chuẩn
thm_agg_funcs = {col: ['mean', 'std'] for col in thm_cols}

train_thm_summary = train_temp.groupby('sequence_id').agg(thm_agg_funcs)
test_thm_summary  = test_temp.groupby('sequence_id').agg(thm_agg_funcs)

# Làm phẳng các cột MultiIndex
flattened_thm_cols = []
for sensor in thm_cols:
    for stat in ['mean','std']:
        flattened_thm_cols.append(f"{sensor}_{stat}")

train_thm_summary.columns = flattened_thm_cols
test_thm_summary.columns  = flattened_thm_cols

# 7) THỜI GIAN BAY: mỗi cảm biến i có 64 cột pixel: tof_i_v0 … tof_i_v63
# Chúng ta sẽ tạo một “tof_i_mean_at_ts” cho mỗi dấu thời gian, sau đó tổng hợp theo từng chuỗi.

def compute_tof_sequence_summary(df):
    # Khởi tạo một dict để giữ DataFrames theo từng chuỗi
    seq_summaries = {}

    for i in range(1, 6):
        # Xây dựng danh sách các cột cho cảm biến i
        tof_cols = [f"tof_{i}_v{pix}" for pix in range(64)]
        # Thay thế -1 bằng NaN để chúng không làm lệch giá trị trung bình; chuyển đổi sang float
        ts_grid = df[tof_cols].replace(-1, np.nan).astype(float)
        # Tính toán “trung bình trên tất cả 64 pixel” cho mỗi dấu thời gian
        df[f"tof_{i}_mean_at_ts"] = ts_grid.mean(axis=1)
    
   # Bây giờ, nhóm theo id chuỗi và tính giá trị trung bình và độ lệch chuẩn của các giá trị trung bình đó
    agg_dict = {f"tof_{i}_mean_at_ts": ['mean','std'] for i in range(1, 6)}
    summary = df.groupby('sequence_id').agg(agg_dict)
    # Làm phẳng các cột MultiIndex
    flat_cols = [f"tof_{i}_{stat}" for i in range(1, 6) for stat in ['mean','std']]
    summary.columns = flat_cols
    return summary

train_tof_summary = compute_tof_sequence_summary(train_temp)
test_tof_summary  = compute_tof_sequence_summary(test_temp)

# 8) Hợp nhất các tóm tắt accel, rotation, thm, tof (trên sequence_id)
train_sensor_summary = (
    train_acc_summary
    .join(train_rot_summary, how='outer')
    .join(train_thm_summary, how='outer')
    .join(train_tof_summary, how='outer')
)

test_sensor_summary = (
    test_acc_summary
    .join(test_rot_summary, how='outer')
    .join(test_thm_summary, how='outer')
    .join(test_tof_summary, how='outer')
)

# 9) Thêm cột “Dataset” để chúng ta có thể thực hiện box+hist song song
train_sensor_summary['Dataset'] = 'Train'
test_sensor_summary['Dataset']  = 'Test'

# 10) Ghép nối thành một DataFrame để vẽ đồ thị
combined_sensor_summary = pd.concat(
    [train_sensor_summary, test_sensor_summary],
    axis=0
).reset_index(drop=True)


import numpy as np
import matplotlib.pyplot as plt

# (1) Sáp nhập train_demographics vào train_df nếu chưa thực hiện
train_df = train_df.merge(
    train_dem_df,
    on="subject",
    how="left"
)


def fix_imu(df):
    imu_cols = [c for c in df.columns if c.startswith(("acc_", "rot_"))]
    def _seq_fix(g):
        g[imu_cols] = g[imu_cols].interpolate(limit_direction="both")
        # renormalize quaternion
        q = g[["rot_w","rot_x","rot_y","rot_z"]].to_numpy(float)
        n = (q**2).sum(1)**0.5
        n[n==0] = 1.0
        q = q / n[:,None]
        g[["rot_w","rot_x","rot_y","rot_z"]] = q
        return g
    return df.groupby("sequence_id", group_keys=False).apply(_seq_fix)



def fix_thm(df):
    thm = [f"thm_{i}" for i in range(1,6)]
    return df.groupby("sequence_id", group_keys=False)[thm].apply(
        lambda g: g.interpolate(limit_direction="both").ffill().bfill()
    )


def fix_tof_and_reduce(df):
    # 1) sentinel -> NaN + flag
    tof_pix = [c for c in df.columns if c.startswith("tof_") and "_v" in c]
    df[tof_pix] = df[tof_pix].replace(-1, np.nan)
    # optional flags per sensor
    for i in range(1,6):
        cols = [f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in df]
        df[f"tof_{i}_missing_frac"] = df[cols].isna().mean(axis=1)
        df[f"tof_{i}_mean"] = df[cols].mean(axis=1)  # dimension reduction
    # 2) interpolate the 5 means
    mean_cols = [f"tof_{i}_mean" for i in range(1,6)]
    df[mean_cols] = df.groupby("sequence_id", group_keys=False)[mean_cols].apply(
        lambda g: g.interpolate(limit_direction="both").ffill().bfill()
    )
    # 3) drop heavy pixel columns to save RAM
    df.drop(columns=tof_pix, inplace=True)
    return df


from sklearn.preprocessing import LabelEncoder
le_gesture = LabelEncoder().fit(train_df["gesture"].astype(str))
train_df["gesture_id"] = le_gesture.transform(train_df["gesture"].astype(str))


def minimal_preprocess(df):
    df = df.copy()
    # 1) IMU
    df = fix_imu(df)
    # 2) Thermopile
    thm_fixed = fix_thm(df)  # returns only thm cols
    df[thm_fixed.columns] = thm_fixed
    # 3) ToF -> means + flags, drop pixels
    df = fix_tof_and_reduce(df)
    # 4) Feature cơ bản (ví dụ):
    df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["rot_angle"] = 2*np.arccos(np.clip(df["rot_w"], -1, 1))
    return df


import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
from typing import Tuple, List, Optional
import warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from scipy.spatial.transform import Rotation as R
import kaggle_evaluation.cmi_inference_server

warnings.filterwarnings('ignore')

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
    
    # LightGBM hyperparameters
    LGBM_PARAMS = {
        'objective': 'multiclass',  # Task type
        'num_class': 18,            # Number of classes for multi-class classification
        'n_estimators': 1024,       # Number of trees
        'max_depth': 8,             # Maximum depth of trees
        'learning_rate': 0.025,     # Learning rate
        'colsample_bytree': 0.5,    # Fraction of features to use for each tree
        'n_jobs': -1,               # Number of parallel threads
        'num_leaves': 20,           # Number of leaves per tree
        'random_state': 42,         # Random seed for reproducibility
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 0.1,          # L2 regularization
        'subsample': 0.5,           # Fraction of samples for each tree
        'verbosity': -1,            # Suppress output verbosity
    }



# =============================================================================
# GESTURE MAPPING
# =============================================================================

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
        print("⚠️ GPU not available, using CPU for LightGBM")
        return 'cpu'

# Initialize DEVICE based on the availability of GPU
DEVICE = check_gpu_availability()



# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_lightgbm_model():
    """Build and return a LightGBM model based on configuration"""
    model = LGBMClassifier(**Config.LGBM_PARAMS)
    return model

# =============================================================================
# MODEL TRAINING
# =============================================================================

from sklearn.metrics import f1_score, accuracy_score

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
    print(f"Overall Competition Score: {overall_score:.4f} ± {np.std(cv_scores):.4f}")
    print(f"Overall Binary F1: {overall_binary_f1:.4f}")
    print(f"Overall Macro F1: {overall_macro_f1:.4f}")
    print(f"Fold scores: {['{:.4f}'.format(score) for score in cv_scores]}")
    print(f"{'='*60}")
    
    # Print final confirmation message
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Final CV Score: {overall_score:.4f}")
    print(f"✓ Core innovation: World acceleration transformation")
    print(f"✓ Models ready for inference")
    
    return models, cv_scores, overall_score  # Return all three values



# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def create_prediction_function(models, feature_cols, imu_cols):
    """Create a function for generating predictions using trained models"""
    
    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame):
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



# =============================================================================
# CORE FEATURE ENGINEERING: WORLD ACCELERATION
# =============================================================================

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Convert acceleration from device coordinates to world coordinates
    """
    try:
        # Convert quaternion format from [w, x, y, z] to [x, y, z, w] for scipy
        rot_scipy = rot[:, [1, 2, 3, 0]]
        
        # Verify quaternions are valid (non-zero norm)
        norms = np.linalg.norm(rot_scipy, axis=1)
        if np.any(norms < 1e-8):
            rot_scipy[norms < 1e-8] = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion in scipy format
        
        # Create rotation object and apply transformation
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
        
    except Exception:
        # Fallback to original acceleration if transformation fails
        acc_world = acc.copy()
    
    return acc_world


def extract_statistical_features(data: np.ndarray, prefix: str) -> dict:
    """
    Extract comprehensive statistical features from a 1D time series.
    This function computes common statistical features like mean, std, skewness, kurtosis, etc.
    
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



from sklearn.metrics import f1_score

def competition_metric(y_true, y_pred) -> tuple:
    """Calculate the competition metric (Binary F1 + Macro F1) / 2"""
    
    # Binary F1: BFRB vs non-BFRB (classifying BFRB gestures as 1, others as 0)
    binary_f1 = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )
    
    # Macro F1: Calculating F1 score across all BFRB gestures (0-7)
    macro_f1 = f1_score(
        np.where(y_true <= 7, y_true, 99),  # Map non-BFRB gestures to class 99
        np.where(y_pred <= 7, y_pred, 99),  # Map non-BFRB gestures to class 99
        average="macro", 
        zero_division=0.0,
    )
    
    # Final competition score: average of binary F1 and macro F1
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



# =============================================================================
# SIMPLIFIED FEATURE EXTRACTION
# =============================================================================

def extract_comprehensive_features(sequence: pl.DataFrame, demographics: pl.DataFrame) -> pd.DataFrame:
    """
    Extract features from IMU data with world acceleration transformation.
    This function should return a DataFrame with extracted features.
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
    acc_world_diff = acc_magnitude - world_acc_magnitude
    features.update(extract_statistical_features(acc_world_diff, 'acc_world_diff'))
    
    # Convert to DataFrame
    result_df = pd.DataFrame([features])
    
    # Handle any remaining NaN values
    result_df = result_df.fillna(0)
    
    return result_df



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

    # Extract features for test sequences
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
    
    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Training labels shape: {y_train.shape}")
    print(f"✓ Test features shape: {X_test.shape}")
    print(f"✓ Number of features: {X_train.shape[1] - 1}")  # -1 for sequence_id
    
    # Now remove `sequence_id` from X_train and X_test
    X_train = X_train.drop(columns=['sequence_id'], errors='ignore')
    X_test = X_test.drop(columns=['sequence_id'], errors='ignore')
    
    # Return processed data
    return X_train, y_train, subjects, X_test, test_sequence_ids, imu_cols






# =============================================================================
# INFERENCE SERVER SETUP
# =============================================================================

def setup_inference_server(predict_function):
    """Set up the inference server to serve predictions"""
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict_function)
    
    # If in competition rerun, use Kaggle environment to serve
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(data_paths=(Config.TEST_PATH, Config.TEST_DEMOGRAPHICS_PATH))



# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main function for training, validation, and inference"""
    
    # Load data
    X_train, y_train, subjects, X_test, test_sequence_ids, imu_cols = load_and_prepare_data()

    # Train models
    models, cv_scores, overall_score = train_lightgbm_models(X_train, y_train, subjects)

    # Prepare prediction function
    feature_cols = [col for col in X_train.columns if col != 'sequence_id']
    predict_function = create_prediction_function(models, feature_cols, imu_cols)  # Khởi tạo predict_function

    # Set up inference server
    setup_inference_server(predict_function)

    # Print final success message
    print("\n✓ Training completed successfully!")
    print(f"✓ Final CV Score: {overall_score:.4f}")
    print(f"✓ Models ready for inference")
    
    # Return the required values
    return predict_function, models, cv_scores



if __name__ == "__main__":
    predict_function, trained_models, cv_scores = main()

