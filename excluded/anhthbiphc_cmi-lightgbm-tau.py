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


train_df['gesture'].unique()



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

REVERSE_GESTURE_MAPPER = {v: k for k, v in GESTURE_MAPPER.items()}\

# =============================================================================
# GROUPED LABELS FOR CONTEST METRIC
# =============================================================================

# ID các lớp target (BFRB)
target_ids = set(range(0, 8))  # 0-7

# ID các lớp non-target (non-BFRB)
non_target_ids = set(range(8, 18))  # 8-17

# Mapping id -> grouped_label
id2label_grouped = {i: ('non_target' if i in non_target_ids else REVERSE_GESTURE_MAPPER[i])
                    for i in range(len(GESTURE_MAPPER))}

# Mapping grouped_label -> id
unique_grouped_labels = sorted(set(id2label_grouped.values()))
label2id_grouped = {g: i for i, g in enumerate(unique_grouped_labels)}
id2label_grouped_final = {i: g for g, i in label2id_grouped.items()}

# ID lớp non_target trong grouped mapping
non_target_id = label2id_grouped['non_target']
target_ids_grouped = {i for i in id2label_grouped_final if i != non_target_id}

print("Target IDs (original mapping):", target_ids)
print("Non-target IDs (original mapping):", non_target_ids)
print("Grouped label2id:", label2id_grouped)

# ==== GROUPED LABELS: 8 target + 1 non_target (gộp 10 lớp 8–17) ====

# 1) Lấy danh sách 8 target theo thứ tự ID gốc 0..7
TARGET_LABELS = [name for name, idx in sorted(GESTURE_MAPPER.items(), key=lambda kv: kv[1]) if idx <= 7]

# 2) Định nghĩa nhãn gộp
GROUPED_LABELS = TARGET_LABELS + ['non_target']   # giữ nguyên 8 target, + 1 lớp non_target

# 3) Mapping NHÓM (ghi đè mapping cũ để dùng xuyên suốt pipeline)
label2id = {name: i for i, name in enumerate(GROUPED_LABELS)}
id2label = {i: name for name, i in label2id.items()}

NON_TARGET = 'non_target'
assert NON_TARGET in label2id, "Grouped mapping chưa có non_target!"
non_target_id = label2id[NON_TARGET]
target_ids = {i for i in id2label if i != non_target_id}

# 4) Tạo y_train theo NHÃN GỘP từ tên gesture gốc trong train_df
def to_grouped_label_from_name(name: str) -> str:
    return name if name in TARGET_LABELS else 'non_target'

train_df['gesture_grouped'] = train_df['gesture'].apply(to_grouped_label_from_name)
y_train = train_df['gesture_grouped'].map(label2id).values

# (Tuỳ biến) Nếu bạn có X_train (features) đã tạo sẵn:
# feature_cols = [c for c in X_train.columns if c != 'sequence_id']
# X_features = X_train[feature_cols]

# 5) In kiểm tra
print("TARGET_LABELS:", TARGET_LABELS)
print("GROUPED_LABELS:", GROUPED_LABELS)
print("label2id:", label2id)
print("Counts grouped:", train_df['gesture_grouped'].value_counts().to_dict())
print("OK: non_target_id =", non_target_id, "| target_ids =", target_ids)



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
    NUM_CLASSES = len(label2id) if 'label2id' in globals() else None

    
    # Feature columns
    ACC_COLS = ['acc_x', 'acc_y', 'acc_z']
    ROT_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # LightGBM hyperparameters
    LGBM_PARAMS = {
        'objective':'multiclass',
        'num_class': NUM_CLASSES if NUM_CLASSES is not None else 9,
        'boosting_type':'gbdt',
        'learning_rate':0.05,
        'n_estimators':2000,          # dùng early_stopping để dừng sớm
        'num_leaves':63,
        'max_depth':-1,
        'min_data_in_leaf':40,
        'feature_fraction':0.75,
        'bagging_fraction':0.75,
        'bagging_freq':1,
        'lambda_l1':1.0,
        'lambda_l2':2.0,
        'class_weight':'balanced',       # Suppress output verbosity
    }



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



from sklearn.metrics import f1_score
import numpy as np

def contest_score(y_true_ids, y_pred_ids, id2label, non_target_id):
    # Binary-F1: target vs non_target
    y_true_bin = (np.array(y_true_ids) != non_target_id).astype(int)
    y_pred_bin = (np.array(y_pred_ids) != non_target_id).astype(int)
    f1_binary = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)

    # Macro-F1: gộp tất cả non-target thành một lớp duy nhất
    f1_macro = f1_score(y_true_ids, y_pred_ids, average='macro', zero_division=0)

    return (f1_binary + f1_macro) / 2.0, f1_binary, f1_macro



# === Contest score: (Binary-F1 + Macro-F1)/2 ===
from sklearn.metrics import f1_score
import numpy as np

def contest_score(y_true_ids, y_pred_ids, id2label, non_target_id):
    y_true_bin = (np.array(y_true_ids) != non_target_id).astype(int)
    y_pred_bin = (np.array(y_pred_ids) != non_target_id).astype(int)
    f1_binary = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    f1_macro = f1_score(y_true_ids, y_pred_ids, average='macro', zero_division=0)
    return (f1_binary + f1_macro) / 2.0, f1_binary, f1_macro

def predict_with_tau(proba, non_target_id, target_ids, tau):
    y_pred = []
    for p in proba:
        best_target = max(target_ids, key=lambda k: p[k])
        if p[best_target] < tau:
            y_pred.append(non_target_id)
        else:
            y_pred.append(int(np.argmax(p)))
    return np.array(y_pred, dtype=int)

def tune_tau_for_cv(valid_y_true, valid_proba, non_target_id, target_ids, id2label):
    best = (None, -1, None, None)
    for tau in np.linspace(0.1, 0.9, 17):
        y_pred = predict_with_tau(valid_proba, non_target_id, target_ids, tau)
        score, f1_bin, f1_mac = contest_score(valid_y_true, y_pred, id2label, non_target_id)
        if score > best[1]:
            best = (tau, score, f1_bin, f1_mac)
    return best



# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_lightgbm_model():
    """Build and return a LightGBM model based on configuration"""
    model = LGBMClassifier(**Config.LGBM_PARAMS)
    return model

# =============================================================================
# MODEL TRAINING (CV + OOF proba + tau tuning)
# YÊU CẦU: các biến sau đã có trong scope trước khi gọi:
# - label2id, id2label: mapping nhãn
# - non_target_id: int id của lớp 'non_target'
# - target_ids: set các id lớp target (khác non_target_id)
# - Config.LGBM_PARAMS, Config.N_FOLDS, Config.SEED
# - DEVICE: 'cpu' hoặc 'gpu' (tuỳ bạn detect)
# - competition_metric: HÀM CHẤM ĐIỂM CHUẨN CONTEST (nếu chưa có, dùng contest_score bên dưới)
# =============================================================================

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import numpy as np

def _predict_with_tau(proba, non_target_id, target_ids, tau):
    """
    proba: ndarray (n_samples, n_classes)
    Luật: nếu max(p[target]) < tau -> non_target, else -> argmax toàn bộ
    """
    y_pred = []
    for p in proba:
        best_target_id = max(target_ids, key=lambda k: p[k]) if len(target_ids) > 0 else None
        best_target_p = p[best_target_id] if best_target_id is not None else 0.0
        if best_target_p < tau:
            y_pred.append(non_target_id)
        else:
            y_pred.append(int(np.argmax(p)))
    return np.array(y_pred, dtype=int)

def _tune_tau(y_true_ids, proba, non_target_id, target_ids, id2label):
    best = (None, -1.0, None, None)  # tau, score, f1_bin, f1_macro
    for tau in np.linspace(0.1, 0.9, 17):
        y_pred_ids = _predict_with_tau(proba, non_target_id, target_ids, tau)
        score, f1_bin, f1_mac = competition_metric(y_true_ids, y_pred_ids)
        if score > best[1]:
            best = (float(tau), float(score), float(f1_bin), float(f1_mac))
    return best  # (tau, score, f1_bin, f1_mac)

def train_lightgbm_models(X_train, y_train, subjects):
    """
    Train LightGBM với StratifiedGroupKFold, lưu OOF predict_proba,
    tune ngưỡng tau theo từng fold, tính OOF score theo tau* (median).
    Trả về:
        models: list các model theo fold
        cv_scores: list contest score từng fold (sau khi áp tau-fold)
        overall_score: OOF contest score với tau*
        tau_star: median của các tau theo fold (dùng cho inference)
        feature_cols: danh sách cột feature (để dùng lại ở predict)
    """

    print(f"Training LightGBM models with {Config.N_FOLDS}-fold cross-validation...")

    # 1) Chuẩn bị feature (loại sequence_id nếu có)
    feature_cols = [c for c in X_train.columns if c != 'sequence_id']
    X_features = X_train[feature_cols].copy()

    n_classes = len(np.unique(y_train))
    print(f"Feature columns: {len(feature_cols)} | Num classes: {n_classes}")
    print("Starting cross-validation...")

    # 2) CV setup
    cv = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    models = []
    cv_scores = []
    taus = []

    # OOF proba để tính điểm tổng thể với một tau*
    oof_proba = np.zeros((len(X_features), n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_features, y_train, groups=subjects), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_tr, X_va = X_features.iloc[tr_idx], X_features.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        print(f"Train size: {len(X_tr)}, Val size: {len(X_va)}")

        # 3) Khởi tạo model + device
        lgbm_params = Config.LGBM_PARAMS.copy()
        lgbm_params['device'] = DEVICE  # ví dụ: 'gpu' nếu bạn đã detect
        model = LGBMClassifier(**lgbm_params)

        # 4) Train với early_stopping (metric chỉ để dừng sớm)
        print(f"Training fold {fold} (Device: {DEVICE})...")
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            eval_names=['train', 'valid'],
            eval_metric=['multi_logloss', 'multi_error'],
            callbacks=[
                log_evaluation(period=25),
                early_stopping(stopping_rounds=100, verbose=True)
            ]
        )
        models.append(model)

        # 5) Lưu proba validation
        proba_va = model.predict_proba(X_va)  # (n_val, n_classes)
        oof_proba[va_idx] = proba_va

        # 6) Tune tau theo fold
        tau, score, f1_bin, f1_mac = _tune_tau(y_va, proba_va, non_target_id, target_ids, id2label)
        taus.append(tau)
        cv_scores.append(score)

        # 7) In điểm fold với tau tốt nhất của fold
        print(f"Fold {fold} | tau={tau:.2f} | score={score:.4f} | F1-bin={f1_bin:.4f} | F1-macro={f1_mac:.4f}")

    # 8) Tính OOF score với tau* = median(taus)
    tau_star = float(np.median(taus))
    y_oof_pred = _predict_with_tau(oof_proba, non_target_id, target_ids, tau_star)
    overall_score, overall_binary_f1, overall_macro_f1 = competition_metric(y_train, y_oof_pred)

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS (with tau*)")
    print(f"{'='*60}")
    print(f"Overall Competition Score: {overall_score:.4f} ± {np.std(cv_scores):.4f}")
    print(f"Overall Binary F1: {overall_binary_f1:.4f}")
    print(f"Overall Macro F1: {overall_macro_f1:.4f}")
    print(f"Fold scores (per-fold best tau): {['{:.4f}'.format(s) for s in cv_scores]}")
    print(f"tau* (median of folds): {tau_star:.2f}")
    print(f"{'='*60}")
    print("✓ Training completed successfully!")
    print("✓ Models ready for inference (use prob-averaging + tau*)")

    return models, cv_scores, overall_score, tau_star, feature_cols



# =============================================================================
# PREDICTION FUNCTION (with tau threshold & prob averaging) — PATCHED
# =============================================================================

def create_prediction_function(models, feature_cols, imu_cols, tau_star, label2id, id2label, non_target_label='non_target'):
    import numpy as np
    import polars as pl

    assert isinstance(models, (list, tuple)) and len(models) > 0, "models must be a non-empty list"
    assert isinstance(feature_cols, (list, tuple)) and len(feature_cols) > 0, "feature_cols must be provided"
    assert isinstance(imu_cols, (list, tuple)) and len(imu_cols) > 0, "imu_cols must be provided"
    assert non_target_label in label2id, "non_target label missing in label2id"

    non_target_id = int(label2id[non_target_label])
    target_ids = {int(k) for k in id2label.keys() if int(k) != non_target_id}

    # 18 NHÃN GỐC CỦA KAGGLE (chuẩn hóa format)
    ALLOWED_LABELS = [
        "Above ear - pull hair", "Cheek - pinch skin", "Eyebrow - pull hair", "Eyelash - pull hair",
        "Forehead - pull hairline", "Forehead - scratch", "Neck - pinch skin", "Neck - scratch",
        "Drink from bottle/cup", "Put on earphones", "Read book", "Scratch elbow", "Scratch palm",
        "Scratch with object", "Text on phone", "Touch back of head", "Write name in air", "Write name on leg"
    ]
    NON_TARGET_FALLBACK = "Drink from bottle/cup"  # chọn 1 nhãn non‑target hợp lệ làm mặc định

    def _predict_with_tau(proba_row, tau):
        best_target_id = max(target_ids, key=lambda k: proba_row[k]) if target_ids else None
        best_target_p = proba_row[best_target_id] if best_target_id is not None else 0.0
        if best_target_p < tau:
            return non_target_id
        return int(np.argmax(proba_row))

    def predict(sequence: pl.DataFrame, demographics: pl.DataFrame):
        # 1) filter IMU cols
        avail = sequence.columns
        use_cols = [c for c in imu_cols if c in avail]
        seq_pl = sequence.select(pl.col(use_cols))

        # 2) featurize
        feats = extract_comprehensive_features(seq_pl, demographics)   # -> pandas DF (1 row)

        # 3) align columns
        missing = [c for c in feature_cols if c not in feats.columns]
        for c in missing:
            feats[c] = 0.0
        X = feats[feature_cols]

        # 4) average probabilities
        proba_sum = None
        for m in models:
            p = m.predict_proba(X)  # (1, n_classes)
            proba_sum = p if proba_sum is None else (proba_sum + p)
        proba_avg = proba_sum / len(models)

        # 5) tau rule
        y_id = _predict_with_tau(proba_avg[0], tau_star)
        gesture = id2label[int(y_id)]

        # ==== MAP VỀ 18 NHÃN HỢP LỆ (SỬA LỖI FORMAT) ====
        # Nếu mô hình là 9 lớp (8 target + 'non_target'), khi ra 'non_target' => map về 1 nhãn hợp lệ.
        if gesture == non_target_label:
            gesture = NON_TARGET_FALLBACK
        # Bảo hiểm: nếu vì lý do gì nhãn không đúng 18 nhãn gốc, ép về fallback
        if gesture not in ALLOWED_LABELS:
            gesture = NON_TARGET_FALLBACK
        # =================================================

        return gesture

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
    """Load and prepare training/test features (IMU-only) + grouped labels"""
    print("Loading training data...")
    train_df = pl.read_csv(Config.TRAIN_PATH)
    train_demographics = pl.read_csv(Config.TRAIN_DEMOGRAPHICS_PATH)

    print("Loading test data...")
    test_df = pl.read_csv(Config.TEST_PATH)
    test_demographics = pl.read_csv(Config.TEST_DEMOGRAPHICS_PATH)

    # Common IMU columns (exclude ToF/Thermal)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    imu_cols = [c for c in common_cols if not (c.startswith('thm_') or c.startswith('tof_'))]

    print(f"✓ Using {len(imu_cols)} common IMU columns")
    print(f"✓ Train-only columns: {train_cols - test_cols}")
    print(f"✓ Test-only columns: {test_cols - train_cols}")

    # =========================
    # TRAIN FEATURE EXTRACTION
    # =========================
    print("Extracting features for training sequences...")
    train_features_list = []
    y_train_grouped = []
    train_subjects = []
    train_sequence_ids = []

    # Cần có subject + sequence_id + gesture để gán nhãn/nhóm
    base_train_cols = list(set(imu_cols) | {'subject','sequence_id','gesture'})
    train_sequences = train_df.select(pl.col(base_train_cols)).group_by('sequence_id', maintain_order=True)

    for seq_key, seq_pl in train_sequences:
        # Lấy sequence_id & subject
        sequence_id_val = seq_key[0] if isinstance(seq_key, tuple) else seq_key
        subject_id = seq_pl['subject'][0]
        subject_demo = train_demographics.filter(pl.col('subject') == subject_id)

        # IMU-only cho feature extraction
        imu_only_pl = seq_pl.select(pl.col(imu_cols))
        features_pd = extract_comprehensive_features(imu_only_pl, subject_demo)  # -> pandas DataFrame
        features_pd['sequence_id'] = sequence_id_val
        train_features_list.append(features_pd)

        # Nhãn gốc -> nhãn gộp
        gesture_name = seq_pl['gesture'][0]
        grouped_label = gesture_name if gesture_name in TARGET_LABELS else 'non_target'
        y_train_grouped.append(label2id[grouped_label])

        train_subjects.append(subject_id)
        train_sequence_ids.append(sequence_id_val)

    X_train = pd.concat(train_features_list, ignore_index=True)
    y_train = np.array(y_train_grouped, dtype=int)
    subjects = np.array(train_subjects)

    # =========================
    # TEST FEATURE EXTRACTION
    # =========================
    print("Extracting features for test sequences...")
    test_features_list = []
    test_sequence_ids = []

    base_test_cols = list(set(imu_cols) | {'subject','sequence_id'})
    test_sequences = test_df.select(pl.col(base_test_cols)).group_by('sequence_id', maintain_order=True)

    for seq_key, seq_pl in test_sequences:
        sequence_id_val = seq_key[0] if isinstance(seq_key, tuple) else seq_key
        subject_id = seq_pl['subject'][0]
        subject_demo = test_demographics.filter(pl.col('subject') == subject_id)

        # Lưu ý: ở test không có gesture
        imu_only_pl = seq_pl.select(pl.col(imu_cols))
        features_pd = extract_comprehensive_features(imu_only_pl, subject_demo)
        features_pd['sequence_id'] = sequence_id_val

        test_features_list.append(features_pd)
        test_sequence_ids.append(sequence_id_val)

    X_test = pd.concat(test_features_list, ignore_index=True)

    print(f"✓ Training features shape: {X_train.shape}")
    print(f"✓ Training labels shape: {y_train.shape}")
    print(f"✓ Test features shape: {X_test.shape}")
    print(f"✓ Number of features: {X_train.shape[1] - 1}")  # -1 for sequence_id

    # Bỏ sequence_id khỏi ma trận đặc trưng (giữ riêng để debug nếu muốn)
    X_train = X_train.drop(columns=['sequence_id'], errors='ignore')
    X_test  = X_test.drop(columns=['sequence_id'], errors='ignore')

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
# MAIN EXECUTION PIPELINE (compatible with tau* + grouped labels)
# =============================================================================

def main():
    X_train, y_train, subjects, X_test, test_sequence_ids, imu_cols = load_and_prepare_data()

    models, cv_scores, overall_score, tau_star, feature_cols = train_lightgbm_models(
        X_train=X_train,
        y_train=y_train,
        subjects=subjects
    )

    # Tạo predict_function (đủ tham số)
    predict_function = create_prediction_function(
        models=models,
        feature_cols=feature_cols,
        imu_cols=imu_cols,
        tau_star=tau_star,
        label2id=label2id,    # mapping GỘP (có 'non_target')
        id2label=id2label,    # mapping GỘP
        non_target_label='non_target'
    )
    print("predict_function is callable?", callable(predict_function))  # phải là True

    # Khởi tạo server (chỉ khi callable)
    server = setup_inference_server(predict_function)

    print("\n✓ Training completed successfully!")
    print(f"✓ Final CV Score (OOF, tau*): {overall_score:.4f}")
    return predict_function, models, cv_scores



if __name__ == "__main__":
    predict_function, trained_models, cv_scores = main()

