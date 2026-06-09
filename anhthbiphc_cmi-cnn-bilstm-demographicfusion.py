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


# Detect Behavior with Sensor Data – CNN + Bi-LSTM + Demographics
# ------------------------------------------------------------------
# This is a minimally-intrusive revision of your original notebook.
# The only functional addition is that the seven demographic/anthro-
# pometric columns from train_demographics.csv are merged onto every
# row of the sensor frame and treated as extra numeric channels.
# Nothing else in the pipeline changes, so you can reuse previous
# hyper-parameters and checkpoints if desired.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization,
    LSTM, Bidirectional, GlobalAveragePooling1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import polars as pl
import kaggle_evaluation.cmi_inference_server  # noqa: F401   | Kaggle runner hook

print("Imports loaded")

# ------------------------------------------------------------------
# 1.  LOAD TRAIN SENSOR DATA + DEMOGRAPHICS
# ------------------------------------------------------------------
print("Loading sensor dataset …")
root = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

df = pd.read_csv(f"{root}/train.csv")
print(f"Loaded {len(df):,} rows of sensor frames")

# --- NEW: merge participant demographics on the key `subject` --------
print("Merging demographic attributes …")
demographics = pd.read_csv(f"{root}/train_demographics.csv")
df = df.merge(demographics, on="subject", how="left")

# ------------------------------------------------------------------
# 2.  LABEL-ENCODE GESTURE TARGET
# ------------------------------------------------------------------
label_encoder = LabelEncoder()
df["gesture"] = label_encoder.fit_transform(df["gesture"].astype(str))
np.save("gesture_classes.npy", label_encoder.classes_)

print("Gesture label mapping:")
for idx, lab in enumerate(label_encoder.classes_):
    print(f"  {idx}: {lab}")

# ------------------------------------------------------------------
# 3.  FEATURE LIST CONSTRUCTION
# ------------------------------------------------------------------
# Optionally skip thermal/TOF values → set to False to use them.

drop_thermal_and_tof = False

excluded_cols = {
    "gesture", "sequence_type", "behavior", "orientation",  # train-only targets
    "row_id", "subject", "phase",                            # meta
    "sequence_id", "sequence_counter"                         # ids
}

thermal_tof_cols = [c for c in df.columns if c.startswith(("thm_", "tof_"))]

if drop_thermal_and_tof:
    excluded_cols.update(thermal_tof_cols)
    print(f"Ignoring {len(thermal_tof_cols)} thermopile/TOF channels → set drop_thermal_and_tof=False to use them.")

# --- NEW: demographic numeric columns --------------------------------
demographic_cols = [
    "adult_child", "age", "sex", "handedness",
    "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm",
]

# Combine sensor + demographic feature list
feature_cols = [c for c in df.columns if c not in excluded_cols]
print(f"Using {len(feature_cols)} feature columns for training, including demographics:")
print(sorted(feature_cols)[:15], "…")

# Check missing values
nan_total = df[feature_cols].isna().sum().sum()
print(f"Total NaNs inside feature matrix: {nan_total:,}")

# ------------------------------------------------------------------
# 4.  SEQUENCE BUILDING HELPERS
# ------------------------------------------------------------------

def preprocess_sequence(df_seq: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Fill→scale a *single* sequence dataframe and return float32 numpy."""
    data = df_seq[feature_columns].copy()
    data = data.ffill().bfill().fillna(0.0)
    scaled = StandardScaler().fit_transform(data)   # per-sequence scaler (unchanged)
    return scaled.astype("float32")

print("Constructing padded tensor dataset …")
seq_groups = df.groupby("sequence_id")

X, seq_lengths = [], []
for i, (_, seq) in enumerate(seq_groups):
    if i and i % 500 == 0:
        print(f"  processed {i} sequences …")
    arr = preprocess_sequence(seq, feature_cols)
    X.append(arr)
    seq_lengths.append(arr.shape[0])

pad_len = int(np.percentile(seq_lengths, 90))
print(f"90th-percentile length = {pad_len} → fixed pad length chosen")
np.save("sequence_maxlen.npy", pad_len)

X = pad_sequences(X, maxlen=pad_len, dtype="float32", padding="post", truncating="post")

y = seq_groups["gesture"].first().values
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes=num_classes)

# ------------------------------------------------------------------
# 5.  TRAIN/VAL SPLIT & MODEL
# ------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Building CNN-BiLSTM model …")
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(pad_len, X_train.shape[2])),
    BatchNormalization(),
    Conv1D(64, 3, activation="relu"),
    MaxPooling1D(2),
    Dropout(0.30),

    Conv1D(128, 5, activation="relu"),
    BatchNormalization(),
    Conv1D(128, 5, activation="relu"),
    MaxPooling1D(2),
    Dropout(0.30),

    Conv1D(256, 7, activation="relu"),
    BatchNormalization(),
    Conv1D(256, 7, activation="relu"),
    MaxPooling1D(2),
    Dropout(0.40),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.40),

    GlobalAveragePooling1D(),

    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.50),
    Dense(256, activation="relu"),
    Dropout(0.30),

    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("Training …")
callbacks = [
    ReduceLROnPlateau(patience=4, factor=0.5, verbose=1),
    EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
]
model.fit(X_train, y_train, epochs=60, batch_size=128,
          validation_data=(X_val, y_val), callbacks=callbacks)

model.save("gesture_cnn_model.h5")
print("Training complete; model saved → gesture_cnn_model.h5")

# ------------------------------------------------------------------
# 6.  LOCAL VALIDATION METRIC (Macro F1 / Binary F1)
# ------------------------------------------------------------------
print("Computing validation metrics (Macro F1 / Binary F1) …")

from sklearn.metrics import f1_score

# 1) Dự đoán trên tập validation
probs_val = model.predict(X_val, verbose=0)
labels_val_pred = np.argmax(probs_val, axis=1)
labels_val_true = np.argmax(y_val, axis=1)

# 2) Load tên lớp để mapping
cls = np.load("gesture_classes.npy", allow_pickle=True)

# 2a) Macro F1 (đa lớp)
overall_macro_f1 = f1_score(labels_val_true, labels_val_pred,
                            average="macro", zero_division=0.0)

# 2b) Binary F1 (BFRB vs non-BFRB)
BFRB_NAMES = {
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
}
idx_is_bfrb = np.array([1 if name in BFRB_NAMES else 0 for name in cls], dtype=int)
y_true_bin = idx_is_bfrb[labels_val_true]
y_pred_bin = idx_is_bfrb[labels_val_pred]
overall_binary_f1 = f1_score(y_true_bin, y_pred_bin,
                             average="binary", zero_division=0.0)

# 3) In REPORT
print("=" * 59)
print("VALIDATION RESULTS")
print("=" * 59)
print(f"Overall Binary F1: {overall_binary_f1:.4f}")
print(f"Overall Macro F1:  {overall_macro_f1:.4f}")
print("=" * 59)


# ------------------------------------------------------------------
# 7.  INFERENCE HELPER
# ------------------------------------------------------------------

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Kaggle inference signature: returns predicted gesture string."""
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    seq_df = seq_df.merge(demo_df, on="subject", how="left")

    arr = preprocess_sequence(seq_df, feature_cols)
    maxlen = int(np.load("sequence_maxlen.npy"))
    padded = pad_sequences([arr], maxlen=maxlen, dtype="float32", padding="post", truncating="post")

    mdl = load_model("gesture_cnn_model.h5")
    probs = mdl.predict(padded, verbose=0)
    idx = int(np.argmax(probs, axis=1)[0])
    classes = np.load("gesture_classes.npy", allow_pickle=True)
    return str(classes[idx])


# Launch inference server
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

