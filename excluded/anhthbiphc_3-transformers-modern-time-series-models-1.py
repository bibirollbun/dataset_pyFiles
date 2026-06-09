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


import os, gc, time, math, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42); np.random.seed(42); torch.manual_seed(42)



# Notebook 3 – Transformers & Modern Time-Series Models
# ------------------------------------------------------------
# Goal: Compare attention-based (Transformer/TST/Informer-like) vs CNN/TCN on IMU/ToF/Thermo sensor sequences
# Metrics: Macro-F1; Ablations over sequence length (window=50/100/200)
# Complexity: train time per epoch & #params
# ------------------------------------------------------------

import os, gc, time, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG = np.random.default_rng(42)
random.seed(42); torch.manual_seed(42); np.random.seed(42)




# =====================
# 0) DATA LOADING (Kaggle paths)
# =====================
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

# --- flexible merge keys
CAND_KEYS = ['id','ID','Id','record_id','sample_id','subject_id','participant_id','user_id','person_id','uid','pid','subject','ParticipantID','session_id']

def pick_key(left: pd.DataFrame, right: pd.DataFrame, candidates=CAND_KEYS):
    both = [k for k in candidates if k in left.columns and k in right.columns]
    if both: return both[0], both[0]
    l2 = [c for c in left.columns if ('id' in c.lower()) or ('subject' in c.lower()) or ('session' in c.lower())]
    r2 = [c for c in right.columns if ('id' in c.lower()) or ('subject' in c.lower()) or ('session' in c.lower())]
    inter = list(set(l2).intersection(r2))
    if inter: return inter[0], inter[0]
    raise KeyError("Không tìm được khóa merge giữa 2 bảng – hãy chỉnh lại tên cột khoá.")

lk, rk = pick_key(train_df, train_dem_df)
train_full = train_df.merge(train_dem_df, left_on=lk, right_on=rk, how='left')
lk_t, rk_t = pick_key(test_df, test_dem_df)
test_full = test_df.merge(test_dem_df, left_on=lk_t, right_on=rk_t, how='left')

# --- label detection
LABEL_CANDS = ['label','target','activity','activity_id','behavior','behavior_id','state','class']
label_map = {c.lower(): c for c in train_full.columns}
LABEL_COL = None
for k in LABEL_CANDS:
    if k in label_map: LABEL_COL = label_map[k]; break
if LABEL_COL is None:
    # heuristic: low-cardinality, non-id
    nn = len(train_full)
    opts = []
    for c in train_full.columns:
        lc = c.lower()
        if 'id' in lc or 'time' in lc or 'timestamp' in lc: continue
        u = train_full[c].nunique(dropna=True)
        if 2 <= u <= min(50, max(2, nn//10)):
            opts.append((c,u))
    opts.sort(key=lambda x: x[1])
    LABEL_COL = opts[0][0]
print('LABEL_COL =', LABEL_COL)

# --- time & grouping columns (for sequence build)
TIME_CANDS = ['timestamp','time','frame','step','t','ts']
GROUP_CANDS = [lk, 'session_id','trial_id','sequence_id', 'seq_id', 'record_id', 'sample_id', 'id']

def pick_time_col(df):
    for k in TIME_CANDS:
        for c in df.columns:
            if c.lower() == k: return c
            if k in c.lower(): return c
    return None

TIME_COL = pick_time_col(train_full)
GROUP_COL = None
for g in GROUP_CANDS:
    if isinstance(g,str) and g in train_full.columns:
        GROUP_COL = g; break
if TIME_COL is None:
    # fallback: assume implicit order
    TIME_COL = '___row_index__'
    train_full[TIME_COL] = np.arange(len(train_full))

print({'TIME_COL': TIME_COL, 'GROUP_COL': GROUP_COL})

# --- select sensor columns (numeric)
SENSOR_PREFIXES = ['ax','ay','az','gx','gy','gz','acc','gyro','mag','imu','tof','thm','thermo']
ALL_NUM = [c for c in train_full.columns if pd.api.types.is_numeric_dtype(train_full[c]) and c not in [LABEL_COL]]
SENSOR_COLS = []
for c in ALL_NUM:
    lc = c.lower()
    if any(lc.startswith(p) for p in SENSOR_PREFIXES):
        SENSOR_COLS.append(c)
if not SENSOR_COLS:
    # if dataset uses other names, fallback to all numeric except demographics-like
    SENSOR_COLS = [c for c in ALL_NUM if 'age' not in c.lower()]

print('n_sensor_cols =', len(SENSOR_COLS))

# Encode labels
le = LabelEncoder(); y_text = train_full[LABEL_COL].astype(str).values
y_all = le.fit_transform(y_text).astype(np.int64)
N_CLASS = len(le.classes_)




# =====================
# 1) BUILD SEQUENCES (sliding window)
# =====================

def build_windows(df, sensor_cols, labels, time_col, group_col, win=100, stride=None):
    if stride is None: stride = win//2
    X_list, y_list = [], []
    if group_col is None:
        df_sorted = df.sort_values(time_col)
        arr = df_sorted[sensor_cols].values
        yarr = labels[df_sorted.index]
        for s in range(0, len(df_sorted)-win+1, stride):
            X_list.append(arr[s:s+win])
            # majority label in window
            yy = yarr[s:s+win]
            vals, cnts = np.unique(yy, return_counts=True)
            y_list.append(vals[np.argmax(cnts)])
    else:
        for gid, gdf in df.groupby(group_col):
            gdf = gdf.sort_values(time_col)
            arr = gdf[sensor_cols].values
            yarr = labels[gdf.index]
            if len(gdf) < win: continue
            for s in range(0, len(gdf)-win+1, stride):
                X_list.append(arr[s:s+win])
                yy = yarr[s:s+win]
                vals, cnts = np.unique(yy, return_counts=True)
                y_list.append(vals[np.argmax(cnts)])
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y




# =====================
# 2) DATASET / DATALOADER
# =====================
class SeqDataset(Dataset):
    def __init__(self, X, y, fit_scaler=False, scaler=None):
        # X: [N, L, C]
        self.X = X
        self.y = y
        self.scaler = scaler if scaler is not None else StandardScaler()
        if fit_scaler:
            X2d = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X2d)
        X2d = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X2d).astype(np.float32)
        self.X = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])




# =====================
# 3) MODELS: TCN, Vanilla Transformer, TST, Informer-like
# =====================
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.1):
        super().__init__()
        # pad để giữ nguyên chiều dài: pad = dilation*(k-1)/2  (yêu cầu k lẻ)
        assert k % 2 == 1, "Kernel size k should be odd to keep SAME length."
        pad = (k - 1) * d // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(), nn.Dropout(p)
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: [B,C,L]
        y = self.net(x)    # [B,out_ch,L]  (giữ nguyên L)
        return F.relu(y + self.down(x))


class TCN(nn.Module):
    def __init__(self, in_dim, n_class, chs=[64,64,128], ks=3, p=0.1):
        super().__init__()
        blocks = []
        c = in_dim
        d = 1
        for ch in chs:
            blocks.append(TemporalConvBlock(c, ch, k=ks, d=d, p=p))
            c = ch; d *= 2
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, n_class))
    def forward(self, x):
        # x: [B,L,C] → [B,C,L]
        x = x.transpose(1,2)
        z = self.tcn(x)
        out = self.head(z)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        # x: [B,L,D]
        L = x.size(1)
        return x + self.pe[:, :L, :]

class VanillaTransformer(nn.Module):
    def __init__(self, in_dim, n_class, d_model=128, nhead=4, num_layers=2, dim_ff=256, p=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, n_class))
    def forward(self, x):
        # x: [B,L,C]
        z = self.proj(x)
        z = self.pe(z)
        z = self.enc(z)           # [B,L,D]
        z = z.transpose(1,2)      # [B,D,L]
        out = self.head(z)
        return out

# Time Series Transformer (TST) – channel/token projection + class token
class TST(nn.Module):
    def __init__(self, in_dim, n_class, d_model=128, nhead=4, num_layers=2, dim_ff=256, p=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_class)
    def forward(self, x):
        B = x.size(0)
        z = self.proj(x)
        z = self.pe(z)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1)  # [B,1+L,D]
        z = self.enc(z)
        out = self.head(z[:,0])
        return out

# Informer-like (ProbSparse-style approximation): keep top-k keys per head via attention masking
class InformerLite(nn.Module):
    def __init__(self, in_dim, n_class, d_model=128, nhead=4, num_layers=2, dim_ff=256, p=0.1, topk=32):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=p, batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(d_model, n_class))
        self.topk = topk
    def forward(self, x):
        z = self.proj(x)
        z = self.pe(z)
        # approximate sparsity by randomly masking tokens except top-k by variance (cheap proxy)
        with torch.no_grad():
            var = z.var(dim=-1)              # [B,L]
            k = min(self.topk, z.size(1))
            idx = var.topk(k, dim=1).indices # [B,k]
            mask = torch.zeros(z.size(0), z.size(1), dtype=torch.bool, device=z.device)
            for b in range(z.size(0)):
                mask[b, idx[b]] = True
        # keep tokens
        kept = []
        for b in range(z.size(0)):
            kept.append(z[b][mask[b]])
        # pad back to max k
        kmax = max(t.size(0) for t in kept)
        z2 = torch.zeros(z.size(0), kmax, z.size(-1), device=z.device)
        for b,t in enumerate(kept): z2[b,:t.size(0)] = t
        for lyr in self.layers:
            z2 = lyr(z2)
        z2 = z2.transpose(1,2)
        out = self.head(z2)
        return out




# =====================
# 4) TRAIN LOOP & UTILS
# =====================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in dl:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(yb.cpu().numpy()); y_pred.extend(preds)
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def train_model(model, dl_tr, dl_va, epochs=10, lr=3e-4):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1, best_state = -1, None
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
        f1 = evaluate(model, dl_va)
        # print(f"Epoch {ep}: val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k,v in best_state.items()})
    return best_f1, train_time, count_params(model)




# =====================
# 5) EXPERIMENTS: window sizes & models
# =====================
WINDOWS = [50, 100, 200]
EPOCHS = 8
BATCH = 128
RESULTS = []

for WIN in WINDOWS:
    print(f"\n===== Window {WIN} =====")
    X, y = build_windows(train_full, SENSOR_COLS, y_all, TIME_COL, GROUP_COL, win=WIN, stride=WIN//2)
    # small stratified split
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    ds_tr = SeqDataset(Xtr, ytr, fit_scaler=True)
    ds_va = SeqDataset(Xva, yva, scaler=ds_tr.scaler)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

    C = X.shape[-1]

    # TCN
    m_tcn = TCN(in_dim=C, n_class=N_CLASS)
    f1_tcn, t_tcn, p_tcn = train_model(m_tcn, dl_tr, dl_va, epochs=EPOCHS, lr=1e-3)
    RESULTS.append({'model':'TCN', 'window':WIN, 'f1':f1_tcn, 'time_s':t_tcn, 'params':p_tcn})

    # Vanilla Transformer
    m_tf = VanillaTransformer(in_dim=C, n_class=N_CLASS, d_model=128, nhead=4, num_layers=2)
    f1_tf, t_tf, p_tf = train_model(m_tf, dl_tr, dl_va, epochs=EPOCHS, lr=3e-4)
    RESULTS.append({'model':'Transformer', 'window':WIN, 'f1':f1_tf, 'time_s':t_tf, 'params':p_tf})

    # TST
    m_tst = TST(in_dim=C, n_class=N_CLASS, d_model=128, nhead=4, num_layers=2)
    f1_tst, t_tst, p_tst = train_model(m_tst, dl_tr, dl_va, epochs=EPOCHS, lr=3e-4)
    RESULTS.append({'model':'TST', 'window':WIN, 'f1':f1_tst, 'time_s':t_tst, 'params':p_tst})

    # Informer-like (lite)
    m_inf = InformerLite(in_dim=C, n_class=N_CLASS, d_model=128, nhead=4, num_layers=2, topk=min(64, WIN))
    f1_inf, t_inf, p_inf = train_model(m_inf, dl_tr, dl_va, epochs=EPOCHS, lr=3e-4)
    RESULTS.append({'model':'InformerLite', 'window':WIN, 'f1':f1_inf, 'time_s':t_inf, 'params':p_inf})

    gc.collect(); torch.cuda.empty_cache()

res_df = pd.DataFrame(RESULTS)
print("\nResults (macro-F1):\n", res_df.pivot(index='window', columns='model', values='f1'))
print("\nComplexity (time_s & #params):\n", res_df.groupby(['model','window'])[['time_s','params']].mean())

# Plot F1 vs window
plt.figure(figsize=(6,4))
for m in res_df['model'].unique():
    d = res_df[res_df['model']==m]
    plt.plot(d['window'], d['f1'], marker='o', label=m)
plt.xlabel('Window length'); plt.ylabel('Macro-F1'); plt.title('Scaling window length')
plt.legend(); plt.tight_layout(); plt.show()

print("\nNotebook 3 – Done.")



import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def benchmark_inference(model, dl, device=None, warmup=1):
    device = device or DEVICE
    model.eval()
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            for xb, _ in dl:
                xb = xb.to(device)
                _ = model(xb)
                break
    # measure
    n_samples = 0
    t0 = time.time()
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            _ = model(xb)
            n_samples += xb.size(0)
    t1 = time.time()
    per_sample_ms = (t1 - t0) / max(1, n_samples) * 1000.0
    return per_sample_ms

@torch.no_grad()
def eval_metrics(model, dl, device=None):
    device = device or DEVICE
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    acc   = accuracy_score(y_true, y_pred)
    if N_CLASS == 2:
        binary = f1_score(y_true, y_pred, average='binary')
    else:
        binary = np.nan
    return macro, acc, binary

def pretty_millions(n_params):
    return round(float(n_params) / 1e6, 3)

# =========================
# Thu thập & in bảng kết quả
# =========================
rows = []

# Bạn có thể tùy biến chuỗi mô tả feature:
features_used_desc = f"All sensors ({len(SENSOR_COLS)} ch)"

# Ví dụ lặp lại train nhanh ở đây, hoặc nếu bạn đã có các biến
# f1_xx, t_xx, p_xx và model đã train (m_xx) thì chỉ cần gọi eval/benchmark.
# Dưới đây minh họa cho 4 model đã train trong vòng lặp window WIN:

models_pack = [
    ("TCN",           m_tcn,  f1_tcn,  t_tcn,  p_tcn,  "AdamW"),
    ("Transformer",   m_tf,   f1_tf,   t_tf,   p_tf,   "AdamW"),
    ("TST",           m_tst,  f1_tst,  t_tst,  p_tst,  "AdamW"),
    ("InformerLite",  m_inf,  f1_inf,  t_inf,  p_inf,  "AdamW"),
]

for name, model, f1_val_returned, train_time_s, n_params, optim_name in models_pack:
    # Đo lại metrics và thời gian suy luận cho chuẩn
    macro, val_acc, binary = eval_metrics(model, dl_va, DEVICE)
    infer_ms = benchmark_inference(model, dl_va, DEVICE)

    row = {
        "model":               name,
        "features_used":       features_used_desc,
        "window_size":         WIN,
        "optimizer/solver":    optim_name,
        "params (M)":          pretty_millions(n_params),
        "Binary":              round(binary, 4) if not np.isnan(binary) else np.nan,
        "Macro":               round(macro, 4),
        "Final Score":         round(macro, 4),   # bạn có thể đổi công thức nếu cần
        "val_acc":             round(val_acc, 4),
        "train_time":          round(float(train_time_s), 2),       # giây
        "inference_time":      round(float(infer_ms), 3),           # ms / sample
    }
    rows.append(row)

report_df = pd.DataFrame(rows, columns=[
    "model", "features_used", "window_size", "optimizer/solver", "params (M)",
    "Binary", "Macro", "Final Score", "val_acc", "train_time", "inference_time"
])

# Sắp xếp hiển thị (tuỳ bạn)
report_df = report_df.sort_values(["window_size", "Final Score"], ascending=[True, False]).reset_index(drop=True)
print(report_df)

# Nếu muốn đẹp hơn:
try:
    from IPython.display import display
    display(report_df)
except:
    pass



# =========================
# Pivot Table để so sánh theo window & model
# =========================

# 1) Macro F1: hàng = window_size, cột = model
pivot_macro = report_df.pivot_table(
    index="window_size", columns="model", values="Macro", aggfunc="mean"
)

# 2) Complexity: train_time & params
pivot_complexity = report_df.pivot_table(
    index="model", columns="window_size", values=["train_time","params (M)"], aggfunc="mean"
)

print("\n=== Pivot Macro F1 ===")
display(pivot_macro)

print("\n=== Pivot Complexity (train_time & params) ===")
display(pivot_complexity)


