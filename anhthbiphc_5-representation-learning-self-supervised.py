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


# Notebook 5 – Representation Learning & Self-Supervised
# ------------------------------------------------------------
# Mục tiêu: chứng minh lợi ích của pretraining/SSL cho dữ liệu cảm biến.
# Chọn 2 hướng nhẹ và khả thi:
#   (A) SimCLR cho time-series (augment → encoder → projection → NT-Xent)
#   (B) Masked Modeling (MAE-lite cho chuỗi) – che một phần timestep rồi tái tạo
# Đánh giá:
#   • Linear probe (freeze encoder → LR/MLP)
#   • Fine-tune vs From-scratch
#   • Transfer cross-subject: pretrain trên nhóm subject A, fine-tune/đánh giá nhóm B
# Báo cáo:
#   features_used | window_size | optimizer/solver | params (M) | Binary | Macro | Final Score | val_acc | train_time | inference_time
# ------------------------------------------------------------

import os, gc, time, math, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)




# =====================
# 0) DATA LOADING (CSV thật)
# =====================
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_dem_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_dem_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

# --- flexible merge ---
CAND_KEYS = ['id','ID','Id','record_id','sample_id','subject_id','participant_id','user_id','person_id','uid','pid','subject','ParticipantID','session_id']

def pick_key(left: pd.DataFrame, right: pd.DataFrame, candidates=CAND_KEYS):
    both = [k for k in candidates if k in left.columns and k in right.columns]
    if both: return both[0], both[0]
    l2 = [c for c in left.columns if any(t in c.lower() for t in ['id','subject','session'])]
    r2 = [c for c in right.columns if any(t in c.lower() for t in ['id','subject','session'])]
    inter = list(set(l2).intersection(r2))
    if inter: return inter[0], inter[0]
    raise KeyError('Không tìm được khóa merge. Hãy kiểm tra cột id/subject.')

lk, rk = pick_key(train_df, train_dem_df)
train_full = train_df.merge(train_dem_df, left_on=lk, right_on=rk, how='left')
lk_t, rk_t = pick_key(test_df, test_dem_df)
test_full = test_df.merge(test_dem_df, left_on=lk_t, right_on=rk_t, how='left')

# --- detect label & group (subject) & time col
LABEL_CANDS = ['label','target','activity','activity_id','behavior','behavior_id','state','class']
label_map = {c.lower(): c for c in train_full.columns}
LABEL_COL = None
for k in LABEL_CANDS:
    if k in label_map: LABEL_COL = label_map[k]; break
if LABEL_COL is None:
    # fallback heuristic
    nn = len(train_full); opts = []
    for c in train_full.columns:
        lc = c.lower()
        if 'id' in lc or 'time' in lc or 'timestamp' in lc: continue
        u = train_full[c].nunique(dropna=True)
        if 2 <= u <= min(50, max(2, nn//10)):
            opts.append((c,u))
    opts.sort(key=lambda x: x[1]); LABEL_COL = opts[0][0]

# subject/group
SUBJ_COL = None
for k in ['subject_id','participant_id','user_id','person_id','subject','ParticipantID', lk]:
    if isinstance(k,str) and k in train_full.columns:
        SUBJ_COL = k; break

# time
TIME_COL = None
for k in ['timestamp','time','frame','step','t','ts']:
    for c in train_full.columns:
        if c.lower()==k or k in c.lower(): TIME_COL = c; break
    if TIME_COL: break
if TIME_COL is None:
    TIME_COL = '___row_index__'; train_full[TIME_COL] = np.arange(len(train_full))

# sensor cols
SENSOR_PREFIXES = ['ax','ay','az','gx','gy','gz','acc','gyro','mag','imu','tof','thm','thermo']
ALL_NUM = [c for c in train_full.columns if pd.api.types.is_numeric_dtype(train_full[c]) and c!=LABEL_COL]
SENSOR_COLS = [c for c in ALL_NUM if any(c.lower().startswith(p) for p in SENSOR_PREFIXES)]
if not SENSOR_COLS:
    SENSOR_COLS = [c for c in ALL_NUM if 'age' not in c.lower()]

# labels
le = LabelEncoder(); y_all = le.fit_transform(train_full[LABEL_COL].astype(str).values).astype(np.int64)
N_CLASS = len(le.classes_)

print({'LABEL_COL': LABEL_COL, 'SUBJ_COL': SUBJ_COL, 'TIME_COL': TIME_COL, 'n_sensors': len(SENSOR_COLS), 'n_class': N_CLASS})




# =====================
# 1) BUILD SEQUENCES & CROSS-SUBJECT SPLIT
# =====================

def build_windows(df, sensor_cols, labels, time_col, group_col, win=100, stride=None):
    if stride is None: stride = win//2
    X_list, y_list, g_list = [], [], []
    if group_col is None:
        df_sorted = df.sort_values(time_col)
        arr = df_sorted[sensor_cols].values
        yarr = labels[df_sorted.index]
        for s in range(0, len(df_sorted)-win+1, stride):
            X_list.append(arr[s:s+win])
            yy = yarr[s:s+win]; vals, cnts = np.unique(yy, return_counts=True)
            y_list.append(vals[np.argmax(cnts)])
            g_list.append(0)
    else:
        for gid, gdf in df.groupby(group_col):
            gdf = gdf.sort_values(time_col)
            if len(gdf) < win: continue
            arr = gdf[sensor_cols].values
            yarr = labels[gdf.index]
            for s in range(0, len(gdf)-win+1, stride):
                X_list.append(arr[s:s+win])
                yy = yarr[s:s+win]; vals, cnts = np.unique(yy, return_counts=True)
                y_list.append(vals[np.argmax(cnts)])
                g_list.append(gid)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list)
    return X, y, g

WINDOW = 100   # có thể thử [50,100,200]; để nhanh chọn 100
STRIDE = WINDOW//2
Xseq, yseq, gseq = build_windows(train_full, SENSOR_COLS, y_all, TIME_COL, SUBJ_COL, win=WINDOW, stride=STRIDE)

# standardize theo channel
scaler = StandardScaler()
X2d = Xseq.reshape(-1, Xseq.shape[-1])
X2d = scaler.fit_transform(X2d).astype(np.float32)
Xseq = X2d.reshape(Xseq.shape[0], Xseq.shape[1], Xseq.shape[2])

# ⬇️ THÊM VÀO ĐÂY
Xseq = np.nan_to_num(Xseq, nan=0.0, posinf=0.0, neginf=0.0)
mask_ok = ~np.isnan(Xseq).any(axis=(1,2))
Xseq, yseq, gseq = Xseq[mask_ok], yseq[mask_ok], gseq[mask_ok]

# split cross-subject: 80% subject pretrain, 20% subject transfer
uniq_subj = np.unique(gseq)
if len(uniq_subj) > 1:
    src_subj, tgt_subj = train_test_split(uniq_subj, test_size=0.2, random_state=SEED)
else:
    src_subj, tgt_subj = uniq_subj, uniq_subj  # không có subject → coi như cùng nhóm
src_mask = np.isin(gseq, src_subj)
tgt_mask = np.isin(gseq, tgt_subj)

Xs, ys = Xseq[src_mask], yseq[src_mask]
Xt, yt = Xseq[tgt_mask], yseq[tgt_mask]

# target: split nhỏ để fine-tune & eval
idx_tr, idx_va = train_test_split(np.arange(len(yt)), test_size=0.5, stratify=yt, random_state=SEED)
Xt_tr, yt_tr = Xt[idx_tr], yt[idx_tr]
Xt_va, yt_va = Xt[idx_va], yt[idx_va]

print({'pretrain_src_windows': len(Xs), 'transfer_train': len(Xt_tr), 'transfer_val': len(Xt_va)})




# =====================
# 2) DATASETS
# =====================
class SSLDataset(Dataset):
    def __init__(self, X, y=None, aug=True):
        self.X = X
        self.y = y
        self.aug = aug

    def _jitter(self, x, sigma=0.02):
        return x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)

    def _scaling(self, x, sigma=0.1):
        f = np.random.normal(1.0, sigma, size=(x.shape[-1],)).astype(np.float32)
        return x * f

    def _time_mask(self, x, p=0.1):
        x = x.copy()
        L = x.shape[0]
        m = max(1, int(L * p))
        s = np.random.randint(0, L - m + 1)
        x[s:s + m] = 0.0
        return x

    def _augment_pair(self, x):
        a = self._jitter(self._scaling(x))
        b = self._time_mask(self._jitter(x))
        return a, b

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.aug:
            x1, x2 = self._augment_pair(x)
            x1 = np.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
            x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.from_numpy(x1), torch.from_numpy(x2), -1 if self.y is None else int(self.y[idx])
        else:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.from_numpy(x), -1 if self.y is None else int(self.y[idx])

    def __len__(self):
        return self.X.shape[0]

class ClsDataset(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), int(self.y[i])

BATCH_SSL = 128
BATCH_CLS = 128




# =====================
# 3) ENCODER & HEADS
# =====================
class TCNEncoder(nn.Module):
    def __init__(self, in_ch, hidden=128, emb=128, ks=5):
        super().__init__()
        pad = (ks-1)//2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, ks, padding=pad), nn.ReLU(), nn.Conv1d(64, 64, ks, padding=pad), nn.ReLU(),
            nn.Conv1d(64, 128, ks, padding=pad), nn.ReLU(), nn.Conv1d(128, hidden, ks, padding=pad), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden, emb)
    def forward(self, x):  # x: [B,L,C]
        x = x.transpose(1,2)
        z = self.net(x)
        z = self.pool(z).squeeze(-1)
        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
    def forward(self, z): return self.net(z)

class ClassifierHead(nn.Module):
    def __init__(self, d, n_class):
        super().__init__(); self.fc = nn.Linear(d, n_class)
    def forward(self, z): return self.fc(z)




# =====================
# 4) SIMCLR PRETRAIN
# =====================
class NTXentLoss(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__(); self.temp=temp
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
        B = z1.size(0)
        z = torch.cat([z1,z2], dim=0)  # [2B, d]
        sim = torch.mm(z, z.t()) / self.temp
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        targets = torch.arange(B, device=z.device)
        targets = torch.cat([targets+B, targets], dim=0)
        loss = F.cross_entropy(sim, targets)
        return loss

def simclr_pretrain(Xsrc, epochs=10, lr=1e-3):
    ds = SSLDataset(Xsrc, aug=True)
    dl = DataLoader(ds, batch_size=BATCH_SSL, shuffle=True, num_workers=0, drop_last=True)
    enc = TCNEncoder(in_ch=Xsrc.shape[-1], emb=128)
    head = ProjectionHead(128)
    enc.to(DEVICE); head.to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=lr)
    crit = NTXentLoss(temp=0.2)
    t0 = time.time()
    for ep in range(1, epochs+1):
        enc.train(); head.train()
        for x1, x2, _ in dl:
            x1 = x1.to(DEVICE); x2 = x2.to(DEVICE)
            opt.zero_grad()
            z1 = head(enc(x1)); z2 = head(enc(x2))
            loss = crit(z1, z2)
            loss.backward(); opt.step()
    t = time.time()-t0
    params = sum(p.numel() for p in enc.parameters() if p.requires_grad) + \
             sum(p.numel() for p in head.parameters() if p.requires_grad)
    return enc, t, params





# =====================
# 5) MASKED MODELING (MAE-lite)
# =====================
class MaskedAutoencoder(nn.Module):
    def __init__(self, in_ch, d=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch,64,5,padding=2), nn.ReLU(),
            nn.Conv1d(64,128,5,padding=2), nn.ReLU(),
            nn.Conv1d(128,d,5,padding=2), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(d,128,5,padding=2), nn.ReLU(),
            nn.Conv1d(128,64,5,padding=2), nn.ReLU(),
            nn.Conv1d(64,in_ch,5,padding=2)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x, mask=None):  # x: [B,L,C]
        x = x.transpose(1,2)
        if mask is not None:
            x = x * mask.transpose(1,2)
        z = self.enc(x)
        recon = self.dec(z).transpose(1,2)
        emb = self.pool(z).squeeze(-1)
        return recon, emb

def mae_pretrain(Xsrc, mask_p=0.3, epochs=8, lr=1e-3):
    class MAEDataset(Dataset):
        def __init__(self, X): self.X=X
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return torch.from_numpy(self.X[i])
    ds = MAEDataset(Xsrc)
    dl = DataLoader(ds, batch_size=BATCH_SSL, shuffle=True, num_workers=0)
    mae = MaskedAutoencoder(in_ch=Xsrc.shape[-1], d=128).to(DEVICE)
    opt = torch.optim.AdamW(mae.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(1, epochs+1):
        mae.train()
        for xb in dl:
            xb = xb.to(DEVICE)
            B,L,C = xb.shape
            mask = (torch.rand(B,L,1, device=xb.device) > mask_p).float()
            opt.zero_grad()
            recon, _ = mae(xb, mask)
            loss = F.mse_loss(recon, xb)
            loss.backward(); opt.step()
    t = time.time()-t0
    params = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    return mae, t, params




# =====================
# 6) LINEAR PROBE, FINE-TUNE, FROM-SCRATCH
# =====================
@torch.no_grad()
def embed_with_encoder(encoder, X):
    encoder.eval()
    Z = []
    dl = DataLoader(ClsDataset(X, np.zeros(len(X))), batch_size=BATCH_CLS, shuffle=False)
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        if isinstance(encoder, MaskedAutoencoder):
            _, emb = encoder(xb, None)
        else:
            emb = encoder(xb)
        Z.append(emb.detach().cpu().numpy())
    Z = np.concatenate(Z, axis=0)

    # ⬇️ Thêm dòng này để loại bỏ NaN/Inf
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    return Z


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def linear_probe(encoder, Xtr, ytr, Xva, yva):
    Ztr = embed_with_encoder(encoder, Xtr)
    Zva = embed_with_encoder(encoder, Xva)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=1000, solver='saga')
    )
    t0 = time.time(); clf.fit(Ztr, ytr); t = time.time()-t0
    ypred = clf.predict(Zva)
    macro = f1_score(yva, ypred, average='macro'); acc = accuracy_score(yva, ypred)
    return macro, acc, t, clf


class ClsFineTune(nn.Module):
    def __init__(self, encoder, n_class):
        super().__init__(); self.encoder = encoder; self.head = ClassifierHead(128, n_class)
    def forward(self, x):
        if isinstance(self.encoder, MaskedAutoencoder):
            _, z = self.encoder(x, None)
        else:
            z = self.encoder(x)
        return self.head(z)

def train_classifier(model, Xtr, ytr, Xva, yva, epochs=10, lr=1e-3, freeze_encoder=False):
    ds_tr = ClsDataset(Xtr, ytr); ds_va = ClsDataset(Xva, yva)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_CLS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_CLS, shuffle=False)
    if freeze_encoder:
        for p in model.encoder.parameters(): p.requires_grad=False
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in dl_tr:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(); logits = model(xb); loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
        # val
        model.eval(); y_true=[]; y_pred=[]
        with torch.no_grad():
            for xb,yb in dl_va:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE)
                logits = model(xb); y_pred.extend(logits.argmax(1).cpu().numpy()); y_true.extend(yb.cpu().numpy())
        macro = f1_score(y_true, y_pred, average='macro')
        if macro>best:
            best=macro; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    t = time.time()-t0
    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    acc = accuracy_score(y_true, y_pred)
    # inference time / sample
    n=0; t0=time.time();
    with torch.no_grad():
        for xb,yb in dl_va:
            xb=xb.to(DEVICE); _=model(xb); n+=xb.size(0)
    infer_ms = (time.time()-t0)/max(1,n)*1000.0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best, acc, t, infer_ms, params




# =====================
# 7) RUN: PRETRAIN → EVAL
# =====================
RESULTS = []
features_used_desc = f"All sensors ({Xseq.shape[-1]} ch)"
WIN = WINDOW

# (A) SimCLR
enc_simclr, t_pre_sim, p_pre_sim = simclr_pretrain(Xs, epochs=8, lr=1e-3)
# Linear probe
macro_lp, acc_lp, t_lp, clf_lp = linear_probe(enc_simclr, Xt_tr, yt_tr, Xt_va, yt_va)
# Fine-tune vs From-scratch
model_ft = ClsFineTune(enc_simclr, N_CLASS)
macro_ft, acc_ft, t_ft, infer_ft, p_ft = train_classifier(model_ft, Xt_tr, yt_tr, Xt_va, yt_va, epochs=8, lr=1e-3, freeze_encoder=False)
model_scratch = ClsFineTune(TCNEncoder(in_ch=Xseq.shape[-1], emb=128), N_CLASS)
macro_sc, acc_sc, t_sc, infer_sc, p_sc = train_classifier(model_scratch, Xt_tr, yt_tr, Xt_va, yt_va, epochs=8, lr=1e-3, freeze_encoder=False)

RESULTS += [
    {"model":"SimCLR:LinearProbe","features_used":features_used_desc,"window_size":WIN,"optimizer/solver":"AdamW","params (M)":round(p_pre_sim/1e6,3),"Binary":np.nan,"Macro":round(macro_lp,4),"Final Score":round(macro_lp,4),"val_acc":round(acc_lp,4),"train_time":round(t_pre_sim+t_lp,2),"inference_time":np.nan},
    {"model":"SimCLR:FineTune","features_used":features_used_desc,"window_size":WIN,"optimizer/solver":"AdamW","params (M)":round(p_ft/1e6,3),"Binary":np.nan,"Macro":round(macro_ft,4),"Final Score":round(macro_ft,4),"val_acc":round(acc_ft,4),"train_time":round(t_pre_sim+t_ft,2),"inference_time":round(infer_ft,3)},
    {"model":"FromScratch","features_used":features_used_desc,"window_size":WIN,"optimizer/solver":"AdamW","params (M)":round(p_sc/1e6,3),"Binary":np.nan,"Macro":round(macro_sc,4),"Final Score":round(macro_sc,4),"val_acc":round(acc_sc,4),"train_time":round(t_sc,2),"inference_time":round(infer_sc,3)},
]

# (B) MAE-lite (masked modeling)
mae, t_pre_mae, p_pre_mae = mae_pretrain(Xs, mask_p=0.3, epochs=6, lr=1e-3)
macro_lp2, acc_lp2, t_lp2, _ = linear_probe(mae, Xt_tr, yt_tr, Xt_va, yt_va)
model_ft2 = ClsFineTune(mae, N_CLASS)
macro_ft2, acc_ft2, t_ft2, infer_ft2, p_ft2 = train_classifier(model_ft2, Xt_tr, yt_tr, Xt_va, yt_va, epochs=8, lr=1e-3, freeze_encoder=False)

RESULTS += [
    {"model":"MAE:LinearProbe","features_used":features_used_desc,"window_size":WIN,"optimizer/solver":"AdamW","params (M)":round(p_pre_mae/1e6,3),"Binary":np.nan,"Macro":round(macro_lp2,4),"Final Score":round(macro_lp2,4),"val_acc":round(acc_lp2,4),"train_time":round(t_pre_mae+t_lp2,2),"inference_time":np.nan},
    {"model":"MAE:FineTune","features_used":features_used_desc,"window_size":WIN,"optimizer/solver":"AdamW","params (M)":round(p_ft2/1e6,3),"Binary":np.nan,"Macro":round(macro_ft2,4),"Final Score":round(macro_ft2,4),"val_acc":round(acc_ft2,4),"train_time":round(t_pre_mae+t_ft2,2),"inference_time":round(infer_ft2,3)},
]

report_df = pd.DataFrame(RESULTS, columns=[
    'model','features_used','window_size','optimizer/solver','params (M)','Binary','Macro','Final Score','val_acc','train_time','inference_time'
])
print(report_df)
try:
    from IPython.display import display
    display(report_df)
except: pass




# =====================
# 8) VISUALIZE LATENT (t-SNE)
# =====================
# So sánh latent của SimCLR (fine-tuned encoder) vs raw (PCA-2D đơn giản)
with torch.no_grad():
    Z_sim = embed_with_encoder(model_ft.encoder, Xt_va)

# lấy raw 2D bằng TSNE cho công bằng
N_VIS = min(1000, len(Xt_va))
sel = np.random.RandomState(SEED).choice(len(Xt_va), size=N_VIS, replace=False)
Z_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, random_state=SEED).fit_transform(Z_sim[sel])
plt.figure(figsize=(5,4))
scatter = plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=yt_va[sel], s=10, cmap='tab10')
plt.title('t-SNE of SSL Embedding (SimCLR Fine-tuned)'); plt.tight_layout(); plt.show()

print('\nNotebook 5 – Done.')

