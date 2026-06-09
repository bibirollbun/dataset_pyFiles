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
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)




# =====================
# 0) DATA LOADING
# =====================
train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_dem_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_dem_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

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

# detect nhãn
LABEL_CANDS = ['label','target','activity','activity_id','behavior','behavior_id','state','class']
label_map = {c.lower(): c for c in train_full.columns}
LABEL_COL = None
for k in LABEL_CANDS:
    if k in label_map: LABEL_COL = label_map[k]; break
if LABEL_COL is None:
    nn = len(train_full); opts = []
    for c in train_full.columns:
        lc = c.lower()
        if 'id' in lc or 'time' in lc or 'timestamp' in lc: continue
        u = train_full[c].nunique(dropna=True)
        if 2 <= u <= min(50, max(2, nn//10)):
            opts.append((c,u))
    opts.sort(key=lambda x: x[1]); LABEL_COL = opts[0][0]

# chọn ToF cột
TOF_COLS = [c for c in train_full.columns if c.lower().startswith('tof') and c != LABEL_COL]
if len(TOF_COLS) == 0:
    raise RuntimeError('Không tìm thấy cột ToF (prefix "tof"). Hãy kiểm tra tên cột trong train.csv')

# time & subject
TIME_COL = None
for k in ['timestamp','time','frame','step','t','ts']:
    for c in train_full.columns:
        if c.lower()==k or k in c.lower(): TIME_COL = c; break
    if TIME_COL: break
if TIME_COL is None:
    TIME_COL = '___row_index__'; train_full[TIME_COL] = np.arange(len(train_full))
SUBJ_COL = None
for k in ['subject_id','participant_id','user_id','person_id','subject','ParticipantID', lk]:
    if isinstance(k,str) and k in train_full.columns:
        SUBJ_COL = k; break

# Label encode
le = LabelEncoder(); y_all = le.fit_transform(train_full[LABEL_COL].astype(str).values).astype(np.int64)
N_CLASS = len(le.classes_)




# =====================
# 1) BUILD ToF WINDOWS (T x H x W)
# =====================

def infer_grid_shape(n_channels):
    # cố gắng đoán H=W=sqrt(n) -> 8x8 phổ biến
    r = int(round(math.sqrt(n_channels)))
    if r*r == n_channels: return r, r
    # nếu không phải hình vuông, giả sử 8x8 nếu n>=64, else 4x4
    if n_channels >= 64: return 8,8
    return 4,4

H,W = infer_grid_shape(len(TOF_COLS))
print({'tof_channels': len(TOF_COLS), 'grid': (H,W)})

# ánh xạ kênh → (H,W): sort tên cột theo số tự nhiên ở cuối nếu có
import re

def sort_tof_cols(cols):
    def key(c):
        m = re.search(r'(\d+)$', c)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)

TOF_COLS = sort_tof_cols(TOF_COLS)

WINDOW = 100
STRIDE = WINDOW//2


def build_tof_windows(df, tof_cols, labels, time_col, group_col, H, W, win=100, stride=50):
    X_list, y_list, g_list = [], [], []
    if group_col is None:
        df = df.sort_values(time_col)
        arr = df[tof_cols].values.astype(np.float32)
        for s in range(0, len(df)-win+1, stride):
            block = arr[s:s+win]
            if block.shape[0] != win: continue
            # chuẩn hóa theo kênh trong cửa sổ
            block = (block - np.nanmean(block, axis=0)) / (np.nanstd(block, axis=0) + 1e-6)
            block = np.nan_to_num(block, 0.0, 0.0, 0.0)
            # reshape -> [T,H,W]
            try:
                grid = block.reshape(win, H, W)
            except Exception:
                # nếu không khớp, pad/crop về H*W
                K = H*W
                grid = block[:, :K]
                if grid.shape[1] < K:
                    pad = np.zeros((win, K-grid.shape[1]), dtype=np.float32)
                    grid = np.concatenate([grid, pad], axis=1)
                grid = grid.reshape(win, H, W)
            X_list.append(grid)
            yy = labels[df.index.values[s:s+win]]
            vals, cnts = np.unique(yy, return_counts=True)
            y_list.append(vals[np.argmax(cnts)])
            g_list.append(0)
    else:
        for gid, gdf in df.groupby(group_col):
            gdf = gdf.sort_values(time_col)
            if len(gdf) < win: continue
            arr = gdf[tof_cols].values.astype(np.float32)
            for s in range(0, len(gdf)-win+1, stride):
                block = arr[s:s+win]
                if block.shape[0] != win: continue
                block = (block - np.nanmean(block, axis=0)) / (np.nanstd(block, axis=0) + 1e-6)
                block = np.nan_to_num(block, 0.0, 0.0, 0.0)
                try:
                    grid = block.reshape(win, H, W)
                except Exception:
                    K = H*W
                    grid = block[:, :K]
                    if grid.shape[1] < K:
                        pad = np.zeros((win, K-grid.shape[1]), dtype=np.float32)
                        grid = np.concatenate([grid, pad], axis=1)
                    grid = grid.reshape(win, H, W)
                X_list.append(grid)
                yy = labels[gdf.index.values[s:s+win]]
                vals, cnts = np.unique(yy, return_counts=True)
                y_list.append(vals[np.argmax(cnts)])
                g_list.append(gid)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    g = np.array(g_list)
    # làm sạch
    X = np.nan_to_num(X, 0.0, 0.0, 0.0)
    return X, y, g

XtoF, yseq, gseq = build_tof_windows(train_full, TOF_COLS, y_all, TIME_COL, SUBJ_COL, H, W, win=WINDOW, stride=STRIDE)
print({'windows': len(XtoF), 'shape_each': XtoF.shape[1:]})

# split transfer (cross-subject nếu có)
uniq_g = np.unique(gseq)
if len(uniq_g) > 1:
    src_g, tgt_g = train_test_split(uniq_g, test_size=0.2, random_state=SEED)
else:
    src_g, tgt_g = uniq_g, uniq_g
mask_src = np.isin(gseq, src_g)
mask_tgt = np.isin(gseq, tgt_g)
Xs, ys = XtoF[mask_src], yseq[mask_src]
Xt, yt = XtoF[mask_tgt], yseq[mask_tgt]
idx_tr, idx_va = train_test_split(np.arange(len(yt)), test_size=0.5, stratify=yt, random_state=SEED)
Xt_tr, yt_tr = Xt[idx_tr], yt[idx_tr]
Xt_va, yt_va = Xt[idx_va], yt[idx_va]




# =====================
# 2) DATASETS
# =====================
class ToFDataset(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        x = np.nan_to_num(x, 0.0, 0.0, 0.0)
        return torch.from_numpy(x), int(self.y[i])

BATCH=64

ds_tr = ToFDataset(Xt_tr, yt_tr)
ds_va = ToFDataset(Xt_va, yt_va)

dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)




# =====================
# 3) GRAPH CONSTRUCTION (8x8 grid)
# =====================

def grid_adjacency(H, W, mode='4n', weight='uniform'):
    N = H*W
    A = np.zeros((N,N), dtype=np.float32)
    def idx(r,c): return r*W + c
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if mode=='4n' else [
        (-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)
    ]
    for r in range(H):
        for c in range(W):
            i = idx(r,c)
            for dr,dc in nbrs:
                rr,cc = r+dr, c+dc
                if 0<=rr<H and 0<=cc<W:
                    j = idx(rr,cc)
                    if weight=='uniform':
                        A[i,j] = 1.0
                    else:
                        d = math.sqrt(dr*dr+dc*dc)
                        A[i,j] = 1.0/(d+1e-6)
    # thêm self-loop
    A += np.eye(N, dtype=np.float32)
    # chuẩn hoá sym: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = 1.0/np.sqrt(D + 1e-6)
    A_hat = (A * D_inv_sqrt[:,None]) * D_inv_sqrt[None,:]
    return torch.from_numpy(A_hat)

A_HAT = grid_adjacency(H,W, mode='8n', weight='uniform').to(DEVICE)
NODES = H*W




# =====================
# 4) MODELS
# =====================
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.w = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, X, A_hat):  # X: [B,N,F]
        Xw = self.w(X)            # [B,N,out]
        return torch.matmul(A_hat, Xw)  # broadcast [N,N] x [B,N,out]

class SimpleGCN(nn.Module):
    def __init__(self, nodes, in_feat=1, hid=64, out=128, nclass=10):
        super().__init__()
        self.gc1 = GraphConv(in_feat, hid)
        self.gc2 = GraphConv(hid, out)
        self.temporal = nn.GRU(out, out, batch_first=True)
        self.cls = nn.Linear(out, nclass)
    def forward(self, x):  # x: [B,T,H,W]
        B,T,H,W = x.shape
        x = x.view(B,T,H*W,1)          # feature=1 per node
        # graph conv từng timestep
        outs=[]
        for t in range(T):
            Xt = x[:,t]                # [B,N,1]
            h = F.relu(self.gc1(Xt, A_HAT))
            h = F.relu(self.gc2(h, A_HAT))
            outs.append(h)
        Z = torch.stack(outs, dim=1)   # [B,T,N,out]
        Z = Z.mean(dim=2)              # pool nodes → [B,T,out]
        Z,_ = self.temporal(Z)         # [B,T,out]
        Z = Z.mean(dim=1)              # pool time → [B,out]
        return self.cls(Z)

class SimpleGAT(nn.Module):
    def __init__(self, nodes, in_feat=1, hid=64, out=128, nclass=10, heads=4):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, hid, bias=False)
        self.att1 = nn.Parameter(torch.randn(heads, nodes, nodes))
        self.fc2 = nn.Linear(hid, out, bias=False)
        self.att2 = nn.Parameter(torch.randn(heads, nodes, nodes))
        self.temporal = nn.GRU(out, out, batch_first=True)
        self.cls = nn.Linear(out, nclass)
    def att_agg(self, X, Aparam):  # X:[B,N,F], Aparam:[H,N,N]
        # softmax theo neighbors
        A = torch.softmax(Aparam, dim=-1)  # [H,N,N]
        Xh = torch.einsum('hij,bjf->bhif', A, X)  # [H,B,N,F]→ [B,H,N,F]
        return Xh.mean(dim=1)                   # avg heads → [B,N,F]
    def forward(self, x):
        B,T,H,W = x.shape
        x = x.view(B,T,H*W,1)
        outs=[]
        for t in range(T):
            Xt = x[:,t]
            h = F.relu(self.fc1(Xt))
            h = self.att_agg(h, self.att1)
            h = F.relu(self.fc2(h))
            h = self.att_agg(h, self.att2)
            outs.append(h)
        Z = torch.stack(outs, dim=1)    # [B,T,N,F]
        Z = Z.mean(dim=2)               # pool nodes
        Z,_ = self.temporal(Z)
        Z = Z.mean(dim=1)
        return self.cls(Z)

class CNN2DTemporal(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((H,W))
        )
        self.temporal = nn.GRU(H*W*32, 128, batch_first=True)
        self.cls = nn.Linear(128, nclass)
    def forward(self, x):  # x:[B,T,H,W]
        B,T,Hh,Wh = x.shape
        x = x.unsqueeze(2)             # [B,T,1,H,W]
        feats=[]
        for t in range(T):
            f = self.frame(x[:,t])     # [B,32,H,W]
            feats.append(f.view(B,-1))
        Z = torch.stack(feats, dim=1)  # [B,T,32*H*W]
        Z,_ = self.temporal(Z)
        Z = Z.mean(dim=1)
        return self.cls(Z)

class CNN3D(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1,16,(3,3,3),padding=1), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=1), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(32,64,(3,3,3),padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.cls = nn.Linear(64, nclass)
    def forward(self, x):  # x:[B,T,H,W]
        x = x.unsqueeze(1)  # [B,1,T,H,W]
        z = self.backbone(x).view(x.size(0), -1)
        return self.cls(z)




# =====================
# 5) TRAIN / EVAL UTILS
# =====================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model, dl):
    model.eval(); y_true, y_pred = [], []
    for xb,yb in dl:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        y_pred.extend(logits.argmax(1).detach().cpu().numpy())
        y_true.extend(yb.detach().cpu().numpy())
    macro = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return macro, acc


def train_model(model, dl_tr, dl_va, epochs=8, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in dl_tr:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad();
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        macro, acc = evaluate(model, dl_va)
        if macro>best:
            best=macro; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    t_train = time.time()-t0
    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    # inference time / sample
    n=0; t0=time.time();
    with torch.no_grad():
        for xb,_ in dl_va:
            xb = xb.to(DEVICE); _ = model(xb); n += xb.size(0)
    infer_ms = (time.time()-t0)/max(1,n)*1000.0
    params = count_params(model)
    macro, acc = evaluate(model, dl_va)
    return macro, acc, t_train, infer_ms, params




# =====================
# 6) RUN EXPERIMENTS
# =====================
RESULTS = []
features_used_desc = f"ToF grid {H}x{W}"
WIN = WINDOW

# GCN
m_gcn = SimpleGCN(nodes=H*W, nclass=N_CLASS)
macro, acc, ttrain, tinfer, params = train_model(m_gcn, dl_tr, dl_va, epochs=8, lr=1e-3)
RESULTS.append({
    'model':'GCN','features_used':features_used_desc,'window_size':WIN,
    'optimizer/solver':'AdamW','params (M)':round(params/1e6,3),'Binary':np.nan,
    'Macro':round(macro,4),'Final Score':round(macro,4),'val_acc':round(acc,4),
    'train_time':round(ttrain,2),'inference_time':round(tinfer,3)
})

gc.collect(); torch.cuda.empty_cache()

# GAT
m_gat = SimpleGAT(nodes=H*W, nclass=N_CLASS, heads=4)
macro, acc, ttrain, tinfer, params = train_model(m_gat, dl_tr, dl_va, epochs=8, lr=1e-3)
RESULTS.append({
    'model':'GAT','features_used':features_used_desc,'window_size':WIN,
    'optimizer/solver':'AdamW','params (M)':round(params/1e6,3),'Binary':np.nan,
    'Macro':round(macro,4),'Final Score':round(macro,4),'val_acc':round(acc,4),
    'train_time':round(ttrain,2),'inference_time':round(tinfer,3)
})

gc.collect(); torch.cuda.empty_cache()

# 2D CNN baseline (frame CNN + temporal pooling)
m_cnn2d = CNN2DTemporal(nclass=N_CLASS)
macro, acc, ttrain, tinfer, params = train_model(m_cnn2d, dl_tr, dl_va, epochs=8, lr=1e-3)
RESULTS.append({
    'model':'CNN2D+GRU','features_used':features_used_desc,'window_size':WIN,
    'optimizer/solver':'AdamW','params (M)':round(params/1e6,3),'Binary':np.nan,
    'Macro':round(macro,4),'Final Score':round(macro,4),'val_acc':round(acc,4),
    'train_time':round(ttrain,2),'inference_time':round(tinfer,3)
})

gc.collect(); torch.cuda.empty_cache()

# 3D CNN
m_cnn3d = CNN3D(nclass=N_CLASS)
macro, acc, ttrain, tinfer, params = train_model(m_cnn3d, dl_tr, dl_va, epochs=8, lr=1e-3)
RESULTS.append({
    'model':'CNN3D','features_used':features_used_desc,'window_size':WIN,
    'optimizer/solver':'AdamW','params (M)':round(params/1e6,3),'Binary':np.nan,
    'Macro':round(macro,4),'Final Score':round(macro,4),'val_acc':round(acc,4),
    'train_time':round(ttrain,2),'inference_time':round(tinfer,3)
})

report_df = pd.DataFrame(RESULTS, columns=[
    'model','features_used','window_size','optimizer/solver','params (M)','Binary','Macro','Final Score','val_acc','train_time','inference_time'
])
print(report_df)
try:
    from IPython.display import display
    display(report_df)
except: pass

print('\nNotebook 6 – Done.')


