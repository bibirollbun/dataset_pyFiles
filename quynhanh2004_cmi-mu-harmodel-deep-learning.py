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


import random
import numpy as np
import torch
import os

def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

SEED = 42
seed_everything(seed=SEED)

import pandas as pd
import polars as pl
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

import kaggle_evaluation.cmi_inference_server
from matplotlib import pyplot as plt


print("Loading datasets...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# Encode labels
label_encoder = LabelEncoder()
train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
gesture_classes = label_encoder.classes_

bfrb_gestures = [
    'Above ear - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin'
]
bfrb_indices = label_encoder.transform(bfrb_gestures)

imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
tof_thm_cols = [c for c in train_df.columns if c.startswith('thm_') or c.startswith('tof_')]

# Reorder so that IMU features come first
feature_cols = imu_cols + tof_thm_cols
imu_dim = len(imu_cols)
tof_thm_dim = len(tof_thm_cols)

print(f"IMU features: {imu_dim}, TOF/Thermal features: {tof_thm_dim}, Total features: {len(feature_cols)}")

# Check for missing values
nan_counts = train_df[feature_cols].isna().sum().sum()
print("Total NaNs in train features:", nan_counts)


# to remove hand dependency in IMU data
# im not sure if the rotation is on the x axis but this give me the best CV
def apply_symmetry(data):
    transformed = data.copy()
    transformed['acc_z'] = -transformed['acc_z']
    transformed['acc_y'] = -transformed['acc_y']
    
    transformed['rot_w'] = transformed['rot_w']
    transformed['rot_x'] = transformed['rot_x']
    transformed['rot_y'] = -transformed['rot_y']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed


train_df = train_df.merge(
    train_dem_df,
    on='subject',
    how='left',
    validate='many_to_one'
)

right_handed_mask = train_df['handedness'] == 1
train_df.loc[right_handed_mask, imu_cols] = apply_symmetry(train_df.loc[right_handed_mask, imu_cols])


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


sequences = train_df.groupby('sequence_id')
X_list = []
lengths = []
y_list = []

sequence_info = []
for i, (seq_id, seq) in enumerate(sequences):
    seq_data = seq[feature_cols].ffill().bfill().fillna(0).values
    X_list.append(seq_data)
    lengths.append(seq_data.shape[0])
    sequence_info.append({
        'sequence_id': seq_id,
        'subject': seq['subject'].iloc[0],
        'gesture': seq['gesture'].iloc[0]
    })

pad_len = int(np.percentile(lengths, 90))
print(f"Pad/truncate all sequences to length {pad_len} (90th percentile).")

seq_df = pd.DataFrame(sequence_info)
X_array = keras_pad_sequences(
    X_list,
    maxlen=pad_len,
    dtype='float32',
    padding='post',
    truncating='post'
)  # shape: (n_samples, pad_len, total_features)

y_array = seq_df['gesture'].values  # shape: (n_samples,)

num_classes = len(np.unique(y_array))
y_array = np.eye(num_classes)[y_array]  # shape: (n_samples, num_classes)

# Transpose to (n_samples, features, seq_len) for PyTorch
X_array = np.transpose(X_array, (0, 2, 1))


class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: np.ndarray of shape (n_samples, features, seq_len)
        y: np.ndarray of shape (n_samples, num_classes) or None for test
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Squeeze: global average pooling over time dimension
        se = x.mean(dim=2)                      # (batch, channels)
        se = self.relu(self.fc1(se))            # (batch, channels//reduction)
        se = self.sigmoid(self.fc2(se))         # (batch, channels)
        se = se.unsqueeze(2)                    # (batch, channels, 1)
        return x * se                           # scale channels

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels, reduction=8)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        shortcut = self.shortcut(x)                                 # (batch, out_channels, seq_len)
        out = self.conv1(x)                                          # (batch, out_channels, seq_len)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)                                        # (batch, out_channels, seq_len)
        out = self.bn2(out)

        out = self.se(out)                                           # SE scaling

        out = out + shortcut                                         # skip connection
        out = self.relu(out)

        out = self.pool(out)                                         # (batch, out_channels, seq_len//pool_size)
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.score_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        scores = torch.tanh(self.score_fc(x))            # (batch, seq_len, 1)
        scores = scores.squeeze(2)                       # (batch, seq_len)
        weights = F.softmax(scores, dim=1)               # (batch, seq_len)
        weights = weights.unsqueeze(2)                   # (batch, seq_len, 1)
        weighted = x * weights                           # (batch, seq_len, features)
        context = weighted.sum(dim=1)                    # (batch, features)
        return context
        
class IMU_HARModel(nn.Module):
    def __init__(self, total_features, imu_dim, pad_len, num_classes):
        super(IMU_HARModel, self).__init__()
        # IMU branch
        self.resblock1 = ResidualSEBlock(imu_dim, 64, kernel_size=3, pool_size=2, dropout_rate=0.1)
        self.resblock2 = ResidualSEBlock(64, 128, kernel_size=3, pool_size=2, dropout_rate=0.1)

        # After pooling twice, seq_len reduced by factor of 4
        reduced_len = pad_len // 4
        merged_channels = 128 #+ 128  # from IMU and TTF here we use IMU only

        self.bigru = nn.GRU(
            input_size=merged_channels,
            hidden_size=merged_channels//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Attention
        self.attention = Attention(input_dim=merged_channels)

        # Dense head
        self.fc1 = nn.Linear(merged_channels, 128, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop_fc1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 64, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop_fc2 = nn.Dropout(0.1)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, total_features, seq_len)
        x_imu = x[:, :imu_dim, :]           # (batch, imu_dim, seq_len)
        x_ttf = x[:, imu_dim:, :]           # (batch, rest_dim, seq_len)

        # IMU branch
        b1 = self.resblock1(x_imu)          # (batch, 64, seq_len/2)
        b1 = self.resblock2(b1)             # (batch, 128, seq_len/4)

        # b2 is reserved for tof branch when will be using it

        # Concatenate branches along channel dimension
        merged = b1 # torch.cat([b1, b2], dim=1)  # (batch, 256, seq_len/4)

        # Prepare for GRU: (batch, seq_len/4, 256)
        merged = merged.permute(0, 2, 1)

        # BiGRU
        lstm_out, _ = self.bigru(merged)       # (batch, seq_len/4, 256)

        # Attention
        context = self.attention(lstm_out)    # (batch, 256)

        # Dense head
        x = self.fc1(context)                 # (batch, 256)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)

        x = self.fc2(x)                       # (batch, 128)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)

        out = self.out(x)                     # (batch, num_classes)
        return out

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def soft_cross_entropy(pred, soft_targets):
    """
    pred: (batch, num_classes) raw scores (no softmax)
    soft_targets: (batch, num_classes) probabilities
    """
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
    return loss

def mixup_data(x, y, alpha=0.2):
    """
    Return mixed inputs and mixed targets (one-hot) for mixup.
    x: Tensor of shape (batch_size, features, seq_len)
    y: Tensor of shape (batch_size, num_classes)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


seed_everything(seed=SEED)

criterion = soft_cross_entropy

n_splits = 5
batch_size = 128
gkf = GroupKFold(n_splits=n_splits)

fold_metrics = []
best_fold_metrics = []
best_models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_array, y_array, groups=seq_df['subject'])):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{n_splits}")
    print(f"Train subjects: {len(np.unique(seq_df.iloc[train_idx]['subject']))}")
    print(f"Val subjects: {len(np.unique(seq_df.iloc[val_idx]['subject']))}")
    print(f"{'='*50}")
    
    train_dataset = SequenceDataset(X_array[train_idx], y_array[train_idx])
    val_dataset = SequenceDataset(X_array[val_idx], y_array[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    seed_everything(seed=SEED + fold)
    model = IMU_HARModel(
        total_features=len(feature_cols),
        imu_dim=imu_dim,
        pad_len=pad_len,
        num_classes=num_classes,
    ).to(device)
    
    # Optimizer et scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5 * steps_per_epoch,
        T_mult=2,
        eta_min=1e-5,
    )
    
    # Early stopping
    best_metric = -np.inf
    best_binary_f1 = -np.inf
    best_macro_f1 = -np.inf
    patience = 15
    epochs_no_improve = 0
    num_epochs = 100
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
    
            # Apply mixup
            mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=0.2)
    
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = criterion(outputs, mixed_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            train_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
        train_loss /= total
    
        # Validation phase
        model.eval()
        val_loss = 0.0
        total = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                total += batch_x.size(0)
                
                # Get predicted class indices
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                # Get true class indices from one-hot
                trues = torch.argmax(batch_y, dim=1).cpu().numpy()
                
                all_true.append(trues)
                all_pred.append(preds)
        
        val_loss /= total
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        
        # Compute competition metrics
        # Binary classification: BFRB (1) vs non-BFRB (0)
        binary_true = np.isin(all_true, bfrb_indices).astype(int)
        binary_pred = np.isin(all_pred, bfrb_indices).astype(int)
        binary_f1 = f1_score(binary_true, binary_pred)
        
        # Collapse non-BFRB gestures into a single class
        collapsed_true = np.where(
            np.isin(all_true, bfrb_indices),
            all_true,
            len(bfrb_gestures)  # Single non-BFRB class
        )
        collapsed_pred = np.where(
            np.isin(all_pred, bfrb_indices),
            all_pred,
            len(bfrb_gestures)  # Single non-BFRB class
        )
        
        # Macro F1 on collapsed classes
        macro_f1 = f1_score(collapsed_true, collapsed_pred, average='macro')
        final_metric = (binary_f1 + macro_f1) / 2
        
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"  Binary F1 = {binary_f1:.4f}, Macro F1 = {macro_f1:.4f}, Final Metric = {final_metric:.4f}")
        
        if final_metric > best_metric:
            best_metric = final_metric
            best_binary_f1 = binary_f1
            best_macro_f1 = macro_f1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"  New best metric! Saving model...")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
    
    torch.save(best_model_state, f"best_model_fold{fold}.pth")
    best_models.append(best_model_state)
    
    fold_metrics.append({
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'final_metric': final_metric
    })
    
    best_fold_metrics.append({
        'binary_f1': best_binary_f1,
        'macro_f1': best_macro_f1,
        'final_metric': best_metric
    })
    
    print(f"\nFold {fold + 1} completed.")
    print(f"Final validation metrics - Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final: {final_metric:.4f}")
    print(f"Best validation metrics - Binary F1: {best_binary_f1:.4f}, Macro F1: {best_macro_f1:.4f}, Final: {best_metric:.4f}")

print("\n" + "="*50)
print("Cross-Validation Results")
print("="*50)

# Statistiques pour les meilleures métriques
best_binary_f1 = [m['binary_f1'] for m in best_fold_metrics]
best_macro_f1 = [m['macro_f1'] for m in best_fold_metrics]
best_metrics = [m['final_metric'] for m in best_fold_metrics]

print("\nBest Fold-wise Metrics:")
for i, (bf1, mf1, fm) in enumerate(zip(best_binary_f1, best_macro_f1, best_metrics)):
    print(f"Fold {i+1}: Binary F1 = {bf1:.4f}, Macro F1 = {mf1:.4f}, Final = {fm:.4f}")

print("\nGlobal Statistics (Best Metrics):")
print(f"Mean Best Final Metric: {np.mean(best_metrics):.4f} ± {np.std(best_metrics):.4f}")
print(f"Mean Best Binary F1: {np.mean(best_binary_f1):.4f} ± {np.std(best_binary_f1):.4f}")
print(f"Mean Best Macro F1: {np.mean(best_macro_f1):.4f} ± {np.std(best_macro_f1):.4f}")


model_ensemble = []
for fold in range(5):
    model = IMU_HARModel(
        total_features=len(feature_cols),
        imu_dim=imu_dim,
        pad_len=pad_len,
        num_classes=num_classes,
    ).to(device)
    checkpoint = torch.load(f"best_model_fold{fold}.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model_ensemble.append(model)


def preprocess_sequence(df_seq: pd.DataFrame):
    """
    Process a single sequence DataFrame (pandas):
    - Forward/backward fill missing
    - Scale using loaded scaler
    - Pad/truncate to pad_len
    - Return torch.Tensor of shape (1, features, seq_len)
    """
    data = df_seq[feature_cols].ffill().bfill().fillna(0).values
    # Pad/truncate
    padded = keras_pad_sequences(
        [data],
        maxlen=pad_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )[0]  # (pad_len, total_features)
    # Transpose to (features, pad_len)
    tensor = torch.from_numpy(padded.T).unsqueeze(0).float()  # (1, features, pad_len)
    return tensor
    
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    sequence: polars DataFrame for a single sequence
    demographics: unused in this model
    Returns: predicted gesture string
    """
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    
    df_seq = df_seq.merge(
    df_demo,
    on='subject',
    how='left',
    validate='many_to_one',
    )
    right_handed_mask = df_seq['handedness'] == 1
    df_seq.loc[right_handed_mask, imu_cols] = apply_symmetry(df_seq.loc[right_handed_mask, imu_cols])

    x_tensor = preprocess_sequence(df_seq).to(device)
    
    all_outputs = []
    with torch.no_grad():
        for model in model_ensemble:
            outputs = model(x_tensor).softmax(dim=-1)
            all_outputs.append(outputs)

    avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    pred_idx = torch.argmax(avg_outputs, dim=1).item()
    
    return str(gesture_classes[pred_idx])


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




