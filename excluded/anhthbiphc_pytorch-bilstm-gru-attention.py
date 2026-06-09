


import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path
import warnings
import shutil
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold # Still from sklearn

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import polars as pl


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


def seed_everything(seed):
    """
    Seeds various random number generators for reproducibility in PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # Set Python hash seed
    random.seed(seed)                        # Seed Python's random module
    np.random.seed(seed)                     # Seed NumPy
    torch.manual_seed(seed)                  # Seed PyTorch for CPU operations
    torch.cuda.manual_seed(seed)             # Seed PyTorch for all GPU operations (if available)
    torch.cuda.manual_seed_all(seed)         # Seed all GPUs if multiple are used

    # Ensure deterministic operations for cuDNN (GPU backend)
    # This can sometimes slightly reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # Disable cuDNN benchmarking for reproducibility

# Call the function to set the seed
seed_everything(seed=42)


# (Competition metric will only be imported when TRAINing)
TRAIN = True
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# used when TRAIN=False
PRETRAINED_DIR = Path("/kaggle/input/pretrained-model")

# artifacts will be saved here
EXPORT_DIR = Path("./")                 

BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40

print("▶ imports ready · pytorch", torch.__version__)


# --- Tensor Manipulations ---
def time_sum(x):
    return torch.sum(x, dim=1)

def squeeze_last_axis(x):
    return x.squeeze(-1) # or torch.squeeze(x, dim=-1)

def expand_last_axis(x):
    return x.unsqueeze(-1) # or torch.unsqueeze(x, dim=-1)

# --- SE Block (Squeeze-and-Excitation Block) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze operation
        # For PyTorch (B, C, L), AdaptiveAvgPool1d(1) gives (B, C, 1)
        # Then squeeze to (B, C)
        b, c, _ = x.size() # Get batch size and channels
        se = self.avg_pool(x).view(b, c) # (B, C, 1) -> (B, C)

        # Excitation operation
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1) # Reshape to (B, C, 1) for multiplication

        return x * se # Element-wise multiplication


# --- Residual CNN Block with SE ---
class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
        super(ResidualSECNNBlock, self).__init__()
        # Use a list to store sequential layers for repeated blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(out_filters) # Apply SE block

        # Shortcut connection
        self.shortcut_conv = None
        if in_filters != out_filters:
            self.shortcut_conv = nn.Sequential(
                nn.Conv1d(in_filters, out_filters, 1, padding='same', bias=False),
                nn.BatchNorm1d(out_filters)
            )

        self.max_pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(drop)

        # L2 regularization (weight decay) is typically handled by the optimizer in PyTorch
        # We define it here to show where it would apply, but it's passed to optim.Adam etc.
        self.wd = wd

    def forward(self, x):
        shortcut = x

        # Conv layers + Batch Norm + ReLU
        x = self.conv_block(x)

        # SE Block
        x = self.se_block(x)

        # Shortcut connection handling
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)

        # Add (residual connection)
        x = x + shortcut # Keras 'add'

        x = F.relu(x) # Activation after addition

        x = self.max_pool(x)
        x = self.dropout(x)
        return x


# --- Attention Layer ---
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.score_dense = nn.Linear(input_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # Calculate raw scores; This will give (batch_size, sequence_length, 1)
        score = self.score_dense(inputs)
        score = self.tanh(score)

        # Squeeze last axis to get (batch_size, sequence_length)
        score = squeeze_last_axis(score)

        # Calculate attention weights
        weights = F.softmax(score, dim=1)

        # Expand last axis back to (batch_size, sequence_length, 1)
        weights = expand_last_axis(weights)

        # Apply weights to inputs
        context = inputs * weights

        # Sum along the time axis
        context = time_sum(context)

        return context


# --- Preprocessing Function ---
class MixupDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """
        Initializes the MixupDataset for PyTorch.

        Args:
            X (torch.Tensor): The input features. Expected shape: (num_samples, sequence_length, num_features).
                              This dataset will NOT perform internal permutation for Conv1D.
            y (torch.Tensor): The labels. Expected shape: (num_samples, num_classes) for one-hot (float),
                              or (num_samples,) for integer labels (long).
            alpha (float): Mixup alpha parameter for beta distribution.
        """
        self.X = X
        self.y = y

        self.alpha = alpha

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Generates one sample of data, applying Mixup by mixing the current sample
        with a randomly selected other sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (mixed_X, mixed_y) as PyTorch tensors, where mixed_X is
                   (sequence_length, num_features).
        """
        x1, y1 = self.X[idx], self.y[idx]

        # Select a random sample from the dataset for mixing
        rand_idx = random.randint(0, len(self.X) - 1)
        x2, y2 = self.X[rand_idx], self.y[rand_idx]

        # Generate lambda from Beta distribution. Ensure alpha > 0.
        # This dataset is only for training where MIXUP_ALPHA > 0
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(0, min(1, lam))

        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2

        return x_mixed, y_mixed


# --- A Simple CNN Block for the ToF Branch ---
class SimpleCNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, pool_size=2, drop=0.2):
        super(SimpleCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )
    def forward(self, x):
        return self.block(x)

# --- Two-Branch Model ---
class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim, tof_dim, n_classes):
        super(TwoBranchModel, self).__init__()
        
        # Store input dimensions for splitting in the forward pass
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        # --- IMU Branch (ResidualSECNNBlock is well-designed) ---
        self.imu_branch_block1 = ResidualSECNNBlock(imu_dim, 64, 3, drop=0.1)
        self.imu_branch_block2 = ResidualSECNNBlock(64, 128, 5, drop=0.1)

        # --- TOF Branch (Using our cleaner block) ---
        self.tof_branch_block1 = SimpleCNNBlock(tof_dim, 64, drop=0.2)
        self.tof_branch_block2 = SimpleCNNBlock(64, 128, drop=0.2)
        
        # --- Shared Recurrent Layer (Choose ONE: LSTM or GRU) ---
        merged_cnn_features = 128 + 128 # 128 from IMU, 128 from TOF
        
        self.recurrent_layer = nn.LSTM(
            input_size=merged_cnn_features, 
            hidden_size=128, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Recurrent layer output features: 2 * hidden_size
        recurrent_output_features = 128 * 2

        # --- Attention Layer ---
        # The input dimension now correctly matches the output of our single recurrent layer
        self.attention_layer = AttentionLayer(input_dim=recurrent_output_features)
        
        # --- Classifier Head ---
        # Input dimension matches the output of the attention layer
        self.classifier_head = nn.Sequential(
            nn.Linear(recurrent_output_features, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, x):
        # Input x shape: (batch_size, pad_len, imu_dim + tof_dim)
        
        # Split branches
        imu = x[:, :, :self.imu_dim]
        tof = x[:, :, self.imu_dim:]

        # Permute for Conv1D: (batch, seq_len, features) -> (batch, features, seq_len)
        imu = imu.permute(0, 2, 1)
        tof = tof.permute(0, 2, 1)

        # --- Process Branches ---
        # IMU Branch
        x1 = self.imu_branch_block1(imu)
        x1 = self.imu_branch_block2(x1) # Shape: (batch, 128, new_seq_len)

        # TOF Branch
        x2 = self.tof_branch_block1(tof)
        x2 = self.tof_branch_block2(x2) # Shape: (batch, 128, new_seq_len)
        
        # --- Merge and Process Sequentially ---
        # Concatenate branches along the feature dimension
        merged = torch.cat([x1, x2], dim=1) # Shape: (batch, 256, new_seq_len)
        
        # Permute for Recurrent Layer: (batch, features, seq_len) -> (batch, seq_len, features)
        merged = merged.permute(0, 2, 1)

        # Pass through the single recurrent layer
        recurrent_out, _ = self.recurrent_layer(merged) # Shape: (batch, seq_len, 256)
        
        # Apply attention to the recurrent output
        attention_out = self.attention_layer(recurrent_out) # Shape: (batch, 256)
        
        # --- Classifier ---
        # Pass the clean, attention-weighted features to the classifier
        classified = self.classifier_head(attention_out)
        
        # --- Output ---
        # Get the raw logits. DO NOT apply softmax here.
        logits = self.output_layer(classified)
        
        return logits


# Instantiate the model
model = TwoBranchModel(pad_len=127, imu_dim=7, tof_dim=325, n_classes=18)



def save_checkpoint(state, is_best, directory=".", filename="latest_checkpoint.pth", best_filename="best_checkpoint.pth"):
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    if is_best:
        # Only copy if it's the best, don't delete previous 'best_checkpoint.pth' if no improvement
        shutil.copyfile(filepath, os.path.join(directory, best_filename))
        print(f"    New best model saved to {best_filename}")


DELETE_EXISTING_CHECKPOINTS = True

if TRAIN:
    # --- Data Loading and Preprocessing ---
    if DELETE_EXISTING_CHECKPOINTS:
        print(f"--- '{DELETE_EXISTING_CHECKPOINTS=}' is True. Checking for existing checkpoints in {EXPORT_DIR} to remove... ---")
        latest_checkpoint_path = os.path.join(EXPORT_DIR, "latest_checkpoint.pth")
        best_checkpoint_path = os.path.join(EXPORT_DIR, "best_checkpoint.pth")
        
        # Check and delete latest_checkpoint.pth
        if os.path.isfile(latest_checkpoint_path):
            os.remove(latest_checkpoint_path)
            print(f"  Removed existing checkpoint: {latest_checkpoint_path}")
        else:
            print(f"  No existing latest_checkpoint.pth found at {latest_checkpoint_path}.")

        # Check and delete best_checkpoint.pth
        if os.path.isfile(best_checkpoint_path):
            os.remove(best_checkpoint_path)
            print(f"  Removed existing best_checkpoint.pth: {best_checkpoint_path}")
        else:
            print(f"  No existing best_checkpoint.pth found at {best_checkpoint_path}.")
    
    # --- Data Loading and Preprocessing ---
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(os.path.join(EXPORT_DIR, "gesture_classes.npy"), le.classes_)
    gesture_classes = le.classes_
    print("  Calculating engineered IMU features...")
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    print("  Calculating ToF features with vectorized NumPy...")
    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    tof_data_np = df[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df), 5, 64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice'); warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    tof_agg_cols = []
    for i in range(1, 6):
        df[f'tof_{i}_mean'], df[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
        df[f'tof_{i}_min'], df[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]
        tof_agg_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    imu_cols = [c for c in df.columns if c.startswith(('acc_', 'rot_'))]
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    final_feature_cols = imu_cols + thm_cols + tof_agg_cols
    imu_dim_final = len(imu_cols)
    tof_thm_aggregated_dim_final = len(thm_cols) + len(tof_agg_cols)
    np.save(os.path.join(EXPORT_DIR, "feature_cols.npy"), np.array(final_feature_cols))
    print("  Building, scaling, and padding sequences...")
    X_list, y_list, lens = [], [], []
    for seq_id, seq_df in df.groupby('sequence_id'):
        X_list.append(seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32'))
        y_list.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(seq_df))
        
    # FIX: Rename StandardScaler to feature_scaler
    feature_scaler = StandardScaler().fit(np.concatenate(X_list, axis=0))
    joblib.dump(feature_scaler, os.path.join(EXPORT_DIR, "scaler.pkl"))
    X_scaled_list = [feature_scaler.transform(x) for x in X_list]
    
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(os.path.join(EXPORT_DIR, "sequence_maxlen.npy"), pad_len)
    X_padded = np.zeros((len(X_scaled_list), pad_len, len(final_feature_cols)), dtype='float32')
    for i, seq in enumerate(X_scaled_list):
        seq_len = min(len(seq), pad_len)
        X_padded[i, :seq_len] = seq[:seq_len]
    y_np = np.array(y_list)
    y_one_hot = F.one_hot(torch.from_numpy(y_np).long(), num_classes=len(gesture_classes)).float().numpy()
    X_tr, X_val, y_tr_oh, y_val_oh, y_tr_int, y_val_int = train_test_split(
        X_padded, y_one_hot, y_np, test_size=0.1, random_state=82, stratify=y_np
    )
    X_tr_t, X_val_t = torch.from_numpy(X_tr).float(), torch.from_numpy(X_val).float()
    y_tr_t, y_val_t = torch.from_numpy(y_tr_oh).float(), torch.from_numpy(y_val_oh).float()
    train_dataset = MixupDataset(X_tr_t, y_tr_t, alpha=MIXUP_ALPHA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(gesture_classes)), y=y_tr_int)
    class_weight = torch.from_numpy(cw_vals).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = class_weight.to(device)
    print(f"Using device: {device}")
    model = TwoBranchModel(pad_len, imu_dim_final, tof_thm_aggregated_dim_final, len(gesture_classes))
    checkpoint_path = os.path.join(EXPORT_DIR, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"--- Resuming training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in saved_state_dict.keys()):
            print("   Checkpoint is from a compiled model. Cleaning keys...")
            cleaned_state_dict = {key.replace('_orig_mod.', ''): value for key, value in saved_state_dict.items()}
        else:
            cleaned_state_dict = saved_state_dict
        model.load_state_dict(cleaned_state_dict)
    else:
        checkpoint = None 
        print("--- No checkpoint found, starting training from scratch. ---")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    lr_scheduler = LambdaLR(optimizer, lambda step: 0.5*(1+np.cos(np.pi*step/(EPOCHS*len(train_loader)))))

    grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        grad_scaler.load_state_dict(checkpoint['scaler_state_dict']) # FIX: use grad_scaler here
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"--- Optimizer and scheduler states loaded. Resuming from epoch {start_epoch}. ---")
    else:
        start_epoch = 0
        best_val_accuracy = -1.0
        
    try:
        model = torch.compile(model)
        print("  Model compiled successfully.")
    except Exception as e:
        print(f"Could not compile model: {e}. Running uncompiled.")
    
    # --- Training Loop ---
    print("  Starting model training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for data, targets_one_hot in train_loader:
            data, targets_one_hot = data.to(device), targets_one_hot.to(device)
            targets_idx = torch.argmax(targets_one_hot, dim=1)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(data)
                smoothed_targets = targets_one_hot * (1 - 0.1) + (0.1 / len(gesture_classes))
                per_sample_loss = F.cross_entropy(outputs, smoothed_targets, reduction='none')
                loss = (per_sample_loss * class_weight[targets_idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets_idx.size(0)
            correct_predictions += (predicted == targets_idx).sum().item()
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        model.eval()
        val_loss, val_correct_predictions, val_total_samples = 0.0, 0, 0
        with torch.no_grad():
            for data, targets_one_hot in val_loader:
                data, targets_one_hot = data.to(device), targets_one_hot.to(device)
                targets_idx = torch.argmax(targets_one_hot, dim=1)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = model(data)
                    v_loss = F.cross_entropy(outputs, targets_idx, weight=class_weight)
                val_loss += v_loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += targets_idx.size(0)
                val_correct_predictions += (predicted == targets_idx).sum().item()
        
        val_epoch_loss = val_loss / val_total_samples
        val_epoch_accuracy = val_correct_predictions / val_total_samples

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_accuracy:.4f}")

        is_best = val_epoch_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_epoch_accuracy
            epochs_no_improve = 0
            print(f"  Validation accuracy improved to {best_val_accuracy:.4f}. Saving best model checkpoint.")
        else:
            epochs_no_improve += 1
        
        if hasattr(model, '_orig_mod'):
            model_state_to_save = model._orig_mod.state_dict()
        else:
            model_state_to_save = model.state_dict()
            
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model_state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': grad_scaler.state_dict(),
            'best_val_accuracy': best_val_accuracy
        }
        save_checkpoint(checkpoint_state, is_best=is_best, directory=EXPORT_DIR)

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping after {PATIENCE} epochs without improvement.")
            break
    print("✔ Training done.")

else:
    # =====================================================================================
    # INFERENCE PIPELINE (FOR KAGGLE SUBMISSION)
    # =====================================================================================
    print("▶ INFERENCE MODE")
    PRETRAINED_DIR = EXPORT_DIR
    
    print("  Loading artifacts...")
    final_feature_cols = np.load(os.path.join(PRETRAINED_DIR, "feature_cols.npy"), allow_pickle=True).tolist()
    pad_len = int(np.load(os.path.join(PRETRAINED_DIR, "sequence_maxlen.npy")))
    gesture_classes = np.load(os.path.join(PRETRAINED_DIR, "gesture_classes.npy"), allow_pickle=True)
    
    # FIX: Load the correct scaler into the correct variable name
    feature_scaler = joblib.load(os.path.join(PRETRAINED_DIR, "scaler.pkl"))

    imu_dim_final = len([c for c in final_feature_cols if c.startswith(('acc_', 'rot_'))])
    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final

    print("  Loading model...")
    best_model_path = os.path.join(PRETRAINED_DIR, "best_checkpoint.pth")
    model = TwoBranchModel(pad_len, imu_dim_final, tof_thm_aggregated_dim_final, len(gesture_classes))
    
    checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
    
    # Use the same robust key-cleaning logic for inference
    saved_state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in saved_state_dict.keys()):
        cleaned_state_dict = {key.replace('_orig_mod.', ''): value for key, value in saved_state_dict.items()}
    else:
        cleaned_state_dict = saved_state_dict
    
    model.load_state_dict(cleaned_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("  Model, feature_scaler, and artifacts loaded – ready for evaluation.")


# =============================================================================
#  EVALUATION METRICS (Overall Accuracy, Precision, Recall, F1)
# =============================================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("\n=== Final Evaluation on Validation Set ===")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for data, targets_one_hot in val_loader:
        data, targets_one_hot = data.to(device), targets_one_hot.to(device)
        targets_idx = torch.argmax(targets_one_hot, dim=1)
        outputs = model(data)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets_idx.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Overall Accuracy
overall_acc = accuracy_score(all_labels, all_preds)

# Precision, Recall, F1 (macro and binary - binary không phù hợp với multi-class nên dùng weighted/macro)
macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

weighted_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
weighted_rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"Overall Accuracy : {overall_acc:.4f}")
print(f"Macro Precision  : {macro_prec:.4f}")
print(f"Macro Recall     : {macro_rec:.4f}")
print(f"Macro F1-score   : {macro_f1:.4f}")
print(f"Weighted Precision: {weighted_prec:.4f}")
print(f"Weighted Recall   : {weighted_rec:.4f}")
print(f"Weighted F1-score : {weighted_f1:.4f}")

# Optional: Full classification report
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=gesture_classes))



def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predicts the gesture for a single sequence using the trained PyTorch model.
    This is an optimized version with vectorized feature engineering.
    """
    df_seq = sequence.to_pandas()

    # --- OPTIMIZATION 1: Vectorized Feature Engineering ---
    # IMU Features (this part was already efficient)
    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
    df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)

    # ToF Features (using the fast NumPy method)
    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    tof_data_np = df_seq[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df_seq), 5, 64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice'); warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    for i in range(1, 6):
        df_seq[f'tof_{i}_mean'], df_seq[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
        df_seq[f'tof_{i}_min'], df_seq[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]

    # --- OPTIMIZATION 2: Efficient Column Reordering & Preprocessing ---
    # Select columns in the correct order and fill any missing values
    mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')

    # --- FIX: Use the correct 'feature_scaler' variable ---
    mat_scaled = feature_scaler.transform(mat_unscaled)

    # --- Padding ---
    padded_sequence = np.zeros((pad_len, len(final_feature_cols)), dtype='float32')
    seq_len = min(len(mat_scaled), pad_len)
    padded_sequence[:seq_len] = mat_scaled[:seq_len]

    # --- Prepare for Model ---
    model_input = torch.from_numpy(padded_sequence).float().unsqueeze(0).to(device)

    # --- Model Inference ---
    model.eval()
    with torch.no_grad():
        outputs = model(model_input) # Get raw logits
        # --- OPTIMIZATION 3: No need for extra softmax ---
        predicted_idx = torch.argmax(outputs, dim=1).item()

    # --- Map to Gesture Class ---
    return str(gesture_classes[predicted_idx])


import kaggle_evaluation.cmi_inference_server

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




