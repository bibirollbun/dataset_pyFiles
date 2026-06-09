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


import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, time, gc
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm



class SequenceDataset(Dataset):
    def __init__(self, df, window_size=64, step=32, features=None):
        self.seq_groups = [g for _, g in df.groupby("sequence_id")]
        self.window_size = window_size
        self.step = step
        self.features = features

    def __len__(self):
        return sum([(len(g)-self.window_size)//self.step + 1 for g in self.seq_groups])

    def __getitem__(self, idx):
        # Find correct sequence
        cum = 0
        for g in self.seq_groups:
            num = (len(g)-self.window_size)//self.step + 1
            if idx < cum + num:
                start = (idx - cum) * self.step
                seq = g[self.features].iloc[start:start+self.window_size].to_numpy(dtype=np.float32)
                return torch.tensor(seq)
            cum += num



# ====== TS2Vec Encoder ======
class TS2VecEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, emb_dim=64):
        super().__init__()
        if input_dim == 0:
            self.encoder = None
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, emb_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(1)
            )
    def forward(self, x):
        if self.encoder is None:
            return None
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.encoder(x).squeeze(-1)  # [B, emb_dim]
        return x


class FusionAttn(nn.Module):
    def __init__(self, imu_dim, tof_dim, thm_dim, num_classes, emb_dim=64):
        super().__init__()
        self.encoder_imu = TS2VecEncoder(imu_dim, emb_dim=emb_dim) if imu_dim > 0 else None
        self.encoder_tof = TS2VecEncoder(tof_dim, emb_dim=emb_dim) if tof_dim > 0 else None
        self.encoder_thm = TS2VecEncoder(thm_dim, emb_dim=emb_dim) if thm_dim > 0 else None

        fusion_input = emb_dim * sum([imu_dim > 0, tof_dim > 0, thm_dim > 0])
        assert fusion_input > 0, "No modality provided to FusionAttn."
        self.fusion = nn.Linear(fusion_input, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def encode(self, imu_x=None, tof_x=None, thm_x=None):
        embs = []
        if self.encoder_imu is not None and imu_x is not None and imu_x.size(-1) > 0:
            embs.append(self.encoder_imu(imu_x))
        if self.encoder_tof is not None and tof_x is not None and tof_x.size(-1) > 0:
            embs.append(self.encoder_tof(tof_x))
        if self.encoder_thm is not None and thm_x is not None and thm_x.size(-1) > 0:
            embs.append(self.encoder_thm(thm_x))

        if len(embs) == 0:
            raise ValueError("encode() received no valid modality tensors.")

        fused = torch.cat(embs, dim=1)
        fused = torch.relu(self.fusion(fused))  # [B, emb_dim]
        return fused

    def forward(self, imu_x=None, tof_x=None, thm_x=None):
        fused = self.encode(imu_x, tof_x, thm_x)
        return self.classifier(fused)  # logits



# ====== Contrastive loss ======
def contrastive_loss(z1, z2, temperature=0.1):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(len(z1)).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


window_sizes = [32, 64]
feature_sets = {
    "IMU": [c for c in train_df.columns if c.startswith(("acc_", "rot_"))],
    #"Thermo": [c for c in train_df.columns if c.startswith("thm_")],
    #"ToF": [c for c in train_df.columns if c.startswith("tof_") and not c.endswith(tuple([f"_v{i}" for i in range(64)]))],
    #"All": [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_")) and "_v" not in c],
}

results = []


device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
batch_size = 256

# ====== Cột cảm biến ======
imu_cols = [c for c in train_df.columns if c.startswith("acc_") or c.startswith("rot_")]
tof_cols = [c for c in train_df.columns if c.startswith("tof_")]
thm_cols = [c for c in train_df.columns if c.startswith("thm_")]

print(f"IMU cols: {len(imu_cols)}, ToF cols: {len(tof_cols)}, Thm cols: {len(thm_cols)}")

# Đảm bảo le_gesture đã được khởi tạo
from sklearn.preprocessing import LabelEncoder
le_gesture = LabelEncoder()
le_gesture.fit(train_df['gesture'])  # Fit LabelEncoder cho cột 'gesture'
num_classes = len(le_gesture.classes_)

# Kiểm tra số lớp của le_gesture
print(f"Number of classes in 'gesture': {len(le_gesture.classes_)}")

# Đảm bảo feature_sets và window_sizes đã được định nghĩa
# feature_sets và window_sizes cần phải có để thực thi vòng lặp tiếp theo

# Mô hình FusionAttn
model = FusionAttn(len(imu_cols), len(tof_cols), len(thm_cols), num_classes=len(le_gesture.classes_)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ====== Training Loop ======
results = []

# Huấn luyện FusionAttn
for features_used, feature_cols in feature_sets.items():
    for ws in window_sizes:
        print(f"\n=== Training FusionAttn ({features_used}, window={ws}) ===")

        # Tách cột theo feature_cols hiện tại
        feat_imu = [c for c in feature_cols if c.startswith(("acc_", "rot_"))]
        feat_tof = [c for c in feature_cols if c.startswith("tof_")]
        feat_thm = [c for c in feature_cols if c.startswith("thm_")]

        d_imu, d_tof, d_thm = len(feat_imu), len(feat_tof), len(feat_thm)

        # Dataset/DataLoader theo đúng feature_cols
        trainset = SequenceDataset(train_df, window_size=ws, step=ws//2, features=feature_cols)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Model/optimizer khởi tạo theo dims hiện tại
        model = FusionAttn(d_imu, d_tof, d_thm, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        for ep in range(EPOCHS):
            model.train()
            total_loss = 0
            for X in trainloader:  # X: [B, T, D_sel]
    # Cắt theo d_imu, d_tof, d_thm của feature set hiện tại
                i0 = 0
                imu_x = X[:, :, i0:i0+d_imu] if d_imu > 0 else None; i0 += d_imu
                tof_x = X[:, :, i0:i0+d_tof] if d_tof > 0 else None; i0 += d_tof
                thm_x = X[:, :, i0:i0+d_thm] if d_thm > 0 else None

                if imu_x is not None: imu_x = imu_x.to(device)
                if tof_x is not None: tof_x = tof_x.to(device)
                if thm_x is not None: thm_x = thm_x.to(device)

    # Augment chỉ khi không None
                x1_imu = imu_x + 0.01*torch.randn_like(imu_x) if imu_x is not None else None
                x2_imu = imu_x + 0.01*torch.randn_like(imu_x) if imu_x is not None else None
                x1_tof = tof_x + 0.01*torch.randn_like(tof_x) if tof_x is not None else None
                x2_tof = tof_x + 0.01*torch.randn_like(tof_x) if tof_x is not None else None
                x1_thm = thm_x + 0.01*torch.randn_like(thm_x) if thm_x is not None else None
                x2_thm = thm_x + 0.01*torch.randn_like(thm_x) if thm_x is not None else None

    # --- Quan trọng: dùng EMBEDDING cho contrastive loss ---
                z1 = model.encode(x1_imu, x1_tof, x1_thm)   # [B, emb_dim]
                z2 = model.encode(x2_imu, x2_tof, x2_thm)   # [B, emb_dim]
                loss = contrastive_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                total_loss += loss.item()
            print(f"Epoch {ep+1}/{EPOCHS} - Loss: {total_loss/len(trainloader):.4f}")
        
        train_time = round((time.time() - start_time)/60, 2)
        print(f"Training completed for window size {ws}. Time: {train_time} minutes.")



from sklearn.linear_model import LogisticRegression
features = [c for c in train_df.columns if c.startswith(("acc_","rot_","thm_","tof_"))]

model.eval()
encoder = TS2VecEncoder(len(features)).to(device)

encoder.eval()
seq_embeds, seq_labels = [], []
infer_times = []


for sid, g in train_df.groupby("sequence_id"):
    x = torch.tensor(g[features].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(x)          # ✅ không permute nữa!
        t0 = time.time()
        t1 = time.time()
    infer_times.append((t1 - t0) * 1000)  # ms per sequence
    seq_embeds.append(z.cpu().numpy().squeeze())
    #seq_embeds.append(z)
    seq_labels.append(g["gesture_id"].iloc[0])
X, y = np.array(seq_embeds), np.array(seq_labels)

# Remove or fill NaN values in embedding matrix
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
inference_time_ms = np.mean(infer_times)

clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(X, y)
y_pred = clf.predict(X)
macroF1 = f1_score(y, y_pred, average="macro")
binaryF1 = f1_score((y>0).astype(int), (y_pred>0).astype(int), average="binary")
acc = accuracy_score(y, y_pred)
final_score = (binaryF1 + macroF1)/2

print(f"Macro F1: {macroF1:.4f} | Binary F1: {binaryF1:.4f}")


feature_sets = {
    "IMU": [c for c in train_df.columns if c.startswith(("acc_", "rot_"))],
    #"Thermo": [c for c in train_df.columns if c.startswith("thm_")],
    #"ToF": [c for c in train_df.columns if c.startswith("tof_") and "_v" not in c],
    #"All": [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_")) and "_v" not in c],
}

window_sizes = [32, 64, 128]

results = []

for features_used, feature_cols in feature_sets.items():
    for ws in window_sizes:
        print(f"\n=== LinearEval ({features_used}, window={ws}) ===")

        # 1️⃣ khởi tạo encoder khớp feature set
        encoder = TS2VecEncoder(input_dim=len(feature_cols), hidden_dim=128).to(device)
        encoder.eval()

        seq_embeds, seq_labels, infer_times = [], [], []
        with torch.no_grad():
            for sid, g in train_df.groupby("sequence_id"):
                x_np = g[feature_cols].to_numpy(dtype=np.float32)
                if len(x_np) < ws:  # skip sequence quá ngắn
                    continue
                x = torch.from_numpy(x_np[:ws]).unsqueeze(0).to(device)
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                z = encoder(x)
                if device == "cuda": torch.cuda.synchronize()
                t1 = time.time()
                infer_times.append((t1 - t0) * 1000.0)
                seq_embeds.append(z.squeeze(0).cpu().numpy())
                seq_labels.append(g["gesture_id"].iloc[0])

        if len(seq_embeds) == 0:
            print("⚠️ Skip — no valid sequences.")
            continue

        X = np.nan_to_num(np.array(seq_embeds), nan=0.0)
        y = np.array(seq_labels)

        if np.unique(y).size < 2:
            print("⚠️ Not enough classes — skipped.")
            continue

        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X, y)
        y_pred = clf.predict(X)
        macroF1 = f1_score(y, y_pred, average="macro")
        binaryF1 = f1_score((y>0).astype(int), (y_pred>0).astype(int), average="binary")
        acc = accuracy_score(y, y_pred)
        final_score = (binaryF1 + macroF1)/2
        inference_time_ms = np.mean(infer_times)

        results.append({
            "model_name": "FusionAttn (Fine-tuned)",
            "features_used": features_used,
            "window_size": ws,
            "Binary F1": binaryF1,
            "Macro F1": macroF1,
            "Final Score": final_score,
            "val_acc": acc,
            "inference_time (ms/seq)": round(inference_time_ms, 4)
        })






import pandas as pd
results_df = pd.DataFrame(results)
display(results_df)

best_each = results_df.loc[results_df.groupby("features_used")["Final Score"].idxmax()]
print("\nBest model per feature type:")
display(best_each)




# Save results to CSV
results_df.to_csv("fusionattn_results.csv", index=False)
print("✅ FusionAttn results saved to fusionattn_results.csv")



import pandas as pd

results_df = pd.DataFrame(results)
display(results_df)

best_each = results_df.loc[results_df.groupby("features_used")["Final Score"].idxmax()]
print("\nBest model per feature type:")
display(best_each)

# Save results to CSV
results_df.to_csv("fusionattn_results.csv", index=False)
print("✅ FusionAttn results saved to fusionattn_results.csv")



import numpy as np
import joblib
import torch
import os

# Giả sử:
# - model: FusionAttn đã train xong (fine-tuned)
# - feature_cols: list các cột đang dùng (ở Notebook 5 bạn đang dùng "All")
# - le_gesture: LabelEncoder fit trên train_df['gesture']
# - ws: window_size đang dùng cho model (ví dụ 100)

ARTIFACT_DIR = "/kaggle/working"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1) Lưu danh sách feature columns
np.save(os.path.join(ARTIFACT_DIR, "feature_cols.npy"),
        np.array(feature_cols, dtype=object))
print("✔ Saved feature_cols.npy")

# 2) Lưu danh sách lớp (gesture classes)
np.save(os.path.join(ARTIFACT_DIR, "gesture_classes.npy"),
        le_gesture.classes_)
print("✔ Saved gesture_classes.npy")

# 3) Lưu sequence length (window size)
np.save(os.path.join(ARTIFACT_DIR, "sequence_maxlen.npy"),
        np.array([ws]))
print("✔ Saved sequence_maxlen.npy")

# 4) (Optional) Lưu scaler nếu bạn có dùng chuẩn hóa
if 'scaler' in globals() and scaler is not None:
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    print("✔ Saved scaler.pkl")
else:
    print("ℹ️ No scaler to save (scaler is None or not defined).")

# 5) Lưu checkpoint FusionAttn
CKPT_PATH = os.path.join(ARTIFACT_DIR, f"FusionAttn_IMU_L{ws}.pth")
torch.save(model.state_dict(), CKPT_PATH)
print(f"✔ Saved FusionAttn checkpoint to {CKPT_PATH}")


