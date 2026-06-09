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
        self.seq_groups = []
        for sid, g in df.groupby("sequence_id"):
            if len(g) >= window_size:      # ✅ chỉ lấy sequence đủ dài
                self.seq_groups.append(g)
        self.window_size = window_size
        self.step = step
        self.features = features

    def __len__(self):
        total = 0
        for g in self.seq_groups:
            num = max((len(g) - self.window_size) // self.step + 1, 0)
            total += num
        return total

    def __getitem__(self, idx):
        cum = 0
        for g in self.seq_groups:
            num = max((len(g) - self.window_size) // self.step + 1, 0)
            if idx < cum + num:
                start = (idx - cum) * self.step
                seq = g[self.features].iloc[start:start+self.window_size].to_numpy(dtype=np.float32)
                return torch.tensor(seq)
            cum += num
        raise IndexError



class TS2VecEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)
        z = self.encoder(x).squeeze(-1)
        return self.projector(z)



def contrastive_loss(z1, z2, temp=0.1):
    z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(len(z1)).to(z1.device)
    return nn.functional.cross_entropy(logits, labels)



# === Configs ===
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
window_sizes = [32, 64, 128]
feature_sets = {
    "IMU": [c for c in train_df.columns if c.startswith(("acc_", "rot_"))],
    "Thermo": [c for c in train_df.columns if c.startswith("thm_")],
    #"ToF": [c for c in train_df.columns if c.startswith("tof_") and not c.endswith(tuple([f"_v{i}" for i in range(64)]))],
    "All": [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_")) and "_v" not in c],
}

results = []


device = "cuda" if torch.cuda.is_available() else "cpu"
features = [c for c in train_df.columns if c.startswith(("acc_","rot_","thm_","tof_"))]

#trainset = SequenceDataset(train_df, window_size=64, step=32, features=features)
#trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

encoder = TS2VecEncoder(len(features)).to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

for features_used, feature_cols in feature_sets.items():
    for ws in window_sizes:
        print(f"\n=== Training TS2Vec ({features_used}, window={ws}) ===")
        trainset = SequenceDataset(train_df, window_size=ws, step=ws//2, features=feature_cols)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        encoder = TS2VecEncoder(len(feature_cols)).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

        start_time = time.time()
        for ep in range(EPOCHS):
            encoder.train()
            total_loss = 0
            for x in trainloader:
                x = x.to(device)
                x1 = x + 0.01 * torch.randn_like(x)
                x2 = x + 0.01 * torch.randn_like(x)
                z1, z2 = encoder(x1), encoder(x2)
                loss = contrastive_loss(z1, z2)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {ep+1}/{EPOCHS} - Loss: {total_loss/len(trainloader):.4f}")
        train_time = round((time.time() - start_time)/60, 2)




print(x.shape)  # -> (1, T, D)



from sklearn.linear_model import LogisticRegression

encoder.eval()
seq_embeds, seq_labels = [], []
infer_times = []
encoder = TS2VecEncoder(input_dim=332, hidden_dim=128)


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
    "Thermo": [c for c in train_df.columns if c.startswith("thm_")],
    #"ToF": [c for c in train_df.columns if c.startswith("tof_") and "_v" not in c],
    "All": [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_")) and "_v" not in c],
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
            "model_name": "TS2Vec (LinearEval)",
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

results_df.to_csv("ts2vec_linear_eval_results.csv", index=False)


