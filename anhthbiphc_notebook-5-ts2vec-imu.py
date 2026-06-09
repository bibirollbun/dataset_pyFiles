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


# Táº£i táº­p dá»¯ liá»‡u
#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p huáº¥n luyá»‡n (Train Sensor Data): 
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")

#ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p huáº¥n luyá»‡n (giá»›i tÃ­nh, tuá»•i, ...): 
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p kiá»ƒm tra:
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")

# ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p kiá»ƒm tra:
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
        print("  âœ… No missing values.")
    print()



#Kiá»ƒm tra trÃ¹ng láº·p (Duplicate Rows): .duplicated(): Tráº£ vá»� Series boolean, dÃ²ng nÃ o trÃ¹ng láº·p sáº½ lÃ  True.
#                                     .sum(): Ä�áº¿m tá»•ng sá»‘ dÃ²ng trÃ¹ng láº·p.
#Kiá»ƒm tra xem dá»¯ liá»‡u cÃ³ báº£n ghi nÃ o bá»‹ trÃ¹ng khÃ´ng â€” quan trá»�ng cho viá»‡c lÃ m sáº¡ch dá»¯ liá»‡u.

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong train_df
train_duplicates = train_df.duplicated().sum()

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong  test_df
test_duplicates = test_df.duplicated().sum()

# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong train_dem_df (optional)
train_dem_duplicates = train_dem_df.duplicated().sum()
# Ä�áº¿m cÃ¡c hÃ ng trÃ¹ng láº·p trong test_dem_df (optional)
test_dem_duplicates = test_dem_df.duplicated().sum()

# In sá»‘ lÆ°á»£ng dÃ²ng trÃ¹ng:
print(f"Number of duplicate rows in train_df: {train_duplicates}")
print(f"Number of duplicate rows in test_df: {test_duplicates}")
print(f"Number of duplicate rows in train_dem_df: {train_dem_duplicates}")
print(f"Number of duplicate rows in test_dem_df: {test_dem_duplicates}")


def null_percent(df):
    per=((df.isnull().sum()/len(df))*100).round(2)
    return per[per>0]

print("Nan Values in Train data")
print(null_percent(train_df))


# --- 2.4 Há»£p nháº¥t dá»¯ liá»‡u nhÃ¢n kháº©u há»�c vÃ  cáº£m biáº¿n ---
merged_train = pd.merge(train_df, train_dem_df, on="subject", how="left")
merged_test = pd.merge(test_df, test_dem_df, on="subject", how="left")

print("Merged Train Shape:", merged_train.shape)
print("Merged Test Shape:", merged_test.shape)
display(merged_train.head(2))



# --- Thá»‘ng kÃª mÃ´ táº£ toÃ n bá»™ merged_train ---
merged_train.describe().T



#Kiá»ƒm tra thiáº¿u giÃ¡ trá»‹ & thá»‘ng kÃª cáº£m biáº¿n: 
#Danh sÃ¡ch cÃ¡c cá»™t khÃ´ng thuá»™c cáº£m biáº¿n (cÃ³ thá»ƒ lÃ  ID, thÃ´ng tin khÃ¡c). 

excluded_prefixes = ('acc_', 'rot_', 'thm_', 'tof_')
sensor_cols = [col for col in train_df.columns if not col.startswith(excluded_prefixes)]

# Sensor Data Summary for TRAIN
#isnull().sum(): Ä�áº¿m sá»‘ giÃ¡ trá»‹ bá»‹ thiáº¿u.
missing_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    '[TRAIN] Missing Count': train_df[sensor_cols].isnull().sum().values,
    '[TRAIN] Missing %': (train_df[sensor_cols].isnull().sum().values / len(train_df)) * 100
})

#nunique(): Ä�áº¿m sá»‘ lÆ°á»£ng giÃ¡ trá»‹ duy nháº¥t.
unique_sensor_train = pd.DataFrame({
    'Feature': sensor_cols,
    'Unique Values [TRAIN]': train_df[sensor_cols].nunique().values
})

#dtypes: Láº¥y kiá»ƒu dá»¯ liá»‡u cá»§a tá»«ng cá»™t.
dtypes_sensor = pd.DataFrame({
    'Feature': sensor_cols,
    'Data Type': train_df[sensor_cols].dtypes.values
})

# Merge all summaries (NO test set)
#merge: Gá»™p cÃ¡c báº£ng thá»‘ng kÃª thÃ nh báº£ng duy nháº¥t theo Feature.
sensor_summary = missing_sensor_train \
    .merge(unique_sensor_train, on='Feature', how='left') \
    .merge(dtypes_sensor, on='Feature', how='left')

# Display styled DataFrame (mask NaNs just for styling)
#fillna(0): Ä�iá»�n giÃ¡ trá»‹ thiáº¿u báº±ng 0 (cho Ä‘áº¹p máº¯t khi hiá»ƒn thá»‹).
#.style.background_gradient: TÃ´ mÃ u ná»�n theo giÃ¡ trá»‹ giÃºp dá»… nhÃ¬n.
styled_df = sensor_summary.fillna(0)
styled_df.style.background_gradient(cmap='viridis')


#Thá»‘ng kÃª tÆ°Æ¡ng tá»± cho nhÃ¢n kháº©u há»�c: TÆ°Æ¡ng tá»± nhÆ° pháº§n thá»‘ng kÃª cáº£m biáº¿n nhÆ°ng Ã¡p dá»¥ng cho dá»¯ liá»‡u nhÃ¢n kháº©u há»�c.

# Cá»™t nhÃ¢n kháº©u há»�c (khÃ´ng loáº¡i trá»«)
dem_cols = train_dem_df.columns

# GiÃ¡ trá»‹ bá»‹ thiáº¿u trong dá»¯ liá»‡u nhÃ¢n kháº©u há»�c cá»§a train 
missing_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    '[TRAIN DEMO] Missing Count': train_dem_df[dem_cols].isnull().sum().values,
    '[TRAIN DEMO] Missing %': (train_dem_df[dem_cols].isnull().sum().values / len(train_dem_df)) * 100
})

# GiÃ¡ trá»‹ duy nháº¥t Ä‘Æ°á»£c tÃ­nh trong dá»¯ liá»‡u nhÃ¢n kháº©u há»�c cá»§a train 
unique_demo_train = pd.DataFrame({
    'Feature': dem_cols,
    'Unique Values [TRAIN DEMO]': train_dem_df[dem_cols].nunique().values
})

# Data types
dtypes_demo = pd.DataFrame({
    'Feature': dem_cols,
    'Data Type': train_dem_df[dem_cols].dtypes.values
})

# TÃ³m táº¯t há»£p nháº¥t (chá»‰ dÃ nh cho Ä‘Ã o táº¡o)
demo_summary = (
    missing_demo_train
    .merge(unique_demo_train, on='Feature', how='left')
    .merge(dtypes_demo, on='Feature', how='left')
)

# Hiá»ƒn thá»‹ tÃ³m táº¯t theo phong cÃ¡ch
demo_summary.style.background_gradient(cmap='viridis')


import numpy as np
import pandas as pd

# 1) Sao chÃ©p Ä‘Ã o táº¡o vÃ  kiá»ƒm tra Ä‘á»ƒ chÃºng ta khÃ´ng sá»­a Ä‘á»•i DataFrame gá»‘c
train_temp = train_df.copy()
test_temp  = test_df.copy()

# 2) GIA Tá»�C Káº¾: tÃ­nh toÃ¡n Ä‘á»™ lá»›n táº¡i má»—i dáº¥u thá»�i gian
train_temp['acc_mag'] = np.sqrt(
    train_temp['acc_x']**2 + train_temp['acc_y']**2 + train_temp['acc_z']**2
)
test_temp['acc_mag'] = np.sqrt(
    test_temp['acc_x']**2 + test_temp['acc_y']**2 + test_temp['acc_z']**2
)

# 3) ROTATION: tÃ­nh toÃ¡n â€œgÃ³c quayâ€� tá»« thÃ nh pháº§n quaternion w
# (LÆ°u Ã½: rot_w náº±m trong [-1,1], do Ä‘Ã³ arccos há»£p lá»‡. ChÃºng tÃ´i bá»� qua NaN náº¿u cÃ³.)
train_temp['rot_angle'] = 2 * np.arccos(train_temp['rot_w'].clip(-1,1))
test_temp['rot_angle']  = 2 * np.arccos(test_temp['rot_w'].clip(-1,1))

# 4) NhÃ³m theo sequence_id vÃ  tá»•ng há»£p cÃ¡c tÃ³m táº¯t gia tá»‘c káº¿
acc_agg_funcs = {
    'acc_mag': ['mean', 'std', 'max']
}
train_acc_summary = train_temp.groupby('sequence_id').agg(acc_agg_funcs)
test_acc_summary  = test_temp.groupby('sequence_id').agg(acc_agg_funcs)

# LÃ m pháº³ng cá»™t MultiIndex
train_acc_summary.columns = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]
test_acc_summary.columns  = ['acc_mag_' + stat for stat in ['mean', 'std', 'max']]

# 5) NhÃ³m theo sequence_id vÃ  tá»•ng há»£p tÃ³m táº¯t vÃ²ng quay
rot_agg_funcs = {
    'rot_angle': ['mean', 'std', 'max']
}
train_rot_summary = train_temp.groupby('sequence_id').agg(rot_agg_funcs)
test_rot_summary  = test_temp.groupby('sequence_id').agg(rot_agg_funcs)

train_rot_summary.columns = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]
test_rot_summary.columns  = ['rot_angle_' + stat for stat in ['mean', 'std', 'max']]

# 6) NHIá»†T Ä�á»˜: nÄƒm cáº£m biáº¿n thm_1 â€¦ thm_5
thm_cols = [f"thm_{i}" for i in range(1, 6)]

# XÃ¡c Ä‘á»‹nh hÃ m tá»•ng há»£p: trung bÃ¬nh + Ä‘á»™ lá»‡ch chuáº©n
thm_agg_funcs = {col: ['mean', 'std'] for col in thm_cols}

train_thm_summary = train_temp.groupby('sequence_id').agg(thm_agg_funcs)
test_thm_summary  = test_temp.groupby('sequence_id').agg(thm_agg_funcs)

# LÃ m pháº³ng cÃ¡c cá»™t MultiIndex
flattened_thm_cols = []
for sensor in thm_cols:
    for stat in ['mean','std']:
        flattened_thm_cols.append(f"{sensor}_{stat}")

train_thm_summary.columns = flattened_thm_cols
test_thm_summary.columns  = flattened_thm_cols

# 7) THá»œI GIAN BAY: má»—i cáº£m biáº¿n i cÃ³ 64 cá»™t pixel: tof_i_v0 â€¦ tof_i_v63
# ChÃºng ta sáº½ táº¡o má»™t â€œtof_i_mean_at_tsâ€� cho má»—i dáº¥u thá»�i gian, sau Ä‘Ã³ tá»•ng há»£p theo tá»«ng chuá»—i.

def compute_tof_sequence_summary(df):
    # Khá»Ÿi táº¡o má»™t dict Ä‘á»ƒ giá»¯ DataFrames theo tá»«ng chuá»—i
    seq_summaries = {}

    for i in range(1, 6):
        # XÃ¢y dá»±ng danh sÃ¡ch cÃ¡c cá»™t cho cáº£m biáº¿n i
        tof_cols = [f"tof_{i}_v{pix}" for pix in range(64)]
        # Thay tháº¿ -1 báº±ng NaN Ä‘á»ƒ chÃºng khÃ´ng lÃ m lá»‡ch giÃ¡ trá»‹ trung bÃ¬nh; chuyá»ƒn Ä‘á»•i sang float
        ts_grid = df[tof_cols].replace(-1, np.nan).astype(float)
        # TÃ­nh toÃ¡n â€œtrung bÃ¬nh trÃªn táº¥t cáº£ 64 pixelâ€� cho má»—i dáº¥u thá»�i gian
        df[f"tof_{i}_mean_at_ts"] = ts_grid.mean(axis=1)
    
   # BÃ¢y giá»�, nhÃ³m theo id chuá»—i vÃ  tÃ­nh giÃ¡ trá»‹ trung bÃ¬nh vÃ  Ä‘á»™ lá»‡ch chuáº©n cá»§a cÃ¡c giÃ¡ trá»‹ trung bÃ¬nh Ä‘Ã³
    agg_dict = {f"tof_{i}_mean_at_ts": ['mean','std'] for i in range(1, 6)}
    summary = df.groupby('sequence_id').agg(agg_dict)
    # LÃ m pháº³ng cÃ¡c cá»™t MultiIndex
    flat_cols = [f"tof_{i}_{stat}" for i in range(1, 6) for stat in ['mean','std']]
    summary.columns = flat_cols
    return summary

train_tof_summary = compute_tof_sequence_summary(train_temp)
test_tof_summary  = compute_tof_sequence_summary(test_temp)

# 8) Há»£p nháº¥t cÃ¡c tÃ³m táº¯t accel, rotation, thm, tof (trÃªn sequence_id)
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

# 9) ThÃªm cá»™t â€œDatasetâ€� Ä‘á»ƒ chÃºng ta cÃ³ thá»ƒ thá»±c hiá»‡n box+hist song song
train_sensor_summary['Dataset'] = 'Train'
test_sensor_summary['Dataset']  = 'Test'

# 10) GhÃ©p ná»‘i thÃ nh má»™t DataFrame Ä‘á»ƒ váº½ Ä‘á»“ thá»‹
combined_sensor_summary = pd.concat(
    [train_sensor_summary, test_sensor_summary],
    axis=0
).reset_index(drop=True)


import numpy as np
import matplotlib.pyplot as plt

# (1) SÃ¡p nháº­p train_demographics vÃ o train_df náº¿u chÆ°a thá»±c hiá»‡n
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
    # 4) Feature cÆ¡ báº£n (vÃ­ dá»¥):
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
            if len(g) >= window_size:      # âœ… chá»‰ láº¥y sequence Ä‘á»§ dÃ i
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



import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.07):
    """
    Contrastive loss (InfoNCE-style) with temperature scaling.
    """
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # Similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size).long().to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss



# === Configs ===
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 500
window_sizes = [128, 256, 512]
feature_sets = {
    "IMU": [c for c in train_df.columns if c.startswith(("acc_", "rot_"))],
    #"Thermo": [c for c in train_df.columns if c.startswith("thm_")],
    #"ToF": [c for c in train_df.columns if c.startswith("tof_") and not c.endswith(tuple([f"_v{i}" for i in range(64)]))],
    #"All": [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_")) and "_v" not in c],
}

results = []


import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
features = [c for c in train_df.columns if c.startswith(("acc_", "rot_", "thm_", "tof_"))]

encoder = TS2VecEncoder(len(features)).to(device)
optimizer = optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)

for features_used, feature_cols in feature_sets.items():
    for ws in window_sizes:
        print(f"\n=== Training TS2Vec ({features_used}, window={ws}) ===")

        trainset = SequenceDataset(train_df, window_size=ws, step=ws//2, features=feature_cols)
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

        encoder = TS2VecEncoder(len(feature_cols)).to(device)
        optimizer = optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)

        start_time = time.time()

        # âœ… Augmentation Ä‘á»‹nh nghÄ©a bÃªn ngoÃ i
        def augment(x, noise_std=0.03, drop_prob=0.2):
            x = x + noise_std * torch.randn_like(x)
            mask = (torch.rand_like(x[..., 0]) > drop_prob).float().unsqueeze(-1)
            return x * mask

        # âœ… Huáº¥n luyá»‡n thá»±c sá»±
        for ep in range(EPOCHS):
            encoder.train()
            total_loss = 0
            for x in trainloader:
                x = x.to(device)

                # ğŸ’¡ Táº¡o 2 phiÃªn báº£n augment khÃ¡c nhau
                x1, x2 = augment(x), augment(x)

                # ğŸ”¥ Chuáº©n hoÃ¡ embedding
                z1, z2 = F.normalize(encoder(x1), dim=-1), F.normalize(encoder(x2), dim=-1)

                # ğŸ§  Contrastive loss cÃ³ temperature
                loss = contrastive_loss(z1, z2, temperature=0.05)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {ep+1}/{EPOCHS} - Loss: {total_loss/len(trainloader):.4f}")

        train_time = round((time.time() - start_time)/60, 2)
        torch.save(encoder.state_dict(), f"ts2vec_{features_used}_window{ws}.pth")
        print(f"âœ… Saved encoder for {features_used}, window={ws}")



#print(x.shape)  # -> (1, T, D)



from sklearn.linear_model import LogisticRegression

encoder = TS2VecEncoder(len(feature_cols)).to(device)
encoder.load_state_dict(torch.load(f"ts2vec_{features_used}_window{ws}.pth", map_location=device))
encoder.eval()
seq_embeds, seq_labels = [], []
infer_times = []


for sid, g in train_df.groupby("sequence_id"):
    x = torch.tensor(g[feature_cols].to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():        z = encoder(x)          # âœ… khÃ´ng permute ná»¯a!
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



results = []

for features_used, feature_cols in feature_sets.items():
    for ws in window_sizes:
        print(f"\n=== LinearEval ({features_used}, window={ws}) ===")

        # 1ï¸�âƒ£ Khá»Ÿi táº¡o encoder khá»›p feature set
        encoder = TS2VecEncoder(input_dim=len(feature_cols), hidden_dim=128).to(device)
        encoder.eval()

        seq_embeds, seq_labels, infer_times = [], [], []

        with torch.no_grad():
            stride = ws // 2  # 50% overlap
            
            # âœ… Láº·p qua tá»«ng sequence_id
            for sid, g in train_df.groupby("sequence_id"):
                x_np = g[feature_cols].to_numpy(dtype=np.float32)
                
                # â�Œ Bá»� qua náº¿u quÃ¡ ngáº¯n
                if len(x_np) < ws:
                    continue

                # ğŸ”� Chia thÃ nh nhiá»�u sub-windows cÃ³ overlap
                for start in range(0, len(x_np) - ws + 1, stride):
                    sub_x = x_np[start:start + ws]
                    x = torch.from_numpy(sub_x).unsqueeze(0).to(device)

                    if device == "cuda": torch.cuda.synchronize()
                    t0 = time.time()
                    z = encoder(x)
                    if device == "cuda": torch.cuda.synchronize()
                    t1 = time.time()

                    infer_times.append((t1 - t0) * 1000.0)
                    seq_embeds.append(z.squeeze(0).cpu().numpy())
                    seq_labels.append(g["gesture_id"].iloc[0])

        # âš ï¸� Kiá»ƒm tra káº¿t quáº£ thu Ä‘Æ°á»£c
        if len(seq_embeds) == 0:
            print("âš ï¸� Skip â€” no valid sequences.")
            continue

        X = np.nan_to_num(np.array(seq_embeds), nan=0.0)
        y = np.array(seq_labels)

        if np.unique(y).size < 2:
            print("âš ï¸� Not enough classes â€” skipped.")
            continue

        # ğŸ§  Huáº¥n luyá»‡n Linear classifier
        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X, y)
        y_pred = clf.predict(X)

        macroF1 = f1_score(y, y_pred, average="macro")
        binaryF1 = f1_score((y > 0).astype(int), (y_pred > 0).astype(int), average="binary")
        acc = accuracy_score(y, y_pred)
        final_score = (binaryF1 + macroF1) / 2
        inference_time_ms = np.mean(infer_times)

        # âœ… LÆ°u káº¿t quáº£
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

        print(f"âœ… Added result: {features_used}, ws={ws}, Final Score={final_score:.4f}")



import pandas as pd
results_df = pd.DataFrame(results)
display(results_df)

best_each = results_df.loc[results_df.groupby("features_used")["Final Score"].idxmax()]
print("\nBest model per feature type:")
display(best_each)

results_df.to_csv("ts2vec_linear_eval_results.csv", index=False)


