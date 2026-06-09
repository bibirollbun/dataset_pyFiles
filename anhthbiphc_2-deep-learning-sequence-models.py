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


# %% [setup] ThÆ° viá»‡n & kiá»ƒm tra mÃ´i trÆ°á»�ng
import os, math, json, time, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
    print('ChÆ°a cÃ³ PyTorch. CÃ i torch Ä‘á»ƒ cháº¡y cÃ¡c Ã´ huáº¥n luyá»‡n. Lá»—i:', e)

DEVICE = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)
    if DEVICE=='cuda': torch.cuda.manual_seed_all(SEED)
print({'torch': HAS_TORCH, 'device': DEVICE})



# %% [config]
CONFIG = {
    'DATA_PATH': '',  # vÃ­ dá»¥: '/mnt/data/your_dataset.npz'

    # Synthetic dataset náº¿u khÃ´ng cÃ³ dá»¯ liá»‡u tháº­t
    'N_SAMPLES': 900,
    'N_CLASSES': 6,
    'T': 64,
    'IMU_CH': 6,
    'THERM_CH': 8,
    'TOF_H': 8, 'TOF_W': 8,

    # Huáº¥n luyá»‡n
    'BATCH_SIZE': 64,
    'MAX_EPOCHS': 10,
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-4,
    'EARLY_STOP_PATIENCE': 5,
    'VAL_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,

    # Regularization
    'DROPOUT': 0.2,
    'USE_LAYER_NORM': True,

    # Danh sÃ¡ch thÃ­ nghiá»‡m cáº§n cháº¡y
    'RUN_SET': [
        'IMU_CNN1D', 'IMU_LSTM', 'IMU_GRU', 'IMU_BiLSTM',
        'TOF_CNN2D',
        'MULTI_CONCAT_CNN1D+CNN2D'
    ]
}
CONFIG



# %% [data]
from typing import Dict

def make_synthetic(cfg: Dict):
    N, C, T = cfg['N_SAMPLES'], cfg['N_CLASSES'], cfg['T']
    IMU_CH, THERM_CH = cfg['IMU_CH'], cfg['THERM_CH']
    H, W = cfg['TOF_H'], cfg['TOF_W']
    y = np.random.randint(0, C, size=N)
    imu = np.random.randn(N, T, IMU_CH).astype('float32')
    thermo = (0.3*np.random.randn(N, T, THERM_CH) + 0.1).astype('float32')
    tof = (np.random.randn(N, T, H, W)*0.2 + 2.5).astype('float32')
    # pattern theo lá»›p
    for cls in range(C):
        idx = y==cls
        imu[idx] += (cls - C/2) * 0.15
        thermo[idx] += (C/2 - cls) * 0.1
        rr, cc = np.mgrid[:H, :W]
        center_r, center_c = (cls % H), (cls % W)
        mask = np.exp(-((rr-center_r)**2 + (cc-center_c)**2)/(2*(H/4)**2)).astype('float32')
        tof[idx] += mask[None, None, :, :]
    return imu, thermo, tof, y.astype('int64')

DATA_PATH = CONFIG['DATA_PATH']
if DATA_PATH and os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH)
    imu = data['imu'].astype('float32')
    thermo = data['thermo'].astype('float32')
    tof = data['tof'].astype('float32')
    y = data['label'].astype('int64')
    print('Loaded real dataset:', DATA_PATH)
else:
    imu, thermo, tof, y = make_synthetic(CONFIG)
    print('Using synthetic dataset')

N, T, IMU_CH = imu.shape
THERM_CH = thermo.shape[-1]
H, W = tof.shape[2], tof.shape[3]
N_CLASSES = int(y.max())+1
print({'N': N, 'T': T, 'IMU_CH': IMU_CH, 'THERM_CH': THERM_CH, 'ToF': (H, W), 'Classes': N_CLASSES})



# %% [split]
idx = np.arange(N)
np.random.shuffle(idx)

TEST = int(CONFIG['TEST_SPLIT'] * N)
VAL  = int(CONFIG['VAL_SPLIT']  * N)

test_idx  = idx[:TEST]
val_idx   = idx[TEST:TEST+VAL]
train_idx = idx[TEST+VAL:]

def take(a, ind):
    return a[ind]

split = {
    'train': {
        'imu': take(imu, train_idx),
        'thermo': take(thermo, train_idx),
        'tof': take(tof, train_idx),
        'y': take(y, train_idx),
    },
    'val': {
        'imu': take(imu, val_idx),
        'thermo': take(thermo, val_idx),
        'tof': take(tof, val_idx),
        'y': take(y, val_idx),
    },
    'test': {
        'imu': take(imu, test_idx),
        'thermo': take(thermo, test_idx),
        'tof': take(tof, test_idx),
        'y': take(y, test_idx),
    },
}

for k in split:
    print(k, {kk: v.shape for kk, v in split[k].items()})



# %% [dataset]
if HAS_TORCH:
    class SeqDataset(Dataset):
        def __init__(self, pack):
            self.imu = pack['imu']        # (N, T, IMU_CH)
            self.thermo = pack['thermo']  # (N, T, THERM_CH)
            self.tof = pack['tof']        # (N, T, H, W)
            self.y = pack['y']            # (N,)
        def __len__(self):
            return self.y.shape[0]
        def __getitem__(self, i):
            return {
                'imu': torch.from_numpy(self.imu[i]),
                'thermo': torch.from_numpy(self.thermo[i]),
                'tof': torch.from_numpy(self.tof[i]),
                'y': torch.tensor(int(self.y[i])),
            }

    loaders = {}
    for name in ['train', 'val', 'test']:
        ds = SeqDataset(split[name])
        loaders[name] = DataLoader(
            ds,
            batch_size=CONFIG['BATCH_SIZE'],
            shuffle=(name == 'train')
        )
else:
    print('ChÆ°a cÃ³ PyTorch â€“ bá»� qua bÆ°á»›c Dataset/Dataloader')



# %% [models]
if HAS_TORCH:
    # ---- LayerNorm cho tensor 1D (B, C, T) ----
    class LayerNorm1D(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.ln = nn.LayerNorm(n_channels)
        def forward(self, x):
            # x: [B, C, T] -> [B, T, C] -> LN -> [B, C, T]
            x = x.permute(0, 2, 1)
            x = self.ln(x)
            return x.permute(0, 2, 1)

    # ---- CNN1D Encoder ----
    class CNN1DEncoder(nn.Module):
        def __init__(self, in_ch, hidden=64, dropout=0.0, use_ln=True):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.ln = LayerNorm1D(hidden) if use_ln else nn.Identity()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            # x: [B, T, C] -> [B, C, T]
            x = x.transpose(1, 2)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.ln(x)
            x = self.pool(x).squeeze(-1)  # [B, hidden]
            return self.drop(x)

    # ---- LSTM/GRU Encoders ----
    class LSTMEncoder(nn.Module):
        def __init__(self, in_ch, hidden=64, bidir=False, dropout=0.0):
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=in_ch, hidden_size=hidden,
                batch_first=True, bidirectional=bidir,
                dropout=dropout if bidir else 0.0
            )
            self.out_ch = hidden * (2 if bidir else 1)
        def forward(self, x):
            out, _ = self.rnn(x)         # [B, T, H*dir]
            return out[:, -1, :]         # láº¥y timestep cuá»‘i

    class GRUEncoder(nn.Module):
        def __init__(self, in_ch, hidden=64, bidir=False, dropout=0.0):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=in_ch, hidden_size=hidden,
                batch_first=True, bidirectional=bidir,
                dropout=dropout if bidir else 0.0
            )
            self.out_ch = hidden * (2 if bidir else 1)
        def forward(self, x):
            out, _ = self.rnn(x)
            return out[:, -1, :]

    # ---- Attention pooling (tuá»³ chá»�n, dÃ¹ng á»Ÿ pháº§n Explainability) ----
    class AttentionPooling1D(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.w = nn.Linear(in_ch, 1)
        def forward(self, x):
            # x: [B, T, C]
            alpha = torch.softmax(self.w(x).squeeze(-1), dim=1)  # [B, T]
            ctx = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)    # [B, C]
            return ctx, alpha

    # ---- CNN1D + LSTM (Hybrid) ----
    class CNN1D_LSTM_Encoder(nn.Module):
        def __init__(self, in_ch, hidden=64, dropout=0.0, use_ln=True):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2)
            self.ln = LayerNorm1D(hidden) if use_ln else nn.Identity()
            self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            h = x.transpose(1, 2)               # [B, C, T]
            h = F.relu(self.conv(h))            # [B, H, T]
            h = self.ln(h).transpose(1, 2)      # [B, T, H]
            out, _ = self.rnn(h)                # [B, T, H]
            return self.drop(out[:, -1, :])     # [B, H]

    # ---- CNN2D Encoder cho ToF ----
    class CNN2DEncoder(nn.Module):
        def __init__(self, in_ch=1, hidden=64, dropout=0.0):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, hidden, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            # x: [B, T, 1, H, W] -> xá»­ lÃ½ tá»«ng frame rá»“i trung bÃ¬nh theo thá»�i gian
            B, T, C, H, W = x.shape
            h = x.reshape(B*T, C, H, W)
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = self.pool(h).squeeze(-1).squeeze(-1)   # [B*T, hidden]
            h = h.view(B, T, -1).mean(dim=1)           # [B, hidden]
            return self.drop(h)

    # ---- Ä�áº§u phÃ¢n loáº¡i ----
    class ClassifierHead(nn.Module):
        def __init__(self, in_ch, n_classes, dropout=0.0):
            super().__init__()
            hid = in_ch//2 if in_ch >= 64 else max(32, in_ch)
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_ch, hid), nn.ReLU(),
                nn.Linear(hid, n_classes)
            )
        def forward(self, x):
            return self.net(x)

    # ---- MÃ´ hÃ¬nh hoÃ n chá»‰nh theo modality ----
    class ModelIMU(nn.Module):
        def __init__(self, kind, in_ch, n_classes, cfg):
            super().__init__()
            dp, ln = cfg['DROPOUT'], cfg['USE_LAYER_NORM']
            if kind == 'CNN1D':
                self.enc = CNN1DEncoder(in_ch, dropout=dp, use_ln=ln); enc_out = 64
            elif kind == 'LSTM':
                self.enc = LSTMEncoder(in_ch, hidden=64, bidir=False, dropout=dp); enc_out = self.enc.out_ch
            elif kind == 'GRU':
                self.enc = GRUEncoder(in_ch, hidden=64, bidir=False, dropout=dp); enc_out = self.enc.out_ch
            elif kind == 'BiLSTM':
                self.enc = LSTMEncoder(in_ch, hidden=64, bidir=True, dropout=dp); enc_out = self.enc.out_ch
            elif kind == 'CNN1D+LSTM':
                self.enc = CNN1D_LSTM_Encoder(in_ch, hidden=64, dropout=dp, use_ln=ln); enc_out = 64
            else:
                raise ValueError('Unknown kind for ModelIMU')
            self.head = ClassifierHead(enc_out, n_classes, dropout=dp)
        def forward(self, imu):
            return self.head(self.enc(imu))

    class ModelToF(nn.Module):
        def __init__(self, n_classes, cfg):
            super().__init__()
            dp = cfg['DROPOUT']
            self.enc = CNN2DEncoder(in_ch=1, hidden=64, dropout=dp)
            self.head = ClassifierHead(64, n_classes, dropout=dp)
        def forward(self, tof):
            return self.head(self.enc(tof))

    class ModelMultimodalConcat(nn.Module):
        def __init__(self, imu_kind, thermo_kind, use_tof, in_imu, in_thermo, n_classes, cfg):
            super().__init__()
            dp = cfg['DROPOUT']
            # encoders
            self.enc_imu = ModelIMU(imu_kind, in_imu, n_classes, cfg).enc
            self.enc_thermo = ModelIMU(thermo_kind, in_thermo, n_classes, cfg).enc
            self.use_tof = use_tof
            if use_tof:
                self.enc_tof = CNN2DEncoder(in_ch=1, hidden=64, dropout=dp)
            # suy ra kÃ­ch thÆ°á»›c Ä‘áº·c trÆ°ng
            d_imu = 64 if imu_kind in ['CNN1D','CNN1D+LSTM'] else (128 if imu_kind=='BiLSTM' else 64)
            d_th  = 64 if thermo_kind in ['CNN1D','CNN1D+LSTM'] else (128 if thermo_kind=='BiLSTM' else 64)
            d_tof = 64 if use_tof else 0
            self.head = ClassifierHead(d_imu + d_th + d_tof, n_classes, dropout=dp)
        def forward(self, imu, thermo, tof):
            feats = [self.enc_imu(imu), self.enc_thermo(thermo)]
            if self.use_tof:
                feats.append(self.enc_tof(tof))
            return self.head(torch.cat(feats, dim=-1))
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua Ä‘á»‹nh nghÄ©a mÃ´ hÃ¬nh')



# %% [train_utils]
if HAS_TORCH:
    from typing import Dict, Tuple, List

    # ---- Macro-F1 tiá»‡n Ã­ch ----
    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    # ---- EarlyStopping Ä‘Æ¡n giáº£n ----
    class EarlyStopping:
        def __init__(self, patience: int = 5, min_delta: float = 0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.best = None
            self.count = 0
            self.stop = False

        def step(self, value: float):
            # value: val_loss hiá»‡n táº¡i
            if self.best is None or value < self.best - self.min_delta:
                self.best = value
                self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.stop = True

    # ---- Cháº¡y má»™t epoch (train hoáº·c val) ----
    def run_epoch(model, loader, criterion, optimizer=None):
        """
        Tráº£ vá»�: (mean_loss, macro_f1)
        """
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        total_loss = 0.0
        ys, yhs = [], []

        for b in loader:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float().unsqueeze(2)  # [B, T, 1, H, W]
            y_b   = b['y'].to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            # Chá»�n Ä‘Æ°á»�ng Ä‘i forward theo loáº¡i mÃ´ hÃ¬nh
            if isinstance(model, ModelIMU):
                logits = model(imu_b)                       # IMU-only
            elif isinstance(model, ModelToF):
                 logits = model(tof_b)                       # ToF-only
            else:
                 logits = model(imu_b, th_b, tof_b)

            loss = criterion(logits, y_b)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y_b.size(0)
            ys.append(y_b.detach().cpu().numpy())
            yhs.append(logits.detach().cpu().argmax(dim=1).numpy())

        ys = np.concatenate(ys)
        yhs = np.concatenate(yhs)
        return total_loss / len(loader.dataset), macro_f1(ys, yhs)

    # ---- Huáº¥n luyá»‡n Ä‘áº§y Ä‘á»§ vá»›i EarlyStopping + LR scheduler ----
    def fit_model(model, loaders: Dict[str, torch.utils.data.DataLoader], cfg: Dict, tag: str = 'model'):
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        early = EarlyStopping(patience=cfg['EARLY_STOP_PATIENCE'])

        hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [], 'lr': []}
        best_state = None
        best_val = float('inf')

        for ep in range(cfg['MAX_EPOCHS']):
            tr_loss, tr_f1 = run_epoch(model, loaders['train'], criterion, optimizer)
            vl_loss, vl_f1 = run_epoch(model, loaders['val'],   criterion, optimizer=None)

            scheduler.step(vl_loss)
            lr_now = optimizer.param_groups[0]['lr']

            hist['epoch'].append(ep + 1)
            hist['train_loss'].append(tr_loss)
            hist['val_loss'].append(vl_loss)
            hist['train_f1'].append(tr_f1)
            hist['val_f1'].append(vl_f1)
            hist['lr'].append(lr_now)

            # LÆ°u tráº¡ng thÃ¡i tá»‘t nháº¥t theo val_loss
            if vl_loss < best_val:
                best_val = vl_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # In log ngáº¯n gá»�n
            print(f"[EP {ep+1:02d}] "
                  f"tr_loss={tr_loss:.4f} val_loss={vl_loss:.4f} "
                  f"tr_f1={tr_f1:.4f} val_f1={vl_f1:.4f} lr={lr_now:.2e}")

            # Early stopping
            early.step(vl_loss)
            if early.stop:
                print("Dá»«ng sá»›m (EarlyStopping).")
                break

        # Táº£i láº¡i trá»�ng sá»‘ tá»‘t nháº¥t
        if best_state is not None:
            model.load_state_dict(best_state)

        return model, hist
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua utilities huáº¥n luyá»‡n')



# %% [experiments]
import pandas as pd
from pathlib import Path

results = []
histories = {}

if HAS_TORCH:
    @torch.no_grad()
    def evaluate_test(model, loader):
        model.eval()
        ys, yhs = [], []
        for b in loader:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float().unsqueeze(2)  # [B, T, 1, H, W]
            y_b   = b['y'].to(DEVICE)

            if isinstance(model, ModelIMU):
                logits = model(imu_b)
            elif isinstance(model, ModelToF):
                logits = model(tof_b)
            else:
                logits = model(imu_b, th_b, tof_b)

            ys.append(y_b.cpu().numpy())
            yhs.append(logits.argmax(dim=1).cpu().numpy())

        ys = np.concatenate(ys)
        yhs = np.concatenate(yhs)
        return f1_score(ys, yhs, average='macro')

    RUN_SET = CONFIG['RUN_SET']
    for tag in RUN_SET:
        if tag.startswith('IMU_'):
            kind = tag.split('_', 1)[1]
            model = ModelIMU(kind, in_ch=IMU_CH, n_classes=N_CLASSES, cfg=CONFIG)
            modality = 'IMU'
        elif tag == 'TOF_CNN2D':
            model = ModelToF(n_classes=N_CLASSES, cfg=CONFIG)
            modality = 'ToF'
        elif tag.startswith('MULTI_CONCAT'):
            model = ModelMultimodalConcat(
                imu_kind='CNN1D',
                thermo_kind='CNN1D',
                use_tof=True,
                in_imu=IMU_CH, in_thermo=THERM_CH,
                n_classes=N_CLASSES, cfg=CONFIG
            )
            modality = 'Multimodal'
        else:
            print(f'Bá»� qua tag khÃ´ng há»— trá»£: {tag}')
            continue

        model, hist = fit_model(model, loaders, CONFIG, tag)
        hist_df = pd.DataFrame(hist)
        histories[tag] = hist_df

        tr_f1 = hist_df['train_f1'].iloc[-1]
        val_f1 = hist_df['val_f1'].iloc[-1]
        te_f1 = evaluate_test(model, loaders['test'])

        results.append({
            'exp': tag,
            'modality': modality,
            'train_f1': tr_f1,
            'val_f1': val_f1,
            'test_f1': te_f1
        })
        print(f"==> {tag}: test macro-F1 = {te_f1:.4f}")

    res_df = pd.DataFrame(results).sort_values('test_f1', ascending=False)
    display(res_df)

    # LÆ°u báº£ng káº¿t quáº£ (tÃ¹y chá»�n)
    out_dir = Path('/mnt/data')
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / 'dl_results.csv'
    res_df.to_csv(res_path, index=False)
    print('Ä�Ã£ lÆ°u báº£ng káº¿t quáº£ táº¡i:', res_path)
else:
    print('ChÆ°a cÃ³ PyTorch â€“ bá»� qua pháº§n cháº¡y thÃ­ nghiá»‡m')



# %% [regularization]
if HAS_TORCH:
    def quick_run_imu_cnn1d(dropout, use_ln):
        cfg2 = CONFIG.copy()
        cfg2['DROPOUT'] = dropout
        cfg2['USE_LAYER_NORM'] = use_ln
        model = ModelIMU('CNN1D', in_ch=IMU_CH, n_classes=N_CLASSES, cfg=cfg2)
        model, hist = fit_model(model, loaders, cfg2, tag=f'IMU_CNN1D_dp{dropout}_ln{use_ln}')
        return pd.DataFrame(hist)

    # Cháº¡y 2 cáº¥u hÃ¬nh: khÃ´ng regularization vs cÃ³ regularization
    h1 = quick_run_imu_cnn1d(dropout=0.0, use_ln=False)
    h2 = quick_run_imu_cnn1d(dropout=0.5, use_ln=True)

    # Váº½ loss curves
    plt.figure()
    plt.plot(h1['epoch'], h1['train_loss'], label='train_loss no-reg')
    plt.plot(h1['epoch'], h1['val_loss'], label='val_loss no-reg')
    plt.plot(h2['epoch'], h2['train_loss'], label='train_loss reg')
    plt.plot(h2['epoch'], h2['val_loss'], label='val_loss reg')
    plt.title('Regularization: Loss Curves')
    plt.legend(); plt.tight_layout(); plt.show()

    # Váº½ macro-F1 curves
    plt.figure()
    plt.plot(h1['epoch'], h1['train_f1'], label='train_f1 no-reg')
    plt.plot(h1['epoch'], h1['val_f1'], label='val_f1 no-reg')
    plt.plot(h2['epoch'], h2['train_f1'], label='train_f1 reg')
    plt.plot(h2['epoch'], h2['val_f1'], label='val_f1 reg')
    plt.title('Regularization: Macro-F1 Curves')
    plt.legend(); plt.tight_layout(); plt.show()
else:
    print('ChÆ°a cÃ³ PyTorch â€“ bá»� qua kiá»ƒm tra regularization')



# %% [explainability]
if HAS_TORCH:
    # ---- MÃ´ hÃ¬nh nhá»� vá»›i Attention ----
    class IMUWithAttention(nn.Module):
        def __init__(self, in_ch, n_classes, cfg):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
            self.proj = nn.Linear(64, 64)
            self.attn = AttentionPooling1D(64)
            self.head = ClassifierHead(64, n_classes, dropout=cfg['DROPOUT'])
        def forward(self, x):
            h = x.transpose(1, 2)
            h = F.relu(self.conv(h))        # [B, 64, T]
            h = h.transpose(1, 2)           # [B, T, 64]
            h2 = self.proj(h)
            ctx, alpha = self.attn(h2)
            return self.head(ctx), alpha

    # Saliency 1D cho IMU
    batch = next(iter(loaders['val']))
    x = batch['imu'].to(DEVICE).float()
    x.requires_grad_(True)
    model_plain = ModelIMU('CNN1D', in_ch=IMU_CH, n_classes=N_CLASSES, cfg=CONFIG).to(DEVICE)
    model_plain.eval()
    logits = model_plain(x)
    cls = logits.argmax(dim=1)
    sel = logits[range(x.size(0)), cls].sum()
    sel.backward()
    sal = x.grad.detach().abs().mean(dim=0).cpu().numpy()  # [T, C]
    plt.figure()
    plt.imshow(sal.T, aspect='auto')
    plt.title('IMU Saliency (channels Ã— time)')
    plt.xlabel('time'); plt.ylabel('channels')
    plt.tight_layout(); plt.show()

    # Saliency 2D cho ToF
    x2 = batch['tof'].to(DEVICE).float().unsqueeze(2)
    x2.requires_grad_(True)
    model_tof = ModelToF(n_classes=N_CLASSES, cfg=CONFIG).to(DEVICE)
    model_tof.eval()
    logits2 = model_tof(x2)
    cls2 = logits2.argmax(dim=1)
    sel2 = logits2[range(x2.size(0)), cls2].sum()
    sel2.backward()
    grad = x2.grad.detach().abs().mean(dim=1).squeeze(1).cpu().numpy()  # [B, H, W]
    plt.figure()
    plt.imshow(grad[0], interpolation='nearest')
    plt.title('ToF Saliency (spatial) - sample 0')
    plt.tight_layout(); plt.show()

    # Attention weights minh hoáº¡
    model_attn = IMUWithAttention(IMU_CH, N_CLASSES, CONFIG).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = Adam(model_attn.parameters(), lr=CONFIG['LR'])
    for _ in range(2):  # train nhanh 2 epoch demo
        model_attn.train()
        for b in loaders['train']:
            x_train = b['imu'].to(DEVICE).float()
            y_train = b['y'].to(DEVICE)
            opt.zero_grad()
            logits, _ = model_attn(x_train)
            loss = crit(logits, y_train)
            loss.backward()
            opt.step()

    model_attn.eval()
    with torch.no_grad():
        _, alpha = model_attn(x)
    a = alpha[0].detach().cpu().numpy()
    plt.figure()
    plt.plot(np.arange(a.shape[0]), a)
    plt.title('Attention weights theo thá»�i gian (IMU sample 0)')
    plt.xlabel('time'); plt.ylabel('weight')
    plt.tight_layout(); plt.show()
else:
    print('ChÆ°a cÃ³ PyTorch â€“ bá»� qua explainability')



# %% [end]
print("Notebook 2 Ä‘Ã£ hoÃ n táº¥t. Káº¿t quáº£ Ä‘Ã£ Ä‘Æ°á»£c lÆ°u ra /mnt/data/dl_results.csv (náº¿u cháº¡y Ä‘áº§y Ä‘á»§).")
print("So sÃ¡nh báº£ng káº¿t quáº£ nÃ y vá»›i baseline á»Ÿ Notebook 1 Ä‘á»ƒ minh chá»©ng deep learning vÆ°á»£t baseline.")

