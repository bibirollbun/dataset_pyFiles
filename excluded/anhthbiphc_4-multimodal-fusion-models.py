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


# %% [setup]
import os, random, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from sklearn.metrics import f1_score
    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
    print('PyTorch chÆ°a sáºµn sÃ ng, cáº§n cÃ i torch Ä‘á»ƒ huáº¥n luyá»‡n. Lá»—i:', e)

# Thá»‘ng kÃª
try:
    from scipy.stats import ttest_rel, wilcoxon
    HAS_SCIPY = True
except Exception as e:
    HAS_SCIPY = False
    print('ChÆ°a cÃ³ SciPy: pháº§n kiá»ƒm Ä‘á»‹nh thá»‘ng kÃª sáº½ bá»� qua. Lá»—i:', e)

DEVICE = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
BASE_SEED = 123
random.seed(BASE_SEED); np.random.seed(BASE_SEED)
if HAS_TORCH:
    torch.manual_seed(BASE_SEED)
    if DEVICE=='cuda': torch.cuda.manual_seed_all(BASE_SEED)
print({'torch': HAS_TORCH, 'device': DEVICE, 'scipy': HAS_SCIPY})



# %% [config_and_data]
CONFIG = {
    'DATA_PATH': '',  # vÃ­ dá»¥ '/mnt/data/your_dataset.npz'
    'N_SAMPLES': 900,
    'N_CLASSES': 6,
    'T': 64,
    'IMU_CH': 6,
    'THERM_CH': 8,
    'TOF_H': 8, 'TOF_W': 8,

    # huáº¥n luyá»‡n
    'BATCH_SIZE': 64,
    'MAX_EPOCHS': 8,
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-4,
    'EARLY_STOP_PATIENCE': 4,

    # láº·p láº¡i Ä‘á»ƒ test thá»‘ng kÃª
    'N_REPEATS': 3,                 # tÄƒng lÃªn 5â€“10 náº¿u muá»‘n kiá»ƒm Ä‘á»‹nh máº¡nh hÆ¡n
    'SEEDS': [123, 321, 777],       # sáº½ tá»± cáº¯t theo N_REPEATS

    # kiáº¿n trÃºc
    'EMB_D': 64,                    # chiá»�u embedding cho má»—i modality
    'FUSE_HIDDEN': 128,             # hidden cho aggregator/head
    'DROPOUT': 0.2,

    # thÃ­ nghiá»‡m chÃ­nh (ablation + fusion)
    'RUNS': [
        # Baselines Ä‘Æ¡n kÃªnh
        ('IMU_ONLY',         {'imu':True,  'thermo':False,'tof':False, 'fusion':'late'}),
        ('THERMO_ONLY',      {'imu':False, 'thermo':True, 'tof':False, 'fusion':'late'}),
        ('TOF_ONLY',         {'imu':False, 'thermo':False,'tof':True,  'fusion':'late'}),

        # Ablation: Late Fusion
        ('LATE_IMU+THERMO',  {'imu':True,  'thermo':True, 'tof':False, 'fusion':'late'}),
        ('LATE_IMU+TOF',     {'imu':True,  'thermo':False,'tof':True,  'fusion':'late'}),
        ('LATE_ALL',         {'imu':True,  'thermo':True, 'tof':True,  'fusion':'late'}),

        # Early Fusion (All)
        ('EARLY_ALL',        {'imu':True,  'thermo':True, 'tof':True,  'fusion':'early'}),

        # Attention Fusion (All) - cÃ³ visualization weights
        ('ATTN_ALL',         {'imu':True,  'thermo':True, 'tof':True,  'fusion':'attn'}),

        # Mixture of Experts (All)
        ('MOE_ALL',          {'imu':True,  'thermo':True, 'tof':True,  'fusion':'moe'}),

        # Ensemble soft voting (train 3 Ä‘Æ¡n kÃªnh trÆ°á»›c rá»“i vote)
        ('ENSEMBLE_SOFT',    {'imu':True,  'thermo':True, 'tof':True,  'fusion':'ensemble'}),
    ]
}

def make_synthetic(cfg):
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
        cr, cc0 = (cls % H), (cls % W)
        mask = np.exp(-((rr-cr)**2 + (cc-cc0)**2)/(2*(H/4)**2)).astype('float32')
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



# %% [split_dataloader]
idx = np.arange(N); np.random.shuffle(idx)
TEST = int(0.15 * N)
VAL  = int(0.15 * N)
test_idx  = idx[:TEST]
val_idx   = idx[TEST:TEST+VAL]
train_idx = idx[TEST+VAL:]

def take(a, ind): return a[ind]
split = {
    'train': {'imu': take(imu, train_idx), 'thermo': take(thermo, train_idx), 'tof': take(tof, train_idx), 'y': take(y, train_idx)},
    'val':   {'imu': take(imu, val_idx),   'thermo': take(thermo, val_idx),   'tof': take(tof, val_idx),   'y': take(y, val_idx)},
    'test':  {'imu': take(imu, test_idx),  'thermo': take(thermo, test_idx),  'tof': take(tof, test_idx),  'y': take(y, test_idx)},
}
for k in split:
    print(k, {kk: v.shape for kk, v in split[k].items()})

if HAS_TORCH:
    class SeqDataset(Dataset):
        def __init__(self, pack):
            self.imu = pack['imu']; self.thermo = pack['thermo']; self.tof = pack['tof']; self.y = pack['y']
        def __len__(self): return self.y.shape[0]
        def __getitem__(self, i):
            return {
                'imu': torch.from_numpy(self.imu[i]),
                'thermo': torch.from_numpy(self.thermo[i]),
                'tof': torch.from_numpy(self.tof[i]),
                'y': torch.tensor(int(self.y[i]))
            }

    loaders = {}
    for name in ['train', 'val', 'test']:
        ds = SeqDataset(split[name])
        loaders[name] = DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=(name=='train'))
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua Dataloader')



# %% [models]
if HAS_TORCH:
    # ----- Embedding 1D cho IMU/Thermo (giá»¯ T) -----
    class Embed1D(nn.Module):
        def __init__(self, in_ch, d, dropout=0.0):
            super().__init__()
            self.proj = nn.Linear(in_ch, d)
            self.drop = nn.Dropout(dropout)
        def forward(self, x):           # x: [B, T, C]
            return self.drop(self.proj(x))  # [B, T, d]

    # ----- CNN2D per-frame cho ToF (giá»¯ T) -----
    class FrameCNN2D(nn.Module):
        def __init__(self, out_d=64, dropout=0.0):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, out_d, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.drop = nn.Dropout(dropout)
        def forward(self, tof):  # tof: [B, T, H, W]
            B, T, H, W = tof.shape
            x = tof.unsqueeze(2)                   # [B, T, 1, H, W]
            h = x.reshape(B*T, 1, H, W)
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = self.pool(h).squeeze(-1).squeeze(-1)  # [B*T, out_d]
            h = self.drop(h).view(B, T, -1)           # [B, T, out_d]
            return h

    # ----- Vector encoders (mean theo T) -----
    class VecEncoder1D(nn.Module):
        def __init__(self, in_ch, d, dropout=0.0):
            super().__init__()
            self.emb = Embed1D(in_ch, d, dropout)
        def forward(self, x):               # x: [B, T, C]
            h = self.emb(x)                 # [B, T, d]
            return h.mean(dim=1)            # [B, d]

    class VecEncoderToF(nn.Module):
        def __init__(self, d, dropout=0.0):
            super().__init__()
            self.enc = FrameCNN2D(d, dropout)
        def forward(self, tof):             # tof: [B, T, H, W]
            h = self.enc(tof)               # [B, T, d]
            return h.mean(dim=1)            # [B, d]

    # ----- Head MLP -----
    class HeadMLP(nn.Module):
        def __init__(self, in_d, n_classes, hidden=128, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_d, hidden), nn.ReLU(),
                nn.Linear(hidden, n_classes)
            )
        def forward(self, x): return self.net(x)

    # ----- Early Fusion -----
    class EarlyFusionModel(nn.Module):
        def __init__(self, use_imu, use_thermo, use_tof, cfg):
            super().__init__()
            d = cfg['EMB_D']; hid = cfg['FUSE_HIDDEN']; dr = cfg['DROPOUT']
            self.use_imu, self.use_thermo, self.use_tof = use_imu, use_thermo, use_tof
            if use_imu:    self.emb_imu    = Embed1D(IMU_CH, d, dr)
            if use_thermo: self.emb_thermo = Embed1D(THERM_CH, d, dr)
            if use_tof:    self.emb_tof    = FrameCNN2D(d, dr)
            Dsum = (d if use_imu else 0) + (d if use_thermo else 0) + (d if use_tof else 0)
            # Aggregator theo thá»�i gian: Conv1D + GAP
            self.conv = nn.Conv1d(Dsum, hid, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = HeadMLP(hid, N_CLASSES, hidden=hid, dropout=dr)

        def forward(self, imu, thermo, tof):
            feats = []
            if self.use_imu:    feats.append(self.emb_imu(imu))      # [B,T,d]
            if self.use_thermo: feats.append(self.emb_thermo(thermo))
            if self.use_tof:    feats.append(self.emb_tof(tof))
            h = torch.cat(feats, dim=-1)          # [B, T, Dsum]
            h = h.transpose(1, 2)                 # [B, Dsum, T]
            h = F.relu(self.conv(h))              # [B, hid, T]
            h = self.pool(h).squeeze(-1)          # [B, hid]
            return self.head(h)

    # ----- Late Fusion -----
    class LateFusionModel(nn.Module):
        def __init__(self, use_imu, use_thermo, use_tof, cfg):
            super().__init__()
            d = cfg['EMB_D']; hid = cfg['FUSE_HIDDEN']; dr = cfg['DROPOUT']
            self.use_imu, self.use_thermo, self.use_tof = use_imu, use_thermo, use_tof
            if use_imu:    self.vec_imu    = VecEncoder1D(IMU_CH, d, dr)
            if use_thermo: self.vec_thermo = VecEncoder1D(THERM_CH, d, dr)
            if use_tof:    self.vec_tof    = VecEncoderToF(d, dr)
            Dsum = (d if use_imu else 0) + (d if use_thermo else 0) + (d if use_tof else 0)
            self.head = HeadMLP(Dsum, N_CLASSES, hidden=hid, dropout=dr)

        def forward(self, imu, thermo, tof):
            vecs = []
            if self.use_imu:    vecs.append(self.vec_imu(imu))
            if self.use_thermo: vecs.append(self.vec_thermo(thermo))
            if self.use_tof:    vecs.append(self.vec_tof(tof))
            h = torch.cat(vecs, dim=-1)    # [B, Dsum]
            return self.head(h)

    # ----- Attention Fusion (xuáº¥t weights) -----
    class AttentionFusionModel(nn.Module):
        def __init__(self, use_imu, use_thermo, use_tof, cfg):
            super().__init__()
            d = cfg['EMB_D']; hid = cfg['FUSE_HIDDEN']; dr = cfg['DROPOUT']
            self.use_imu, self.use_thermo, self.use_tof = use_imu, use_thermo, use_tof
            if use_imu:    self.vec_imu    = VecEncoder1D(IMU_CH, d, dr)
            if use_thermo: self.vec_thermo = VecEncoder1D(THERM_CH, d, dr)
            if use_tof:    self.vec_tof    = VecEncoderToF(d, dr)
            self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
            self.head  = HeadMLP(d, N_CLASSES, hidden=hid, dropout=dr)

        def forward(self, imu, thermo, tof, return_weights=False):
            vecs = []
            names = []
            if self.use_imu:    vecs.append(self.vec_imu(imu));        names.append('IMU')
            if self.use_thermo: vecs.append(self.vec_thermo(thermo));  names.append('Thermo')
            if self.use_tof:    vecs.append(self.vec_tof(tof));        names.append('ToF')
            V = torch.stack(vecs, dim=1)        # [B, M, d]
            s = self.score(V)                   # [B, M, 1]
            w = torch.softmax(s.squeeze(-1), dim=1)  # [B, M]
            z = (w.unsqueeze(-1) * V).sum(dim=1)     # [B, d]
            logits = self.head(z)
            if return_weights:
                return logits, w, names
            return logits

    # ----- Mixture of Experts (logits fusion) -----
    class MoEModel(nn.Module):
        def __init__(self, use_imu, use_thermo, use_tof, cfg):
            super().__init__()
            d = cfg['EMB_D']; hid = cfg['FUSE_HIDDEN']; dr = cfg['DROPOUT']
            self.use_imu, self.use_thermo, self.use_tof = use_imu, use_thermo, use_tof
            self.vecs = nn.ModuleDict()
            self.experts = nn.ModuleList()
            names = []
            if use_imu:
                self.vecs['imu'] = VecEncoder1D(IMU_CH, d, dr);  self.experts.append(HeadMLP(d, N_CLASSES, hidden=hid, dropout=dr)); names.append('imu')
            if use_thermo:
                self.vecs['thermo'] = VecEncoder1D(THERM_CH, d, dr); self.experts.append(HeadMLP(d, N_CLASSES, hidden=hid, dropout=dr)); names.append('thermo')
            if use_tof:
                self.vecs['tof'] = VecEncoderToF(d, dr); self.experts.append(HeadMLP(d, N_CLASSES, hidden=hid, dropout=dr)); names.append('tof')
            self.names = names
            self.gate = nn.Sequential(
                nn.Linear(d*len(names), hid), nn.ReLU(), nn.Linear(hid, len(names))
            )

        def forward(self, imu, thermo, tof, return_weights=False):
            vec_list = []
            logit_list = []
            for nm, enc in self.vecs.items():
                if nm=='imu':      v = enc(imu)
                elif nm=='thermo': v = enc(thermo)
                else:              v = enc(tof)
                vec_list.append(v)
            V = torch.stack(vec_list, dim=1)                 # [B, M, d]
            # experts
            for i, v in enumerate(vec_list):
                logit_list.append(self.experts[i](v))        # [B, C]
            L = torch.stack(logit_list, dim=1)               # [B, M, C]
            # gate
            g = self.gate(V.reshape(V.size(0), -1))          # [B, M]
            w = torch.softmax(g, dim=1)                      # [B, M]
            logits = (w.unsqueeze(-1) * L).sum(dim=1)        # [B, C]
            if return_weights:
                return logits, w, self.names
            return logits
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua models')



# %% [train_utils]
if HAS_TORCH:
    def macro_f1(y_true, y_pred): return f1_score(y_true, y_pred, average='macro')

    class EarlyStopping:
        def __init__(self, patience=4, min_delta=0.0):
            self.patience = patience; self.min_delta = min_delta
            self.best = None; self.count = 0; self.stop = False
        def step(self, value):
            if self.best is None or value < self.best - self.min_delta:
                self.best = value; self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience: self.stop = True

    def run_epoch(model, loader, criterion, optimizer=None):
        train = optimizer is not None
        model.train() if train else model.eval()
        total = 0.0; ys=[]; yhs=[]
        for b in loader:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float()
            y_b   = b['y'].to(DEVICE)
            if train: optimizer.zero_grad()
            # Há»— trá»£ Attention/MoE tráº£ vá»� (logits, weights, names)
            out = model(imu_b, th_b, tof_b)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y_b)
            if train:
                loss.backward(); optimizer.step()
            total += loss.item() * y_b.size(0)
            ys.append(y_b.detach().cpu().numpy())
            yhs.append(logits.detach().cpu().argmax(dim=1).cpu().numpy())
        ys = np.concatenate(ys); yhs = np.concatenate(yhs)
        return total / len(loader.dataset), macro_f1(ys, yhs)

    def fit_model(model, loaders, cfg):
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optim = Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
        early = EarlyStopping(patience=cfg['EARLY_STOP_PATIENCE'])
        best_state = None; best_val = float('inf')
        hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        t0 = time.monotonic()
        for ep in range(cfg['MAX_EPOCHS']):
            tr_loss, tr_f1 = run_epoch(model, loaders['train'], criterion, optim)
            vl_loss, vl_f1 = run_epoch(model, loaders['val'],   criterion, None)
            hist['epoch'].append(ep+1); hist['train_loss'].append(tr_loss); hist['val_loss'].append(vl_loss)
            hist['train_f1'].append(tr_f1); hist['val_f1'].append(vl_f1)
            if vl_loss < best_val:
                best_val = vl_loss
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f"[EP {ep+1:02d}] tr_loss={tr_loss:.4f} val_loss={vl_loss:.4f} tr_f1={tr_f1:.4f} val_f1={vl_f1:.4f}")
            early.step(vl_loss)
            if early.stop:
                print("Early stopping."); break
        t1 = time.monotonic()
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist, (t1 - t0)

    @torch.no_grad()
    def evaluate_test(model, loader):
        model.eval(); ys=[]; yhs=[]
        for b in loader:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float()
            y_b   = b['y'].to(DEVICE)
            out = model(imu_b, th_b, tof_b)
            logits = out[0] if isinstance(out, tuple) else out
            ys.append(y_b.cpu().numpy())
            yhs.append(logits.argmax(dim=1).cpu().numpy())
        ys = np.concatenate(ys); yhs = np.concatenate(yhs)
        return f1_score(ys, yhs, average='macro')
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua train utils')



# %% [experiments]
from pathlib import Path

if HAS_TORCH:
    def make_model(use_imu, use_thermo, use_tof, fusion, cfg):
        if fusion == 'early':
            return EarlyFusionModel(use_imu, use_thermo, use_tof, cfg)
        if fusion == 'late':
            return LateFusionModel(use_imu, use_thermo, use_tof, cfg)
        if fusion == 'attn':
            return AttentionFusionModel(use_imu, use_thermo, use_tof, cfg)
        if fusion == 'moe':
            return MoEModel(use_imu, use_thermo, use_tof, cfg)
        if fusion == 'ensemble':
            return None  # xá»­ lÃ½ riÃªng
        raise ValueError('Unknown fusion: '+fusion)

    def param_count(model): return sum(p.numel() for p in model.parameters())

    # Train models Ä‘Æ¡n kÃªnh Ä‘á»ƒ dÃ¹ng cho Ensemble soft voting
    def train_unimodal_baselines(loaders, cfg):
        uni = {}
        # IMU
        m_imu = LateFusionModel(True, False, False, cfg).to(DEVICE)
        m_imu, _, _ = fit_model(m_imu, loaders, cfg); uni['IMU'] = m_imu
        # Thermo
        m_th = LateFusionModel(False, True, False, cfg).to(DEVICE)
        m_th, _, _ = fit_model(m_th, loaders, cfg); uni['THERMO'] = m_th
        # ToF
        m_tof = LateFusionModel(False, False, True, cfg).to(DEVICE)
        m_tof, _, _ = fit_model(m_tof, loaders, cfg); uni['TOF'] = m_tof
        return uni

    @torch.no_grad()
    def eval_ensemble_soft(uni_models, loader):
        for k in uni_models: uni_models[k].eval()
        ys=[]; yhs=[]
        for b in loader:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float()
            y_b   = b['y'].to(DEVICE)
            probs = []
            # má»—i model tráº£ vá»� logits tá»« modality tÆ°Æ¡ng á»©ng
            logits_imu  = uni_models['IMU'](imu_b, th_b, tof_b)
            logits_th   = uni_models['THERMO'](imu_b, th_b, tof_b)
            logits_tof  = uni_models['TOF'](imu_b, th_b, tof_b)
            for lg in [logits_imu, logits_th, logits_tof]:
                if isinstance(lg, tuple): lg = lg[0]
                probs.append(torch.softmax(lg, dim=1))
            p = torch.stack(probs, dim=0).mean(dim=0)   # soft voting
            yhat = p.argmax(dim=1)
            ys.append(y_b.cpu().numpy()); yhs.append(yhat.cpu().numpy())
        ys = np.concatenate(ys); yhs = np.concatenate(yhs)
        return f1_score(ys, yhs, average='macro')

    results = []
    histories = {}

    for name, cfg_run in CONFIG['RUNS']:
        use_imu, use_th, use_tof = cfg_run['imu'], cfg_run['thermo'], cfg_run['tof']
        fusion = cfg_run['fusion']
        print(f"\n=== Run {name} | fusion={fusion} | use: IMU={use_imu}, Thermo={use_th}, ToF={use_tof} ===")

        if fusion == 'ensemble':
            uni = train_unimodal_baselines(loaders, CONFIG)
            te_f1 = eval_ensemble_soft(uni, loaders['test'])
            results.append({'exp': name, 'fusion': fusion, 'modalities': f"{use_imu}{use_th}{use_tof}",
                            'params': int(sum(param_count(m) for m in uni.values())),
                            'train_time_sec': np.nan, 'val_f1': np.nan, 'test_f1': float(te_f1)})
            print(f"==> Ensemble soft voting: test F1 = {te_f1:.4f}")
            continue

        model = make_model(use_imu, use_th, use_tof, fusion, CONFIG).to(DEVICE)
        nparam = param_count(model)
        model, hist, train_time = fit_model(model, loaders, CONFIG)
        te_f1 = evaluate_test(model, loaders['test'])

        histories[name] = pd.DataFrame(hist)
        results.append({
            'exp': name, 'fusion': fusion, 'modalities': f"{use_imu}{use_th}{use_tof}",
            'params': int(nparam), 'train_time_sec': float(train_time),
            'val_f1': float(hist['val_f1'][-1]), 'test_f1': float(te_f1)
        })
        print(f"==> {name}: test macro-F1 = {te_f1:.4f}")

    res_df = pd.DataFrame(results).sort_values('test_f1', ascending=False)
    display(res_df)
    out_dir = Path('/mnt/data'); out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / 'multimodal_results.csv'
    res_df.to_csv(out_p, index=False)
    print('Ä�Ã£ lÆ°u káº¿t quáº£ táº¡i:', out_p)
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua experiments')



# %% [viz_attention]
if HAS_TORCH:
    # Khá»Ÿi táº¡o láº¡i model attention & náº¡p trá»�ng sá»‘ tá»‘t nháº¥t báº±ng cÃ¡ch huáº¥n luyá»‡n ngáº¯n
    attn_model = AttentionFusionModel(True, True, True, CONFIG).to(DEVICE)
    attn_model, _, _ = fit_model(attn_model, loaders, CONFIG)
    # Thu weights trÃªn val
    attn_model.eval()
    ws = []
    names_ref = None
    with torch.no_grad():
        for b in loaders['val']:
            imu_b = b['imu'].to(DEVICE).float()
            th_b  = b['thermo'].to(DEVICE).float()
            tof_b = b['tof'].to(DEVICE).float()
            logits, w, names = attn_model(imu_b, th_b, tof_b, return_weights=True)
            ws.append(w.cpu().numpy())
            if names_ref is None: names_ref = names
    W = np.concatenate(ws, axis=0)  # [N_val, M]
    mean_w = W.mean(axis=0)
    perc = 100*mean_w/mean_w.sum()
    for nm, pi in zip(names_ref, perc):
        print(f"{nm}: {pi:.1f}%")

    # Váº½ bar
    plt.figure()
    plt.bar(names_ref, perc)
    plt.title('Attention weights trung bÃ¬nh theo modality (val)')
    plt.ylabel('Pháº§n trÄƒm (%)')
    plt.tight_layout(); plt.show()
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua visualization attention')



# (Ä�Ã£ thá»±c hiá»‡n trong Section 6 â€” khÃ´ng cáº§n thÃªm code á»Ÿ Ä‘Ã¢y)
print("MoE vÃ  Ensemble Ä‘Ã£ Ä‘Æ°á»£c cháº¡y trong pháº§n thÃ­ nghiá»‡m.")



# %% [stats_tests]
if HAS_TORCH and HAS_SCIPY:
    seeds = CONFIG['SEEDS'][:CONFIG['N_REPEATS']]
    f1_imu, f1_fuse = [], []

    def set_seed(s):
        random.seed(s); np.random.seed(s)
        torch.manual_seed(s); 
        if DEVICE=='cuda': torch.cuda.manual_seed_all(s)

    for s in seeds:
        print(f"\n== Re-run with seed {s} ==")
        set_seed(s)
        # Rebuild loaders (shuffle khÃ¡c)
        idx = np.arange(N); np.random.shuffle(idx)
        TEST = int(0.15 * N); VAL = int(0.15 * N)
        test_idx, val_idx, train_idx = idx[:TEST], idx[TEST:TEST+VAL], idx[TEST+VAL:]
        split2 = {
            'train': {'imu': imu[train_idx], 'thermo': thermo[train_idx], 'tof': tof[train_idx], 'y': y[train_idx]},
            'val':   {'imu': imu[val_idx],   'thermo': thermo[val_idx],   'tof': tof[val_idx],   'y': y[val_idx]},
            'test':  {'imu': imu[test_idx],  'thermo': thermo[test_idx],  'tof': tof[test_idx],  'y': y[test_idx]},
        }
        class DS(Dataset):
            def __init__(self, p): self.p=p
            def __len__(self): return self.p['y'].shape[0]
            def __getitem__(self, i):
                return {'imu': torch.from_numpy(self.p['imu'][i]),
                        'thermo': torch.from_numpy(self.p['thermo'][i]),
                        'tof': torch.from_numpy(self.p['tof'][i]),
                        'y': torch.tensor(int(self.p['y'][i]))}
        loaders2 = {}
        for nm in ['train','val','test']:
            loaders2[nm] = DataLoader(DS(split2[nm]), batch_size=CONFIG['BATCH_SIZE'], shuffle=(nm=='train'))

        # IMU-only
        m_imu = LateFusionModel(True, False, False, CONFIG).to(DEVICE)
        m_imu, _, _ = fit_model(m_imu, loaders2, CONFIG)
        f1i = evaluate_test(m_imu, loaders2['test'])
        f1_imu.append(f1i)

        # Attention Fusion All
        m_attn = AttentionFusionModel(True, True, True, CONFIG).to(DEVICE)
        m_attn, _, _ = fit_model(m_attn, loaders2, CONFIG)
        f1a = evaluate_test(m_attn, loaders2['test'])
        f1_fuse.append(f1a)

        print(f"Seed {s}: IMU-only={f1i:.4f} | ATTN_ALL={f1a:.4f}")

    print("\n== Kiá»ƒm Ä‘á»‹nh thá»‘ng kÃª (paired) ==")
    t_res = ttest_rel(f1_fuse, f1_imu)
    w_res = wilcoxon(f1_fuse, f1_imu, zero_method='wilcox', correction=False, alternative='two-sided')
    print("Paired t-test:    t=%.4f, p=%.6f" % (t_res.statistic, t_res.pvalue))
    print("Wilcoxon signed:  W=%.4f, p=%.6f" % (w_res.statistic, w_res.pvalue))
    print("CÃ¡c vector F1:", "\n  IMU:", f1_imu, "\n  ATTN:", f1_fuse)
else:
    print("Thiáº¿u torch hoáº·c scipy â€” bá»� qua kiá»ƒm Ä‘á»‹nh thá»‘ng kÃª.")



# %% [end]
print("Notebook 4 hoÃ n táº¥t âœ…. Káº¿t quáº£ chÃ­nh lÆ°u táº¡i /mnt/data/multimodal_results.csv")


