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

DEVICE = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
SEED = 42
random.seed(SEED); np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)
    if DEVICE=='cuda': torch.cuda.manual_seed_all(SEED)
print({'torch': HAS_TORCH, 'device': DEVICE})



# %% [config]
CONFIG = {
    'DATA_PATH': '',  # vÃ­ dá»¥ '/mnt/data/your_dataset.npz' cÃ³ keys: imu, thermo, label
    'N_SAMPLES': 900,
    'N_CLASSES': 6,
    'T_MAX': 200,        # Ä‘á»™ dÃ i chuá»—i tá»‘i Ä‘a Ä‘á»ƒ sinh synthetic
    'IMU_CH': 6,
    'THERM_CH': 8,

    # windows Ä‘á»ƒ so sÃ¡nh
    'WINDOWS': [50, 100, 200],

    # huáº¥n luyá»‡n
    'BATCH_SIZE': 64,
    'MAX_EPOCHS': 8,
    'LR': 2e-3,
    'WEIGHT_DECAY': 1e-4,
    'EARLY_STOP_PATIENCE': 4,
    'VAL_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,

    # mÃ´ hÃ¬nh Transformer
    'D_MODEL': 128,
    'NHEAD': 4,
    'NUM_LAYERS': 2,
    'DROPOUT': 0.1,
    'INFORMER_REDUCTION': 4,  # giáº£m chiá»�u dÃ i chuá»—i cho K/V

    # chá»�n mÃ´ hÃ¬nh Ä‘á»ƒ cháº¡y
    'RUN_SET': [
        'Transformer', 'TST', 'InformerLite', 'TCN', 'LSTM', 'CNN1D'
    ]
}
CONFIG



# %% [data]
def make_synthetic(cfg):
    N, C, T = cfg['N_SAMPLES'], cfg['N_CLASSES'], cfg['T_MAX']
    IMU_CH, THERM_CH = cfg['IMU_CH'], cfg['THERM_CH']

    # TÃ­n hiá»‡u ngáº«u nhiÃªn vá»›i pattern theo lá»›p
    y = np.random.randint(0, C, size=N)
    imu = np.random.randn(N, T, IMU_CH).astype('float32')
    thermo = (0.5*np.random.randn(N, T, THERM_CH) + 0.2).astype('float32')

    for cls in range(C):
        idx = y == cls
        imu[idx]    += (cls - C/2) * 0.12
        thermo[idx] += (C/2 - cls) * 0.08

    x = np.concatenate([imu, thermo], axis=2)  # [N, T, IMU_CH+THERM_CH]
    return x, y.astype('int64')

DATA_PATH = CONFIG['DATA_PATH']
if DATA_PATH and os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH)
    imu = data['imu'].astype('float32')       # [N, T, IMU_CH]
    thermo = data['thermo'].astype('float32') # [N, T, THERM_CH]
    y = data['label'].astype('int64')         # [N]
    assert imu.shape[:2] == thermo.shape[:2], "IMU vÃ  Thermo pháº£i cÃ¹ng (N, T)"
    x = np.concatenate([imu, thermo], axis=2)
    print('Loaded real dataset:', DATA_PATH)
else:
    x, y = make_synthetic(CONFIG)
    print('Using synthetic dataset')

N, T_MAX, C_TOTAL = x.shape
N_CLASSES = int(y.max()) + 1
print({'N': N, 'T_MAX': T_MAX, 'C_TOTAL': C_TOTAL, 'Classes': N_CLASSES})



# %% [split_and_dataloaders]
# Chia táº­p má»™t láº§n theo chá»‰ sá»‘
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
    'train': {'x': take(x, train_idx), 'y': take(y, train_idx)},
    'val':   {'x': take(x, val_idx),   'y': take(y, val_idx)},
    'test':  {'x': take(x, test_idx),  'y': take(y, test_idx)},
}
for k in split:
    print(k, {kk: v.shape for kk, v in split[k].items()})

# Dataset/Dataloader theo tá»«ng window
if HAS_TORCH:
    class SeqOnlyDataset(Dataset):
        def __init__(self, x, y, window: int):
            self.x = x
            self.y = y
            self.window = int(window)

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, i):
            xi = self.x[i]                 # [T_MAX, C_TOTAL]
            T  = xi.shape[0]
            w  = self.window
            if w <= T:
                xi = xi[T-w:]              # láº¥y Ä‘oáº¡n cuá»‘i Ä‘á»™ dÃ i w
            else:
                pad = np.zeros((w - T, xi.shape[1]), dtype=xi.dtype)
                xi = np.concatenate([pad, xi], axis=0)
            yi = int(self.y[i])
            return torch.from_numpy(xi), torch.tensor(yi)

    def build_loaders_for_window(window: int):
        loaders = {}
        for name in ['train', 'val', 'test']:
            ds = SeqOnlyDataset(split[name]['x'], split[name]['y'], window)
            loaders[name] = DataLoader(
                ds,
                batch_size=CONFIG['BATCH_SIZE'],
                shuffle=(name == 'train')
            )
        return loaders
else:
    print('ChÆ°a cÃ³ torch â€“ bá»� qua Dataset/Dataloader cho tá»«ng window')



# %% [models]
if HAS_TORCH:
    # ===== Positional embedding há»�c Ä‘Æ°á»£c =====
    class LearnablePositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.max_len = max_len

        def forward(self, x):  # x: [B, T, D]
            T = x.size(1)
            if T > self.max_len:
                raise ValueError(f"Seq len {T} > max_len {self.max_len}")
            return x + self.pos[:, :T, :]

    # ===== Vanilla Transformer (Encoder) =====
    class TransformerEncoderClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1, max_len=512):
            super().__init__()
            self.proj = nn.Linear(in_ch, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pos = LearnablePositionalEncoding(d_model, max_len=max_len+1)  # +1 cho token CLS
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        def forward(self, x):  # x: [B, T, C]
            B = x.size(0)
            h = self.proj(x)                       # [B, T, D]
            cls = self.cls.expand(B, 1, -1)        # [B, 1, D]
            h = torch.cat([cls, h], dim=1)         # [B, 1+T, D]
            h = self.pos(h)
            h = self.enc(h)
            z = h[:, 0, :]                         # token CLS
            return self.head(z)

    # ===== TST: Conv1D embedding + Transformer Encoder =====
    class TSTEncoderClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1, max_len=512):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pos = LearnablePositionalEncoding(d_model, max_len=max_len+1)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        def forward(self, x):  # x: [B, T, C]
            B = x.size(0)
            h = x.transpose(1, 2)                 # [B, C, T]
            h = self.conv(h).transpose(1, 2)      # [B, T, D]
            cls = self.cls.expand(B, 1, -1)
            h = torch.cat([cls, h], dim=1)
            h = self.pos(h)
            h = self.enc(h)
            z = h[:, 0, :]
            return self.head(z)

    # ===== Informer-lite: self-attention rÃºt gá»�n K/V =====
    class InformerLiteBlock(nn.Module):
        def __init__(self, d_model, nhead=4, reduction=4, dropout=0.1):
            super().__init__()
            self.reduction = reduction
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):  # x: [B, T, D]
            B, T, D = x.shape
            # NÃ©n K/V theo thá»�i gian báº±ng adaptive pooling (xáº¥p xá»‰ giáº£m Ä‘á»™ dÃ i)
            S = max(1, (T + self.reduction - 1) // self.reduction)
            kv = F.adaptive_avg_pool1d(x.transpose(1, 2), S).transpose(1, 2)  # [B, S, D]
            q = self.q_proj(x); k = self.k_proj(kv); v = self.v_proj(kv)
            h, _ = self.attn(q, k, v)                  # [B, T, D]
            x = self.norm1(x + self.drop(h))
            h2 = self.ff(x)
            x = self.norm2(x + self.drop(h2))
            return x

    class InformerLiteClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, d_model=128, nhead=4, num_layers=2, dropout=0.1, reduction=4, max_len=512):
            super().__init__()
            self.proj = nn.Linear(in_ch, d_model)
            self.pos  = LearnablePositionalEncoding(d_model, max_len=max_len+1)
            self.cls  = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList(
                [InformerLiteBlock(d_model, nhead, reduction, dropout) for _ in range(num_layers)]
            )
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        def forward(self, x):  # x: [B, T, C]
            B = x.size(0)
            h = self.proj(x)                       # [B, T, D]
            cls = self.cls.expand(B, 1, -1)        # [B, 1, D]
            h = torch.cat([cls, h], dim=1)         # [B, 1+T, D]
            h = self.pos(h)
            for blk in self.blocks:
                h = blk(h)                         # giá»¯ Ä‘á»™ dÃ i 1+T
            z = h[:, 0, :]
            return self.head(z)

    # ===== TCN (padding 'same' Ä‘á»ƒ giá»¯ nguyÃªn chiá»�u T) =====
    class TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.1):
            super().__init__()
            # padding 'same' cho kernel láº»: p = d*(k-1)/2
            assert k % 2 == 1, "Chá»�n kernel k láº» (vd k=3,5,7) Ä‘á»ƒ padding 'same'."
            pad_same = (k - 1) * d // 2

            self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=k, padding=pad_same, dilation=d)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad_same, dilation=d)
            self.drop  = nn.Dropout(dropout)
            self.down  = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):  # x: [B, C_in, T]
            h = F.relu(self.conv1(x))      # [B, C_out, T]
            h = self.drop(h)
            h = F.relu(self.conv2(h))      # [B, C_out, T]
            h = self.drop(h)
            return F.relu(h + self.down(x))

    class TCNClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, channels=(64, 64, 64), k=3, dropout=0.1):
            super().__init__()
            layers = []
            c_prev = in_ch
            for i, c in enumerate(channels):
                layers.append(TemporalBlock(c_prev, c, k=k, d=2**i, dropout=dropout))
                c_prev = c
            self.net  = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(c_prev, n_classes)

        def forward(self, x):  # x: [B, T, C]
            h = x.transpose(1, 2)          # [B, C, T]
            h = self.net(h)                # [B, C, T]
            h = self.pool(h).squeeze(-1)   # [B, C]
            return self.head(h)

    # ===== CNN1D (baseline) =====
    class CNN1DClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, hidden=64, dropout=0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.drop  = nn.Dropout(dropout)
            self.head  = nn.Linear(hidden, n_classes)

        def forward(self, x):  # x: [B, T, C]
            h = x.transpose(1, 2)          # [B, C, T]
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = self.pool(h).squeeze(-1)   # [B, hidden]
            h = self.drop(h)
            return self.head(h)

    # ===== LSTM (baseline) =====
    class LSTMClassifier(nn.Module):
        def __init__(self, in_ch, n_classes, hidden=128, bidir=False, dropout=0.1):
            super().__init__()
            self.rnn  = nn.LSTM(
                in_ch, hidden, batch_first=True,
                bidirectional=bidir, dropout=dropout if bidir else 0.0
            )
            out = hidden * (2 if bidir else 1)
            self.head = nn.Linear(out, n_classes)

        def forward(self, x):  # x: [B, T, C]
            h, _ = self.rnn(x)             # [B, T, H*dir]
            return self.head(h[:, -1, :])  # dÃ¹ng timestep cuá»‘i
else:
    print("ChÆ°a cÃ³ torch â€“ bá»� qua Ä‘á»‹nh nghÄ©a mÃ´ hÃ¬nh")



# %% [train_eval]
if HAS_TORCH:
    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    class EarlyStopping:
        def __init__(self, patience=4, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.best = None
            self.count = 0
            self.stop = False
        def step(self, value):
            if self.best is None or value < self.best - self.min_delta:
                self.best = value; self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.stop = True

    def param_count(model):
        return sum(p.numel() for p in model.parameters())

    def run_epoch(model, loader, criterion, optimizer=None):
        train = optimizer is not None
        model.train() if train else model.eval()
        total_loss = 0.0; ys=[]; yhs=[]
        for xb, yb in loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            ys.append(yb.detach().cpu().numpy())
            yhs.append(logits.detach().cpu().argmax(dim=1).cpu().numpy())
        ys = np.concatenate(ys); yhs = np.concatenate(yhs)
        return total_loss / len(loader.dataset), macro_f1(ys, yhs)

    def fit_model(model, loaders, cfg):
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
        early = EarlyStopping(patience=cfg['EARLY_STOP_PATIENCE'])
        best_state = None; best_val = float('inf')
        hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        t0 = time.monotonic()
        for ep in range(cfg['MAX_EPOCHS']):
            tr_loss, tr_f1 = run_epoch(model, loaders['train'], criterion, optimizer)
            vl_loss, vl_f1 = run_epoch(model, loaders['val'], criterion, None)
            hist['epoch'].append(ep+1)
            hist['train_loss'].append(tr_loss)
            hist['val_loss'].append(vl_loss)
            hist['train_f1'].append(tr_f1)
            hist['val_f1'].append(vl_f1)
            if vl_loss < best_val:
                best_val = vl_loss
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f"[EP {ep+1:02d}] tr_loss={tr_loss:.4f} val_loss={vl_loss:.4f} tr_f1={tr_f1:.4f} val_f1={vl_f1:.4f}")
            early.step(vl_loss)
            if early.stop:
                print("Early stopping")
                break
        t1 = time.monotonic()
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist, (t1 - t0)

    def evaluate_test(model, loader):
        model.eval(); ys=[]; yhs=[]
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE).float()
                yb = yb.to(DEVICE)
                logits = model(xb)
                ys.append(yb.cpu().numpy())
                yhs.append(logits.argmax(dim=1).cpu().numpy())
        ys = np.concatenate(ys); yhs = np.concatenate(yhs)
        return f1_score(ys, yhs, average='macro')

    def make_model(tag, in_ch, n_classes):
        d = CONFIG['D_MODEL']; h = CONFIG['NHEAD']; L = CONFIG['NUM_LAYERS']; dr = CONFIG['DROPOUT']; red = CONFIG['INFORMER_REDUCTION']
        if tag == 'Transformer':
            return TransformerEncoderClassifier(in_ch, n_classes, d_model=d, nhead=h, num_layers=L, dropout=dr, max_len=max(CONFIG['WINDOWS']))
        if tag == 'TST':
            return TSTEncoderClassifier(in_ch, n_classes, d_model=d, nhead=h, num_layers=L, dropout=dr, max_len=max(CONFIG['WINDOWS']))
        if tag == 'InformerLite':
            return InformerLiteClassifier(in_ch, n_classes, d_model=d, nhead=h, num_layers=L, dropout=dr, reduction=red, max_len=max(CONFIG['WINDOWS']))
        if tag == 'TCN':
            return TCNClassifier(in_ch, n_classes, channels=[64,64,64], k=3, dropout=dr)
        if tag == 'CNN1D':
            return CNN1DClassifier(in_ch, n_classes, hidden=64, dropout=dr)
        if tag == 'LSTM':
            return LSTMClassifier(in_ch, n_classes, hidden=128, bidir=False, dropout=dr)
        raise ValueError("Unknown model tag: "+tag)

    # ==== Cháº¡y toÃ n bá»™ thÃ­ nghiá»‡m ====
    results = []
    histories = {}
    for W in CONFIG['WINDOWS']:
        loaders = build_loaders_for_window(W)
        in_ch = C_TOTAL
        for tag in CONFIG['RUN_SET']:
            print(f"\n=== Running {tag} @ window={W} ===")
            model = make_model(tag, in_ch, N_CLASSES)
            nparam = param_count(model)
            model, hist, train_time = fit_model(model, loaders, CONFIG)
            te_f1 = evaluate_test(model, loaders['test'])
            histories[(tag, W)] = pd.DataFrame(hist)
            results.append({
                'model': tag,
                'window': W,
                'params': int(nparam),
                'train_time_sec': float(train_time),
                'val_f1': float(hist['val_f1'][-1]),
                'test_f1': float(te_f1),
            })
    res_df = pd.DataFrame(results).sort_values(['window','test_f1'], ascending=[True, False])
    display(res_df)

    # lÆ°u káº¿t quáº£
    from pathlib import Path
    out_dir = Path('/mnt/data'); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / 'ts_results.csv'
    res_df.to_csv(res_path, index=False)
    print("Ä�Ã£ lÆ°u báº£ng káº¿t quáº£ táº¡i:", res_path)
else:
    print("ChÆ°a cÃ³ torch â€“ bá»� qua pháº§n huáº¥n luyá»‡n/Ä‘Ã¡nh giÃ¡")



# %% [scaling_plots]
import pandas as pd
import matplotlib.pyplot as plt

# Cá»‘ gáº¯ng Ä‘á»�c káº¿t quáº£ Ä‘Ã£ lÆ°u
try:
    df = pd.read_csv('/mnt/data/ts_results.csv')
except FileNotFoundError:
    print("ChÆ°a tÃ¬m tháº¥y '/mnt/data/ts_results.csv'. HÃ£y cháº¡y Section 6 trÆ°á»›c.")
    df = None

if df is not None and len(df) > 0:
    models = df['model'].unique().tolist()
    for m in models:
        sub = df[df['model'] == m].sort_values('window')
        plt.figure()
        plt.plot(sub['window'].values, sub['test_f1'].values, marker='o')
        plt.title(f'F1 (test) vs window length â€“ {m}')
        plt.xlabel('window length')
        plt.ylabel('macro-F1')
        plt.tight_layout()
        plt.show()

    # Báº£ng tá»•ng há»£p nhanh: best theo má»—i model
    summary = (df.sort_values('test_f1', ascending=False)
                 .groupby('model', as_index=False)
                 .first()[['model','window','params','train_time_sec','test_f1']])
    print("TÃ³m táº¯t (best theo tá»«ng mÃ´ hÃ¬nh):")
    display(summary)



# %% [end]
print("Notebook 3 Ä‘Ã£ hoÃ n táº¥t âœ…")
print("Káº¿t quáº£ báº£ng so sÃ¡nh cÃ¡c mÃ´ hÃ¬nh Ä‘Æ°á»£c lÆ°u á»Ÿ /mnt/data/ts_results.csv")


