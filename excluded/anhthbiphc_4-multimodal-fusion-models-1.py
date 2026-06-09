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


# Notebook 4 â€“ Multimodal Fusion Models (Core ğŸš€)
# ------------------------------------------------------------
# Goal: Quantify added value of ToF & Thermopile via multimodal fusion
# - Early Fusion (concat)
# - Late Fusion (per-modality encoders â†’ concat â†’ MLP)
# - Attention Fusion (modality attention)
# - Mixture of Experts (MoE)
# - Simple Ensembles (soft-voting / stacking)
# - Compare IMU-only vs Fusion (IMU+Thermo+ToF)
# - Attention visualization & Statistical tests across CV folds
# ------------------------------------------------------------

import os, gc, math, random, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline

from scipy.stats import ttest_rel, wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)




# =====================
# 0) DATA LOADING
# =====================
# Expect Kaggle paths (adjust if needed)
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

# --- Flexible merge key picker ---
CAND_KEYS = ['id','ID','Id','record_id','sample_id','subject_id','participant_id','user_id','person_id','uid','pid','subject','ParticipantID']

def pick_key(left: pd.DataFrame, right: pd.DataFrame, candidates=CAND_KEYS):
    both = [k for k in candidates if k in left.columns and k in right.columns]
    if both:
        return both[0], both[0]
    l2 = [c for c in left.columns if ('id' in c.lower()) or ('subject' in c.lower())]
    r2 = [c for c in right.columns if ('id' in c.lower()) or ('subject' in c.lower())]
    inter = list(set(l2).intersection(r2))
    if inter:
        return inter[0], inter[0]
    cols_left = list(left.columns)[:50]
    cols_right = list(right.columns)[:50]
    raise KeyError(
        "KhÃ´ng tÃ¬m Ä‘Æ°á»£c khÃ³a merge. "
        "HÃ£y kiá»ƒm tra tÃªn cá»™t khÃ³a trong train/train_dem.\n"
        f"Cá»™t train: {cols_left}...\n"
        f"Cá»™t train_dem: {cols_right}..."
    )

lk, rk = pick_key(train_df, train_dem_df)
train_full = train_df.merge(train_dem_df, left_on=lk, right_on=rk, how='left', validate='m:1')
lk_t, rk_t = pick_key(test_df, test_dem_df)
test_full = test_df.merge(test_dem_df, left_on=lk_t, right_on=rk_t, how='left', validate='m:1')

# --- Detect label column ---
LABEL_CANDIDATES = ['label','target','y','class','classes','activity','activity_id','act','state','behavior','behavior_id','behaviour','category','cat','event','event_id']
label_map = {c.lower(): c for c in train_full.columns}
LABEL_COL = None
for cand in LABEL_CANDIDATES:
    if cand in label_map:
        LABEL_COL = label_map[cand]
        break
if LABEL_COL is None:
    # last resort: choose small-cardinality non-id col
    nn = len(train_full)
    cand2 = []
    for c in train_full.columns:
        lc = c.lower()
        if 'id' in lc or 'time' in lc or 'timestamp' in lc:
            continue
        u = train_full[c].nunique(dropna=True)
        if 2 <= u <= min(50, max(2, nn//10)):
            cand2.append((c,u))
    cand2 = sorted(cand2, key=lambda x: x[1])
    if cand2:
        LABEL_COL = cand2[0][0]
        print(f"[Heuristic] LABEL_COL='{LABEL_COL}' (nunique={cand2[0][1]})")
    else:
        raise KeyError("KhÃ´ng tÃ¬m tháº¥y cá»™t nhÃ£n. HÃ£y Ä‘áº·t LABEL_COL thá»§ cÃ´ng.")

print('LABEL_COL =', LABEL_COL)




# =====================
# 1) MODALITY SPLIT & FEATURE MATRIX
# =====================
IMU_PREFIXES = ['ax','ay','az','gx','gy','gz','acc','gyro','mag','imu']
TOF_PREFIXES = ['tof']
THERMO_PREFIXES = ['thm','thermo']

num_cols_all = []
cat_cols_all = []
for c in train_full.columns:
    if c == LABEL_COL: 
        continue
    if pd.api.types.is_numeric_dtype(train_full[c]):
        num_cols_all.append(c)
    else:
        cat_cols_all.append(c)

# simple modality detector by prefix

def pick_by_prefix(cols, prefixes):
    out = []
    for c in cols:
        lc = c.lower()
        if any(lc.startswith(p) for p in prefixes):
            out.append(c)
    return out

IMU_COLS = pick_by_prefix(num_cols_all, IMU_PREFIXES)
TOF_COLS = pick_by_prefix(num_cols_all, TOF_PREFIXES)
THERMO_COLS = pick_by_prefix(num_cols_all, THERMO_PREFIXES)
META_NUM_COLS = [c for c in num_cols_all if c not in set(IMU_COLS+TOF_COLS+THERMO_COLS)]
META_CAT_COLS = cat_cols_all

print({'imu': len(IMU_COLS), 'tof': len(TOF_COLS), 'thermo': len(THERMO_COLS), 'meta_num': len(META_NUM_COLS), 'meta_cat': len(META_CAT_COLS)})

# Encode labels
le = LabelEncoder()
y_text = train_full[LABEL_COL].astype(str).values
y = le.fit_transform(y_text)
NUM_CLASSES = len(le.classes_)
print('Classes:', list(le.classes_))

# One-hot cat meta (bounded cardinality)
MAX_OHE_CARD = 20
meta_cat_df = train_full[META_CAT_COLS].copy() if META_CAT_COLS else pd.DataFrame(index=train_full.index)
if not meta_cat_df.empty:
    meta_cat_df = meta_cat_df.fillna('NA')
    for c in meta_cat_df.columns:
        vc = meta_cat_df[c].value_counts()
        keep = set(vc.head(MAX_OHE_CARD).index)
        meta_cat_df[c] = meta_cat_df[c].where(meta_cat_df[c].isin(keep), other='OTHER')
    META_CAT_OHE = pd.get_dummies(meta_cat_df, drop_first=True, dtype=np.int8)
else:
    META_CAT_OHE = pd.DataFrame(index=train_full.index)

# Build per-modality numeric matrices
IMU = train_full[IMU_COLS].astype(np.float32) if IMU_COLS else pd.DataFrame(index=train_full.index)
TOF = train_full[TOF_COLS].astype(np.float32) if TOF_COLS else pd.DataFrame(index=train_full.index)
THERMO = train_full[THERMO_COLS].astype(np.float32) if THERMO_COLS else pd.DataFrame(index=train_full.index)
META_NUM = train_full[META_NUM_COLS].astype(np.float32) if META_NUM_COLS else pd.DataFrame(index=train_full.index)

# Add simple per-row ToF statistic (spatial var) if present
if not TOF.empty:
    IMPLICIT_STAT = TOF.var(axis=1).astype(np.float32)
    IMPLICIT_STAT.name = 'tof_spatial_var'
else:
    IMPLICIT_STAT = pd.Series(np.zeros(len(train_full), dtype=np.float32), index=train_full.index, name='tof_spatial_var')

# Standardize numeric blocks separately (fit inside CV later); here keep raw, scaler will be in dataset/loader




# =====================
# 2) DATASET & DATALOADER
# =====================
class MultiModalFrame:
    def __init__(self, imu, tof, thermo, meta_num, meta_cat_ohe, y=None):
        self.imu = imu.values if not imu.empty else None
        self.tof = tof.values if not tof.empty else None
        self.thermo = thermo.values if not thermo.empty else None
        self.meta_num = meta_num.values if not meta_num.empty else None
        self.meta_cat = meta_cat_ohe.values if not meta_cat_ohe.empty else None
        self.extra = IMPLICIT_STAT.values.reshape(-1,1)
        self.y = y

class MultiModalDataset(Dataset):
    def __init__(self, mm: MultiModalFrame, scalers=None, fit=False):
        self.mm = mm
        self.scalers = scalers or {}
        # fit scalers per block
        for name, block in [('imu', mm.imu), ('tof', mm.tof), ('thermo', mm.thermo), ('meta_num', mm.meta_num), ('extra', mm.extra)]:
            if block is None: 
                continue
            if fit:
                sc = StandardScaler()
                sc.fit(block)
                self.scalers[name] = sc
            elif name not in self.scalers:
                sc = StandardScaler(); sc.fit(block)
                self.scalers[name] = sc
        
        # transform
        self.Xs = {}
        for name, block in [('imu', mm.imu), ('tof', mm.tof), ('thermo', mm.thermo), ('meta_num', mm.meta_num), ('extra', mm.extra)]:
            if block is None:
                self.Xs[name] = None
            else:
                self.Xs[name] = self.scalers[name].transform(block).astype(np.float32)
        self.Xs['meta_cat'] = mm.meta_cat.astype(np.float32) if mm.meta_cat is not None else None
        self.y = mm.y.astype(np.int64) if mm.y is not None else None
        
    def __len__(self):
        # use imu length if exists else any
        for name in ['imu','tof','thermo','meta_num','extra','meta_cat']:
            X = self.Xs.get(name)
            if X is not None:
                return X.shape[0]
        return 0
    
    def __getitem__(self, idx):
        batch = {}
        for name in ['imu','tof','thermo','meta_num','extra','meta_cat']:
            X = self.Xs.get(name)
            batch[name] = torch.from_numpy(X[idx]) if X is not None else torch.zeros(0)
        y = torch.tensor(self.y[idx]) if self.y is not None else torch.tensor(-1)
        return batch, y




# =====================
# 3) MODELS
# =====================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[128,64], out_dim=None, p=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        if out_dim is not None:
            layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Early Fusion: simply concat all available blocks â†’ MLP
class EarlyFusionModel(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.dims = dims
        in_dim = sum([d for d in dims.values() if d > 0])
        self.backbone = MLP(in_dim, hidden=[256, 128, 64])
        self.head = nn.Linear(64, num_classes)

    def forward(self, batch):
        feats = []
        for name in ['imu', 'tof', 'thermo', 'meta_num', 'extra', 'meta_cat']:
            if self.dims.get(name, 0) <= 0:
                continue
            x = batch[name]
            feats.append(x)
        x = torch.cat(feats, dim=-1)
        z = self.backbone(x)
        logits = self.head(z)
        return logits, {'att': None}


# Late Fusion: per-modality encoders then concat â†’ MLP
class LateFusionModel(nn.Module):
    def __init__(self, dims, num_classes, emb=64):
        super().__init__()
        self.enc = nn.ModuleDict()
        for name in ['imu','tof','thermo','meta_num','extra','meta_cat']:
            d = dims.get(name,0)
            if d>0:
                self.enc[name] = MLP(d, hidden=[128,emb])
        self.fuse = MLP(emb*len(self.enc), hidden=[128,64])
        self.head = nn.Linear(64, num_classes)
    def forward(self, batch):
        embs = []
        for name, enc in self.enc.items():
            x = batch[name]
            embs.append(enc(x))
        x = torch.cat(embs, dim=-1)
        z = self.fuse(x)
        logits = self.head(z)
        return logits, {'att': None}

# Attention Fusion: learn attention over modality embeddings
class AttentionFusionModel(nn.Module):
    def __init__(self, dims, num_classes, emb=64):
        super().__init__()
        self.enc = nn.ModuleDict()
        names = []
        for name in ['imu','tof','thermo','meta_num','extra','meta_cat']:
            d = dims.get(name,0)
            if d>0:
                self.enc[name] = MLP(d, hidden=[128,emb])
                names.append(name)
        self.names = names
        self.query = nn.Parameter(torch.randn(emb))  # global query vector
        self.fuse = MLP(emb, hidden=[128,64])
        self.head = nn.Linear(64, num_classes)
    def forward(self, batch):
        embs = []
        for name in self.names:
            embs.append(self.enc[name](batch[name]))  # [B, emb]
        H = torch.stack(embs, dim=1)                 # [B, M, emb]
        q = self.query[None,None,:].expand(H.size(0),1,-1)  # [B,1,emb]
        # scaled dot-product attention over modalities
        att = torch.matmul(q, H.transpose(1,2)) / math.sqrt(H.size(-1))  # [B,1,M]
        att = torch.softmax(att, dim=-1)                                 # [B,1,M]
        ctx = torch.matmul(att, H).squeeze(1)                            # [B,emb]
        z = self.fuse(ctx)
        logits = self.head(z)
        return logits, {'att': att.squeeze(1)}   # return attention weights per sample

# Mixture of Experts: gating over modality experts
class MoEModel(nn.Module):
    def __init__(self, dims, num_classes, emb=64):
        super().__init__()
        self.experts = nn.ModuleDict()
        names = []
        for name in ['imu','tof','thermo','meta_num','extra','meta_cat']:
            d = dims.get(name,0)
            if d>0:
                self.experts[name] = MLP(d, hidden=[128,emb], out_dim=num_classes)
                names.append(name)
        self.names = names
        self.gate = MLP(sum([dims[n] for n in names]), hidden=[128,64], out_dim=len(names))
    def forward(self, batch):
        inputs = []
        expert_logits = []
        for name in self.names:
            x = batch[name]
            inputs.append(x)
            expert_logits.append(self.experts[name](x))  # [B,C]
        Xcat = torch.cat(inputs, dim=-1)
        g = self.gate(Xcat)                               # [B,M]
        w = torch.softmax(g, dim=-1)                      # [B,M]
        # weighted sum of expert logits
        L = torch.stack(expert_logits, dim=1)             # [B,M,C]
        logits = torch.sum(w.unsqueeze(-1) * L, dim=1)    # [B,C]
        return logits, {'att': w}




# =====================
# 4) TRAIN / EVAL UTILS
# =====================
class TorchRunner:
    def __init__(self, model, lr=1e-3, class_weights=None):
        self.model = model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    def step(self, batch, y):
        for k in batch:
            batch[k] = batch[k].to(device)
        logits, extras = self.model(batch)
        loss = self.criterion(logits, y.to(device))
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        return loss, preds, extras

    def fit(self, dl_train, dl_val, epochs=8, verbose=False):
        best_f1, best_state = -1, None
        att_collector = []
        for ep in range(1, epochs+1):
            self.model.train()
            for (xb, yb) in dl_train:
                self.opt.zero_grad()
                loss, _, _ = self.step(xb, yb)
                loss.backward(); self.opt.step()
            # val
            self.model.eval()
            y_true, y_pred = [], []
            att_samples = []
            with torch.no_grad():
                for (xb, yb) in dl_val:
                    loss, preds, extras = self.step(xb, yb)
                    y_true.extend(yb.numpy().tolist())
                    y_pred.extend(preds.tolist())
                    if extras.get('att') is not None:
                        att_samples.append(extras['att'].cpu().numpy())
            f1 = f1_score(y_true, y_pred, average='macro')
            if verbose:
                print(f"Epoch {ep}: val F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k,v in self.model.state_dict().items()}
                if att_samples:
                    att_collector = np.concatenate(att_samples, axis=0)  # [N,M]
        # load best
        if best_state is not None:
            self.model.load_state_dict({k: v.to(device) for k,v in best_state.items()})
        return best_f1, att_collector

# helper: build dims dict from dataset sample

def infer_dims(dataset: MultiModalDataset):
    sample, _ = dataset[0]
    dims = {k: (v.numel()) for k,v in sample.items() if v.numel()>0}
    return dims




# =====================
# 5) CROSS-VALIDATION & EXPERIMENTS
# =====================
BATCH_SIZE = 256
EPOCHS = 8
FOLDS = 5

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Build global MultiModalFrame once
mm_all = MultiModalFrame(IMU, TOF, THERMO, META_NUM, META_CAT_OHE, y=y)

results = { 'IMU_only': [], 'EarlyFusion_All': [], 'LateFusion_All': [], 'AttentionFusion_All': [], 'MoE_All': [] }
att_weights_all_folds = []  # store for visualization

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    print(f"\n===== Fold {fold}/{FOLDS} =====")
    # slice frame
    def slice_block(arr, idx):
        if arr is None: return None
        return arr[idx]
    mm_tr = MultiModalFrame(
        IMU.iloc[tr_idx] if not IMU.empty else IMU,
        TOF.iloc[tr_idx] if not TOF.empty else TOF,
        THERMO.iloc[tr_idx] if not THERMO.empty else THERMO,
        META_NUM.iloc[tr_idx] if not META_NUM.empty else META_NUM,
        META_CAT_OHE.iloc[tr_idx] if not META_CAT_OHE.empty else META_CAT_OHE,
        y=y[tr_idx]
    )
    mm_va = MultiModalFrame(
        IMU.iloc[va_idx] if not IMU.empty else IMU,
        TOF.iloc[va_idx] if not TOF.empty else TOF,
        THERMO.iloc[va_idx] if not THERMO.empty else THERMO,
        META_NUM.iloc[va_idx] if not META_NUM.empty else META_NUM,
        META_CAT_OHE.iloc[va_idx] if not META_CAT_OHE.empty else META_CAT_OHE,
        y=y[va_idx]
    )

    ds_tr = MultiModalDataset(mm_tr, fit=True)
    ds_va = MultiModalDataset(mm_va, scalers=ds_tr.scalers, fit=False)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dims = infer_dims(ds_tr)
    # Class weights for imbalance
    cls_w = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y[tr_idx]).astype(np.float32)
    cls_w_t = torch.tensor(cls_w)

    # ===== Baseline: IMU-only (Early Fusion on IMU block only)
    dims_imu = {k: (v if k=='imu' else 0) for k,v in dims.items()}
    m_imu = EarlyFusionModel(dims_imu, NUM_CLASSES)
    run_imu = TorchRunner(m_imu, lr=3e-3, class_weights=cls_w_t)
    f1_imu, _ = run_imu.fit(dl_tr, dl_va, epochs=EPOCHS)
    results['IMU_only'].append(f1_imu)

    # ===== Early Fusion (All)
    m_early = EarlyFusionModel(dims, NUM_CLASSES)
    run_early = TorchRunner(m_early, lr=3e-3, class_weights=cls_w_t)
    f1_early, _ = run_early.fit(dl_tr, dl_va, epochs=EPOCHS)
    results['EarlyFusion_All'].append(f1_early)

    # ===== Late Fusion (All)
    m_late = LateFusionModel(dims, NUM_CLASSES)
    run_late = TorchRunner(m_late, lr=2e-3, class_weights=cls_w_t)
    f1_late, _ = run_late.fit(dl_tr, dl_va, epochs=EPOCHS)
    results['LateFusion_All'].append(f1_late)

    # ===== Attention Fusion (All)
    m_att = AttentionFusionModel(dims, NUM_CLASSES)
    run_att = TorchRunner(m_att, lr=2e-3, class_weights=cls_w_t)
    f1_att, att_w = run_att.fit(dl_tr, dl_va, epochs=EPOCHS)
    results['AttentionFusion_All'].append(f1_att)
    if att_w is not None and att_w.size>0:
        att_weights_all_folds.append(att_w)  # [N,M]

    # ===== Mixture of Experts (All)
    m_moe = MoEModel(dims, NUM_CLASSES)
    run_moe = TorchRunner(m_moe, lr=2e-3, class_weights=cls_w_t)
    f1_moe, _ = run_moe.fit(dl_tr, dl_va, epochs=EPOCHS)
    results['MoE_All'].append(f1_moe)

    # Free GPU RAM
    del m_imu, m_early, m_late, m_att, m_moe, run_imu, run_early, run_late, run_att, run_moe
    gc.collect(); torch.cuda.empty_cache()




# =====================
# 6) ENSEMBLE (soft voting of LateFusion + Attention)
# =====================
# For simplicity here we report mean of fold F1s from models; a strict ensemble needs storing fold predictions.
# If needed, extend runner to save val probabilities per fold and average.

# =====================
# 7) REPORT & STATS
# =====================

res_df = pd.DataFrame({k: np.array(v) for k,v in results.items()})
print("\nCV Macro-F1 per fold:\n", res_df)
print("\nCV Means:")
print(res_df.mean().sort_values(ascending=False))

# Paired tests: Fusion vs IMU-only
for k in ['EarlyFusion_All','LateFusion_All','AttentionFusion_All','MoE_All']:
    try:
        t_p = ttest_rel(res_df[k], res_df['IMU_only']).pvalue
        w_p = wilcoxon(res_df[k], res_df['IMU_only']).pvalue
        print(f"Stats {k} > IMU_only: t-test p={t_p:.4g}, Wilcoxon p={w_p:.4g}")
    except Exception as e:
        print(f"Stats failed for {k}: {e}")




# =====================
# 8) ATTENTION VISUALIZATION
# =====================
if att_weights_all_folds:
    A = np.vstack(att_weights_all_folds)  # [N_total, M]
    mean_att = A.mean(axis=0)
    # Map modality order used by AttentionFusion
    names = [n for n in ['imu','tof','thermo','meta_num','extra','meta_cat'] if (n in dims and dims[n]>0)]
    plt.figure(figsize=(5,3))
    plt.bar(names, mean_att)
    plt.title('Mean modality attention weights (val across folds)')
    plt.ylabel('Weight')
    plt.ylim(0,1)
    plt.show()
else:
    print("No attention weights collected (maybe model had no attention-enabled mods).")

# =====================
# 9) ABLATION STUDY (systematic)
# =====================
# IMU-only, IMU+Thermo, IMU+ToF, All (using Early Fusion for speed)

abl_results = { 'IMU_only': [], 'IMU_Thermo': [], 'IMU_ToF': [], 'All': [] }

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    mm_tr = MultiModalFrame(
        IMU.iloc[tr_idx] if not IMU.empty else IMU,
        TOF.iloc[tr_idx] if not TOF.empty else TOF,
        THERMO.iloc[tr_idx] if not THERMO.empty else THERMO,
        META_NUM.iloc[tr_idx] if not META_NUM.empty else META_NUM,
        META_CAT_OHE.iloc[tr_idx] if not META_CAT_OHE.empty else META_CAT_OHE,
        y=y[tr_idx]
    )
    mm_va = MultiModalFrame(
        IMU.iloc[va_idx] if not IMU.empty else IMU,
        TOF.iloc[va_idx] if not TOF.empty else TOF,
        THERMO.iloc[va_idx] if not THERMO.empty else THERMO,
        META_NUM.iloc[va_idx] if not META_NUM.empty else META_NUM,
        META_CAT_OHE.iloc[va_idx] if not META_CAT_OHE.empty else META_CAT_OHE,
        y=y[va_idx]
    )
    ds_tr = MultiModalDataset(mm_tr, fit=True)
    ds_va = MultiModalDataset(mm_va, scalers=ds_tr.scalers, fit=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dims_all = infer_dims(ds_tr)

    # helpers to zero-out dims for ablations
    def dims_mask(base, keep):
        return {k: (base[k] if k in keep else 0) for k in base}

    configs = {
        'IMU_only': ['imu'],
        'IMU_Thermo': ['imu','thermo'],
        'IMU_ToF': ['imu','tof'],
        'All': [k for k in dims_all.keys()]
    }

    for name, keep in configs.items():
        dims_cfg = dims_mask(dims_all, set(keep))
        model = EarlyFusionModel(dims_cfg, NUM_CLASSES)
        runner = TorchRunner(model, lr=3e-3)
        f1, _ = runner.fit(dl_tr, dl_va, epochs=EPOCHS)
        abl_results[name].append(f1)

# Summarize ablation
abl_df = pd.DataFrame({k: np.array(v) for k,v in abl_results.items()})
print("\nAblation (F1 per fold):\n", abl_df)
print("\nAblation means:")
print(abl_df.mean().sort_values(ascending=False))

# Done.
print("\nNotebook 4 â€“ Multimodal Fusion complete.")


