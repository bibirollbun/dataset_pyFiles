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


# %% [setup] Imports & environment checks
import os, sys, math, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from numpy.fft import rfft
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Optional packages
HAS_XGB = False
HAS_LGBM = False
HAS_SHAP = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass
try:
    import shap
    HAS_SHAP = True
except Exception:
    pass

print({'xgboost': HAS_XGB, 'lightgbm': HAS_LGBM, 'shap': HAS_SHAP})


# %% [data] Read CSV or simulate
DATA_PATH = ''  # vÃ­ dá»¥: '/mnt/data/train.csv' hoáº·c Ä‘á»ƒ rá»—ng Ä‘á»ƒ dÃ¹ng synthetic
LABEL_COL = 'label'

def simulate_dataset(n=1200, n_imu=6, n_tof=8, n_classes=5):
    # IMU features: 6 kÃªnh (ax, ay, az, gx, gy, gz)
    imu = np.random.randn(n, n_imu)
    # ToF pixels (vÃ­ dá»¥ 8 tia)
    tof = np.abs(np.random.randn(n, n_tof) * 0.5 + 2.5)
    # latent class influence
    y = np.random.randint(0, n_classes, size=n)
    for c in range(n_classes):
        idx = y == c
        imu[idx] += (c - n_classes/2) * 0.15
        tof[idx] += (n_classes/2 - c) * 0.1
    cols_imu = [f"ax", "ay", "az", "gx", "gy", "gz"][:n_imu]
    cols_tof = [f"tof_{i+1}" for i in range(n_tof)]
    df = pd.DataFrame(np.hstack([imu, tof]), columns=cols_imu+cols_tof)
    df[LABEL_COL] = y
    return df

if DATA_PATH and os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}")
else:
    df_raw = simulate_dataset()
    print("Using synthetic dataset (override DATA_PATH to use your data)")

df_raw.head()


# %% [fe] Auto-detect columns & feature builders
def detect_columns(df):
    cols = list(df.columns)
    cols = [c for c in cols if c != LABEL_COL]
    imu_prefixes = ['acc', 'gyro', 'mag', 'imu']
    imu_exact = {'ax','ay','az','gx','gy','gz'}
    tof_prefixes = ['tof', 'distance']
    imu_cols, tof_cols, others = [], [], []
    for c in cols:
        cl = c.lower()
        if cl in imu_exact or any(cl.startswith(p) for p in imu_prefixes):
            imu_cols.append(c)
        elif any(cl.startswith(p) for p in tof_prefixes):
            tof_cols.append(c)
        else:
            others.append(c)
    return imu_cols, tof_cols, others

imu_cols, tof_cols, other_cols = detect_columns(df_raw)
print('IMU cols:', imu_cols)
print('ToF cols:', tof_cols)
print('Other cols:', other_cols)

def fft_energy(vec):
    # NÄƒng lÆ°á»£ng cá»§a phá»• FFT rá»�i ráº¡c (bá»� DC component)
    v = np.asarray(vec)
    if v.ndim == 0:
        return 0.0
    spec = np.abs(rfft(v))
    if spec.size <= 1:
        return 0.0
    return float(np.sum(spec[1:]**2))

def build_row_features(df, imu_cols, tof_cols):
    X = pd.DataFrame(index=df.index)
    # Basic stats for each IMU channel
    for c in imu_cols:
        X[f'{c}_mean'] = df[c]
        # std/ skew trÃªn 1 giÃ¡ trá»‹ khÃ´ng Ä‘á»‹nh nghÄ©a -> set 0 theo hÃ ng if 1 value
        X[f'{c}_std'] = 0.0
        X[f'{c}_skew'] = 0.0
        # FFT energy trÃªn tá»«ng hÃ ng (giÃ¡ trá»‹ Ä‘Æ¡n) lÃ  0
        X[f'{c}_fft_energy'] = 0.0
    # ToF spatial variance (phÆ°Æ¡ng sai giá»¯a cÃ¡c pixel ToF á»Ÿ cÃ¹ng hÃ ng)
    if len(tof_cols) > 1:
        X['tof_spatial_var'] = df[tof_cols].var(axis=1)
        X['tof_spatial_mean'] = df[tof_cols].mean(axis=1)
    elif len(tof_cols) == 1:
        X['tof_spatial_var'] = 0.0
        X['tof_spatial_mean'] = df[tof_cols[0]]
    return X

# Build features (row-wise baseline). Náº¿u báº¡n cÃ³ dáº¡ng time-series theo cá»­a sá»•,
# hÃ£y thay báº±ng hÃ m groupby(window_id) vÃ  tÃ­nh cÃ¡c stats tÆ°Æ¡ng tá»±.
X_fe = build_row_features(df_raw, imu_cols, tof_cols)
y = df_raw[LABEL_COL].values
X_fe.head()


# %% [models] Define models & evaluator
def get_models():
    models = {
        'LogReg': Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', LogisticRegression(max_iter=2000, n_jobs=None, solver='saga', random_state=RANDOM_STATE))
        ]),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric='mlogloss',
            random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        models['HGB'] = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE)
    return models

def evaluate_models(X, y, n_splits=5):
    results = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(f1_score, average='macro')
    for name, model in get_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        results.append({'model': name, 'macro_f1_mean': scores.mean(), 'macro_f1_std': scores.std(), 'n_splits': n_splits})
    return pd.DataFrame(results).sort_values('macro_f1_mean', ascending=False)

results_preview = evaluate_models(X_fe, y)
results_preview


# %% [compare] IMU-only vs ToF-only vs Multimodal
def subset_features(X, use_imu=True, use_tof=True):
    cols = []
    if use_imu:
        cols += [c for c in X.columns if any(c.startswith(prefix) for prefix in [
            'ax_', 'ay_', 'az_', 'gx_', 'gy_', 'gz_', 'acc', 'gyro', 'mag', 'imu'
        ])]
    if use_tof:
        cols += [c for c in X.columns if c.startswith('tof_') or c.startswith('tof')]
        cols += [c for c in X.columns if c in ['tof_spatial_var','tof_spatial_mean']]
    cols = list(dict.fromkeys(cols))
    return X[cols] if cols else X

experiments = {}
X_imu = subset_features(X_fe, use_imu=True, use_tof=False)
X_tof = subset_features(X_fe, use_imu=False, use_tof=True)
X_multi = subset_features(X_fe, use_imu=True, use_tof=True)

experiments['IMU_only'] = evaluate_models(X_imu, y)
if X_tof.shape[1] > 0:
    experiments['ToF_only'] = evaluate_models(X_tof, y)
experiments['Multimodal'] = evaluate_models(X_multi, y)

for name, dfres in experiments.items():
    print('\n===', name, '===')
    display(dfres)


# %% [fs] Mutual Information & PCA selection
from sklearn.impute import SimpleImputer

def mi_select(X, y, k=64):
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    mi = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE, discrete_features=False)
    order = np.argsort(mi)[::-1][:min(k, X.shape[1])]
    cols_sel = X.columns[order]
    return X[cols_sel], pd.Series(mi, index=X.columns).sort_values(ascending=False)

def pca_transform(X, var_keep=0.95):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=var_keep, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    return Xp, pca, scaler

# Run on Multimodal as default
Xmm = X_multi.copy()
Xmi, mi_scores = mi_select(Xmm, y, k=64)
print('MI-selected shape:', Xmi.shape)
Xp, pca, scaler = pca_transform(Xmm, var_keep=0.95)
print('PCA shape:', Xp.shape)

# Evaluate
res_mi = evaluate_models(Xmi, y)
res_pca = evaluate_models(pd.DataFrame(Xp), y)
display(res_mi)
display(res_pca)

# Plot MI scores (Top 20)
top_mi = mi_scores.head(20)
plt.figure()
top_mi[::-1].plot(kind='barh')
plt.title('Top-20 Mutual Information scores')
plt.tight_layout()
plt.show()


# %% [viz] Permutation Importance & SHAP (optional)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xmm, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
pi = permutation_importance(rf, X_test, y_test, scoring=make_scorer(f1_score, average='macro'), n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
pi_mean = pd.Series(pi.importances_mean, index=Xmm.columns).sort_values(ascending=False)
top20 = pi_mean.head(20)

plt.figure()
top20[::-1].plot(kind='barh')
plt.title('Permutation Importance (RF) â€“ Top 20')
plt.tight_layout()
plt.show()

if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        # summary plot (requires matplotlib, generated inline)
        shap.summary_plot(shap_values, X_test, show=True)
    except Exception as e:
        print('SHAP failed:', e)
else:
    print('SHAP not available; skipping SHAP plots.')


# %% [save] Export result tables
out_dir = Path('/mnt/data')
out_dir.mkdir(parents=True, exist_ok=True)

def concat_results(experiments):
    rows = []
    for name, dfres in experiments.items():
        tmp = dfres.copy()
        tmp.insert(0, 'feature_set', name)
        rows.append(tmp)
    return pd.concat(rows, axis=0, ignore_index=True)

all_res = concat_results(experiments)
res_path = out_dir / 'baseline_results.csv'
mi_path = out_dir / 'mi_scores.csv'
all_res.to_csv(res_path, index=False)
mi_scores.to_csv(mi_path, header=['mi_score'])
print('Saved:', res_path)
print('Saved:', mi_path)

