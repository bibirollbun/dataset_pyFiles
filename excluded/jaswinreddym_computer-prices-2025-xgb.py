# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
from sklearn.model_selection import KFold
from collections import defaultdict

TRAIN_PATH = '/kaggle/input/computer-prices-2025/computer_prices_all.csv'
TEST_PATH  = '/kaggle/input/computer-prices-2025/computer_prices_test.csv'

train = pd.read_csv(TRAIN_PATH, index_col='ID')
test  = pd.read_csv(TEST_PATH, index_col='ID')

train_raw = train.copy()
test_raw  = test.copy()

# Target
TARGET = 'price'
y = train[TARGET].copy()


# ---------- small safe helpers ----------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b): 
        return np.nan
    return a / b

# ---------- CPU parsing ----------
def parse_cpu_model(cpu_model_str, cpu_brand_col=None):
    """
    Return (cpu_brand, cpu_family, cpu_model_number, cpu_freq_ghz)
    - cpu_brand_col: if the dataset already has cpu_brand filled, prefer that.
    """
    s = str(cpu_model_str).strip().lower() if pd.notna(cpu_model_str) else ''
    brand = None
    fam = None
    num = np.nan

    # prefer an explicit cpu_brand column if passed as string
    if cpu_brand_col is not None and pd.notna(cpu_brand_col):
        brand = str(cpu_brand_col).strip().lower()

    # detect brand keywords in model string if brand not provided
    if brand is None:
        if 'intel' in s: brand = 'intel'
        elif 'amd' in s: brand = 'amd'
        elif 'apple' in s: brand = 'apple'
        elif 'qualcomm' in s: brand = 'qualcomm'

    # families: i3/i5/i7/i9, ryzen\d, m1/m2/m3, xeon
    m = re.search(r'\b(i[3579]|i[1-9])\b', s)
    if m:
        fam = m.group(0)
    else:
        m = re.search(r'\bryzen[\s-]*([1-9])\b', s)
        if m:
            fam = 'ryzen' + m.group(1)
        else:
            m = re.search(r'\bm(\d)\b', s)   # m1,m2
            if m:
                fam = 'm' + m.group(1)
            else:
                if 'xeon' in s:
                    fam = 'xeon'

    # numeric model code like '12600', '1365' etc.
    nums = re.findall(r'(\d{2,5})', s)
    if nums:
        # prefer the longest token (likely the model number)
        num = int(sorted(nums, key=len, reverse=True)[0])

    return brand, fam, num

# ---------- Storage parsing adapted to explicit columns ----------
def parse_storage_from_columns(storage_type, storage_gb, storage_drive_count):
    """
    Dataset already has storage_type, storage_gb, storage_drive_count.
    Return numeric features:
      - storage_total_gb (float)
      - storage_is_ssd (0/1)
      - storage_is_hdd (0/1)
      - storage_is_nvme (0/1)
      - storage_drive_count (int)
    Handles missing values robustly.
    """
    # normalize types
    stype = None
    if pd.notna(storage_type):
        stype = str(storage_type).strip().lower()
    total_gb = safe_float(storage_gb)
    drive_count = int(storage_drive_count) if (pd.notna(storage_drive_count) and float(storage_drive_count).is_integer()) else np.nan

    is_ssd = 0
    is_hdd = 0
    is_nvme = 0
    if stype:
        if 'ssd' in stype:
            is_ssd = 1
        if 'hdd' in stype:
            is_hdd = 1
        if 'nvme' in stype or 'nvme' in stype:
            is_nvme = 1
        # Hybrid/dual -> mark both flags if ambiguous
        if 'hybrid' in stype:
            # hybrid often means both ssd & hdd or sshd: mark both
            is_ssd = 1
            is_hdd = 1

    # fallbacks
    if pd.isna(total_gb):
        total_gb = np.nan

    return total_gb, is_ssd, is_hdd, is_nvme, drive_count

# ---------- GPU parsing ----------
def parse_gpu(gpu_model_str, gpu_brand_col=None):
    """
    Return (gpu_brand, gpu_vram_gb)
    """
    s = str(gpu_model_str).lower() if pd.notna(gpu_model_str) else ''
    brand = None
    vram = np.nan

    # prefer provided gpu_brand column if present
    if gpu_brand_col is not None and pd.notna(gpu_brand_col):
        brand = str(gpu_brand_col).strip().lower()

    if brand is None:
        if 'nvidia' in s: brand = 'nvidia'
        elif ('amd' in s and 'radeon' in s) or 'radeon' in s: brand = 'amd'
        elif 'intel' in s: brand = 'intel'
        elif 'apple' in s: brand = 'apple'

    m = re.search(r'(\d+)\s*gb', s)
    if m:
        try:
            vram = int(m.group(1))
        except:
            vram = np.nan
    else:
        # sometimes VRAM is like '6gb', or '6 gb'; we covered that, else leave NaN
        vram = np.nan

    return brand, vram

# ---------- Resolution parsing (column named 'resolution') ----------
def parse_resolution(res_str):
    """
    Parse strings like "1920x1080" or "3440x1440" and return (res_w, res_h, aspect_ratio)
    """
    if pd.isna(res_str):
        return np.nan, np.nan, np.nan
    s = str(res_str).lower()
    m = re.search(r'(\d{3,4})\s*[x×]\s*(\d{3,4})', s)
    if m:
        try:
            w = int(m.group(1)); h = int(m.group(2))
            aspect = safe_div(w, h)
            return w, h, aspect
        except:
            return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan

# ---------- RAM extraction (we already have ram_gb numeric in CSV) ----------
def extract_ram_gb(ram_col_value, ram_string_col=None):
    """
    Prefer numeric ram_gb column if present. Else try to parse string.
    """
    if pd.notna(ram_col_value):
        try:
            return float(ram_col_value)
        except:
            pass
    if ram_string_col is not None and pd.notna(ram_string_col):
        s = str(ram_string_col).lower()
        nums = re.findall(r'(\d+)\s*gb', s)
        if nums:
            return sum(int(n) for n in nums)
    return np.nan

# ---------- Apply engineered features to a dataframe ----------
def fe_dataframe(df):
    """
    Input: df (pandas DataFrame) with columns as found in the CSV.
    Returns: df with new engineered columns (keeps original columns by default).
    """
    df = df.copy()

    # --- brand/model ---
    if 'model' in df.columns and 'brand' not in df.columns:
        df['brand'] = df['model'].astype(str).str.split().str[0].replace('nan', np.nan)

    # --- CPU: prefer cpu_brand column if it exists; parse cpu_model for family/num/freq ---
    if 'cpu_model' in df.columns:
        cpu_parsed = df.apply(lambda r: pd.Series(parse_cpu_model(r.get('cpu_model', np.nan), cpu_brand_col=r.get('cpu_brand', np.nan))),
                              axis=1)
        cpu_parsed.columns = ['cpu_brand_ex', 'cpu_family_ex', 'cpu_model_num_ex']
        df = pd.concat([df, cpu_parsed], axis=1)
    else:
        # if cpu_model doesn't exist, still try to normalize cpu_brand
        if 'cpu_brand' in df.columns:
            df['cpu_brand_ex'] = df['cpu_brand'].astype(str).str.lower()

    # If explicit numeric CPU columns exist (cpu_cores, cpu_threads, base/boost ghz) keep them and cast numeric
    for c in ['cpu_cores', 'cpu_threads', 'cpu_base_ghz', 'cpu_boost_ghz']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- Storage: use storage_type + storage_gb + storage_drive_count ---
    if 'storage_type' in df.columns or 'storage_gb' in df.columns:
        stor_parsed = df.apply(lambda r: pd.Series(parse_storage_from_columns(
            r.get('storage_type', np.nan), r.get('storage_gb', np.nan), r.get('storage_drive_count', np.nan)
        )), axis=1)
        stor_parsed.columns = ['storage_total_gb', 'storage_is_ssd', 'storage_is_hdd', 'storage_is_nvme', 'storage_drive_count_ex']
        df = pd.concat([df, stor_parsed], axis=1)
        # keep original storage_drive_count as numeric if exists
        if 'storage_drive_count' in df.columns:
            df['storage_drive_count'] = pd.to_numeric(df['storage_drive_count'], errors='coerce')

    # --- RAM ---
    if 'ram_gb' in df.columns:
        df['ram_gb'] = pd.to_numeric(df['ram_gb'], errors='coerce')
    else:
        # try to parse some 'ram' textual column if present
        if 'ram' in df.columns:
            df['ram_gb'] = df['ram'].apply(lambda v: extract_ram_gb(v, v))

    # --- GPU ---
    if 'gpu_model' in df.columns or 'gpu_brand' in df.columns:
        gpu_parsed = df.apply(lambda r: pd.Series(parse_gpu(r.get('gpu_model', np.nan), gpu_brand_col=r.get('gpu_brand', np.nan))), axis=1)
        gpu_parsed.columns = ['gpu_brand_ex', 'gpu_vram_gb']
        df = pd.concat([df, gpu_parsed], axis=1)
        # coerce to numeric
        df['gpu_vram_gb'] = pd.to_numeric(df['vram_gb'], errors='coerce')

    # --- resolution / display ---
    if 'resolution' in df.columns:
        res_parsed = df['resolution'].apply(lambda s: pd.Series(parse_resolution(s)))
        res_parsed.columns = ['res_w', 'res_h', 'res_aspect']
        df = pd.concat([df, res_parsed], axis=1)
    if 'display_size_in' in df.columns:
        df['display_size_in'] = pd.to_numeric(df['display_size_in'], errors='coerce')
        # ppi if both res and size are present
        df['ppi'] = df.apply(lambda r:
                             np.sqrt((r.get('res_w') or 0)**2 + (r.get('res_h') or 0)**2) / r['display_size_in']
                             if pd.notna(r.get('res_w')) and pd.notna(r['display_size_in']) else np.nan, axis=1)

    # --- other numeric conversions ---
    for c in ['weight_kg', 'battery_wh', 'charger_watts', 'psu_watts', 'refresh_hz', 'warranty_months', 'release_year']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- warranty years and age ---
    if 'warranty_months' in df.columns:
        df['warranty_years'] = df['warranty_months'] / 12.0
    if 'release_year' in df.columns:
        current_year = pd.Timestamp.now().year
        df['age_years'] = current_year - df['release_year']

    # --- boolean flags and simple derived features ---
    # storage drive booleans already created; ensure integer type
    for c in ['storage_is_ssd', 'storage_is_hdd', 'storage_is_nvme']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # flag if discrete presence of battery (useful for desktops vs laptops)
    if 'battery_wh' in df.columns:
        df['has_battery'] = (~df['battery_wh'].isna()) & (df['battery_wh'] > 0)
        df['has_battery'] = df['has_battery'].astype(int)

    # brand normalization
    if 'brand' in df.columns:
        df['brand_simple'] = df['brand'].astype(str).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)

    # normalize os
    if 'os' in df.columns:
        df['os_simple'] = df['os'].astype(str).str.lower().replace({
            'windows 10': 'windows', 'windows 11': 'windows', 'macos': 'macos',
        })

    # final: replace empty strings with NaN for object cols
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].replace({'': np.nan, 'nan': np.nan})

    return df


train_fe = fe_dataframe(train)
test_fe  = fe_dataframe(test)

# ---------- raw columns to keep/drop ----------
to_drop = [
    'model',        # brand / model_line kept
    'cpu_model',    # cpu_brand_ex / cpu_family_ex kept
    'gpu_model',    # gpu_brand_ex kept
    'display_resolution',  # res_w / res_h kept,
]

to_drop = [c for c in to_drop if c in train_fe.columns]
train_fe = train_fe.drop(columns=to_drop, errors='ignore')
test_fe  = test_fe.drop(columns=to_drop, errors='ignore')


# ---------- Align columns and prepare categorical lists ----------
# We'll treat object columns as categorical candidates
def list_categoricals(df):
    return [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']

cat_cols = sorted(list(set(list_categoricals(train_fe) + list_categoricals(test_fe))))
# Exclude target
if TARGET in cat_cols:
    cat_cols.remove(TARGET)


# ---------- Numeric filling ----------
# Build a numeric column list
num_cols = [c for c in train_fe.columns if c not in cat_cols + [TARGET]]
# Make sure numeric dtype
for df in (train_fe, test_fe):
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

# Fill numeric missing with median (train median for train and test to avoid leakage on test)
medians = train_fe[num_cols].median()
train_fe[num_cols] = train_fe[num_cols].fillna(medians)
test_fe[num_cols]  = test_fe[num_cols].fillna(medians)


# ---------- Target encoding ----------
def kfold_target_encode(train_df, train_target, test_df, cols, n_splits=5, random_state=42, smoothing=20):
    """
    K-fold out-of-fold target encoding for 'train_df' and mapping to 'test_df'.
    Returns train_enc_df (same index order as train_df) and test_enc_df (same index order as test_df).
    smoothing: larger -> stronger shrinkage toward global mean.
    """
    train_enc = pd.DataFrame(index=train_df.index)
    test_enc = pd.DataFrame(index=test_df.index)
    global_mean = train_target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for col in cols:
        oof = pd.Series(index=train_df.index, dtype=float)
        for tr_idx, val_idx in kf.split(train_df):
            tr_vals = train_df.iloc[tr_idx][col]
            tr_target = train_target.iloc[tr_idx]
            # compute group stats on tr
            grp = tr_target.groupby(tr_vals).agg(['mean','count']).rename(columns={'mean':'grp_mean','count':'grp_count'})
            # smoothing
            smooth = (grp['grp_mean'] * grp['grp_count'] + global_mean * smoothing) / (grp['grp_count'] + smoothing)
            # map to val_idx
            oof.iloc[val_idx] = train_df.iloc[val_idx][col].map(smooth)
        # for any remaining NaNs (rare), fill with global_mean
        oof.fillna(global_mean, inplace=True)
        train_enc[col + '_te'] = oof

        # fit on full train for test mapping
        full_grp = train_target.groupby(train_df[col]).agg(['mean','count']).rename(columns={'mean':'grp_mean','count':'grp_count'})
        smooth_full = (full_grp['grp_mean'] * full_grp['grp_count'] + global_mean * smoothing) / (full_grp['grp_count'] + smoothing)
        mapped_test = test_df[col].map(smooth_full).fillna(global_mean)
        test_enc[col + '_te'] = mapped_test
    return train_enc, test_enc

train_te, test_te = kfold_target_encode(train_fe, y, test_fe, cat_cols, n_splits=5, smoothing=20)


# ---------- Construct final feature sets ----------
# Use: numeric engineered cols (num_cols), target-encoded cols (train_te/test_te), one-hot low-card (train_ohe/test_ohe)
X = pd.concat([train_fe[num_cols].reset_index(drop=True),
               train_te.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_fe[num_cols].reset_index(drop=True),
                    test_te.reset_index(drop=True)], axis=1)
X = X.drop(['cpu_model_num_ex', 'storage_total_gb', 'release_year', 'gpu_vram_gb', 'storage_drive_count_ex'], axis=1)
X_test = X_test.drop(['cpu_model_num_ex', 'storage_total_gb', 'release_year', 'gpu_vram_gb', 'storage_drive_count_ex'], axis=1)
# Keep original indexes
X.index = train_fe.index
X_test.index = test_fe.index

# Align column order
X = X.sort_index(axis=1)
X_test = X_test[X.columns]

# Final types and sanity check
print("X shape:", X.shape)
print("X_test shape:", X_test.shape)
print("y shape:", y.shape)
print("Numeric columns examples:", num_cols[:10])
print("Target-encoded cols:", [c + '_te' for c in cat_cols])


# 1) Compare numeric feature means between train and test
num_feats = X.select_dtypes('number').columns
train_means = X[num_feats].mean()
test_means = X_test[num_feats].mean()
diff = (train_means - test_means).abs().sort_values(ascending=False)
print("Top 20 feature mean differences (train vs test):")
print(diff)

# 2) Check duplicates
dupes = X.duplicated().sum()
print("Number of duplicate train rows:", dupes)

# 3) Check target distribution skew / outliers
print("Target: mean, std, min, max:", y.mean(), y.std(), y.min(), y.max())


# --- XGBoost (Optuna) ---
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import optuna
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
N_FOLDS = 5
OPTUNA_TRIALS = 50  # change to 100+ for more thorough tuning


# Convert y to 1d array
y = pd.Series(y).reset_index(drop=True)
X = X.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# -----------------------
# Utility: RMSE scoring
# -----------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# XGBoost tuning with Optuna (K-Fold CV with early stopping per fold)
print("Starting Optuna study for XGBoost...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

def xgb_cv_rmse(params, X_df, y_ser, n_splits=N_FOLDS, random_state=RANDOM_SEED):
    """Manual KFold CV that trains XGB with early stopping on each fold and returns mean RMSE."""
    rmses = []
    for train_idx, val_idx in kf.split(X_df):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_tr, y_val = y_ser.iloc[train_idx], y_ser.iloc[val_idx]

        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=2000,
            verbosity=0,
            random_state=random_state,
            early_stopping_rounds=50,
            **params
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        preds = model.predict(X_val, iteration_range=(0, model.best_iteration + 1))
        rmses.append(rmse(y_val, preds))
    return float(np.mean(rmses))

def objective(trial):
    # search space
    param = {
        'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0), #0.1
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20), #20
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 20.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 20.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise','lossguide']),
    }
    score = xgb_cv_rmse(param, X, y, n_splits=N_FOLDS, random_state=RANDOM_SEED)
    # Optuna minimizes by default
    return score

# Ucomment the lines below for optune tuning
# study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
# study.optimize(objective, n_trials=6, show_progress_bar=True)

# print("Best XGBoost trial:")
# print(study.best_trial.params)
# print(f"Best CV RMSE (XGB by Optuna): {study.best_value:.6f}")

# best_params = study.best_trial.params


best_params = {'eta': 0.028180680291847244, 'max_depth': 3, 'subsample': 0.8421165132560784, 
                'colsample_bytree': 0.4961372443656412, 'min_child_weight': 3, 'reg_alpha': 0.0004033251408150122, 
                'reg_lambda': 2.088568224702048e-08, 'grow_policy': 'depthwise'}


# Fit final XGBoost on full train with a small holdout for early stopping
print("Fitting final XGBoost on full training set (with small validation split for early stopping)...")
X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED)

final_xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=5000,
    verbosity=0,
    random_state=RANDOM_SEED,
    early_stopping_rounds=50,
    **best_params
)

final_xgb.fit(
    X_tr_full, y_tr_full,
    eval_set=[(X_val_full, y_val_full)],
    verbose=50
)



# Predictions
print("Predicting on test set with XGBoost...")
pred_xgb = final_xgb.predict(X_test, iteration_range=(0, final_xgb.best_iteration + 1))


submission_xgb = pd.DataFrame({
    'ID': test.index,
    'price': pred_xgb
})

submission_xgb.to_csv('submission_xgb.csv', index=False)
print("Wrote submissions.")


# Feature importances from XGBoost
importances = pd.Series(final_xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop XGBoost feature importances:")
print(importances.head(20))

