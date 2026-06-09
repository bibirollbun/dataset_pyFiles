!pip install -qq pytabkit
!pip install -qq scikit-learn==1.5.2



import xgboost as xgb
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_all.csv')
test = pd.read_csv('/kaggle/input/computer-prices-2025/computer_prices_test.csv')

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

cols_to_drop = [
    'bluetooth',
    'warranty_months',
    'storage_drive_count',
    'wifi',
    'model',
    'cpu_model',
    'display_size_in',
    'weight_kg',
    'cpu_boost_ghz',
    'battery_wh'
]

train.head(3)



def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        if col == 'team_scoring_next':
            continue

        col_type = df[col].dtype

        # Skip non-numeric columns
        if not np.issubdtype(col_type, np.number):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # ---- Integer types ----
        if np.issubdtype(col_type, np.integer):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # ---- Float types ----
        else:
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem

    if verbose:
        print(
            f"[INFO] Mem. usage decreased from {start_mem:.2f} MB "
            f"to {end_mem:.2f} MB ({reduction:.2f}% reduction)"
        )

    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



for df in [train,test]:
    df['brand'] = df['model'].str.split().str[0]  
    df['cpu_brand'] = df['cpu_model'].str.split().str[0] 
    df.drop(cols_to_drop,axis=1,inplace=True)


TARGET = 'price'
categorical_features = [
    'device_type', 'brand', 'model', 'form_factor', 'os',
    'cpu_brand', 'cpu_model', 'cpu_tier',
    'gpu_brand', 'gpu_model', 'gpu_tier',
    'storage_type', 'display_type', 'resolution',
    'wifi', 'bluetooth'
]

numerical_features = [
    'release_year', 'cpu_cores', 'cpu_threads', 'cpu_base_ghz', 'cpu_boost_ghz',
    'vram_gb', 'ram_gb', 'storage_gb', 'storage_drive_count',
    'display_size_in', 'refresh_hz', 'battery_wh', 'charger_watts',
    'psu_watts', 'weight_kg', 'warranty_months'
]

# Filter to only existing columns
categorical_features = [col for col in categorical_features if col in train.columns]
numerical_features = [col for col in numerical_features if col in train.columns]


BASE = [col for col in train.columns if col not in ['id', TARGET]]

from itertools import combinations

INTER = []

for col1, col2 in combinations(BASE, 2):
    new_col_name = f'{col1}_{col2}'
    INTER.append(new_col_name)
    for df in [train, test]:
        df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
        
print(f'{len(INTER)} Features.')


FEATURES = BASE + INTER


X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]


params = {'batch_size': 'auto',
          'patience': 32,
          'allow_amp': True,
          'arch_type': 'tabm-mini',
          'tabm_k': 14,
          'gradient_clipping_norm': 1.0, 
          'share_training_batches': False,
          'lr': 0.0029993695720154537,
          'weight_decay': 0.023742083301699905,
          'n_blocks': 3,
          'd_block': 448, 
          'dropout': 0, 
          'num_emb_type': 'pwl',
          'd_embedding': 32,
          'num_emb_n_bins': 119,
         }




def fix_dtypes_for_tabm(df, cat_cols):
    for col in df.columns:
        if col not in cat_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df



import os, sys
from contextlib import contextmanager
from pytabkit import TabM_D_Regressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import io
from contextlib import redirect_stdout

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout



N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Prepare OOF and test predictions
TabM_oof_preds = np.zeros(len(X))
TabM_test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f'--- Fold {fold}/{N_SPLITS} ---')

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = test[FEATURES].copy()

    # Fix dtypes
    X_train = fix_dtypes_for_tabm(X_train, categorical_features).fillna(0.0)
    X_val   = fix_dtypes_for_tabm(X_val, categorical_features).fillna(0.0)
    X_test_fold = fix_dtypes_for_tabm(X_test_fold, categorical_features).fillna(0.0)

    

    # ----------------------------
    # Train TabM
    # ----------------------------
    print("Model: TabM")
    with redirect_stdout(io.StringIO()):  # suppress stdout
        TabM_model = TabM_D_Regressor(**params)
        TabM_model.fit(X_train, y_train, X_val, y_val, cat_col_names=categorical_features)

    TabM_oof_preds[val_idx] = TabM_model.predict(X_val)
    TabM_test_preds += TabM_model.predict(X_test_fold) / N_SPLITS
    print(f"Fold {fold} TABM RMSE: {mean_squared_error(y_val, TabM_oof_preds[val_idx], squared=False):.4f}")
    
   
# ----------------------------
# Overall OOF RMSE
# ----------------------------
print("====================")
print(f"Overall OOF TABM RMSE: {mean_squared_error(y, TabM_oof_preds, squared=False):.4f}")
print("====================")



sub = pd.read_csv(f"/kaggle/input/computer-prices-2025/sample_submission.csv")
sub['price'] = TabM_test_preds
sub.to_csv("submission.csv",index=False)

