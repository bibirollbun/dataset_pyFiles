# ============================================================
# REAL ESTATE DEMAND PREDICTION - Kaggle Notebook Single Cell (Upgraded + Fixed)
# End-to-end: Data -> FE -> Backtesting -> LGBM/CatBoost/XGBoost -> Per-horizon Blend -> Horizon -> Submission
# Key fixes:
# - Avoid filling categorical columns with numeric sentinels
# - Separate frames per model: LGBM uses 'category' dtypes; Cat/XGB use integer codes
# ============================================================

import os
import gc
import warnings
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [15, 6]
pd.set_option('display.max_columns', None)
np.random.seed(42)

# ----------------------- User-tunable parameters -----------------------
DATA_PATH = '/kaggle/input/china-real-estate-demand-prediction/'

# Validation settings
N_FOLDS = 5         # number of time-based folds (for one-step backtest reporting)
VAL_WINDOW = 3      # validation window (months) per fold
EARLY_STOP = 500    # early stopping rounds

# Feature engineering
LAG_FEATURES = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)

# CSI control
CSI_TOP_K = 50          # variance-based top-K keywords before PCA (None to skip)
CSI_PCA_K = 16          # number of CSI PCA components to keep (None to skip PCA)
CSI_USE_PCA_ONLY = True # if True, drop raw CSI keywords after adding PCA comps
ADD_CSI_TOTAL = True

# Target transform and metrics
Y_TRANSFORM = 'log1p'  # 'log1p' or 'none'
MAPE_EPS = 1.0

# LightGBM objective
LGBM_OBJECTIVE = 'tweedie'   # 'tweedie' | 'poisson' | 'regression_l1' | 'regression'
TWEEDIE_POWER = 1.5          # typical range [1.1, 1.5]

# Model toggles
FORCE_DISABLE_CAT = False    # set True to force disable CatBoost (no GPU)
FORCE_DISABLE_XGB = False    # set True to force disable XGBoost if unavailable/slow
USE_MULTI_HORIZON_ML = True  # per-horizon ML with per-horizon blending
ENABLE_DECEMBER_BUMP = True  # apply December seasonal scaling based on historical stats

# Ensemble search (for LGBM/Cat/XGB blending)
OPTIMIZE_ENSEMBLE_WEIGHTS = True
WEIGHT_OPTIM_STEP = 0.05     # 0.05 is finer than 0.1, but slower

# Post-processing: per-sector clipping
CLIP_P99_FACTOR = 1.6        # upper cap multiplier for sector p99
CLIP_MIN_MEDIAN_FACTOR = 3.0 # ensure cap at least this multiple of sector median

# Plotting toggle
ENABLE_PLOTS = True
# ----------------------------------------------------------------------

# Optional CatBoost (graceful fallback)
try:
    if FORCE_DISABLE_CAT:
        raise ImportError("CatBoost forced disabled")
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# Optional XGBoost (graceful fallback)
try:
    if FORCE_DISABLE_XGB:
        raise ImportError("XGBoost forced disabled")
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ====================== Utility & Metrics ======================
def reduce_memory_usage(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=['int', 'int64', 'int32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start:.2f} MB -> {end:.2f} MB ({100*(start-end)/max(start,1e-9):.1f}%)")
    return df

def safe_mape(y_true, y_pred, eps=MAPE_EPS):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def rmsle(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_pred + eps) - np.log1p(y_true + eps))**2)))

def compute_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'RMSLE': float(rmsle(y_true, y_pred)),
        'sMAPE': float(smape(y_true, y_pred)),
        'MAPE_safe': float(safe_mape(y_true, y_pred)),
    }

def set_lgbm_categoricals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns and df[c].dtype.name != 'category':
            df[c] = df[c].astype('category')
    return df

def prepare_frames_for_models(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]):
    """
    Returns:
    - X_lgb: DataFrame with categorical dtypes preserved for LGBM; numeric columns cleaned.
    - X_num: DataFrame where categorical columns are converted to integer codes; fully numeric and cleaned for Cat/XGB.
    """
    X = df[feature_cols].copy()

    # LGBM frame: only clean numeric columns
    X_lgb = X.copy()
    num_cols = [c for c in X_lgb.columns if X_lgb[c].dtype.name != 'category']
    X_lgb[num_cols] = X_lgb[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-2)
    # ensure no NaN in categoricals
    for c in cat_cols:
        if c in X_lgb.columns and X_lgb[c].dtype.name == 'category' and X_lgb[c].isna().any():
            X_lgb[c] = X_lgb[c].cat.add_categories(['__NA__']).fillna('__NA__')

    # Numeric frame for Cat/XGB: convert categories to codes first, then clean
    X_num = X.copy()
    for c in cat_cols:
        if c in X_num.columns:
            if X_num[c].dtype.name == 'category':
                X_num[c] = X_num[c].cat.codes.astype('int32')
            else:
                # if not category yet, cast to int if possible
                X_num[c] = pd.to_numeric(X_num[c], errors='coerce').fillna(-1).astype('int32')
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(-2)

    return X_lgb, X_num

# ====================== Data IO & Preprocess ======================
def load_data() -> Dict[str, pd.DataFrame]:
    print(Fore.BLUE + Style.BRIGHT + "--- LOAD DATA ---" + Style.RESET_ALL)
    base = os.environ.get('DATA_PATH', DATA_PATH)
    data = {
        'ci': pd.read_csv(f'{base}train/city_indexes.csv'),
        'csi': pd.read_csv(f'{base}train/city_search_index.csv'),
        'sp': pd.read_csv(f'{base}train/sector_POI.csv'),
        'train_lt': pd.read_csv(f'{base}train/land_transactions.csv'),
        'train_pht': pd.read_csv(f'{base}train/pre_owned_house_transactions.csv'),
        'train_nht': pd.read_csv(f'{base}train/new_house_transactions.csv'),
        'test': pd.read_csv(f'{base}test.csv'),
        'sample_sub': pd.read_csv(f'{base}sample_submission.csv'),
    }
    print(Fore.GREEN + "✓ Loaded" + Style.RESET_ALL)
    return data

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    month_codes = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df = df.copy()
    if 'month' in df.columns:
        mstr = df['month'].astype(str)
        df['year'] = mstr.str.slice(0, 4).astype(int)
        tail = mstr.str.slice(5)
        is_alpha = tail.str.contains('[A-Za-z]', regex=True).fillna(False)
        df.loc[is_alpha, 'month_num'] = tail[is_alpha].map(month_codes).astype('Int64')
        df.loc[~is_alpha, 'month_num'] = pd.to_numeric(tail[~is_alpha], errors='coerce').astype('Int64')
        df['month_num'] = df['month_num'].fillna(1).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month_num'] - 1
        df.drop(columns=['month'], inplace=True, errors='ignore')
    if 'sector' in df.columns:
        df['sector_id'] = df['sector'].astype(str).str.slice(7).astype(int)
        df.drop(columns=['sector'], inplace=True, errors='ignore')
    return df

def process_city_search_index(csi: pd.DataFrame) -> pd.DataFrame:
    csi = preprocess_dataframe(csi).reset_index(drop=True)
    pt = csi.pivot_table(index='time', columns='keyword', values='search_volume', aggfunc='sum').fillna(0)

    # variance filter
    if CSI_TOP_K and pt.shape[1] > CSI_TOP_K:
        top_cols = pt.var().sort_values(ascending=False).head(CSI_TOP_K).index
        pt = pt[top_cols]

    # optional PCA reduction
    pca_df = None
    if CSI_PCA_K and CSI_PCA_K > 0 and pt.shape[1] > 0:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(pt.values)
        pca = PCA(n_components=min(CSI_PCA_K, pt.shape[1]))
        comps = pca.fit_transform(X)
        pca_cols = [f'csi_pca_{i+1}' for i in range(comps.shape[1])]
        pca_df = pd.DataFrame(comps, index=pt.index, columns=pca_cols)

    # rename raw CSI
    pt_renamed = pt.copy()
    pt_renamed.columns = [f'csi_{str(c).replace(" ", "_")}' for c in pt_renamed.columns]

    # build final CSI block
    blocks = []
    if not CSI_USE_PCA_ONLY and pt_renamed.shape[1] > 0:
        blocks.append(pt_renamed)
    if pca_df is not None:
        blocks.append(pca_df)
    if ADD_CSI_TOTAL:
        base_for_total = pt_renamed if pt_renamed.shape[1] > 0 else pt
        csi_total = pd.DataFrame({'csi_total': base_for_total.sum(axis=1)}, index=pt.index)
        blocks.append(csi_total)

    out = pd.concat(blocks, axis=1) if blocks else pt_renamed
    return out.sort_index()

def create_master_dataframe(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    print(Fore.BLUE + Style.BRIGHT + "\n--- BUILD MASTER DF ---" + Style.RESET_ALL)
    train_nht = preprocess_dataframe(data['train_nht'])
    train_pht = preprocess_dataframe(data['train_pht'])
    train_lt = preprocess_dataframe(data['train_lt'])
    sp = preprocess_dataframe(data['sp'])
    csi_agg = process_city_search_index(data['csi'])

    test = data['test'].copy()
    sp_id = test['id'].astype(str).str.split('_', expand=True)
    test['month'] = sp_id[0]
    test['sector'] = sp_id[1]
    test = preprocess_dataframe(test)

    ci = data['ci'].copy()
    if 'month' in ci.columns:
        mstr = ci['month'].astype(str)
        ci['year'] = mstr.str.slice(0, 4).astype(int)
        ci.drop(columns=['month'], inplace=True)
    elif 'city_indicator_data_year' in ci.columns:
        ci.rename(columns={'city_indicator_data_year': 'year'}, inplace=True)
    if 'year' not in ci.columns:
        raise ValueError("city_indexes.csv must have year/month/city_indicator_data_year")
    ci['year'] = pd.to_numeric(ci['year'], errors='coerce').astype('Int64').fillna(0).astype(int)

    max_time_train = int(train_nht['time'].max())
    max_time_test = int(test['time'].max())
    base_idx = pd.MultiIndex.from_product([range(0, max_time_test+1), range(1, 97)], names=['time', 'sector_id'])
    df_master = pd.DataFrame(index=base_idx).reset_index()
    df_master['year'] = (df_master['time'] // 12) + 2019
    df_master['month_num'] = (df_master['time'] % 12) + 1

    df_master = df_master.merge(train_nht, on=['time', 'sector_id', 'year'], how='left')
    df_master = df_master.merge(train_pht, on=['time', 'sector_id', 'year'], how='left', suffixes=('', '_pht'))
    df_master = df_master.merge(train_lt, on=['time', 'sector_id', 'year'], how='left', suffixes=('', '_lt'))
    df_master = df_master.merge(sp, on='sector_id', how='left')
    df_master = df_master.merge(ci, on='year', how='left')
    df_master = df_master.merge(csi_agg, on='time', how='left')

    train_idx = df_master['time'] <= max_time_train
    if 'amount_new_house_transactions' in df_master.columns:
        df_master.loc[train_idx, 'amount_new_house_transactions'] = df_master.loc[train_idx, 'amount_new_house_transactions'].fillna(0)
    txn_cols = [c for c in df_master.columns if ('transaction' in c or 'area' in c)]
    if txn_cols:
        df_master[txn_cols] = df_master[txn_cols].fillna(0)
    csi_cols = [c for c in df_master.columns if c.startswith('csi_') or c.startswith('csi_pca_') or c == 'csi_total']
    if csi_cols:
        df_master[csi_cols] = df_master[csi_cols].fillna(0)

    df_master = reduce_memory_usage(df_master, verbose=True)
    print(f"✓ Master DF: {df_master.shape}")
    del train_pht, train_lt, sp, csi_agg, ci
    gc.collect()
    return df_master, max_time_train, test

# ====================== Features ======================
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['quarter'] = (df['month_num'] - 1) // 3 + 1
    df['is_year_end'] = (df['month_num'] == 12).astype(int)
    df['is_year_start'] = (df['month_num'] == 1).astype(int)
    # cyclical month
    df['month_sin'] = np.sin(2 * np.pi * (df['month_num'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month_num'] - 1) / 12.0)
    return df

def create_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print(Fore.BLUE + "Create lag/rolling..." + Style.RESET_ALL)
    df = df.set_index(['time', 'sector_id']).sort_index()
    grouped = df.groupby('sector_id', sort=False)

    feats = [
        'amount_new_house_transactions', 'price_new_house_transactions',
        'amount_pre_owned_house_transactions', 'transaction_amount'
    ]
    csi_cols = [c for c in df.columns if c.startswith('csi_') or c.startswith('csi_pca_') or c == 'csi_total']
    # prioritize a small core
    core_csi = []
    if 'csi_total' in csi_cols:
        core_csi.append('csi_total')
    core_csi.extend([c for c in csi_cols if c != 'csi_total'][:min(12, len(csi_cols))])
    feats.extend(core_csi)

    new_cols = {}
    for col in feats:
        if col not in df.columns:
            continue
        for lag in LAG_FEATURES:
            new_cols[f'{col}_lag{lag}'] = grouped[col].shift(lag)
        shifted = grouped[col].shift(1)
        for w in ROLLING_WINDOWS:
            new_cols[f'{col}_roll_mean{w}'] = shifted.rolling(w).mean()
            new_cols[f'{col}_roll_std{w}'] = shifted.rolling(w).std()
            new_cols[f'{col}_roll_max{w}'] = shifted.rolling(w).max()
            new_cols[f'{col}_roll_min{w}'] = shifted.rolling(w).min()

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).reset_index()
    df = reduce_memory_usage(df)
    print(Fore.GREEN + f"✓ New features: {len(new_cols)}" + Style.RESET_ALL)
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'price_new_house_transactions_lag1' in df.columns and 'price_new_house_transactions_lag3' in df.columns:
        df['price_momentum'] = (df['price_new_house_transactions_lag1'] /
                                (df['price_new_house_transactions_lag3'] + 1))
    if 'amount_new_house_transactions_lag1' in df.columns and 'amount_new_house_transactions_lag6' in df.columns:
        df['volume_trend'] = (df['amount_new_house_transactions_lag1'] /
                              (df['amount_new_house_transactions_lag6'] + 1))
    return df

# ====================== Split & Transform ======================
def time_folds(unique_times, n_folds=N_FOLDS, val_window=VAL_WINDOW):
    times = np.sort(unique_times)
    folds = []
    end_idx = len(times)
    for k in range(n_folds, 0, -1):
        val_end = end_idx - (n_folds - k) * val_window
        val_start = max(val_end - val_window, 0)
        if val_end <= 0 or val_start <= 0:
            continue
        val_times = times[val_start:val_end]
        train_times = times[:val_start]
        if len(val_times) == 0 or len(train_times) == 0:
            continue
        folds.append((train_times, val_times))
    if not folds and len(times) > 1:
        folds = [(times[:-1], times[-1:])]
    return folds

def y_to_model(y: pd.Series) -> pd.Series:
    if Y_TRANSFORM == 'log1p':
        return np.log1p(np.maximum(y, 0))
    return y

def y_from_model(y_pred: np.ndarray) -> np.ndarray:
    if Y_TRANSFORM == 'log1p':
        return np.maximum(np.expm1(y_pred), 0.0)
    return y_pred

# ====================== CV & Train (one-step backtest reporting) ======================
def lgbm_cv_train(df_train, feature_cols):
    params = dict(
        metric='l1',
        n_estimators=10000,
        learning_rate=0.01,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.15,
        lambda_l2=0.15,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
        seed=42
    )
    if LGBM_OBJECTIVE == 'tweedie':
        params['objective'] = 'tweedie'
        params['tweedie_variance_power'] = TWEEDIE_POWER
    elif LGBM_OBJECTIVE == 'poisson':
        params['objective'] = 'poisson'
    elif LGBM_OBJECTIVE == 'regression_l1':
        params['objective'] = 'regression_l1'
    else:
        params['objective'] = 'regression'

    folds = time_folds(np.sort(df_train['time'].unique()))
    cv_metrics = []
    val_preds_all = []
    y_val_all = []

    # ensure categorical dtypes for LGBM
    cat_cols = [c for c in ['sector_id', 'year', 'month_num', 'quarter'] if c in df_train.columns]
    df_train = set_lgbm_categoricals(df_train, cat_cols)

    for tr_times, va_times in folds:
        tr_df = df_train[df_train['time'].isin(tr_times)]
        va_df = df_train[df_train['time'].isin(va_times)]

        X_tr_lgb, _ = prepare_frames_for_models(tr_df, feature_cols, cat_cols)
        X_va_lgb, _ = prepare_frames_for_models(va_df, feature_cols, cat_cols)
        y_tr = y_to_model(tr_df['amount_new_house_transactions'])
        y_va = va_df['amount_new_house_transactions']

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_to_model(y_va))],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=0)]
        )
        va_pred = y_from_model(model.predict(X_va_lgb, num_iteration=model.best_iteration_))
        va_pred = np.maximum(va_pred, 0.0)
        cv_metrics.append(compute_metrics(y_va.values, va_pred))
        val_preds_all.append(va_pred)
        y_val_all.append(y_va.values)

    return params, cv_metrics, val_preds_all, y_val_all

def cat_cv_train(df_train, feature_cols):
    if not HAS_CAT:
        return None, [], [], []
    folds = time_folds(np.sort(df_train['time'].unique()))
    cv_metrics = []
    val_preds_all = []
    y_val_all = []

    cat_cols = [c for c in ['sector_id', 'year', 'month_num', 'quarter'] if c in df_train.columns]
    df_train = set_lgbm_categoricals(df_train, cat_cols)

    for tr_times, va_times in folds:
        tr_df = df_train[df_train['time'].isin(tr_times)]
        va_df = df_train[df_train['time'].isin(va_times)]

        _, X_tr_num = prepare_frames_for_models(tr_df, feature_cols, cat_cols)
        _, X_va_num = prepare_frames_for_models(va_df, feature_cols, cat_cols)
        y_tr = y_to_model(tr_df['amount_new_house_transactions'])
        y_va = va_df['amount_new_house_transactions'].values

        cb = CatBoostRegressor(
            iterations=20000,
            learning_rate=0.02,
            depth=8,
            loss_function='RMSE',
            eval_metric='RMSE',
            l2_leaf_reg=2.0,
            random_strength=0.3,
            od_type='Iter',
            od_wait=EARLY_STOP,
            random_seed=42,
            task_type='GPU' if os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None else 'CPU',
            verbose=False
        )
        cb.fit(X_tr_num, y_tr, eval_set=(X_va_num, y_to_model(pd.Series(y_va))), use_best_model=True, verbose=False)
        va_pred = y_from_model(cb.predict(X_va_num))
        va_pred = np.maximum(va_pred, 0.0)
        cv_metrics.append(compute_metrics(y_va, va_pred))
        val_preds_all.append(va_pred)
        y_val_all.append(y_va)

    return cb.get_params(), cv_metrics, val_preds_all, y_val_all

def xgb_cv_train(df_train, feature_cols):
    if not HAS_XGB:
        return None, [], [], []
    folds = time_folds(np.sort(df_train['time'].unique()))
    cv_metrics = []
    val_preds_all = []
    y_val_all = []

    use_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None
    tree_method = 'gpu_hist' if use_gpu else 'hist'

    cat_cols = [c for c in ['sector_id', 'year', 'month_num', 'quarter'] if c in df_train.columns]
    df_train = set_lgbm_categoricals(df_train, cat_cols)

    from xgboost import XGBRegressor
    for tr_times, va_times in folds:
        tr_df = df_train[df_train['time'].isin(tr_times)]
        va_df = df_train[df_train['time'].isin(va_times)]

        _, X_tr_num = prepare_frames_for_models(tr_df, feature_cols, cat_cols)
        _, X_va_num = prepare_frames_for_models(va_df, feature_cols, cat_cols)
        y_tr = y_to_model(tr_df['amount_new_house_transactions'])
        y_va = va_df['amount_new_house_transactions'].values

        xgb = XGBRegressor(
            n_estimators=6000,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.3,
            random_state=42,
            n_jobs=-1,
            tree_method=tree_method
        )
        xgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_to_model(pd.Series(y_va)))], verbose=False)
        va_pred = y_from_model(xgb.predict(X_va_num))
        va_pred = np.maximum(va_pred, 0.0)
        cv_metrics.append(compute_metrics(y_va, va_pred))
        val_preds_all.append(va_pred)
        y_val_all.append(y_va)

    return xgb.get_params(), cv_metrics, val_preds_all, y_val_all

def optimize_weights(val_y_list: List[np.ndarray], preds_list: List[List[np.ndarray]], step=WEIGHT_OPTIM_STEP):
    if len(preds_list) == 1:
        return [1.0]
    V = np.concatenate(val_y_list, axis=0)
    aligned_preds = [np.concatenate(p, axis=0) for p in preds_list]

    best_w, best_loss = None, 1e18
    steps = int(1/step)
    def gen_weights(k, remain, cur):
        if k == len(aligned_preds)-1:
            yield cur + [remain]
        else:
            for i in range(steps+1):
                w = i*step
                if w <= remain + 1e-9:
                    yield from gen_weights(k+1, remain-w, cur+[w])

    for w in gen_weights(0, 1.0, []):
        mix = sum(wi*pi for wi, pi in zip(w, aligned_preds))
        loss = smape(V, mix)
        if loss < best_loss:
            best_loss, best_w = loss, w
    return best_w

# ====================== Heuristic & Helpers ======================
def build_month_codes():
    return {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def add_time_and_sector_fields(df, month_codes):
    df = df.copy()
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if 'month' not in df.columns and 'month_text' in df.columns:
        df['month'] = df['month_text'].str.slice(5, None).map(month_codes)
        df['year'] = df['month_text'].str.slice(0, 4).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    elif 'month' in df.columns:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df

def build_amount_matrix(train_nht_pd, month_codes):
    df = train_nht_pd.copy()
    if 'sector_id' not in df.columns or 'time' not in df.columns:
        df = add_time_and_sector_fields(df, month_codes)
    amount_col = 'amount_new_house_transactions'
    if 'nht_amount_new_house_transactions' in df.columns:
        amount_col = 'nht_amount_new_house_transactions'
    pivot = df.set_index(['time', 'sector_id'])[amount_col].unstack()
    pivot = pivot.fillna(0)
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sectors]
    return pivot

def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.85, clip_high=1.4):
    is_dec = (a_tr.index.values % 12) == 11
    dec_means = a_tr[is_dec].mean(axis=0)
    nondec_means = a_tr[~is_dec].mean(axis=0)
    dec_counts = a_tr[is_dec].notna().sum(axis=0)
    raw = dec_means / (nondec_means + eps)
    overall = float(dec_means.mean() / (nondec_means.mean() + eps))
    raw = raw.where(dec_counts >= min_dec_obs, overall)
    raw = raw.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return raw.clip(lower=clip_low, upper=clip_high).to_dict()

# ====================== Multi-horizon ML (per-horizon blending) ======================
def lgb_params_base():
    p = dict(
        metric='l1',
        n_estimators=10000,
        learning_rate=0.01,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.15,
        lambda_l2=0.15,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=20,
        verbose=-1,
        n_jobs=-1,
        seed=42
    )
    if LGBM_OBJECTIVE == 'tweedie':
        p['objective'] = 'tweedie'
        p['tweedie_variance_power'] = TWEEDIE_POWER
    elif LGBM_OBJECTIVE == 'poisson':
        p['objective'] = 'poisson'
    elif LGBM_OBJECTIVE == 'regression_l1':
        p['objective'] = 'regression_l1'
    else:
        p['objective'] = 'regression'
    return p

def optimize_weights_simple(y_val: np.ndarray, preds_list: List[np.ndarray], step=WEIGHT_OPTIM_STEP):
    if len(preds_list) == 1:
        return [1.0]
    best_w, best_loss = None, 1e18
    steps = int(1/step)
    def gen_weights(k, remain, cur):
        if k == len(preds_list)-1:
            yield cur + [remain]
        else:
            for i in range(steps+1):
                w = i*step
                if w <= remain + 1e-9:
                    yield from gen_weights(k+1, remain-w, cur+[w])
    for w in gen_weights(0, 1.0, []):
        mix = sum(wi*pi for wi, pi in zip(w, preds_list))
        loss = smape(y_val, mix)
        if loss < best_loss:
            best_loss, best_w = loss, w
    return best_w

def train_predict_multi_horizon_blended(df_proc: pd.DataFrame, feature_cols: List[str], max_time_train: int, test_times: np.ndarray):
    """
    For each horizon H (t_test = max_time_train + H):
      - Create label_hH
      - Split train/val by time (last VAL_WINDOW months as val)
      - Train LGBM (+ CatBoost/XGBoost if available)
      - Optimize per-horizon blend weights by sMAPE on val
      - Predict test row at t_test using blended outputs
    Returns pred_matrix indexed by test_times, columns sector_id (1..96).
    """
    horizons = sorted(list(set(int(t - max_time_train) for t in test_times)))
    # Add labels
    df_mh = df_proc.sort_values(['sector_id', 'time']).copy()
    for H in horizons:
        df_mh[f'label_h{H}'] = df_mh.groupby('sector_id')['amount_new_house_transactions'].shift(-H)

    pred_mat = pd.DataFrame(index=test_times, columns=sorted(df_proc['sector_id'].unique()), dtype=float)
    pred_mat.index.name = 'time'

    use_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None
    tree_method = 'gpu_hist' if (HAS_XGB and use_gpu) else 'hist'

    # Ensure LGBM categorical dtypes exist
    cat_cols_common = [c for c in ['sector_id', 'year', 'month_num', 'quarter'] if c in df_proc.columns]
    df_proc = set_lgbm_categoricals(df_proc, cat_cols_common)
    df_mh = set_lgbm_categoricals(df_mh, cat_cols_common)

    from xgboost import XGBRegressor as _XGB  # local alias if available

    for H in horizons:
        y_col = f'label_h{H}'
        tr_all = df_mh[(df_mh['time'] <= max_time_train) & df_mh[y_col].notna()].copy()
        if tr_all.empty:
            continue
        times = np.sort(tr_all['time'].unique())
        if len(times) <= VAL_WINDOW:
            tr_times = times[:-1]
            va_times = times[-1:]
        else:
            tr_times = times[:-VAL_WINDOW]
            va_times = times[-VAL_WINDOW:]
        if len(tr_times) == 0 or len(va_times) == 0:
            tr_times = times[:-1]
            va_times = times[-1:]

        tr_df = tr_all[tr_all['time'].isin(tr_times)]
        va_df = tr_all[tr_all['time'].isin(va_times)]

        # Prepare frames per model
        X_tr_lgb, X_tr_num = prepare_frames_for_models(tr_df, feature_cols, cat_cols_common)
        X_va_lgb, X_va_num = prepare_frames_for_models(va_df, feature_cols, cat_cols_common)
        y_tr = y_to_model(tr_df[y_col])
        y_va = va_df[y_col].values

        preds_val_models = []
        models = []

        # LGBM
        lp = lgb_params_base()
        lgb_model = lgb.LGBMRegressor(**lp)
        lgb_model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_to_model(pd.Series(y_va)))],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=0)]
        )
        lgb_va = y_from_model(lgb_model.predict(X_va_lgb, num_iteration=lgb_model.best_iteration_))
        lgb_va = np.maximum(lgb_va, 0.0)
        preds_val_models.append(lgb_va)
        models.append(('lgb', lgb_model))

        # CatBoost
        if HAS_CAT:
            cb = CatBoostRegressor(
                iterations=20000,
                learning_rate=0.02,
                depth=8,
                loss_function='RMSE',
                eval_metric='RMSE',
                l2_leaf_reg=2.0,
                random_strength=0.3,
                od_type='Iter',
                od_wait=EARLY_STOP,
                random_seed=42,
                task_type='GPU' if use_gpu else 'CPU',
                verbose=False
            )
            cb.fit(X_tr_num, y_tr, eval_set=(X_va_num, y_to_model(pd.Series(y_va))), use_best_model=True, verbose=False)
            cb_va = y_from_model(cb.predict(X_va_num))
            cb_va = np.maximum(cb_va, 0.0)
            preds_val_models.append(cb_va)
            models.append(('cat', cb))

        # XGBoost
        if HAS_XGB:
            xgb = _XGB(
                n_estimators=6000,
                learning_rate=0.02,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=0.3,
                random_state=42,
                n_jobs=-1,
                tree_method=tree_method
            )
            xgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_to_model(pd.Series(y_va)))], verbose=False)
            xgb_va = y_from_model(xgb.predict(X_va_num))
            xgb_va = np.maximum(xgb_va, 0.0)
            preds_val_models.append(xgb_va)
            models.append(('xgb', xgb))

        # Blend weights for this horizon
        if OPTIMIZE_ENSEMBLE_WEIGHTS and len(preds_val_models) > 1:
            weights = optimize_weights_simple(y_va, preds_val_models, step=WEIGHT_OPTIM_STEP)
        else:
            weights = [1.0] if len(preds_val_models) == 1 else [1.0/len(preds_val_models)]*len(preds_val_models)

        print(Fore.MAGENTA + f"[H={H}] blend weights -> {dict(zip([m for m,_ in models], weights))}" + Style.RESET_ALL)

        # Predict test time tt = max_time_train + H
        tt = max_time_train + H
        te_df = df_proc[df_proc['time'] == tt]
        if te_df.empty:
            continue
        X_te_lgb, X_te_num = prepare_frames_for_models(te_df, feature_cols, cat_cols_common)

        test_preds = []
        for (name, mdl), w in zip(models, weights):
            if name == 'lgb':
                p = y_from_model(mdl.predict(X_te_lgb, num_iteration=mdl.best_iteration_))
            else:
                p = y_from_model(mdl.predict(X_te_num))
            test_preds.append(w * np.maximum(p, 0.0))

        yhat_blend = np.sum(np.vstack(test_preds), axis=0)

        # align by sector_id
        sec = te_df['sector_id'].values
        sec_sorted_idx = np.argsort(sec)
        y_sorted = yhat_blend[sec_sorted_idx]
        sec_sorted = sec[sec_sorted_idx]
        pred_mat.loc[tt, sec_sorted] = y_sorted

    pred_mat = pred_mat.fillna(0.0)
    return pred_mat

# ====================== Post-processing: per-sector clipping ======================
def robust_clip_by_sector(pred_matrix: pd.DataFrame, a_tr: pd.DataFrame, p99_factor=1.6, min_med_factor=3.0):
    """
    Clip predictions per sector to avoid unrealistic spikes.
    Cap per sector = max(p99 * p99_factor, median * min_med_factor).
    """
    caps = {}
    for s in pred_matrix.columns:
        hist = a_tr[s].values
        p99 = np.percentile(hist, 99) if np.any(hist > 0) else 0.0
        med_vals = hist[hist > 0]
        med = np.median(med_vals) if med_vals.size > 0 else 0.0
        cap = max(p99 * p99_factor, med * min_med_factor, 0.0)
        caps[s] = cap if np.isfinite(cap) else 0.0
        pred_matrix[s] = np.minimum(pred_matrix[s].values, caps[s])
    return pred_matrix, caps

# ====================== Build submission helper ======================
def build_submission_from_matrix(pred_matrix: pd.DataFrame, test_df: pd.DataFrame, month_codes: Dict[str, int]):
    test_pd = test_df.copy()
    test_pd['month_text'] = test_pd['id'].str.split('_').str[0]
    test_pd['sector'] = test_pd['id'].str.split('_').str[1]
    test_pd = add_time_and_sector_fields(test_pd, month_codes)

    pm = pred_matrix.copy()
    if pm.index.name != 'time':
        pm.index.name = 'time'
    lookup = pm.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    if 'time' not in lookup.columns and 'level_0' in lookup.columns:
        lookup.rename(columns={'level_0': 'time'}, inplace=True)

    merged = test_pd.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    submission = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'}).copy()
    submission['new_house_transaction_amount'].replace([np.inf, -np.inf], np.nan, inplace=True)
    submission['new_house_transaction_amount'].fillna(0.0, inplace=True)
    return submission

# ====================== MAIN PIPELINE ======================
def run_notebook():
    print(Fore.CYAN + Style.BRIGHT + "="*70)
    print("REAL ESTATE DEMAND PREDICTION - NOTEBOOK E2E (Multi-Horizon Blend+CSI PCA+Clipping) [Fixed]")
    print("="*70 + Style.RESET_ALL)

    data = load_data()
    df_master, max_time_train, test = create_master_dataframe(data)

    # Features
    df_proc = create_time_features(df_master)
    df_proc = create_lag_rolling_features(df_proc)
    df_proc = create_interaction_features(df_proc)

    # Ensure LGBM categoricals upfront
    cat_cols_common = [c for c in ['sector_id', 'year', 'month_num', 'quarter'] if c in df_proc.columns]
    df_proc = set_lgbm_categoricals(df_proc, cat_cols_common)

    # Train rows
    train_full = df_proc[df_proc['time'] <= max_time_train].copy()
    if 'amount_new_house_transactions_lag12' in train_full.columns:
        train_full = train_full.dropna(subset=['amount_new_house_transactions_lag12'])

    # Select features
    exclude = {'amount_new_house_transactions', 'time', 'id'}  # 'season' not created; exclude guard simplified
    feature_cols = [c for c in train_full.columns if c not in exclude]

    # Clean numeric only (do not break categoricals)
    num_cols_global = [c for c in feature_cols if df_proc[c].dtype.name != 'category']
    df_proc[num_cols_global] = df_proc[num_cols_global].replace([np.inf, -np.inf], np.nan).fillna(-2)
    train_full[num_cols_global] = train_full[num_cols_global].replace([np.inf, -np.inf], np.nan).fillna(-2)
    # ensure no NaN in categoricals
    for c in cat_cols_common:
        if c in df_proc.columns and df_proc[c].isna().any():
            df_proc[c] = df_proc[c].cat.add_categories(['__NA__']).fillna('__NA__')
        if c in train_full.columns and train_full[c].isna().any():
            train_full[c] = train_full[c].cat.add_categories(['__NA__']).fillna('__NA__')

    # One-step backtesting (for reporting only)
    print(Fore.BLUE + Style.BRIGHT + "\n--- TIME-BASED BACKTESTING (one-step report) ---" + Style.RESET_ALL)
    lgb_params, lgb_cv_metrics, lgb_val_preds, y_val_all = lgbm_cv_train(train_full, feature_cols)
    for i, m in enumerate(lgb_cv_metrics, 1):
        print(f"LGB Fold{i}: {m}")

    if HAS_CAT:
        _, cat_cv_metrics, cat_val_preds, _ = cat_cv_train(train_full, feature_cols)
        for i, m in enumerate(cat_cv_metrics, 1):
            print(f"CAT Fold{i}: {m}")

    if HAS_XGB:
        _, xgb_cv_metrics, xgb_val_preds, _ = xgb_cv_train(train_full, feature_cols)
        for i, m in enumerate(xgb_cv_metrics, 1):
            print(f"XGB Fold{i}: {m}")

    # Determine test times
    month_codes = build_month_codes()
    test_pd = test.copy()
    test_pd['month_text'] = test_pd['id'].str.split('_').str[0]
    test_pd['sector'] = test_pd['id'].str.split('_').str[1]
    test_pd = add_time_and_sector_fields(test_pd, month_codes)
    test_times = np.sort(test_pd['time'].unique())

    # Horizon prediction: direct multi-horizon ML (per-horizon blended)
    if USE_MULTI_HORIZON_ML:
        print(Fore.BLUE + Style.BRIGHT + "\n--- DIRECT MULTI-HORIZON ML (per-horizon blend) ---" + Style.RESET_ALL)
        pred_matrix = train_predict_multi_horizon_blended(
            df_proc,
            feature_cols,
            max_time_train,
            test_times
        )
        # Optional: December bump based on training stats
        if ENABLE_DECEMBER_BUMP:
            try:
                a_tr = build_amount_matrix(data['train_nht'].copy(), month_codes)
                sector_to_mult = compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.82, clip_high=1.45)
                dec_rows = [t for t in pred_matrix.index.values if (t % 12) == 11]
                if dec_rows:
                    for s in pred_matrix.columns:
                        pred_matrix.loc[dec_rows, s] *= sector_to_mult.get(s, 1.0)
            except Exception as e:
                print("December bump skipped:", e)
        # Robust clipping per sector
        try:
            a_tr = build_amount_matrix(data['train_nht'].copy(), month_codes)
            pred_matrix, caps = robust_clip_by_sector(pred_matrix, a_tr, CLIP_P99_FACTOR, CLIP_MIN_MEDIAN_FACTOR)
        except Exception as e:
            print("Clipping skipped:", e)
    else:
        raise NotImplementedError("Set USE_MULTI_HORIZON_ML=True to use per-horizon blended ML.")

    # Build submission
    submission = build_submission_from_matrix(pred_matrix, test, month_codes)
    out_path = 'submission_notebook_e2e.csv'
    submission.to_csv(out_path, index=False)

    print(Fore.GREEN + Style.BRIGHT + "\n" + "="*70)
    print(f"Saved: {out_path}")
    print(f"Total predictions: {len(submission)} | Non-zero: {(submission['new_house_transaction_amount'] > 0).sum()} | Mean: {submission['new_house_transaction_amount'].mean():.2f}")
    print("="*70 + Style.RESET_ALL)

    if ENABLE_PLOTS:
        try:
            s = submission['new_house_transaction_amount'].astype(float)
            s = s.replace([np.inf,-np.inf], np.nan).dropna()
            plt.figure(figsize=(10,4))
            sns.kdeplot(s, fill=True)
            plt.title('Submission Prediction Distribution (Multi-horizon blended + Clipped)')
            plt.show()
        except Exception as e:
            print("Plot error:", e)

# Execute the pipeline in notebook
run_notebook()

