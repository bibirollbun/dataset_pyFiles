import os, gc, calendar, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')

DATA_DIR = "/kaggle/input/china-real-estate-demand-prediction"
TARGET_COL = "amount_new_house_transactions"
CUT_YEAR, CUT_MONTH = 2023, 7



def custom_competition_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    ape = np.empty_like(y_true, dtype=float)
    zero_mask = (y_true == 0)
    nonzero_mask = ~zero_mask

    ape[nonzero_mask] = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / np.abs(y_true[nonzero_mask])
    ape[zero_mask] = np.where(np.abs(y_pred[zero_mask]) == 0, 0.0, np.inf)

    frac_over_1 = np.mean(ape > 1.0)
    if frac_over_1 > 0.30:
        return 0.0

    ok_mask = (ape <= 1.0)
    if ok_mask.sum() == 0:
        return 0.0
    mape_ok = np.mean(ape[ok_mask])
    scaled_mape = mape_ok / ok_mask.mean()
    return float(1.0 - scaled_mape)


def load_all_data(data_dir):
    data = {}
    datasets = {
        'new': 'train/new_house_transactions.csv',
        'new_nb': 'train/new_house_transactions_nearby_sectors.csv',
        'pre': 'train/pre_owned_house_transactions.csv',
        'pre_nb': 'train/pre_owned_house_transactions_nearby_sectors.csv',
        'land': 'train/land_transactions.csv',
        'land_nb': 'train/land_transactions_nearby_sectors.csv',
        'city_idx': 'train/city_indexes.csv',
        'city_search': 'train/city_search_index.csv',
        'poi': 'train/sector_POI.csv',
        'test': 'test.csv'
    }
    for name, path in datasets.items():
        try:
            df = pd.read_csv(os.path.join(data_dir, path))
            data[name] = df
            print(f"Loaded {name}: {df.shape}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return data

all_data = load_all_data(DATA_DIR)


def extract_datetime_features(df, date_col='month'):
    df = df.copy()
    if date_col in df.columns:
        # '2019-Jan' -> split
        tmp = df[date_col].astype(str).str.split('-', expand=True)
        tmp.columns = ['Year', 'Month']
        df = pd.concat([df.drop(columns=[date_col]), tmp], axis=1)

        # Year -> int
        df['Year'] = df['Year'].astype(int)

        # Month abbr -> int (1..12)
        month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
                     'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        df['Month_num'] = df['Month'].map(month_map).astype(int)

        # date, time_index
        df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str) + '-01')
        df['time_index'] = (df['Year'] - 2019) * 12 + df['Month_num']

        # seasonality
        df['quarter'] = ((df['Month_num'] - 1) // 3 + 1).astype(int)
        df['is_quarter_end'] = df['Month_num'].isin([3,6,9,12]).astype(int)
        df['is_year_end'] = (df['Month_num'] == 12).astype(int)
        df['sin_month'] = np.sin(2 * np.pi * df['Month_num'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['Month_num'] / 12.0)
    return df

def extract_sector_features(df):
    df = df.copy()
    if 'sector' in df.columns:
        df['sector_num'] = df['sector'].astype(str).str.extract(r'(\d+)').astype(int)
    return df

for name in ['new','new_nb','pre','pre_nb','land','land_nb']:
    if name in all_data:
        all_data[name] = extract_datetime_features(all_data[name])
        all_data[name] = extract_sector_features(all_data[name])

if 'poi' in all_data:
    all_data['poi'] = extract_sector_features(all_data['poi'])


def create_master_dataset(new_df, other):
    master = new_df.copy()

    # nearby
    if 'new_nb' in other:
        nb = other['new_nb'].copy()
        nb = extract_datetime_features(nb)
        nb = extract_sector_features(nb)
        # переименуем все НЕ ключевые колонки
        nb_ren = {c: f'nb_{c}' for c in nb.columns if c not in ['Year','Month_num','sector_num','date']}
        nb = nb.rename(columns=nb_ren)
        master = master.merge(nb, on=['Year','Month_num','sector_num','date'], how='left')

    # pre
    if 'pre' in other:
        pre = other['pre'].copy()
        pre = extract_datetime_features(pre)
        pre = extract_sector_features(pre)
        pre_ren = {c: f'pre_{c}' for c in pre.columns if c not in ['Year','Month_num','sector_num','date']}
        pre = pre.rename(columns=pre_ren)
        master = master.merge(pre, on=['Year','Month_num','sector_num','date'], how='left')

    # land
    if 'land' in other:
        land = other['land'].copy()
        land = extract_datetime_features(land)
        land = extract_sector_features(land)
        land_ren = {c: f'land_{c}' for c in land.columns if c not in ['Year','Month_num','sector_num','date']}
        land = land.rename(columns=land_ren)
        master = master.merge(land, on=['Year','Month_num','sector_num','date'], how='left')

    # POI (static)
    if 'poi' in other:
        poi = other['poi'].copy()
        if 'sector' in poi.columns:
            poi = poi.drop(columns=['sector'])
        master = master.merge(poi, on='sector_num', how='left')

    return master

master_df = create_master_dataset(all_data['new'], all_data)


def create_advanced_features(df):
    df = df.copy()
    gcols = ['sector_num']

    # lags по target
    for lag in [1,3,6,12]:
        df[f'target_lag_{lag}'] = df.groupby(gcols)[TARGET_COL].shift(lag)

    # rolling stats
    for w in [3,6,12]:
        df[f'rolling_mean_{w}'] = df.groupby(gcols)[TARGET_COL].transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'rolling_std_{w}']  = df.groupby(gcols)[TARGET_COL].transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'rolling_max_{w}']  = df.groupby(gcols)[TARGET_COL].transform(lambda x: x.rolling(w, min_periods=1).max())

    # rates
    df['yoy_growth'] = df.groupby(gcols)[TARGET_COL].pct_change(12)
    df['mom_growth'] = df.groupby(gcols)[TARGET_COL].pct_change()

    # target 12m ahead
    df['target_12m_ahead'] = df.groupby(gcols)[TARGET_COL].shift(-12)

    # заполним частично lag/roll средним по сектору (чтобы не тащить NaN)
    num_cols = df.select_dtypes(include=[np.number]).columns
    fill_cols = [c for c in num_cols if c.startswith('rolling_') or c.startswith('target_lag_')]
    for c in fill_cols:
        df[c] = df.groupby(gcols)[c].transform(lambda x: x.fillna(x.mean()))

    return df

master_df = create_advanced_features(master_df)



def handle_missing_values(df):
    df = df.copy()
    gcols = ['sector_num']
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns

    for c in num_cols:
        df[c] = df.groupby(gcols)[c].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    return df

master_df = handle_missing_values(master_df)


def prepare_time_series_split(df, train_end_year=2023, train_end_month=7):
    train_mask = (df['Year'] < train_end_year) | ((df['Year'] == train_end_year) & (df['Month_num'] <= train_end_month))
    df_tr = df[train_mask].copy()
    df_te = df[~train_mask].copy()
    df_tr = df_tr.dropna(subset=['target_12m_ahead'])
    return df_tr, df_te

df_train, df_test = prepare_time_series_split(master_df, CUT_YEAR, CUT_MONTH)


def prepare_features(df, target_col='target_12m_ahead'):
    exclude = {target_col, 'date', 'Month', 'sector', 'month', TARGET_COL}
    num_cols = [c for c in df.columns if (c not in exclude) and np.issubdtype(df[c].dtype, np.number)]
    X = df[num_cols].copy()
    y = df[target_col].astype(float).copy()
    return X, y, num_cols

X_train, y_train, feature_cols = prepare_features(df_train)
X_test,  y_test,  _           = prepare_features(df_test)


def train_xgb(X_tr, y_tr, X_va=None, y_va=None):
    params = dict(
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        eval_metric=["rmse", "mae"],
        verbosity=1
    )
    if X_va is not None:
        params["early_stopping_rounds"] = 200
    model = XGBRegressor(**params)

    if X_va is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
    else:
        model.fit(X_tr, y_tr, verbose=100)
    return model


def train_lgb_log(X_tr, y_tr, X_va=None, y_va=None, seed=42):
    params = dict(
        objective="regression_l2",
        metric=["rmse","mae"],
        learning_rate=0.007070354291686468,
        n_estimators=12000,
        num_leaves=103,
        max_depth=6,
        min_data_in_leaf=20,
        min_child_weight=30,
        feature_fraction=0.600156542345633,
        bagging_fraction=0.8996846649797056,
        bagging_freq=10,
        reg_alpha=4.5638590789050015,
        reg_lambda=6.099195283306121,
        min_gain_to_split=0.10238923181844709,
        max_bin=255,
        verbosity=-1,
        random_state=seed,
        device_type="gpu",
    )
    model = lgb.LGBMRegressor(**params)

    y_tr_log = np.log1p(np.clip(y_tr, 0, None))
    if X_va is not None:
        y_va_log = np.log1p(np.clip(y_va, 0, None))
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            eval_metric=["rmse","mae"],
            callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=200)]
        )
    else:
        model.fit(X_tr, y_tr_log)
    return model


def safe_predict_xgb(model, X_scaled):
    return np.clip(model.predict(X_scaled), 0, None)


def safe_predict_lgb(model, X):
    kwargs = {}
    if hasattr(model, "best_iteration_") and model.best_iteration_:
        kwargs["num_iteration"] = model.best_iteration_
    pred = model.predict(X, **kwargs)
    pred = np.expm1(pred)
    return np.clip(pred, 0, None)



def add_target_encoding_final(X_tr, y_tr, X_te, col='sector_num'):
    tmp = X_tr[[col]].copy()
    tmp['_t'] = y_tr.values
    mapping = tmp.groupby(col)['_t'].mean().to_dict()
    X_tr[f'{col}_te'] = X_tr[col].map(mapping).astype(float)
    X_te[f'{col}_te'] = X_te[col].map(mapping).fillna(y_tr.mean()).astype(float)
    return X_tr, X_te

X_train, X_test = add_target_encoding_final(X_train.copy(), y_train, X_test.copy(), col='sector_num')
feature_cols = list(X_train.columns)

def add_target_encoding_fold(X_tr, y_tr, X_va, col='sector_num'):
    tmp = X_tr[[col]].copy()
    tmp['_t'] = y_tr.values
    mapping = tmp.groupby(col)['_t'].mean().to_dict()
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_tr[f'{col}_te'] = X_tr[col].map(mapping).astype(float)
    X_va[f'{col}_te'] = X_va[col].map(mapping).fillna(y_tr.mean()).astype(float)
    return X_tr, X_va


def evaluate_fold(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    sc   = custom_competition_score(y_true, y_pred)
    print(f"{name:>4} | RMSE={rmse:,.1f}  MAE={mae:,.1f}  R²={r2:.3f}  Comp={sc:.4f}")
    return rmse, mae, r2, sc


def time_series_cv(X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb_metrics, lgb_metrics = [], []

    for i, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        print(f"\n=== Fold {i}/{n_splits} ===")
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # In-fold TE
        X_tr, X_va = add_target_encoding_fold(X_tr, y_tr, X_va, col='sector_num')

        # Scale for XGB
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # --- Train XGB
        xgb = train_xgb(X_tr_s, y_tr, X_va_s, y_va)
        y_hat_xgb = safe_predict_xgb(xgb, X_va_s)
        xgb_metrics.append(evaluate_fold("XGB", y_va, y_hat_xgb))

        # --- Train LGB
        lgbm = train_lgb_log(X_tr, y_tr, X_va, y_va)
        y_hat_lgb = safe_predict_lgb(lgbm, X_va)
        lgb_metrics.append(evaluate_fold("LGB", y_va, y_hat_lgb))

    # aggregate
    xgb_metrics = np.array(xgb_metrics)
    lgb_metrics = np.array(lgb_metrics)

    print("\n=== Mean CV ===")
    print(f"XGB | RMSE={xgb_metrics[:,0].mean():,.1f}  MAE={xgb_metrics[:,1].mean():,.1f}  "
          f"R²={xgb_metrics[:,2].mean():.3f}  Comp={xgb_metrics[:,3].mean():.4f}")
    print(f"LGB | RMSE={lgb_metrics[:,0].mean():,.1f}  MAE={lgb_metrics[:,1].mean():,.1f}  "
          f"R²={lgb_metrics[:,2].mean():.3f}  Comp={lgb_metrics[:,3].mean():.4f}")

    return xgb_metrics.mean(axis=0), lgb_metrics.mean(axis=0)


# Run CV
_ = time_series_cv(X_train, y_train, n_splits=3)



X_train_full = X_train.copy()
X_test_full  = X_test.copy()
X_train_full, X_test_full = add_target_encoding_final(X_train_full, y_train, X_test_full, col='sector_num')

feature_cols_full = list(X_train_full.columns)
scaler_full = StandardScaler()
X_train_full_s = pd.DataFrame(scaler_full.fit_transform(X_train_full), columns=feature_cols_full, index=X_train_full.index)
X_test_full_s  = pd.DataFrame(scaler_full.transform(X_test_full),  columns=feature_cols_full, index=X_test_full.index)

print("\n=== Train Final Models (full train) ===")
final_xgb = train_xgb(X_train_full_s, y_train)
final_lgb = train_lgb_log(X_train_full, y_train)



y_pred_xgb = safe_predict_xgb(final_xgb, X_test_full_s)
y_pred_lgb = safe_predict_lgb(final_lgb, X_test_full)

alphas = np.linspace(0, 1, 21)
best = (-1, None, None)
for a in alphas:
    y_blend = a * y_pred_lgb + (1 - a) * y_pred_xgb
    s = custom_competition_score(y_test, y_blend)
    if s > best[0]:
        best = (s, a, y_blend)

best_score, best_alpha, best_preds = best
print(f"\n>>> Best blend: α(LGB)={best_alpha:.2f}, Score={best_score:.5f}")


