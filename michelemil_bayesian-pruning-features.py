!pip install xgboost Scikit-Optimize deap holidays lightgbm catboost optuna scikit-learn pmdarima 


import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import holidays
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# -----------------------
# CONFIG
# -----------------------
BASE_PATH = '/kaggle/input/china-real-estate-demand-prediction/train'
TEST_PATH = '/kaggle/input/china-real-estate-demand-prediction/test.csv'
POI_PATH = f'{BASE_PATH}/sector_POI.csv'
EXTRA_PATH_POLICY = "/kaggle/input/real-estate-global-cn-policy/local_policy_stance.csv"
EXTRA_PATH_MACRO = "/kaggle/input/cn-cpi-hpi-usdrates-boundsrates/cn-china-economic-data-extended.csv"
CITY_INDEXES_PATH = f'{BASE_PATH}/city_indexes.csv'
EXTRA_PATH_STATS = "/kaggle/input/real-estate-global-cn-policy/city_data_policy_mapping.csv"
EXTRA_PATH_CITY_POLICY = "/kaggle/input/real-estate-global-cn-policy/city-restric-cn.csv"
SECTOR_42_PATH = "/kaggle/input/sector-42/submission (32).csv"

INPUT_FEATURES = [
    'sector',
    'amount_new_house_transactions_accel',
    'amount_new_house_transactions_lag_1', 'amount_new_house_transactions_lag_12',
    'amount_new_house_transactions_lag_3', 'amount_new_house_transactions_lag_6',
    'amount_new_house_transactions_nearby_sectors_lag_1',
    'amount_new_house_transactions_nearby_sectors_lag_12',
    'amount_new_house_transactions_nearby_sectors_lag_3',
    'amount_new_house_transactions_nearby_sectors_lag_6',
    'amount_new_house_transactions_nearby_sectors_roll_mean_12',
    'amount_new_house_transactions_nearby_sectors_roll_median_3',
    'amount_new_house_transactions_nearby_sectors_roll_std_6',
    'area_new_house_available_for_sale_lag_1',
    'area_new_house_available_for_sale_lag_12',
    'area_new_house_available_for_sale_lag_3',
    'area_new_house_available_for_sale_lag_6',
    'area_new_house_available_for_sale_roll_mean_6',
    'area_new_house_available_for_sale_roll_median_3',
    'area_new_house_available_for_sale_roll_median_6',
    'area_new_house_available_for_sale_roll_std_12',
    'area_new_house_available_for_sale_roll_std_3',
    'area_new_house_transactions_lag_1', 'area_new_house_transactions_lag_12',
    'area_new_house_transactions_lag_3', 'area_new_house_transactions_lag_6',
    'area_new_house_transactions_roll_mean_12', 'area_new_house_transactions_roll_std_6',
    'area_pre_owned_house_transactions_lag_1', 'area_pre_owned_house_transactions_lag_12',
    'area_pre_owned_house_transactions_lag_3', 'area_pre_owned_house_transactions_lag_6',
    'area_pre_owned_house_transactions_roll_mean_12',
    'area_pre_owned_house_transactions_roll_median_6',
    'cpi_lag_3', 'cpi_lag_6', 'cpi_roll_mean_3', 'cpi_roll_std_12', 'is_holiday_month',
    'month_cos', 'month_num', 'month_sin',
    'num_new_house_available_for_sale_lag_1', 'num_new_house_available_for_sale_lag_12',
    'num_new_house_available_for_sale_lag_3', 'num_new_house_available_for_sale_lag_6',
    'num_new_house_available_for_sale_roll_std_12', 'num_new_house_transactions_lag_1',
    'num_new_house_transactions_lag_12', 'num_new_house_transactions_lag_3',
    'num_new_house_transactions_lag_6', 'num_new_house_transactions_roll_mean_12',
    'num_new_house_transactions_roll_median_6', 'num_pre_owned_house_transactions_lag_1',
    'num_pre_owned_house_transactions_lag_12', 'num_pre_owned_house_transactions_lag_3',
    'num_pre_owned_house_transactions_lag_6', 'num_pre_owned_house_transactions_roll_mean_12',
    'period_new_house_sell_through_lag_1', 'period_new_house_sell_through_lag_12',
    'price_new_house_transactions_lag_1', 'price_new_house_transactions_lag_12',
    'price_new_house_transactions_lag_3', 'price_new_house_transactions_lag_6',
    'price_new_house_transactions_roll_mean_3',
    'price_new_house_transactions_roll_median_12',
    'price_new_house_transactions_roll_median_6',
    'price_new_house_transactions_roll_std_12',
    'price_new_house_transactions_roll_std_3', 'price_new_house_transactions_roll_std_6',
    'price_pre_owned_house_transactions_lag_1',
    'price_pre_owned_house_transactions_lag_12',
    'price_pre_owned_house_transactions_lag_3',
    'price_pre_owned_house_transactions_lag_6',
    'price_pre_owned_house_transactions_roll_median_6',
    'price_pre_owned_house_transactions_roll_std_12', 'search_volume_lag_1',
    'search_volume_lag_12', 'search_volume_lag_3', 'search_volume_lag_6',
    'total_price_per_unit_new_house_transactions_lag_1',
    'total_price_per_unit_new_house_transactions_lag_12',
    'total_price_per_unit_new_house_transactions_lag_3',
    'total_price_per_unit_new_house_transactions_lag_6', 'treasury_rate_lag_1',
    'treasury_rate_lag_12', 'treasury_rate_roll_std_12', 'treasury_rate_roll_std_3',
    'usd_cny_exchange_rate_lag_1', 'usd_cny_exchange_rate_lag_12',
    'usd_cny_exchange_rate_lag_6', 'usd_cny_exchange_rate_lag_3',
    'usd_cny_exchange_rate_roll_std_6', 'year',
]

NUMERICAL_FEATURES = [f for f in INPUT_FEATURES if f != 'sector']
TARGET = 'amount_new_house_transactions'

def safe_merge(df1, df2, on_cols, suffix=''):
    if not isinstance(on_cols, list):
        on_cols = [on_cols]
    return pd.merge(df1, df2, on=on_cols, how='left', suffixes=('', suffix))



# -----------------------
# Helper: Holiday Features
# -----------------------
def add_monthly_holiday_features(df, date_col='month'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    min_year = df[date_col].dt.year.min()
    max_year = df[date_col].dt.year.max() + 1
    cn_holidays = holidays.China(years=range(min_year, max_year + 1))
    holiday_dates = pd.to_datetime(sorted(cn_holidays.keys()))
    holiday_df = pd.DataFrame({'holiday_date': holiday_dates})
    holiday_df['holiday_month'] = holiday_df['holiday_date'].dt.to_period('M')
    holiday_months = holiday_df['holiday_month'].drop_duplicates()
    df['ym'] = df[date_col].dt.to_period('M')
    df['is_holiday_month'] = df['ym'].isin(holiday_months).astype(int)
    df = df.drop(columns=['ym'])
    return df

# -----------------------
# Load & Merge TRAIN Data
# -----------------------
def load_and_merge_data():
    print("? Loading training data...")
    dfs = {}
    file_list = [
        'city_indexes.csv', 'city_search_index.csv', 'land_transactions.csv',
        'land_transactions_nearby_sectors.csv', 'new_house_transactions.csv',
        'new_house_transactions_nearby_sectors.csv', 'pre_owned_house_transactions.csv',
        'pre_owned_house_transactions_nearby_sectors.csv', 'sector_POI.csv'
    ]
    for file in file_list:
        try:
            path = f'{BASE_PATH}/{file}'
            df = pd.read_csv(path)
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            dfs[file.replace('.csv', '')] = df
        except Exception:
            pass

    main_df = dfs['new_house_transactions'].copy()
    main_df = main_df.rename(columns={'month': 'date'})
    main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce')
    main_df['year'] = main_df['date'].dt.year.astype('Int64')
    main_df['month'] = main_df['date'].dt.month.astype('Int64')
    main_df['ym'] = main_df['date'].dt.to_period('M')
    main_df = main_df.dropna(subset=['date']).reset_index(drop=True)
    main_df['sector'] = main_df['sector'].astype(str).str.replace('sector ', '').str.strip()
    main_df['merge_key'] = main_df['date'].dt.strftime('%Y-%m') + '_' + main_df['sector']


    # Merge nearby sectors
    if 'new_house_transactions_nearby_sectors' in dfs:
        df = dfs['new_house_transactions_nearby_sectors'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df = df.rename(columns={'month': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
        df['merge_key'] = df['date'].dt.strftime('%Y-%m') + '_' + df['sector']
        df = df.drop(['date', 'sector'], axis=1, errors='ignore')
        main_df = safe_merge(main_df, df, 'merge_key')

    # Merge land and pre-owned
    for key in ['land_transactions', 'pre_owned_house_transactions']:
        if key in dfs:
            df = dfs[key].copy()
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            df = df.rename(columns={'month': 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
            df['merge_key'] = df['date'].dt.strftime('%Y-%m') + '_' + df['sector']
            cols = [c for c in df.columns if c not in ['date', 'sector', 'merge_key']]
            suffix = '_land' if 'land' in key else '_pre'
            main_df = safe_merge(main_df, df[['merge_key'] + cols], 'merge_key', suffix)

    # Merge POI
    if 'sector_poi' in dfs:
        df = dfs['sector_poi'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
        df = df.drop_duplicates(subset=['sector'], keep='last')
        df = df.drop('month', axis=1, errors='ignore')
        main_df = safe_merge(main_df, df, 'sector')

    # Merge city search index
    if 'city_search_index' in dfs:
        df = dfs['city_search_index'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df = df.rename(columns={'month': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['ym'] = df['date'].dt.to_period('M')
        if 'search_volume' in df.columns:
            agg = df.groupby('ym')['search_volume'].sum().reset_index()
            main_df = safe_merge(main_df, agg, 'ym', suffix='_search')

    # Merge national stats
    if os.path.exists(EXTRA_PATH_STATS):
        stats_df = pd.read_csv(EXTRA_PATH_STATS)
        stats_df.columns = stats_df.columns.str.strip().str.replace(' ', '_').str.lower()
        if 'year' not in stats_df.columns or 'month' not in stats_df.columns:
            raise ValueError("Expected 'year' and 'month' in EXTRA_PATH_STATS.")
        stats_df['date'] = pd.to_datetime(
            stats_df['year'].astype(int).astype(str) + '-' + stats_df['month'].astype(int).astype(str).str.zfill(2),
            errors='coerce'
        )
        stats_df = stats_df.rename(columns={
            'median_price': 'national_median_price',
            'sale_volume': 'national_sale_volume',
            'price_index_yoy': 'national_price_index_yoy',
            'inventory_months': 'national_inventory_months'
        })
        stats_df['ym'] = stats_df['date'].dt.to_period('M')
        main_df['ym'] = main_df['date'].dt.to_period('M')
        main_df = safe_merge(main_df, stats_df.drop(columns=['date', 'year', 'month']), on_cols='ym', suffix='_stats')
        main_df = main_df.drop(columns=['ym'], errors='ignore')

    # Merge macro
    if os.path.exists(EXTRA_PATH_MACRO):
        extra_df = pd.read_csv(EXTRA_PATH_MACRO)
        extra_df.columns = extra_df.columns.str.strip().str.replace(' ', '_').str.lower()
        extra_df['date'] = pd.to_datetime(
            extra_df['year'].astype(str) + '-' + extra_df['month'].astype(str).str.zfill(2),
            errors='coerce'
        )
        extra_df['ym'] = extra_df['date'].dt.to_period('M')
        main_df['ym'] = main_df['date'].dt.to_period('M')
        if {'cpi', 'hpi', 'lpr_5y'} <= set(extra_df.columns):
            extra_df['cpi_norm'] = extra_df['cpi'] / 100.0
            extra_df['hpi_norm'] = extra_df['hpi'] / 111.5332
            extra_df['mortgage_factor'] = 1 + (extra_df['lpr_5y'] / 100.0)
            extra_df['buying_power_index'] = 1.0 / (
                extra_df['hpi_norm'] * extra_df['mortgage_factor'] * extra_df['cpi_norm']
            )
            base_bp = extra_df.loc[(extra_df['year'] == 2016) & (extra_df['month'] == 1), 'buying_power_index']
            if not base_bp.empty:
                extra_df['buying_power_index'] /= base_bp.iloc[0]
            else:
                median_bp = extra_df['buying_power_index'].median()
                extra_df['buying_power_index'] /= median_bp if median_bp > 0 else 1.0
        cols_to_keep = ['ym', 'cpi', 'hpi', 'treasury_rate', 'usd_cny_exchange_rate', 'm2_growth_yoy', 'lpr_1y', 'lpr_5y', 'buying_power_index']
        extra_df = extra_df[[c for c in cols_to_keep if c in extra_df.columns]]
        main_df = safe_merge(main_df, extra_df, 'ym', suffix='_macro')
        main_df = main_df.drop(columns=['ym'], errors='ignore')
        main_df.rename(columns={
            'cpi_macro': 'cpi',
            'usd_cny_exchange_rate_macro': 'usd_cny_exchange_rate',
            'm2_growth_yoy_macro': 'm2_growth_yoy',
            'lpr_5y_macro': 'lpr_5y',
            'treasury_rate_macro': 'treasury_rate'
        }, inplace=True)

    # Merge city indexes
    if os.path.exists(CITY_INDEXES_PATH):
        df = pd.read_csv(CITY_INDEXES_PATH)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        if 'city_indicator_data_year' in df.columns:
            df.rename(columns={'city_indicator_data_year': 'year'}, inplace=True)
            main_df['year'] = main_df['date'].dt.year
            main_df = safe_merge(main_df, df.drop(columns=['city_search_index'], errors='ignore'), 'year')

    # Merge city policy
    if os.path.exists(EXTRA_PATH_CITY_POLICY):
        policy_df = pd.read_csv(EXTRA_PATH_CITY_POLICY)
        policy_df['date_time'] = pd.to_datetime(
            policy_df['year'].astype(str) + '-' + policy_df['month'].astype(str).str.zfill(2),
            errors='coerce'
        )
        policy_df = policy_df.dropna(subset=['date_time'])
        policy_df['sector'] = policy_df['sector'].astype(str).str.replace('sector ', '').str.strip()
        policy_features = ['hukou_policy_level', 'policy_group', 'policy_stance']
        policy_df = policy_df.sort_values(['sector', 'date_time'])
        policy_df_resampled = policy_df.groupby('sector').apply(
            lambda g: g.set_index('date_time')[policy_features].resample('MS').ffill()
        ).reset_index()
        policy_df_resampled = policy_df_resampled.rename(columns={'date_time': 'month'})
        policy_cols = ['month', 'sector'] + policy_features
        policy_df = policy_df_resampled[policy_cols].drop_duplicates(subset=['month', 'sector'])
        main_df = safe_merge(main_df, policy_df, ['month', 'sector'])

    CONCURRENT = [
        'search_volume', 'national_median_price', 'usd_cny_exchange_rate', 'cpi',
        'm2_growth_yoy', 'lpr_5y', 'purchase_restriction_level', 'min_down_payment_ratio_first_home'
    ]
    main_df = main_df.sort_values(['sector', 'month']).reset_index(drop=True)
    cols_to_shift = [c for c in CONCURRENT if c in main_df.columns]
    if cols_to_shift:
        main_df[cols_to_shift] = main_df.groupby('sector')[cols_to_shift].shift(1)

    main_df.drop('merge_key', axis=1, errors='ignore', inplace=True)
    main_df = add_monthly_holiday_features(main_df, date_col='month')
    return main_df

# -----------------------
# Feature Engineering
# -----------------------
def feature_engineer(main_df):
    target = 'amount_new_house_transactions'
    LEAKY_TARGETS = [
        'num_new_house_available_for_sale', 'area_new_house_available_for_sale',
        'num_new_house_transactions', 'area_new_house_transactions',
        'price_new_house_transactions', 'total_price_per_unit_new_house_transactions',
        'period_new_house_sell_through', 'num_pre_owned_house_transactions',
        'area_pre_owned_house_transactions', 'price_pre_owned_house_transactions',
        'amount_new_house_transactions_nearby_sectors',
    ]
    LEAKY_TARGETS = [c for c in LEAKY_TARGETS if c in main_df.columns]
    print(f"\nâœ… LEAKY_TARGETS actually used: {LEAKY_TARGETS}")

    main_df['year'] = main_df['date'].dt.year
    main_df['month_num'] = main_df['date'].dt.month
    main_df['month_sin'] = np.sin(2 * np.pi * main_df['month_num'] / 12)
    main_df['month_cos'] = np.cos(2 * np.pi * main_df['month_num'] / 12)
    main_df['day_of_week'] = main_df['date'].dt.dayofweek
    main_df = main_df.sort_values(['sector', 'date']).reset_index(drop=True)

    lags = [1, 3, 6, 12]
    rolling_windows = [3, 6, 12]
    aux_features_to_lag = [target] + LEAKY_TARGETS + [
        'treasury_rate','search_volume', 'national_median_price', 'usd_cny_exchange_rate', 'cpi',
        'm2_growth_yoy', 'lpr_5y', 'purchase_restriction_level', 'min_down_payment_ratio_first_home'
    ]
    aux_features_to_lag = list(set([f for f in aux_features_to_lag if f in main_df.columns]))

    for feat in aux_features_to_lag:
        for lag in lags:
            lag_col = f'{feat}_lag_{lag}'
            main_df[lag_col] = main_df.groupby('sector')[feat].shift(lag)
            med = main_df.groupby('sector')[lag_col].transform('median')
            main_df[lag_col] = main_df[lag_col].fillna(med).fillna(0)

    for feat in aux_features_to_lag:
        for window in rolling_windows:
            rolling = main_df.groupby('sector')[feat].rolling(window=window, min_periods=1, closed='left')
            main_df[f'{feat}_roll_mean_{window}'] = rolling.mean().reset_index(level=0, drop=True)
            main_df[f'{feat}_roll_std_{window}'] = rolling.std().reset_index(level=0, drop=True)
            main_df[f'{feat}_roll_median_{window}'] = rolling.median().reset_index(level=0, drop=True)

    for feat in [target, 'price_new_house_transactions']:
        lag1_col = f'{feat}_lag_1'
        lag3_col = f'{feat}_lag_3'
        lag6_col = f'{feat}_lag_6'
        if all(c in main_df.columns for c in [lag1_col, lag3_col, lag6_col]):
            main_df[f'{feat}_accel'] = (main_df[lag1_col] - main_df[lag3_col]) - (main_df[lag3_col] - main_df[lag6_col])

    available = [f for f in INPUT_FEATURES if f in main_df.columns]
    missing = [f for f in INPUT_FEATURES if f not in main_df.columns]
    if missing:
        print(f"âš ï¸� Missing features: {missing}")

    X_full = main_df[available].copy()
    y_full = main_df[target].values
    y_full_log = np.log1p(y_full)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full = X_full.fillna(0)
    numeric_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[numeric_cols] = X_full[numeric_cols].clip(lower=-1e10, upper=1e10)
    y_full_log = np.nan_to_num(y_full_log, nan=0.0, posinf=0.0, neginf=0.0)
    return X_full, y_full, y_full_log, main_df

# -----------------------
# sMAPE
# -----------------------
def smape(A, F):
    A = np.array(A, dtype=float)
    F = np.array(F, dtype=float)
    return 100.0 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-8))

# -----------------------
# FULL MATRIX FOR RIDGE
# -----------------------
def month_str_to_time(month_str):
    if isinstance(month_str, pd.Timestamp):
        year = month_str.year
        month_name = month_str.strftime('%b')
    elif '-' in month_str:
        year, month = month_str.split('-')
        year = int(year)
        month_name = month
    else:
        parts = month_str.split()
        year = int(parts[0])
        month_name = parts[1]
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    month_num = month_map[month_name]
    return (year - 2019) * 12 + month_num

def create_training_matrix(df):
    matrix = df.pivot_table(
        index='time',
        columns='sector',
        values='amount_new_house_transactions',
        fill_value=0
    )
    expected_sectors = [f'sector {i}' for i in range(1, 97)]
    for sector in expected_sectors:
        if sector not in matrix.columns:
            matrix[sector] = 0
    matrix = matrix.reindex(sorted(matrix.columns, key=lambda x: int(x.split()[-1])), axis=1)
    return matrix

# ========= Horizon-specific model =========
class HorizonSpecificModel:
    def __init__(self, horizon, ridge_alpha=2.0, logistic_C=0.5, use_isotonic=True, verbose=True):
        self.horizon = int(horizon)
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C = float(logistic_C)
        self.use_isotonic = bool(use_isotonic)
        self.verbose = bool(verbose)
        self.scaler = None
        self.zero_model = None
        self.count_model = None
        self.iso_ = None
        self.zero_threshold_ = 0.5
        self.mu_cap_ = 1e12
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.verbose:
            print(f"\nğŸ”§ Train Horizon H={self.horizon}  X={X.shape}  zero-rate={np.mean(y==0):.1%}")
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X)
        self.zero_model = LogisticRegression(C=self.logistic_C, class_weight='balanced', max_iter=500, random_state=42)
        self.zero_model.fit(Xs, (y == 0).astype(int))
        if self.use_isotonic:
            pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
            self.iso_ = IsotonicRegression(out_of_bounds='clip').fit(pi_raw, (y == 0).astype(int))
        else:
            self.iso_ = None
        mask_pos = y > 0
        if mask_pos.sum() > 10:
            y_log = np.log1p(y[mask_pos])
            self.count_model = Ridge(alpha=self.ridge_alpha, solver='svd')
            self.count_model.fit(Xs[mask_pos], y_log)
            self.mu_cap_ = float(np.quantile(y[mask_pos], 0.995))
        else:
            self.count_model = None
            self.mu_cap_ = 1e12
        self._calibrate_threshold(Xs, y)
        self.is_fitted = True
        return self

    def _calibrate_threshold(self, Xs, y):
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        best_thr, best_sc = 0.5, -1.0
        for t in np.linspace(0.05, 0.65, 13):
            yhat = (1.0 - p0) * mu
            yhat = yhat.copy()
            yhat[p0 >= t] = 0.0
            sc = custom_score(y, yhat)['score']
            if sc > best_sc:
                best_sc, best_thr = sc, float(t)
        self.zero_threshold_ = best_thr
        if self.verbose:
            print(f"  tuned thr={self.zero_threshold_:.2f}  score={best_sc:.3f}")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        yhat = (1.0 - p0) * mu
        yhat[p0 >= self.zero_threshold_] = 0.0
        return np.maximum(yhat, 0.0)

    def predict_soft(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        yhat = (1.0 - p0) * mu
        return np.maximum(yhat, 0.0)
        
    def predict_with_p0(self, X):
        """
        Returns (predictions, p0) where p0 is the calibrated probability of zero.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
        yhat = (1.0 - p0) * mu
        yhat[p0 >= self.zero_threshold_] = 0.0
        return np.maximum(yhat, 0.0), p0    

# ========= Multi-horizon trainer/predictor =========
class MultiHorizonForecaster:
    def __init__(self, horizons=12, ridge_alpha=2.0, logistic_C=0.5, use_isotonic=True, verbose=True, max_gpu_history=18):
        self.horizons = list(range(1, horizons + 1))
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C = float(logistic_C)
        self.use_isotonic = bool(use_isotonic)
        self.verbose = bool(verbose)
        self.max_gpu_history = int(max_gpu_history)
        self.models = {}
        self.feature_cols = None

    def fit(self, train_matrix: pd.DataFrame):
        print(f"\nğŸš€ Training {len(self.horizons)} horizon-specific models ...")
        max_train_time = int(train_matrix.index.max())
        tables = {}
        for h in self.horizons:
            tbl = create_training_data_for_horizon(train_matrix, h, max_gpu_history=self.max_gpu_history, max_train_time=max_train_time)
            if tbl.empty:
                raise ValueError(f"Horizon {h} produced no samples; reduce max_gpu_history or check data.")
            tables[h] = tbl
        self.feature_cols = [c for c in tables[1].columns if c not in ['sector','time','target','target_time']]
        if self.verbose:
            print(f"Feature count: {len(self.feature_cols)}  Samples/H: {len(tables[1])}")
        for h in self.horizons:
            dfh = tables[h]
            X = dfh[self.feature_cols].astype(float).values
            y = dfh['target'].astype(float).values
            model = HorizonSpecificModel(horizon=h, ridge_alpha=self.ridge_alpha,
                                         logistic_C=self.logistic_C, use_isotonic=self.use_isotonic,
                                         verbose=self.verbose)
            model.fit(X, y)
            self.models[h] = model
        print(f"âœ… Trained {len(self.models)} models.")
        return self

    def _features_for_start(self, train_matrix, start_time):
        feats = create_features_at_time(train_matrix, start_time, max_gpu_history=self.max_gpu_history)
        if feats.empty:
            raise ValueError(f"No features at time {start_time} (insufficient gpu_history).")
        return feats

    def predict(self, train_matrix: pd.DataFrame, start_time: int):
        if not self.models:
            raise ValueError("Fit the forecaster first.")
        feats_t = self._features_for_start(train_matrix, start_time)
        sectors = feats_t['sector'].values
        preds = {int(start_time + h): None for h in self.horizons}
        for h in self.horizons:
            tgt_time = int(start_time + h)
            dfh = feats_t.copy()
            tgt_month = ((tgt_time - 1) % 12) + 1
            dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
            dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
            dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
            dfh['horizon'] = h / 12.0
            for c in self.feature_cols:
                if c not in dfh.columns:
                    dfh[c] = 0.0
            X = dfh[self.feature_cols].astype(float).values
            preds[tgt_time] = self.models[h].predict(X)
        out = pd.DataFrame(index=sectors, data={t: preds[t] for t in sorted(preds)}, dtype=float)
        out.index.name = 'sector'
        return out

    def predict_with_p0(self, train_matrix: pd.DataFrame, start_time: int):
        if not self.models:
            raise ValueError("Fit the forecaster first.")
        feats_t = self._features_for_start(train_matrix, start_time)
        sectors = feats_t['sector'].values
        preds = {}
        p0s = {}
        for h in self.horizons:
            tgt_time = int(start_time + h)
            dfh = feats_t.copy()
            tgt_month = ((tgt_time - 1) % 12) + 1
            dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
            dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
            dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
            dfh['horizon'] = h / 12.0
            for c in self.feature_cols:
                if c not in dfh.columns:
                    dfh[c] = 0.0
            X = dfh[self.feature_cols].astype(float).values
            y_pred, p0 = self.models[h].predict_with_p0(X)
            preds[tgt_time] = y_pred
            p0s[tgt_time] = p0
        pred_df = pd.DataFrame(preds, index=sectors).sort_index(axis=1)
        p0_df = pd.DataFrame(p0s, index=sectors).sort_index(axis=1)
        return pred_df, p0_df

# --- single-anchor features ---
def create_features_at_time(train_matrix: pd.DataFrame, time_idx: int, max_gpu_history=18):
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    total = df_panel.sum(axis=1)
    total_roll3 = total.rolling(3, min_periods=1).mean().shift(1)
    total_roll6 = total.rolling(6, min_periods=1).mean().shift(1)
    features_list, sector_list = [], []
    for sector in df_panel.columns:
        # Skip non-sector columns (e.g., if accidentally included)
        if not str(sector).startswith('sector '):
            continue
        series = df_panel[sector].dropna()
        gpu_hist = series[series.index <= time_idx]
        if len(gpu_hist) < max_gpu_history:
            continue
        f = {}
        f['sector_id'] = int(str(sector).replace('sector ', '')) / 100.0
        month_t = ((time_idx - 1) % 12) + 1
        f['month_sin'] = np.sin(2*np.pi*month_t/12)
        f['month_cos'] = np.cos(2*np.pi*month_t/12)
        f['quarter'] = ((month_t - 1) // 3 + 1) / 4.0
        for lag in [1, 2, 3, 6, 12]:
            if len(gpu_hist) >= lag:
                val = float(gpu_hist.iloc[-lag])
                f[f'lag_{lag}_log1p'] = np.log1p(val)
                f[f'lag_{lag}_zero'] = int(val == 0)
            else:
                f[f'lag_{lag}_log1p'] = 0.0
                f[f'lag_{lag}_zero'] = 1
        for w in [3, 6, 12]:
            if len(gpu_hist) >= w:
                roll = gpu_hist.tail(w)
                f[f'roll_{w}_mean_log1p'] = np.log1p(roll.mean())
                f[f'roll_{w}_median_log1p'] = np.log1p(roll.median())
                f[f'roll_{w}_zero_rate'] = float((roll == 0).mean())
                f[f'roll_{w}_trend'] = np.tanh(np.polyfit(range(w), roll.values, 1)[0]) if roll.std() > 0 else 0.0
            else:
                f[f'roll_{w}_mean_log1p'] = 0.0
                f[f'roll_{w}_median_log1p'] = 0.0
                f[f'roll_{w}_zero_rate'] = 1.0
                f[f'roll_{w}_trend'] = 0.0
        f['sector_zero_rate'] = float((gpu_hist == 0).mean())
        nz = gpu_hist[gpu_hist > 0]
        if len(nz) > 2:
            f['nonzero_mean_log1p'] = np.log1p(nz.mean())
            f['nonzero_cv'] = np.tanh(nz.std() / (nz.mean() + 1e-6))
        else:
            f['nonzero_mean_log1p'] = 0.0
            f['nonzero_cv'] = 0.0
        mkt3 = float(total_roll3.loc[time_idx]) if time_idx in total_roll3.index else 0.0
        mkt6 = float(total_roll6.loc[time_idx]) if time_idx in total_roll6.index else 0.0
        f['market_activity_log1p'] = np.log1p(mkt3)
        f['market_trend'] = np.tanh((mkt3 - mkt6) / (mkt6 + 1e-6))
        cz = 0
        for v in gpu_hist.tail(6).iloc[::-1]:
            if v == 0: cz += 1
            else: break
        f['consecutive_zeros_norm'] = min(cz / 6.0, 1.0)
        f['seasonal_zero_interaction'] = f['month_sin'] * f['sector_zero_rate']
        features_list.append(f)
        sector_list.append(sector)
    features_df = pd.DataFrame(features_list)
    features_df['sector'] = sector_list
    features_df['time'] = int(time_idx)
    return features_df

def create_training_data_for_horizon(train_matrix: pd.DataFrame, horizon: int, max_gpu_history=18, max_train_time=None):
    print(f"Creating training data for horizon={horizon} ...")
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    min_t = int(df_panel.index.min() + max_gpu_history)
    max_t_possible = int(df_panel.index.max() - horizon)
    max_t = int(min(max_t_possible, max_train_time)) if max_train_time is not None else max_t_possible
    if max_t < min_t:
        raise ValueError(f"No valid times for horizon {horizon}. Increase data or reduce horizon.")
    valid_times = [int(t) for t in df_panel.index if min_t <= t <= max_t]
    frames = []
    for t in valid_times:
        feat_t = create_features_at_time(df_panel, t, max_gpu_history=max_gpu_history)
        if feat_t.empty:
            continue
        tgt_time = t + horizon
        target_month = ((tgt_time - 1) % 12) + 1
        feat_t['target_month_sin'] = np.sin(2*np.pi*target_month/12)
        feat_t['target_month_cos'] = np.cos(2*np.pi*target_month/12)
        feat_t['target_quarter']   = ((target_month - 1) // 3 + 1) / 4.0
        feat_t['horizon'] = horizon / 12.0
        feat_t['target_time'] = int(tgt_time)
        feat_t['target'] = df_panel.loc[tgt_time, feat_t['sector']].values.astype(float)
        frames.append(feat_t)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        print(f"âš ï¸� Horizon {horizon}: produced 0 samples.")
        return result
    print(f"  â†’ {len(result)} rows, {result.filter(regex='^(?!sector$|time$|target$|target_time$)').shape[1]} features")
    print(f"  zero-rate={np.mean(result['target'].values==0):.1%}  mean={result['target'].mean():.2f}")
    return result

# ========= Post-processing =========
def comp_metric(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ape = np.abs((y_pred - y_true) / np.maximum(y_true, eps))
    frac_bad = (ape > 1.0).mean()
    if frac_bad > 0.30:
        return 0.0
    good_mask = (ape <= 1.0)
    good_frac = good_mask.mean()
    if good_frac <= 0.0:
        return 0.0
    scaled_mape = ape[good_mask].mean() / good_frac
    return float(1.0 - scaled_mape)

def tiny_positive_to_zero(y_pred, lag1, zero_streak6_norm, tau=0.5):
    y_pred = np.asarray(y_pred, float).copy()
    zstreak6 = np.round(np.asarray(zero_streak6_norm, float) * 6.0)
    hard_zero = (lag1 == 0) & (zstreak6 >= 4) & (y_pred < tau)
    y_pred[hard_zero] = 0.0
    return y_pred

def _tune_shrink_and_clip(y_true, y_pred_raw,
                          s_grid=np.linspace(0.70, 1.00, 7),
                          q_caps=(0.95, 0.975, 0.99, 0.995)):
    y_true = np.asarray(y_true, float)
    y_raw  = np.asarray(y_pred_raw, float)
    best = (1.0, np.inf, -1.0)
    preds_pos = y_raw[np.isfinite(y_raw)]
    preds_pos = preds_pos[preds_pos > 0]
    caps = [np.inf]
    if preds_pos.size > 20:
        caps.extend([float(np.quantile(preds_pos, q)) for q in q_caps])
    for s in s_grid:
        base = y_raw * s
        for cap in caps:
            y_pp = np.clip(base, 0.0, cap)
            sc = comp_metric(y_true, y_pp)
            if sc > best[2]:
                best = (float(s), float(cap), float(sc))
    return best

def apply_postprocess(y_pred_raw, s, cap):
    return np.clip(np.asarray(y_pred_raw, float) * float(s), 0.0, float(cap))

def ratio_guard(y_pred, last_val, max_ratio=3.0):
    y_pred = np.asarray(y_pred, float).copy()
    last_val = np.asarray(last_val, float)
    rel_cap = np.where(last_val > 0, max_ratio * last_val, np.inf)
    return np.minimum(y_pred, rel_cap)

def custom_score(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0: raise ValueError("empty array")
    if (y_true < 0).any(): raise ValueError("negative y-true")
    if (~np.isfinite(y_pred)).any(): raise ValueError("infinite y-pred")
    ape = np.abs(y_true - y_pred) / np.maximum(y_true, eps)
    good_mask = ape <= 1.0
    good_rate = float(good_mask.mean())
    if good_rate < 0.7:
        return {'score': 0.0, 'good_rate': good_rate, 'str': f"score=0.000  good_rate={good_rate:.3f}"}
    mape_over_good = float(np.mean(ape[good_mask]))
    scaled_mape = mape_over_good / max(good_rate, eps)
    score = 1.0 - scaled_mape
    return {'score': float(score), 'good_rate': good_rate, 'str': f"score={score:.3f}  good_rate={good_rate:.3f}"}

def generate_final_predictions(
    train_matrix,
    horizons=12,
    ridge_alpha=2.0,
    logistic_C=0.5,
    use_isotonic=True,
    max_gpu_history=18,
):
    last_train = int(train_matrix.index.max())
    print("\n" + "="*80)
    print(f"FINAL: train 1..{last_train}  â†’  predict {last_train+1}..{last_train+horizons}")
    print("="*80)
    forecaster = MultiHorizonForecaster(
        horizons=horizons, ridge_alpha=ridge_alpha,
        logistic_C=logistic_C, use_isotonic=use_isotonic,
        verbose=True, max_gpu_history=max_gpu_history
    )
    forecaster.fit(train_matrix)
    tuned_hard = {}
    tuned_soft = {}
    for h in range(1, horizons + 1):
        anchor_t = last_train - h
        if anchor_t not in train_matrix.index:
            tuned_hard[h] = (1.0, np.inf)
            tuned_soft[h] = (1.0, np.inf)
            continue
        feat_anchor = create_features_at_time(train_matrix, time_idx=anchor_t, max_gpu_history=max_gpu_history)
        sectors = feat_anchor['sector'].values
        feat_anchor = feat_anchor.set_index('sector')
        zstreak_norm_vec = feat_anchor['consecutive_zeros_norm'].astype(float).reindex(sectors).to_numpy()
        lag1_vec = train_matrix.loc[anchor_t, sectors].astype(float).to_numpy()
        tgt_time = anchor_t + h
        dfh = feat_anchor.copy()
        tgt_month = ((tgt_time - 1) % 12) + 1
        dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
        dfh['horizon'] = h / 12.0
        for c in forecaster.feature_cols:
            if c not in dfh.columns:
                dfh[c] = 0.0
        Xh = dfh[forecaster.feature_cols].astype(float).values
        y_raw_hard = forecaster.models[h].predict(Xh)
        y_raw_soft = forecaster.models[h].predict_soft(Xh)
        y_true = train_matrix.loc[tgt_time, sectors].astype(float).to_numpy()
        y_step_h = tiny_positive_to_zero(y_raw_hard, lag1_vec, zstreak_norm_vec, tau=0.5)
        s_h, cap_h, _ = _tune_shrink_and_clip(y_true, y_step_h)
        tuned_hard[h] = (s_h, cap_h)
        y_step_s = tiny_positive_to_zero(y_raw_soft, lag1_vec, zstreak_norm_vec, tau=0.5)
        s_s, cap_s, _ = _tune_shrink_and_clip(y_true, y_step_s)
        tuned_soft[h] = (s_s, cap_s)
    feats_now = create_features_at_time(train_matrix, time_idx=last_train, max_gpu_history=max_gpu_history)
    sectors_now = feats_now['sector'].values
    feats_now = feats_now.set_index('sector')
    zstreak_norm_now = feats_now['consecutive_zeros_norm'].astype(float).reindex(sectors_now).to_numpy()
    lag1_now = train_matrix.loc[last_train, sectors_now].astype(float).to_numpy()
    preds_hard = {}
    preds_soft = {}
    for h in range(1, horizons + 1):
        tgt_time = last_train + h
        dfh = feats_now.copy()
        tgt_month = ((tgt_time - 1) % 12) + 1
        dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
        dfh['horizon'] = h / 12.0
        for c in forecaster.feature_cols:
            if c not in dfh.columns:
                dfh[c] = 0.0
        Xh = dfh[forecaster.feature_cols].astype(float).values
        y_raw_h = forecaster.models[h].predict(Xh)
        y_step_h = tiny_positive_to_zero(y_raw_h, lag1_now, zstreak_norm_now, tau=0.5)
        s_h, cap_h = tuned_hard.get(h, (1.0, np.inf))
        y_pp_h = apply_postprocess(y_step_h, s_h, cap_h)
        y_pp_h = ratio_guard(y_pp_h, lag1_now, max_ratio=3.0)
        preds_hard[tgt_time] = y_pp_h
        y_raw_s = forecaster.models[h].predict_soft(Xh)
        y_step_s = tiny_positive_to_zero(y_raw_s, lag1_now, zstreak_norm_now, tau=0.5)
        s_s, cap_s = tuned_soft.get(h, (1.0, np.inf))
        y_pp_s = apply_postprocess(y_step_s, s_s, cap_s)
        y_pp_s = ratio_guard(y_pp_s, lag1_now, max_ratio=3.0)
        preds_soft[tgt_time] = y_pp_s
    preds_df      = pd.DataFrame(preds_hard, index=sectors_now).sort_index(axis=1)
    preds_df_soft = pd.DataFrame(preds_soft, index=sectors_now).sort_index(axis=1)
    print(f"[hard] shape={preds_df.shape}      zero-rate={(preds_df.values==0).mean():.1%}")
    print(f"[soft] shape={preds_df_soft.shape} zero-rate={(preds_df_soft.values==0).mean():.1%}")
        # --- Build prob_zero_dict from forecaster ---
    prob_zero_dict = {}
    for h in range(1, horizons + 1):
        tgt_time = last_train + h
        dfh = feats_now.copy()
        tgt_month = ((tgt_time - 1) % 12) + 1
        dfh['target_month_sin'] = np.sin(2*np.pi*tgt_month/12)
        dfh['target_month_cos'] = np.cos(2*np.pi*tgt_month/12)
        dfh['target_quarter']   = ((tgt_month - 1) // 3 + 1) / 4.0
        dfh['horizon'] = h / 12.0
        for c in forecaster.feature_cols:
            if c not in dfh.columns:
                dfh[c] = 0.0
        Xh = dfh[forecaster.feature_cols].astype(float).values
        Xs = forecaster.models[h].scaler.transform(Xh)
        pi_raw = forecaster.models[h].zero_model.predict_proba(Xs)[:, 1]
        p0 = forecaster.models[h].iso_.transform(pi_raw) if forecaster.models[h].iso_ is not None else pi_raw
        for i, sec in enumerate(sectors_now):
            prob_zero_dict[(sec, tgt_time)] = float(p0[i])
    
    
    
    return preds_df, preds_df_soft, prob_zero_dict
    #return preds_df, preds_df_soft

# ========= Seasonal Nudger =========
def _month_from_time(t: int) -> int:
    return ((int(t) - 1) % 12) + 1

def _trimmed_mean(a, trim_prop=0.1):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    k = int(np.floor(trim_prop * a.size))
    if k <= 0: return float(np.mean(a))
    a_sorted = np.sort(a)
    return float(np.mean(a_sorted[k: a.size - k]))

def _winsor_mean(a, q=0.10):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    lo, hi = np.quantile(a, q), np.quantile(a, 1 - q)
    a_w = np.clip(a, lo, hi)
    return float(np.mean(a_w))

def _quantile_blend(a, q_lo=0.35, q_hi=0.65):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    lo = np.quantile(a, q_lo)
    hi = np.quantile(a, q_hi)
    return float((lo + hi) / 2.0)

def _ema_on_pairs(pairs, alpha=0.35):
    if len(pairs) == 0: return np.nan
    m = None
    for _, r in pairs:
        r = float(r)
        if not np.isfinite(r) or r <= 0: 
            continue
        if m is None:
            m = r
        else:
            m = alpha * r + (1 - alpha) * m
    return float(m) if (m is not None) else np.nan

def build_seasonal_nudger(train_matrix: pd.DataFrame,
                          window: int = 12,
                          method: str = "harmonic",
                          min_obs_per_month: int = 1,
                          trim_prop: float = 0.10,
                          winsor_q: float = 0.10,
                          q_lo: float = 0.35, q_hi: float = 0.65,
                          ema_alpha: float = 0.35) -> dict:
    df = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    last_t = int(df.index.max())
    win_idx = df.index[df.index > (last_t - window)]
    win = df.loc[win_idx].copy()
    sector_base = win.replace(0, np.nan).mean(axis=0)
    buckets = { (s, m): [] for s in df.columns for m in range(1, 13) }
    buckets_time = { (s, m): [] for s in df.columns for m in range(1, 13) }
    for t in win.index:
        m = _month_from_time(t)
        row = win.loc[t]
        for s, y in row.items():
            b = float(sector_base.get(s, np.nan))
            if not np.isfinite(b) or b <= 0:
                continue
            r = float(y) / b
            if r > 0 and np.isfinite(r):
                buckets[(s, m)].append(r)
                buckets_time[(s, m)].append((int(t), r))
    factors_raw = {}
    for key in buckets.keys():
        arr = buckets[key]
        if len(arr) < min_obs_per_month:
            f = 1.0
        else:
            a = np.asarray(arr, float)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size == 0:
                f = 1.0
            else:
                if method == "harmonic":
                    f = float(a.size / np.sum(1.0 / np.maximum(a, 1e-12)))
                elif method == "median":
                    f = float(np.median(a))
                elif method == "mean":
                    f = float(np.mean(a))
                elif method == "trimmed_mean":
                    f = _trimmed_mean(a, trim_prop=trim_prop)
                elif method == "winsor_mean":
                    f = _winsor_mean(a, q=winsor_q)
                elif method == "quantile_blend":
                    f = _quantile_blend(a, q_lo=q_lo, q_hi=q_hi)
                elif method == "ema":
                    pairs = sorted(buckets_time[key], key=lambda x: x[0])
                    f = _ema_on_pairs(pairs, alpha=ema_alpha)
                    if not np.isfinite(f): f = 1.0
                else:
                    raise ValueError(f"Unknown method: {method}")
        factors_raw[key] = float(f if np.isfinite(f) and f > 0 else 1.0)
    for s in df.columns:
        vals = np.array([factors_raw[(s, m)] for m in range(1, 13)], dtype=float)
        good = np.isfinite(vals) & (vals > 0)
        scale = np.mean(vals[good]) if np.any(good) else 1.0
        scale = scale if (np.isfinite(scale) and scale > 0) else 1.0
        for m in range(1, 13):
            factors_raw[(s, m)] = float(factors_raw[(s, m)] / scale if scale != 0 else 1.0)
    return factors_raw

def shrink_seasonal_factors(factors_raw: dict, strength: float = 0.30, clip: float = 0.10) -> dict:
    lo, hi = 1.0 - clip, 1.0 + clip
    out = {}
    for key, f in factors_raw.items():
        f = float(f) if np.isfinite(f) else 1.0
        f_sh = 1.0 + strength * (f - 1.0)
        out[key] = float(np.clip(f_sh, lo, hi))
    return out

def apply_seasonal_nudge_to_matrix(preds_matrix: pd.DataFrame, factors: dict | None) -> pd.DataFrame:
    if not factors:
        return preds_matrix.copy()
    out = preds_matrix.copy()
    for t in out.index:
        m = _month_from_time(int(t))
        for s in out.columns:
            v = float(out.loc[t, s])
            if v > 0:
                out.loc[t, s] = float(v * factors.get((s, m), 1.0))
    return out

# ========= Adjustment Helpers =========
def adjust_predictions(final_preds, final_preds_positive, adjustment_rules):
    adjusted_preds = final_preds.copy()
    for sector, rules in adjustment_rules.items():
        if sector not in adjusted_preds.index:
            print(f"Warning: {sector} not found in dataframe index")
            continue
        if 'use_positive' in rules and rules['use_positive']:
            for col in rules['use_positive']:
                if col in adjusted_preds.columns:
                    adjusted_preds.loc[sector, col] = final_preds_positive.loc[sector, col]
        if 'use_zero' in rules and rules['use_zero']:
            for col in rules['use_zero']:
                if col in adjusted_preds.columns:
                    adjusted_preds.loc[sector, col] = 0.0
    return adjusted_preds

# -----------------------
# MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    # STEP 1: XGBoost
    train_df = load_and_merge_data()
    X_train, y_train, y_train_log, train_df = feature_engineer(train_df)

    SEED_FEATURE_LIST = INPUT_FEATURES
    MACRO_BASES = ['cpi', 'usd_cny_exchange_rate', 'm2_growth_yoy', 'lpr_5y']

    def select_best_macro_features(train_df, seed_feature_list, macro_base_names):
        selected = {}
        for base in macro_base_names:
            candidates = [f for f in seed_feature_list if f.startswith(base + '_') and f in train_df.columns]
            if candidates:
                selected[base] = candidates[0]
        return selected

    selected_macro_feats = select_best_macro_features(train_df, SEED_FEATURE_LIST, MACRO_BASES)
    print("Selected macro features:", selected_macro_feats)

   # Run Bayesian Optimization and train XGBoost (your existing code)
    print("Running LOCAL Bayesian Optimization...")
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args

    train_df['ym'] = train_df['date'].dt.to_period('M')
    ym_full = train_df['ym'].values
    all_ym = pd.PeriodIndex(train_df['ym'].unique()).sort_values()
    def generate_splits(ym_list, train_size, gap, test_size):
        splits = []
        start = 0
        while start + train_size + gap + test_size <= len(ym_list):
            train_ym = ym_list[start: start + train_size]
            test_ym = ym_list[start + train_size + gap: start + train_size + gap + test_size]
            splits.append((train_ym, test_ym))
            start += test_size
        return splits
    cv_splits = generate_splits(all_ym, 23, 1, 12)

    @use_named_args([
        Integer(270, 359, name='n_estimators'),
        Integer(3, 4, name='max_depth'),
        Real(0.025, 0.028, name='learning_rate'),
        Real(9.5, 10.0, name='reg_alpha'),
        Real(3.6, 3.8, name='reg_lambda'),
        Real(0.59, 0.62, name='subsample')
    ])
    def bo_objective(**params):
        fold_scores = []
        for train_ym, test_ym in cv_splits[:3]:
            train_mask = np.isin(ym_full, train_ym)
            test_mask = np.isin(ym_full, test_ym)
            X_tr, X_te = X_train[train_mask], X_train[test_mask]
            y_tr_log, y_te = y_train_log[train_mask], y_train[test_mask]
            if len(X_tr) == 0 or len(X_te) == 0:
                continue

            num_feats = [f for f in NUMERICAL_FEATURES if f in X_tr.columns]
            cat_feats = ['sector']
            transformers = [
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_feats),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
            ]
            pipe = Pipeline([
                ('preprocessor', ColumnTransformer(transformers, remainder='drop')),
                ('xgb', xgb.XGBRegressor(
                    objective='reg:squarederror',
                    tree_method='hist',
                    random_state=42,
                    **params
                ))
            ])
            pipe.fit(X_tr, y_tr_log)
            preds = np.expm1(pipe.predict(X_te))
            preds = np.clip(preds, 1.0, None)
            fold_scores.append(smape(y_te, preds))
        return np.mean(fold_scores) if fold_scores else 999.0

    result = gp_minimize(
        bo_objective,
        dimensions=[
            Integer(270, 359, name='n_estimators'),
            Integer(3, 4, name='max_depth'),
            Real(0.025, 0.028, name='learning_rate'),
            Real(9.5, 10.0, name='reg_alpha'),
            Real(3.6, 3.8, name='reg_lambda'),
            Real(0.59, 0.62, name='subsample')
        ],
        n_calls=5,
        n_random_starts=4,
        random_state=42,
        verbose=True
    )

    BEST_PARAMS = dict(zip([
        'n_estimators', 'max_depth', 'learning_rate','reg_alpha','reg_lambda','subsample' ], result.x))
    print(f"Best BO sMAPE: {result.fun:.3f}%")

    num_feats = [f for f in NUMERICAL_FEATURES if f in X_train.columns]
    cat_feats = ['sector']
    transformers = [
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ]
    final_pipe = Pipeline([
        ('preprocessor', ColumnTransformer(transformers, remainder='drop')),
        ('xgb', xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            **BEST_PARAMS
        ))
    ])
    final_pipe.fit(X_train, y_train_log)
    joblib.dump(final_pipe, 'xgb_final_submission.joblib')
    # STEP 2: Ridge Ensemble â€” INJECT MACRO AS PSEUDO-SECTORS (97, 98, ...)
    print("Training Ridge Ensemble on FULL matrix...")
    train_data_full = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
    train_data_full['time'] = train_data_full['month'].apply(month_str_to_time)
    train_matrix = create_training_matrix(train_data_full)

    # âœ… INJECT MACRO FEATURES AS EXTRA SECTORS (sector 97, 98, ...)
    '''if selected_macro_feats:
        train_df_for_macro = train_df.copy()
        train_df_for_macro['time'] = train_df_for_macro['month'].apply(month_str_to_time)
        macro_series = train_df_for_macro.groupby('time')[list(selected_macro_feats.values())].first()
        # Rename to sector 97, 98, ...
        next_id = 97
        renamed = {}
        for col in macro_series.columns:
            renamed[col] = f'sector {next_id}'
            print ('rename map'+col+'sectorid'+str(next_id))
            next_id += 1
        macro_series = macro_series.rename(columns=renamed)
        macro_aligned = macro_series.reindex(train_matrix.index).fillna(0.0)
        train_matrix = pd.concat([train_matrix, macro_aligned], axis=1)
        print(f"Injected {len(macro_aligned.columns)} macro pseudo-sectors: {list(renamed.values())}")
    '''
 
    final_preds, final_preds_positive, prob_zero_dict = generate_final_predictions(
        train_matrix=train_matrix,
        horizons=12,
        ridge_alpha=0.73,
        logistic_C=0.07,
        use_isotonic=True,
        max_gpu_history=16,
    )

    

    # Apply adjustments and seasonal nudge
    adjustment_rules = {
        'sector 12': {'use_positive': [68, 69, 71], 'use_zero': []},
        'sector 19': {'use_positive': [72, 76, 78, 79], 'use_zero': []},
        'sector 33': {'use_positive': [], 'use_zero': [69, 70, 71, 72, 74, 75, 76, 77]},
        'sector 53': {'use_positive': [68], 'use_zero': []},
        'sector 58': {'use_positive': [68], 'use_zero': []},
        'sector 89': {'use_positive': [69, 71, 72], 'use_zero': []},
        'sector 70': {'use_positive': [78, 79], 'use_zero': []},
        'sector 17': {'use_positive': [78, 79], 'use_zero': []},
        'sector 72': {'use_positive': [69, 71], 'use_zero': []},
        'sector 73': {'use_positive': [68, 71, 72, 73], 'use_zero': []},
        'sector 75': {'use_positive': [69, 72], 'use_zero': []}
    }
    adjusted_preds = adjust_predictions(final_preds, final_preds_positive, adjustment_rules)
    future_preds = adjusted_preds.T
    factors_raw = build_seasonal_nudger(train_matrix, window=12, method="trimmed_mean")
    factors = shrink_seasonal_factors(factors_raw, strength=0.30, clip=0.10)
    future_preds_nudged = apply_seasonal_nudge_to_matrix(future_preds, factors)

 # -----------------------# -----------------------
# STEP 3: Load test and predict recursively (SIMPLIFIED & FIXED)
# -----------------------
test_df = pd.read_csv(TEST_PATH)
test_df[['month_str', 'sector_str']] = test_df['id'].str.split('_', expand=True)
test_df['month'] = pd.to_datetime(test_df['month_str'], format='%Y %b')
test_df['sector'] = test_df['sector_str'].str.replace('sector ', '').str.strip()

# Basic time features
test_df['year'] = test_df['month'].dt.year
test_df['month_num'] = test_df['month'].dt.month
test_df['month_sin'] = np.sin(2 * np.pi * test_df['month_num'] / 12)
test_df['month_cos'] = np.cos(2 * np.pi * test_df['month_num'] / 12)

# âœ… Compute holiday feature using your existing helper
test_df = add_monthly_holiday_features(test_df, date_col='month')

# Initialize lag dictionary from TRAIN DATA
lag_dict = {}
for feat in NUMERICAL_FEATURES:
    if '_lag_' in feat:
        base_feat = feat.split('_lag_')[0]
        lag_dict[feat] = {}
        for sector in train_df['sector'].unique():
            sector_data = train_df[train_df['sector'] == sector].sort_values('month')
            if base_feat in sector_data.columns:
                vals = sector_data[base_feat].dropna().values
                lag_dict[feat][sector] = list(vals)
            else:
                lag_dict[feat][sector] = []

# Get train medians for fallback imputation
train_medians = train_df[NUMERICAL_FEATURES].median().to_dict()

# Recursive prediction
test_df = test_df.sort_values(['month', 'sector']).reset_index(drop=True)
test_months = sorted(test_df['month'].unique())
predictions = {}

for month in test_months:
    batch = test_df[test_df['month'] == month].copy()
    valid_mask = batch['sector'].isin(train_df['sector'].unique())
    
    # Invalid sectors â†’ predict 0
    for idx, row in batch.loc[~valid_mask].iterrows():
        predictions[row['id']] = 0.0

    valid_batch = batch.loc[valid_mask].copy()
    if len(valid_batch) == 0:
        continue

    # Fill lag features using lag_dict (updated recursively)
    for feat in NUMERICAL_FEATURES:
        if '_lag_' in feat:
            lag_val = int(feat.split('_lag_')[1])
            base_feat = feat.split('_lag_')[0]
            for idx, row in valid_batch.iterrows():
                sector = row['sector']
                hist = lag_dict.get(feat, {}).get(sector, [])
                if len(hist) >= lag_val:
                    valid_batch.at[idx, feat] = hist[-lag_val]
                else:
                    valid_batch.at[idx, feat] = train_medians.get(base_feat, 0.0)

    # Fill any remaining missing features with train medians
    for feat in NUMERICAL_FEATURES:
        if feat not in valid_batch.columns:
            valid_batch[feat] = train_medians.get(feat, 0.0)
        else:
            valid_batch[feat] = valid_batch[feat].fillna(train_medians.get(feat, 0.0))

    valid_batch['sector'] = valid_batch['sector'].astype(str)
    X_batch = valid_batch[INPUT_FEATURES]

    # XGBoost prediction
    xgb_preds = np.expm1(final_pipe.predict(X_batch))
    xgb_preds = np.clip(xgb_preds, 1.0, None)

    # Get Ridge prediction and prob_zero
    time_int = month_str_to_time(f"{month.year} {month.strftime('%b')}")
    ridge_preds = []
    prob_zeros = []
    for _, row in valid_batch.iterrows():
        sec = f"sector {row['sector']}"
        ridge_val = future_preds_nudged.loc[time_int, sec] if sec in future_preds_nudged.columns else 0.0
        p0 = prob_zero_dict.get((sec, time_int), 1.0)
        ridge_preds.append(ridge_val)
        prob_zeros.append(p0)
    ridge_preds = np.array(ridge_preds)
    prob_zeros = np.array(prob_zeros)

    # MANUAL ZERO OVERRIDES
    for i, (idx, row) in enumerate(valid_batch.iterrows()):
        sec = f"sector {row['sector']}"
        if sec == 'sector 33' and time_int in [69, 70, 71, 72, 74, 75, 76, 77]:
            ridge_preds[i] = 0.0
            prob_zeros[i] = 1.0  # ensure override

    # âœ… FIXED: Corrected blending loop (was broken due to indentation)
    final_preds_v1 = np.zeros_like(xgb_preds)
    for i, (idx, row) in enumerate(valid_batch.iterrows()):
        sec = row['sector']
        # Use more Ridge for volatile sectors (e.g., 42, 33), more XGBoost for stable ones
        if sec in ['42', '33', '85']:
            w = 0.7  # trust Ridge more
        else:
            w = 0.4  # trust XGBoost more
        blended = w * ridge_preds[i] + (1 - w) * xgb_preds[i]
        # Apply zero override
        if prob_zeros[i] >= 0.5:
            blended = 0.0
        final_preds_v1[i] = blended

    # HORIZON-AWARE BLENDING (your original logic)
    horizon = test_months.index(month) + 1
    if horizon <= 6:
        blended_preds = 0.55 * xgb_preds + 0.45 * ridge_preds
    else:
        blended_preds = 0.65 * xgb_preds + 0.35 * ridge_preds

    # ZERO OVERRIDE
    final_preds = blended_preds.copy()
    final_preds[prob_zeros >= 0.55] = 0.0

    # Save predictions and update lag history
    for i, (idx, row) in enumerate(valid_batch.iterrows()):
        pred = final_preds[i]
        predictions[row['id']] = pred
        # Update lag history for amount_new_house_transactions features
        for feat in lag_dict:
            if feat.startswith('amount_new_house_transactions'):
                lag_dict[feat][row['sector']].append(pred)
   
    # -----------------------
    # STEP 4: Build submission from predictions and apply Sector 42 override
    # -----------------------
    submission = pd.DataFrame({
        'id': test_df['id'],
        'new_house_transaction_amount': [predictions.get(i, 0.0) for i in test_df['id']]
    })

    # Apply Sector 42 override if file exists
    if os.path.exists(SECTOR_42_PATH):
        sector_42 = pd.read_csv(SECTOR_42_PATH)
        sector_42_only = sector_42[sector_42['id'].str.endswith('sector 42')].copy()
        sector_42_dict = dict(zip(sector_42_only['id'], sector_42_only['new_house_transaction_amount']))
        for idx in submission.index:
            id_val = submission.loc[idx, 'id']
            if id_val in sector_42_dict:
                submission.loc[idx, 'new_house_transaction_amount'] = sector_42_dict[id_val]

    submission.to_csv('submission.csv', index=False)
    print(f"âœ… Submission saved! Mean prediction: {submission['new_house_transaction_amount'].mean():.2f}")


import pandas as pd
import numpy as np
import os
import joblib
import pmdarima as pm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import holidays
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress all warnings for clean execution
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG (Based on your todebug.txt)
# -----------------------
BASE_PATH = '/kaggle/input/china-real-estate-demand-prediction/train'
TEST_PATH = '/kaggle/input/china-real-estate-demand-prediction/test.csv'
POI_PATH = f'{BASE_PATH}/sector_POI.csv'
# NOTE: These paths assume a Kaggle notebook environment. You might need to adjust them.
EXTRA_PATH_MACRO = "/kaggle/input/cn-cpi-hpi-usdrates-boundsrates/cn-china-economic-data-extended.csv"
EXTRA_PATH_STATS = "/kaggle/input/real-estate-global-cn-policy/city_data_policy_mapping.csv"
EXTRA_PATH_CITY_POLICY = "/kaggle/input/real-estate-global-cn-policy/city-restric-cn.csv"
CITY_INDEXES_PATH = f'{BASE_PATH}/city_indexes.csv'

TARGET = 'amount_new_house_transactions'

RAW_TS_FEATURES = [
    'amount_new_house_transactions', 'num_new_house_available_for_sale', 
    'area_new_house_available_for_sale', 'num_new_house_transactions', 
    'area_new_house_transactions', 'price_new_house_transactions', 
    'total_price_per_unit_new_house_transactions', 'period_new_house_sell_through', 
    'num_pre_owned_house_transactions', 'area_pre_owned_house_transactions', 
    'price_pre_owned_house_transactions', 'amount_new_house_transactions_nearby_sectors', 
    'treasury_rate', 'search_volume', 'national_median_price', 
    'usd_cny_exchange_rate', 'cpi', 'm2_growth_yoy', 'lpr_5y', 
    'purchase_restriction_level', 'min_down_payment_ratio_first_home'
]
RAW_TS_FEATURES = list(set(RAW_TS_FEATURES)) # Ensure uniqueness

# Output settings
OUTPUT_DIR = './feature_forecast_artifacts'
PCA_MODEL_PATH = os.path.join(OUTPUT_DIR, 'pca_scaler_model.joblib')
PREDICTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'predicted_exogenous_features.csv')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
TEST_HORIZON_MONTHS = 12 

# -----------------------
# Helper Functions (Actual Data Loading Logic)
# -----------------------
def safe_merge(df1, df2, on_cols, suffix=''):
    if not isinstance(on_cols, list):
        on_cols = [on_cols]
    return pd.merge(df1, df2, on=on_cols, how='left', suffixes=('', suffix))

def add_monthly_holiday_features(df, date_col='month'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    min_year = df[date_col].dt.year.min()
    max_year = df[date_col].dt.year.max() + 1
    cn_holidays = holidays.China(years=range(min_year, max_year + 1))
    holiday_dates = pd.to_datetime(sorted(cn_holidays.keys()))
    holiday_df = pd.DataFrame({'holiday_date': holiday_dates})
    holiday_df['holiday_month'] = holiday_df['holiday_date'].dt.to_period('M')
    holiday_months = holiday_df['holiday_month'].drop_duplicates()
    df['ym'] = df[date_col].dt.to_period('M')
    df['is_holiday_month'] = df['ym'].isin(holiday_months).astype(int)
    df = df.drop(columns=['ym'])
    return df

def load_and_merge_data():
    """Loads and merges all raw time series data based on your original pipeline."""
    
    print("? Loading and merging training data...")
    dfs = {}
    file_list = [
        'city_indexes.csv', 'city_search_index.csv', 'land_transactions.csv',
        'land_transactions_nearby_sectors.csv', 'new_house_transactions.csv',
        'new_house_transactions_nearby_sectors.csv', 'pre_owned_house_transactions.csv',
        'pre_owned_house_transactions_nearby_sectors.csv', 'sector_POI.csv'
    ]
    for file in file_list:
        try:
            path = f'{BASE_PATH}/{file}'
            df = pd.read_csv(path)
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            dfs[file.replace('.csv', '')] = df
        except Exception as e:
            # print(f"Could not load {file}: {e}") # Debugging aid
            pass

    if 'new_house_transactions' not in dfs:
        raise FileNotFoundError("new_house_transactions.csv is required but not found/loaded.")

    main_df = dfs['new_house_transactions'].copy()
    main_df = main_df.rename(columns={'month': 'date'})
    main_df['date'] = pd.to_datetime(main_df['date'], errors='coerce')
    main_df['year'] = main_df['date'].dt.year.astype('Int64')
    main_df['month'] = main_df['date'].dt.month.astype('Int64')
    main_df['ym'] = main_df['date'].dt.to_period('M')
    main_df = main_df.dropna(subset=['date']).reset_index(drop=True)
    main_df['sector'] = main_df['sector'].astype(str).str.replace('sector ', '').str.strip()
    main_df['merge_key'] = main_df['date'].dt.strftime('%Y-%m') + '_' + main_df['sector']


    # Merge nearby sectors
    if 'new_house_transactions_nearby_sectors' in dfs:
        df = dfs['new_house_transactions_nearby_sectors'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df = df.rename(columns={'month': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
        df['merge_key'] = df['date'].dt.strftime('%Y-%m') + '_' + df['sector']
        df = df.drop(['date', 'sector'], axis=1, errors='ignore')
        main_df = safe_merge(main_df, df, 'merge_key')

    # Merge land and pre-owned
    for key in ['land_transactions', 'pre_owned_house_transactions']:
        if key in dfs:
            df = dfs[key].copy()
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            df = df.rename(columns={'month': 'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
            df['merge_key'] = df['date'].dt.strftime('%Y-%m') + '_' + df['sector']
            cols = [c for c in df.columns if c not in ['date', 'sector', 'merge_key']]
            suffix = '_land' if 'land' in key else '_pre'
            main_df = safe_merge(main_df, df[['merge_key'] + cols], 'merge_key', suffix)

    # Merge POI
    if 'sector_poi' in dfs:
        df = dfs['sector_poi'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df['sector'] = df['sector'].astype(str).str.replace('sector ', '').str.strip()
        df = df.drop_duplicates(subset=['sector'], keep='last')
        df = df.drop('month', axis=1, errors='ignore')
        main_df = safe_merge(main_df, df, 'sector')

    # Merge city search index
    if 'city_search_index' in dfs:
        df = dfs['city_search_index'].copy()
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        df = df.rename(columns={'month': 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['ym'] = df['date'].dt.to_period('M')
        if 'search_volume' in df.columns:
            agg = df.groupby('ym')['search_volume'].sum().reset_index()
            main_df = safe_merge(main_df, agg, 'ym', suffix='_search')

    # Merge national stats
    if os.path.exists(EXTRA_PATH_STATS):
        stats_df = pd.read_csv(EXTRA_PATH_STATS)
        stats_df.columns = stats_df.columns.str.strip().str.replace(' ', '_').str.lower()
        stats_df['date'] = pd.to_datetime(
            stats_df['year'].astype(int).astype(str) + '-' + stats_df['month'].astype(int).astype(str).str.zfill(2),
            errors='coerce'
        )
        stats_df = stats_df.rename(columns={
            'median_price': 'national_median_price',
            'sale_volume': 'national_sale_volume',
            'price_index_yoy': 'national_price_index_yoy',
            'inventory_months': 'national_inventory_months'
        })
        stats_df['ym'] = stats_df['date'].dt.to_period('M')
        main_df['ym'] = main_df['date'].dt.to_period('M')
        main_df = safe_merge(main_df, stats_df.drop(columns=['date', 'year', 'month']), on_cols='ym', suffix='_stats')
        main_df = main_df.drop(columns=['ym'], errors='ignore')
        main_df.rename(columns={'search_volume_search': 'search_volume'}, inplace=True)

    # Merge macro
    if os.path.exists(EXTRA_PATH_MACRO):
        extra_df = pd.read_csv(EXTRA_PATH_MACRO)
        extra_df.columns = extra_df.columns.str.strip().str.replace(' ', '_').str.lower()
        extra_df['date'] = pd.to_datetime(
            extra_df['year'].astype(str) + '-' + extra_df['month'].astype(str).str.zfill(2),
            errors='coerce'
        )
        extra_df['ym'] = extra_df['date'].dt.to_period('M')
        main_df['ym'] = main_df['date'].dt.to_period('M')
        # Placeholder for complex feature engineering (as seen in previous code snippets)
        if {'cpi', 'hpi', 'lpr_5y'} <= set(extra_df.columns):
            extra_df['cpi_norm'] = extra_df['cpi'] / 100.0
            extra_df['hpi_norm'] = extra_df['hpi'] / extra_df['hpi'].iloc[0] # Example normalization
            extra_df['mortgage_factor'] = 1 + (extra_df['lpr_5y'] / 100.0)
            extra_df['buying_power_index'] = 1.0 / (
                extra_df['hpi_norm'] * extra_df['mortgage_factor'] * extra_df['cpi_norm']
            )
        cols_to_keep = ['ym', 'cpi', 'hpi', 'treasury_rate', 'usd_cny_exchange_rate', 'm2_growth_yoy', 'lpr_1y', 'lpr_5y', 'buying_power_index']
        extra_df = extra_df[[c for c in cols_to_keep if c in extra_df.columns]]
        main_df = safe_merge(main_df, extra_df, 'ym', suffix='_macro')
        main_df = main_df.drop(columns=['ym'], errors='ignore')
        main_df.rename(columns={
            'cpi_macro': 'cpi',
            'usd_cny_exchange_rate_macro': 'usd_cny_exchange_rate',
            'm2_growth_yoy_macro': 'm2_growth_yoy',
            'lpr_5y_macro': 'lpr_5y',
            'treasury_rate_macro': 'treasury_rate'
        }, inplace=True)

    # Merge city indexes
    if os.path.exists(CITY_INDEXES_PATH):
        df = pd.read_csv(CITY_INDEXES_PATH)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        if 'city_indicator_data_year' in df.columns:
            df.rename(columns={'city_indicator_data_year': 'year'}, inplace=True)
            main_df['year'] = main_df['date'].dt.year
            main_df = safe_merge(main_df, df.drop(columns=['city_search_index'], errors='ignore'), 'year')

    # Merge city policy
    if os.path.exists(EXTRA_PATH_CITY_POLICY):
        policy_df = pd.read_csv(EXTRA_PATH_CITY_POLICY)
        policy_df['date_time'] = pd.to_datetime(
            policy_df['year'].astype(str) + '-' + policy_df['month'].astype(str).str.zfill(2),
            errors='coerce'
        )
        policy_df = policy_df.dropna(subset=['date_time'])
        policy_df['sector'] = policy_df['sector'].astype(str).str.replace('sector ', '').str.strip()
        policy_features = ['hukou_policy_level', 'policy_group', 'policy_stance']
        policy_df = policy_df.sort_values(['sector', 'date_time'])
        policy_df_resampled = policy_df.groupby('sector').apply(
            lambda g: g.set_index('date_time')[policy_features].resample('MS').ffill()
        ).reset_index()
        policy_df_resampled = policy_df_resampled.rename(columns={'date_time': 'month'})
        policy_cols = ['month', 'sector'] + policy_features
        policy_df = policy_df_resampled[policy_cols].drop_duplicates(subset=['month', 'sector'])
        
        # Ensure 'month' is compatible for merge (convert to datetime if necessary)
        policy_df['month'] = pd.to_datetime(policy_df['month'].astype(str))
        main_df = safe_merge(main_df, policy_df, ['month', 'sector'])

    # Shift concurrent features
    CONCURRENT = [
        'search_volume', 'national_median_price', 'usd_cny_exchange_rate', 'cpi',
        'm2_growth_yoy', 'lpr_5y', 'purchase_restriction_level', 
        'min_down_payment_ratio_first_home'
    ]
    main_df = main_df.sort_values(['sector', 'date']).reset_index(drop=True)
    cols_to_shift = [c for c in CONCURRENT if c in main_df.columns]
    if cols_to_shift:
        # Shift concurrent features by one period (month) to avoid leakage
        # NOTE: This operation is done AFTER all merges, ensuring 'date' and 'sector' columns are intact.
        main_df[cols_to_shift] = main_df.groupby('sector')[cols_to_shift].shift(1)

    main_df.drop('merge_key', axis=1, errors='ignore', inplace=True)
    main_df = add_monthly_holiday_features(main_df, date_col='date')
    
    # Final check: Ensure date and sector columns are correctly named and non-null
    main_df.rename(columns={'month_x': 'month'}, inplace=True)
    
    print(f"Loaded {len(main_df)} records.")
    # Return with 'date' and 'sector' columns present
    return main_df

# -----------------------
# ARIMA COMPONENT FORECASTING (Unchanged from previous successful logic)
# -----------------------
def forecast_components(components_df, n_ahead):
    """
    Fits an AutoARIMA model to each Principal Component time series and forecasts 
    the next n_ahead steps.
    """
    print(f"\n? Fitting AutoARIMA to {components_df.shape[1]} principal components...")
    
    forecast_values = {}
    
    for comp in tqdm(components_df.columns, desc="Forecasting PCs"):
        series = components_df[comp].dropna()
        
        try:
            model = pm.auto_arima(
                series, 
                start_p=1, start_q=1, max_p=3, max_q=3,
                m=12, seasonal=True, start_P=0, d=1, D=1,
                trace=False, error_action='ignore', suppress_warnings=True, 
                stepwise=True
            )
            forecast, _ = model.predict(n_periods=n_ahead, return_conf_int=True)
            forecast_values[comp] = forecast.values
            
        except Exception as e:
            # Fallback to Seasonal Naive (S-Naive)
            if len(series) >= 12:
                 last_year_value = series.iloc[-12] if len(series) > 12 else series.iloc[-1]
                 forecast_values[comp] = np.repeat(last_year_value, n_ahead)
            else:
                 last_value = series.iloc[-1] if not series.empty else 0
                 forecast_values[comp] = np.repeat(last_value, n_ahead)


    last_time = components_df.index.max()
    forecast_dates = pd.to_datetime(pd.date_range(start=last_time, periods=n_ahead + 1, freq='MS')).shift(1, freq='MS')[1:]
    forecast_df = pd.DataFrame(forecast_values, index=forecast_dates)
    
    print(f"Generated {n_ahead}-step forecasts for all components using ARIMA/S-Naive.")
    return forecast_df

# -----------------------
# ANALYSIS AND PLOTTING FUNCTION (Unchanged from previous successful logic)
# -----------------------
def run_analysis_and_plot(data, pca_input_features, TARGET, n_components_90, pca_model):
    """
    Performs correlation analysis and plots the diagnostic heatmaps.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Separate features and target for analysis
    # NOTE: data here is expected to be a DataFrame with 'date' and 'sector' columns
    data_clean = data.copy().set_index(['date', 'sector'])
    X = data_clean.drop(columns=[TARGET], errors='ignore')
    
    # --- 1. FEATURE-TARGET CORRELATION HEATMAP ---
    print("\n--- 1. Feature-Target Correlation Heatmap ---")
    corr_matrix_full = data_clean.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix_full, annot=False, cmap='coolwarm', fmt=".2f", 
                linewidths=.5, linecolor='lightgray', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Heatmap (All Time Series)', fontsize=16)
    plt.tight_layout()
    corr_heatmap_path = os.path.join(PLOT_DIR, 'feature_correlation_heatmap.png')
    plt.savefig(corr_heatmap_path)
    plt.close()
    print(f"Saved Feature Correlation Heatmap to {corr_heatmap_path}")
    
    # --- 2. PCA EXPLAINED VARIANCE ---
    print("\n--- 2. PCA Explained Variance Plot ---")
    
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    
    plt.axvline(x=n_components_90, color='r', linestyle=':', label=f'{n_components_90} Components for 90% Variance')
    plt.axhline(y=0.90, color='r', linestyle=':')
    
    plt.title('PCA Cumulative Explained Variance', fontsize=16)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    variance_plot_path = os.path.join(PLOT_DIR, 'pca_explained_variance.png')
    plt.savefig(variance_plot_path)
    plt.close()
    print(f"Saved PCA Explained Variance Plot to {variance_plot_path}")
    
    # --- 3. FEATURE-COMPONENT CORRELATION HEATMAP ---
    print("\n--- 3. Feature-Component Correlation Heatmap ---")
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_pca = pca_model.transform(X_scaled)

    n_components_plot = min(n_components_90, 10) # Plot max 10 for readability
    component_names = [f'PC{i+1}' for i in range(n_components_plot)]
    
    # Create DF of PC Scores
    pca_scores_df = pd.DataFrame(X_pca[:, :n_components_plot], index=X.index, columns=component_names)
    
    # Combine scaled features and PC scores
    features_and_pcs = pd.DataFrame(X_scaled, index=X.index, columns=X.columns).join(pca_scores_df)
    
    # Calculate correlation between original features and PC scores
    corr_features_pcs = features_and_pcs.corr().loc[X.columns, component_names]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_features_pcs, annot=True, cmap='viridis', fmt=".2f", 
                linewidths=.5, linecolor='lightgray', cbar_kws={'label': 'Correlation with Principal Component'})
    plt.title(f'Feature-to-Component Correlation (Top {n_components_plot} PCs)', fontsize=16)
    plt.tight_layout()
    comp_corr_heatmap_path = os.path.join(PLOT_DIR, 'feature_component_correlation_heatmap.png')
    plt.savefig(comp_corr_heatmap_path)
    plt.close()
    print(f"Saved Feature-Component Correlation Heatmap to {comp_corr_heatmap_path}")

# -----------------------
# MAIN PCA PIPELINE
# -----------------------
def run_pca_and_feature_forecast():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    raw_df = load_and_merge_data()
    
    # 2. Reshape and Clean Data for PCA
    print("\n? Preparing data for PCA...")
    
    pca_input_features = [f for f in RAW_TS_FEATURES if f in raw_df.columns]
    
    # **FIXED LINE** explicitly set the multi-index using 'date' and 'sector' columns
    pca_data = raw_df.set_index(['date', 'sector'])[pca_input_features]
    
    # Impute missing values robustly (median of the entire series)
    pca_data = pca_data.fillna(pca_data.median()).fillna(0)
    
    # 3. Fit Scaler and PCA
    print(f"Fitting StandardScaler and PCA on {len(pca_input_features)} features.")
    scaler = StandardScaler() 
    data_scaled = scaler.fit_transform(pca_data)
    
    # Initial PCA to determine components (k)
    pca_initial = PCA(n_components=None)
    pca_initial.fit(data_scaled)
    
    # Determine number of components for 90% explained variance
    cumulative_variance = np.cumsum(pca_initial.explained_variance_ratio_)
    try:
        n_components_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1
    except IndexError:
        n_components_90 = len(pca_input_features) 
    print(f"Found {n_components_90} components explaining >= 90% of variance.")

    # 4. Final PCA Fit and Transform
    pca_final = PCA(n_components=n_components_90)
    components_series = pca_final.fit_transform(data_scaled)
    
    # Aggregate components across sectors for macro forecasting
    components_df_full = pd.DataFrame(components_series, 
                                 index=pca_data.index,
                                 columns=[f'PC_{i}' for i in range(n_components_90)])
    
    # Group by date index to get MACRO components for forecasting
    components_df_macro = components_df_full.groupby(level='date').mean()
    
    # 5. Run Analysis and Plots
    # Pass the index to columns so the plotting function can re-set it
    run_analysis_and_plot(pca_data.reset_index(), pca_input_features, TARGET, n_components_90, pca_final)
    
    # 6. Forecast Components
    forecasted_components_df = forecast_components(components_df_macro, n_ahead=TEST_HORIZON_MONTHS)
    
    # 7. Inverse Transform to Feature Forecasts
    print("\n? Reconstructing original features...")
    historical_sectors = raw_df['sector'].unique()
    sector_replicated_forecast = []
    
    for _, row in forecasted_components_df.iterrows():
        for _ in historical_sectors:
            sector_replicated_forecast.append(row.values)
            
    future_scaled_features = pca_final.inverse_transform(np.array(sector_replicated_forecast))
    future_features_unscaled = scaler.inverse_transform(future_scaled_features)
    
    # 8. Save Artifacts
    print("? Saving fitted models and feature forecasts...")
    
    joblib.dump({'scaler': scaler, 'pca': pca_final, 'raw_features': pca_input_features}, PCA_MODEL_PATH)
    
    future_dates = forecasted_components_df.index
    final_forecast_list = []
    idx = 0
    for date in future_dates:
        date_str = date.strftime('%Y-%m')
        for sector in historical_sectors:
            sector_str = str(sector)
            row_dict = {
                'date': date,
                'sector': sector_str,
                'id': f"{date_str}_{sector_str}"
            }
            for i, feature in enumerate(pca_input_features):
                 row_dict[feature] = future_features_unscaled[idx, i]
            final_forecast_list.append(row_dict)
            idx += 1

    final_forecast_df = pd.DataFrame(final_forecast_list)
    final_forecast_df.to_csv(PREDICTED_FEATURES_PATH, index=False)
    
    print(f"âœ… Saved PCA and Scaler model to: {PCA_MODEL_PATH}")
    print(f"âœ… Saved {len(final_forecast_df)} predicted feature rows to: {PREDICTED_FEATURES_PATH}")
    print("\nâœ¨ Feature Forecasting Pipeline Complete.")

if __name__ == '__main__':
    run_pca_and_feature_forecast()

