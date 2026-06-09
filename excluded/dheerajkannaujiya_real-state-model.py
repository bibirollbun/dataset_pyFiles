import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import calendar
import warnings
warnings.filterwarnings('ignore')

# ==== CUSTOM SCORE (No Changes) ====
def custom_competition_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / np.abs(y_true)
    ape[np.isinf(ape)] = 0; ape[np.isnan(ape)] = 0
    if np.mean(ape > 1.0) > 0.30: return 0.0
    ok_mask = (ape <= 1.0)
    scaled_mape = np.mean(ape[ok_mask]) / np.mean(ok_mask)
    return float(1.0 - scaled_mape)

# ==== CONFIG (No Changes) ====
DATA_DIR = "/kaggle/input/china-real-estate-demand-prediction"
TARGET_COL = "amount_new_house_transactions"
CUT_YEAR, CUT_MONTH = 2024, 7

# ==== LOAD DATA (No Changes) ====
def load_all_data(data_dir):
    data = {}
    datasets = { 'new': 'train/new_house_transactions.csv',
                'new_nb': 'train/new_house_transactions_nearby_sectors.csv',
                'pre': 'train/pre_owned_house_transactions.csv',
                'pre_nb': 'train/pre_owned_house_transactions_nearby_sectors.csv',
                'land': 'train/land_transactions.csv',
                'land_nb': 'train/land_transactions_nearby_sectors.csv',
                'city_idx': 'train/city_indexes.csv',
                'city_search': 'train/city_search_index.csv',
                'poi': 'train/sector_POI.csv',
                'test': 'test.csv' }
    
    for name, path in datasets.items():
        data[name] = pd.read_csv(os.path.join(data_dir, path))
    return data

all_data = load_all_data(DATA_DIR)

# ==== DATETIME & SECTOR (No Changes) ====
def extract_datetime_features(df, date_col='month'):
    df = df.copy(); df[['Year', 'Month']] = df[date_col].str.split('-', expand=True); df['Year'] = df['Year'].astype(int)
    month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    df['Month_num'] = df['Month'].map(month_map)
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_num'].astype(str) + '-01')
    df['time_index'] = (df['Year'] - 2019) * 12 + df['Month_num']; df['quarter'] = (df['Month_num'] - 1) // 3 + 1
    df['sin_month'] = np.sin(2 * np.pi * df['Month_num'] / 12); df['cos_month'] = np.cos(2 * np.pi * df['Month_num'] / 12)
    return df

def extract_sector_features(df):
    df = df.copy(); df['sector_num'] = df['sector'].str.extract(r'(\d+)').astype(int); return df

for name in ['new', 'new_nb', 'pre', 'pre_nb', 'land', 'land_nb']:
    if name in all_data: all_data[name] = extract_datetime_features(all_data[name]); all_data[name] = extract_sector_features(all_data[name])
if 'poi' in all_data: all_data['poi'] = extract_sector_features(all_data['poi'])

# ==== CLUSTERING (No Changes) ====
print("Running K-Means Clustering on POI data...")
poi_df = all_data['poi'].copy()
poi_features = poi_df.select_dtypes(include=np.number).drop(columns=['sector_num'])
poi_features = poi_features.fillna(0)
scaler_poi = StandardScaler()
poi_scaled = scaler_poi.fit_transform(poi_features)
kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
poi_df['sector_cluster'] = kmeans.fit_predict(poi_scaled)
all_data['poi'] = poi_df
print("Clustering complete.")

# ==== DATA INTEGRATION AND MERGING (Original Logic) ====
def create_master_dataset(new_df, other_dfs):
    master_df = new_df.copy()
    # Merge nearby, pre-owned, land, and POI data
    for name, prefix in [('new_nb', 'nb'), ('pre', 'pre'), ('land', 'land')]:
        if name in other_dfs:
            master_df = master_df.merge(other_dfs[name].rename(columns={col: f'{prefix}_{col}' for col in other_dfs[name].columns if col not in ['Year', 'Month_num', 'sector_num', 'date']}), on=['Year', 'Month_num', 'sector_num', 'date'], how='left')
    if 'poi' in other_dfs:
        master_df = master_df.merge(other_dfs['poi'].drop(columns=['sector']), on='sector_num', how='left')
    return master_df

master_df = create_master_dataset(all_data['new'], all_data)

# ==== ADVANCED FEATURE ENGINEERING (No Changes) ====
def create_advanced_features(df):
    df = df.copy(); group_cols = ['sector_num']
    
    # Original lags and rolling
    for lag in [1, 3, 6, 12]:
        df[f'target_lag_{lag}'] = df.groupby(group_cols)[TARGET_COL].shift(lag)
        
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df.groupby(group_cols)[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        df[f'rolling_std_{window}'] = df.groupby(group_cols)[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())

    # EWMA
    print("Creating EWMA features...")
    for window in [3, 6, 12]:
        df[f'target_ewma_{window}'] = df.groupby('sector_num')[TARGET_COL].transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean())
        if 'price_new_house_transactions' in df.columns:
            df[f'price_ewma_{window}'] = df.groupby('sector_num')['price_new_house_transactions'].transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean())

    # Cluster features
    print("Creating Cluster-based features...")
    if 'sector_cluster' in df.columns:
        df['cluster_mean_sales'] = df.groupby(['Year', 'Month_num', 'sector_cluster'])[TARGET_COL].transform('mean')
        df['sales_vs_cluster_ratio'] = df[TARGET_COL] / (df['cluster_mean_sales'] + 1e-6)
    
    # Target creation
    df['target_12m_ahead'] = df.groupby(group_cols)[TARGET_COL].shift(-12)

    # Volatility
    print("Creating Spike-Detector features...")
    for col in [TARGET_COL, 'price_new_house_transactions']:
        if col in df.columns:
            df[f'{col}_yoy_growth'] = (df[col] - df.groupby(group_cols)[col].shift(12)) / (df.groupby(group_cols)[col].shift(12) + 1e-6)

    for window in [6, 12]:
        df[f'target_diff_from_rolling_mean_{window}'] = df[TARGET_COL].shift(1) - df[f'rolling_mean_{window}']

    print("Clipping extreme values...")
    volatile_cols = [col for col in df.columns if 'yoy_growth' in col or 'diff' in col]
    for col in volatile_cols:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower_bound, upper_bound)

    df['target_same_month_last_year'] = df.groupby(group_cols)[TARGET_COL].shift(12)
    df['target_same_month_last_year'] = df.groupby(group_cols)['target_same_month_last_year'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df['target_same_month_last_year'] = df['target_same_month_last_year'].fillna(0)
    
    # Dummies for holidays and policy
    df['is_cny_month'] = ((df['Month_num'] == 1) | (df['Month_num'] == 2)).astype(int)
    df['is_national_day'] = (df['Month_num'] == 10).astype(int)
    df['is_covid_period'] = ((df['Year'] == 2020) | (df['Year'] == 2021)).astype(int)
    df['policy_relax_2023'] = (df['Year'] >= 2023).astype(int)

    # POI aggregate
    print("Aggregating POI...")
    poi_cats = {
        'education': [col for col in df.columns if 'education' in col],
        'medical': [col for col in df.columns if 'medical' in col],
        'transport': [col for col in df.columns if 'transport' in col],
        'leisure': [col for col in df.columns if 'leisure' in col],
        'commercial': [col for col in df.columns if 'commercial' in col]
    }
    for cat, cols in poi_cats.items():
      if cols:
        df[f'total_{cat}'] = df[cols].sum(axis=1)
        df[f'{cat}_mean'] = df[cols].mean(axis=1)
    # Drop original
    poi_drop = [col for col in df.columns if 'dense' in col.lower() and not col.startswith('total_') and col not in ['resident_population_dense']]
    df = df.drop(columns=poi_drop, errors='ignore')

    # Cross-sector weighted
    if 'amount_new_house_transactions_nearby_sectors' in df.columns and 'resident_population_dense' in df.columns:
        df['weighted_nearby_amount'] = df['amount_new_house_transactions_nearby_sectors'] * df['resident_population_dense']

    return df

master_df = create_advanced_features(master_df)

# ==== HANDLE MISSING VALUES (Original) ====
def handle_missing_values(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df.groupby('sector_num')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0))
    df = df.fillna(0); df.replace([np.inf, -np.inf], 0, inplace=True)
    return df

master_df = handle_missing_values(master_df)

# ==== PREPARE TRAINING AND TESTING DATA (Original) ====
def prepare_time_series_split(df, train_end_year=2023, train_end_month=7):
    train_mask = (df['Year'] < train_end_year) | ((df['Year'] == train_end_year) & (df['Month_num'] <= train_end_month))
    df_train = df[train_mask].copy(); df_test = df[~train_mask].copy()
    df_train = df_train.dropna(subset=['target_12m_ahead']); return df_train, df_test

df_train, df_test = prepare_time_series_split(master_df)

def prepare_features(df, target_col='target_12m_ahead'):
    exclude_cols = [target_col, 'date', 'month', 'sector', 'Month', TARGET_COL]
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    X = df[feature_cols]; y = df[target_col]; return X, y, feature_cols

X_train, y_train, feature_cols = prepare_features(df_train)
X_test, y_test, _ = prepare_features(df_test)

y_train_log = np.log1p(y_train); y_test_log = np.log1p(y_test.fillna(0))

scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# ==== CUSTOM METRIC FOR XGB (Updated) ====
def get_custom_score_for_xgb(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    y_true_orig = np.asarray(y_true_orig, dtype=float)
    y_pred_orig = np.asarray(y_pred_orig, dtype=float)
    ape = np.abs(y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + 1e-6)
    ape[np.isinf(ape)] = 0; ape[np.isnan(ape)] = 0
    if np.mean(ape > 1.0) > 0.30:
        return 0.0
    ok_mask = (ape <= 1.0)
    if np.sum(ok_mask) == 0:
        return 0.0
    scaled_mape = np.mean(ape[ok_mask]) / np.mean(ok_mask)
    score = 1.0 - scaled_mape
    return score

def xgb_custom_eval_metric(preds, dtrain):
    labels = dtrain.get_label()
    score = get_custom_score_for_xgb(labels, preds)
    return 'custom_competition_score', score

# ==== PARAMS (No Changes) ====
XGB_PARAMS = {'objective': "reg:squarederror",
              'learning_rate': 0.04930142036355155,
              'max_depth': 7,
              'subsample': 0.7479463764909958,
              'colsample_bytree': 0.9912054217322006,
              'min_child_weight': 10,
              'reg_alpha': 0.0388276999838505,
              'reg_lambda': 0.08053474016385297,
              'random_state': 42,
              'verbosity': 0}

LGB_PARAMS = {'objective': 'huber',
              'n_estimators': 1200, 
              'learning_rate': 0.03632024139954956, 
              'num_leaves': 20, 
              'max_depth': 8, 
              'subsample': 0.7108130739614915,
              'colsample_bytree': 0.730877749280941, 
              'reg_alpha': 0.5581300379523999,
              'reg_lambda': 0.03599441037744031,
              'random_state': 42,
              'n_jobs': -1, 'metric': 'mape', 'callbacks': [lgb.early_stopping(100, verbose=False)]}

CAT_PARAMS = {'iterations': 2000,
              'learning_rate': 0.02,
              'depth': 8,
              'l2_leaf_reg': 3,
              'random_seed': 42,
              'loss_function': 'MAPE',
              'verbose': False}

# ==== TRAIN MODELS (Trees Only) ====
def train_models(X_train, y_train, X_val, y_val):
    # XGB with custom eval
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'val')]
    model_xgb = xgb.train(XGB_PARAMS, dtrain, num_boost_round=1400, 
                          evals=evals, early_stopping_rounds=100, 
                          verbose_eval=False, feval=xgb_custom_eval_metric, maximize=True)
    
    model_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    model_cat = CatBoostRegressor(**CAT_PARAMS)
    model_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    return model_xgb, model_lgb, model_cat

# ==== CV (Trees Only) ====
def time_series_cross_validation(X, y_log, y_original, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ensemble_scores = []

    threshold = 300

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"\n=== Fold {fold + 1} ===")
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_log_cv, y_val_log_cv = y_log.iloc[train_index], y_log.iloc[val_index]
        y_val_original_cv = y_original.iloc[val_index]
        y_train_original_cv = y_original.iloc[train_index]

        # Target encoding
        sector_map = y_train_original_cv.groupby(X_train_cv['sector_num']).mean()
        X_train_cv['sector_target_enc'] = X_train_cv['sector_num'].map(sector_map)
        X_val_cv['sector_target_enc'] = X_val_cv['sector_num'].map(sector_map)

        cluster_map = y_train_original_cv.groupby(X_train_cv['sector_cluster']).mean()
        X_train_cv['cluster_target_enc'] = X_train_cv['sector_cluster'].map(cluster_map)
        X_val_cv['cluster_target_enc'] = X_val_cv['sector_cluster'].map(cluster_map)

        X_train_cv.fillna(y_train_original_cv.mean(), inplace=True)
        X_val_cv.fillna(y_train_original_cv.mean(), inplace=True)
        
        scaler_fold = StandardScaler()
        X_train_s = scaler_fold.fit_transform(X_train_cv)
        X_val_s = scaler_fold.transform(X_val_cv)
        
        model_xgb, model_lgb, model_cat = train_models(X_train_s, y_train_log_cv, X_val_s, y_val_log_cv)
        
        pred_xgb_log = model_xgb.predict(xgb.DMatrix(X_val_s))
        pred_lgb_log = model_lgb.predict(X_val_s)
        pred_cat_log = model_cat.predict(X_val_s)
        
        # Stacking
        val_stack = np.column_stack([pred_xgb_log, pred_lgb_log, pred_cat_log])
        meta = Ridge(alpha=1.0)
        meta.fit(val_stack, y_val_log_cv)
        pred_meta_log = meta.predict(val_stack)
        
        y_pred_ensemble = np.expm1(pred_meta_log)
        y_pred_ensemble[y_pred_ensemble < threshold] = 0
        y_pred_ensemble[y_pred_ensemble < 0] = 0
        
        score_ensemble = custom_competition_score(y_val_original_cv, y_pred_ensemble)
        ensemble_scores.append(score_ensemble)
        
        print(f"Fold {fold + 1} Ensemble Score: {score_ensemble:.4f}")
        
    return np.mean(ensemble_scores)

cv_score_ensemble = time_series_cross_validation(X_train_scaled, y_train_log, y_train, n_splits=5)

print(f"\nAverage Ensemble CV Score: {cv_score_ensemble:.4f}")

# ==== FINAL TRAIN AND SUBMISSION (Trees Only, Full Stacking) ====
print("\n=== Training Final Models ===")
final_model_xgb, final_model_lgb, final_model_cat = train_models(X_train_scaled.values, y_train_log.values, X_test_scaled.values, y_test_log.values)

# Train preds for meta
pred_xgb_train = final_model_xgb.predict(xgb.DMatrix(X_train_scaled.values))
pred_lgb_train = final_model_lgb.predict(X_train_scaled.values)
pred_cat_train = final_model_cat.predict(X_train_scaled.values)

train_preds = np.column_stack([pred_xgb_train, pred_lgb_train, pred_cat_train])

meta_params = {'objective': 'regression_l1', 'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 16, 'max_depth': 4, 'random_state': 42}
final_meta = lgb.LGBMRegressor(**meta_params)
final_meta.fit(train_preds, y_train_log)

# Test preds
pred_xgb_test = final_model_xgb.predict(xgb.DMatrix(X_test_scaled.values))
pred_lgb_test = final_model_lgb.predict(X_test_scaled.values)
pred_cat_test = final_model_cat.predict(X_test_scaled.values)

test_preds = np.column_stack([pred_xgb_test, pred_lgb_test, pred_cat_test])
final_pred_log = final_meta.predict(test_preds)
final_pred = np.expm1(final_pred_log)
final_pred[final_pred < 300] = 0
final_pred[final_pred < 0] = 0

df_test['predicted_amount'] = final_pred

sub_df = all_data['test'].copy()
map_df = df_test[['Year', 'Month_num', 'sector_num', 'predicted_amount']].copy()
mabbr = {i: calendar.month_abbr[i] for i in range(1, 13)}
map_df['id'] = (map_df['Year'].astype(int) + 1).astype(str) + " " + map_df['Month_num'].astype(int).map(mabbr) + "_sector " + map_df['sector_num'].astype(int).astype(str)
sub_df = sub_df.merge(map_df[['id', 'predicted_amount']], on='id', how='left')
sub_df['new_house_transaction_amount'] = sub_df['predicted_amount'].fillna(0)
sub_df[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)

print("Submission created! Check if predictions >0 now.")
gc.collect()

