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


# Kaggle-ready Hybrid Pipeline: EWGM + December bump + LightGBM blending

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# ---------------------------
#  Utility functions from EWGM model
# ---------------------------
def build_month_codes():
    return {
        'Jan': 1,'Feb': 2,'Mar': 3,'Apr': 4,'May': 5,'Jun': 6,
        'Jul': 7,'Aug': 8,'Sep': 9,'Oct': 10,'Nov': 11,'Dec': 12
    }

def split_test_id_column(df):
    parts = df.id.str.split('_', expand=True)
    df['month_text'] = parts[0]
    df['sector'] = parts[1]
    return df

def add_time_and_sector_fields(df, month_codes):
    if 'sector' in df.columns:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if 'month' not in df.columns:
        df['month'] = df['month_text'].str.slice(5, None).map(month_codes)
        df['year'] = df['month_text'].str.slice(0, 4).astype(int)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    else:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df

def load_competition_data():
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
    return train_nht, test

def build_amount_matrix(train_nht, month_codes):
    train_nht = add_time_and_sector_fields(train_nht.copy(), month_codes)
    pivot = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
    pivot = pivot.fillna(0)
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sectors]
    return pivot

def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.8, clip_high=1.5):
    is_december = (a_tr.index.values % 12) == 11
    dec_means = a_tr[is_december].mean(axis=0)
    nondec_means = a_tr[~is_december].mean(axis=0)
    dec_counts = a_tr[is_december].notna().sum(axis=0)
    raw_mult = dec_means / (nondec_means + eps)
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))
    raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
    raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    clipped_mult = raw_mult.clip(lower=clip_low, upper=clip_high)
    return clipped_mult.to_dict()

def apply_december_bump(a_pred, sector_to_mult):
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    for sector in a_pred.columns:
        m = sector_to_mult.get(sector, 1.0)
        a_pred.loc[dec_rows, sector] = a_pred.loc[dec_rows, sector] * m
    return a_pred

def ewgm_per_sector(a_tr, sector, n_lags, alpha):
    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)], dtype=float)
    weights = weights / weights.sum()
    recent_vals = a_tr.tail(n_lags)[sector].values
    if (len(recent_vals) != n_lags) or (recent_vals <= 0).all():
        return 0.0
    mask = recent_vals > 0
    pos_vals = recent_vals[mask]
    pos_w = weights[mask]
    if pos_vals.size == 0:
        return 0.0
    pos_w = pos_w / pos_w.sum()
    log_vals = np.log(pos_vals + 1e-12)
    wlm = np.sum(pos_w * log_vals) / pos_w.sum()
    return float(np.exp(wlm))

def predict_horizon(a_tr, n_lags, alpha, t2):
    idx = np.arange(67, 79)
    cols = a_tr.columns
    a_pred = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for sector in cols:
        if (a_tr.tail(t2)[sector].min() == 0) or (a_tr[sector].sum() == 0):
            a_pred[sector] = 0.0
            continue
        base = ewgm_per_sector(a_tr=a_tr, sector=sector, n_lags=n_lags, alpha=alpha)
        a_pred[sector] = base
    a_pred.index.rename('time', inplace=True)
    return a_pred

def build_submission_df(a_pred, test_raw, month_codes):
    test = split_test_id_column(test_raw.copy())
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    out = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return out

# ---------------------------
# LightGBM simple lag model
# ---------------------------
def run_lightgbm_predictions():
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
    if 'amount_new_house_transactions' in train_nht.columns and 'new_house_transaction_amount' not in train_nht.columns:
        train_nht = train_nht.rename(columns={'amount_new_house_transactions':'new_house_transaction_amount'})
    
    train_nht['month_dt'] = pd.to_datetime(train_nht['month'], errors='coerce')
    test['month_text'] = test['id'].apply(lambda x: str(x).split('_',1)[0].strip())
    test['sector'] = test['id'].apply(lambda x: str(x).split('_',1)[1].strip())
    try:
        test['month_dt'] = pd.to_datetime(test['month_text'], format='%Y %b')
    except:
        test['month_dt'] = pd.to_datetime(test['month_text'] + '-01', errors='coerce')
    
    base = train_nht[['month_dt','sector','new_house_transaction_amount']].copy()
    base = base.sort_values(['sector','month_dt']).reset_index(drop=True)
    for l in [1,2,3,6]:
        base[f'lag{l}'] = base.groupby('sector')['new_house_transaction_amount'].shift(l)
    base['roll3'] = base.groupby('sector')['new_house_transaction_amount'].shift(1).rolling(3, min_periods=1).mean().reset_index(0,drop=True)
    
    train = base[base['new_house_transaction_amount'].notna()].copy()
    train['target'] = train['new_house_transaction_amount'].astype(float)
    test_feat = test.merge(base, on=['month_dt','sector'], how='left')
    
    exclude = ['month','month_dt','sector','id','new_house_transaction_amount','target','month_text']
    features = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train = pd.DataFrame(imp.fit_transform(train[features]), columns=features, index=train.index)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test = pd.DataFrame(imp.transform(test_feat[features]), columns=features, index=test_feat.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
    
    y = np.log1p(train['target'].clip(lower=0))
    
    params = {
        'objective':'regression','learning_rate':0.05,'num_leaves':64,
        'min_data_in_leaf':20,'feature_fraction':0.8,'bagging_fraction':0.8,
        'bagging_freq':5,'seed':42,'verbosity':-1,'metric':'rmse'
    }
    dtrain = lgb.Dataset(X_train, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    return pred

# ---------------------------
# Hybrid pipeline
# ---------------------------
def generate_hybrid_submission():
    month_codes = build_month_codes()
    train_nht, test = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    a_pred = predict_horizon(a_tr=a_tr, n_lags=7, alpha=0.35, t2=4)
    sector_to_mult = compute_december_multipliers(a_tr=a_tr, clip_low=0.8, clip_high=1.5)
    a_pred = apply_december_bump(a_pred=a_pred, sector_to_mult=sector_to_mult)
    sub_ewgm = build_submission_df(a_pred=a_pred, test_raw=test, month_codes=month_codes)
    
    pred_lgbm = run_lightgbm_predictions()
    
    # Blend
    w = 0.4  # weight for LGBM
    final_preds = (1-w) * sub_ewgm['new_house_transaction_amount'].values + w * pred_lgbm
    # Safety clipping vs EWGM baseline
    base_preds = sub_ewgm['new_house_transaction_amount'].values
    final_preds = np.minimum(final_preds, base_preds * 2.5)
    final_preds = np.maximum(final_preds, base_preds * 0.4)
    
    submission = sub_ewgm.copy()
    submission['new_house_transaction_amount'] = final_preds
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print('Hybrid submission saved to /kaggle/working/submission.csv')

if __name__ == '__main__':
    generate_hybrid_submission()


