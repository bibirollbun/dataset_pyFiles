import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
from tqdm.notebook import tqdm
from scipy.optimize import minimize
from scipy.stats import pearsonr
import random

# --------------------------
# Custom scoring function
# --------------------------
def custom_score(y_true, y_pred, eps=1e-12):
    """Scoring function of the competition."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.size == 0:
        raise ValueError('empty array')
    if (y_true < 0).any():
        raise ValueError('negative y_true')
    if (~np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')
    
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))
    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}
    
    good_ape = ape[good_mask]
    mape = np.mean(good_ape)
    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}


# --------------------------
# Load datasets
# --------------------------
ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv')
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv')
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv')
train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

# --------------------------
# Preprocessing
# --------------------------
month_codes = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id
for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1

amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack().fillna(0)
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)].astype(float)

# Seasonality (keep your global version to avoid feature creep)
seasonal = train_nht.groupby('month')['amount_new_house_transactions'].mean()
seasonal = (seasonal / seasonal.mean()).clip(0.5, 2.0).astype(float)

csi_agg = csi.groupby('time')['search_volume'].mean()
csi_trend = csi_agg.tail(6).mean() / csi_agg.tail(12).mean() if not csi_agg.empty else 1.0
csi_trend = float(np.clip(csi_trend, 0.5, 2.0))

# --------------------------
# Grid search for models (same shapes you used)
# --------------------------
t1_options = [1, 3, 6, 9]
t2_options = [1, 2, 3]
csi_scale_options = [0.8, 0.9, 1.0, 1.1]
cv = TimeSeriesSplit(n_splits=4, test_size=12)

top_models = []
print("Starting grid search...")

# We'll also track horizon per validation index across folds
horizon_map_parts = []

for t1 in tqdm(t1_options, desc="Grid Search"):
    weight_options = [
        np.ones(t1),                                       # Equal weights
        np.linspace(1.5, 0.5, t1),                         # Linear decay
        np.array([0.5] + [0.5/(t1-1)]*(t1-1)) if t1 > 1 else np.array([1.0]), # Heavy recent weight
        np.exp(np.linspace(0, -2, t1)),                    # Exponential decay
        1 / (np.arange(1, t1+1)),                          # Inverse rank
        np.exp(-((np.arange(t1) - t1/2)**2) / (2*(t1/4)**2)), # Gaussian center emphasis
        np.concatenate([np.repeat(1.0, t1//2), np.repeat(0.5, t1 - t1//2)]) # Plateau then drop
    ]
    weights_list = [w / w.sum() for w in weight_options]
    
    for t2, weights, csi_scale in product(t2_options, weights_list, csi_scale_options):
        oof_preds, true_values = [], []
        horizons_this_model = []  # (not needed per-model; weâ€™ll build horizons once outside)
        first_fold = True
        for idx_tr, idx_va in cv.split(amount_new_house_transactions):
            a_tr, a_va = amount_new_house_transactions.iloc[idx_tr], amount_new_house_transactions.iloc[idx_va]
            a_tr_safe = a_tr.tail(t1).where(a_tr.tail(t1) > 0, 1.0)
            pred_values = np.exp(np.average(np.log(a_tr_safe), weights=weights, axis=0))
            pred_values = np.clip(pred_values, 0, 1e6)
            
            a_pred = pd.DataFrame(
                np.array([pred_values * seasonal[int(time % 12 + 1)] * (csi_trend * csi_scale) for time in idx_va]),
                index=idx_va, columns=a_tr.columns
            ).clip(0, 1e6)
            a_pred.loc[:, a_tr.columns[a_tr.tail(t2).min(axis=0) == 0]] = 0
            a_pred.index.rename('time', inplace=True)
            
            oof_preds.append(a_pred)
            true_values.append(a_va)

            # Collect horizon info ONCE (based on split, independent of model)
            if first_fold:
                last_train_t = amount_new_house_transactions.index[idx_tr][-1]
                h = pd.Series((idx_va - last_train_t).astype(int), index=idx_va)  # 1..12
                horizon_map_parts.append(h)
            first_fold = False
        
        # Score
        score = custom_score(pd.concat(true_values), pd.concat(oof_preds))['score']
        top_models.append({
            'score': float(score),
            'params': {'t1': t1, 't2': t2, 'weights': weights, 'csi_scale': csi_scale},
            'oof_preds': pd.concat(oof_preds)
        })

# Build the global horizon map aligned to OOF indices
horizon_map = pd.concat(horizon_map_parts).astype(int)  # index = validation time, value = horizon 1..12

# --------------------------
# Select diverse pool (top + correlation-penalized weaker)
# --------------------------
def safe_corr(a, b):
    try:
        c, _ = pearsonr(a, b)
        if np.isnan(c):
            return 1.0
        return float(c)
    except Exception:
        return 1.0

top_models_sorted = sorted(top_models, key=lambda x: x['score'], reverse=True)
strong_models = top_models_sorted[:20]

weak_candidates = top_models_sorted[20:100]
lambda_corr = 0.1
diversity_rank = []
for m in weak_candidates:
    m_flat = m['oof_preds'].values.flatten()
    corrs = [safe_corr(m_flat, s['oof_preds'].values.flatten()) for s in strong_models]
    mean_corr = float(np.mean(corrs)) if len(corrs) else 1.0
    adjusted = m['score'] - lambda_corr * mean_corr
    diversity_rank.append((m, adjusted, mean_corr))

diverse_weak_models = [m for (m, _, _) in sorted(diversity_rank, key=lambda x: x[1], reverse=True)[:22]]
diverse_pool = strong_models + diverse_weak_models
print(f"Selected pool size: {len(diverse_pool)} ({len(diverse_pool) - len(diverse_weak_models)} strong + {len(diverse_weak_models)} diverse-weak)")

# --------------------------
# Collect OOF preds and true values
# --------------------------
true_values_full = pd.concat([
    amount_new_house_transactions.iloc[idx_va] 
    for _, idx_va in cv.split(amount_new_house_transactions)
]).reset_index(drop=True)

oof_predictions = [
    m['oof_preds'].reset_index(drop=True) for m in diverse_pool
]

# Build horizon map (steps ahead relative to validation set)
horizon_map = pd.Series(
    np.arange(len(true_values_full)), 
    index=true_values_full.index
)
n_models = len(diverse_pool)

# --------------------------
# Horizon bucket optimizer (linear ensemble, per-bucket)
# --------------------------
def objective_linear(weights, preds_list, y_true_df, alpha_l2=0.02, beta_corr=0):
    # Combine predictions
    ensembled = sum(w * p for w, p in zip(weights, preds_list))
    score = custom_score(y_true_df, ensembled)['score']
    # Correlation penalty weighted by w_i w_j
    preds_matrix = np.array([p.values.flatten() for p in preds_list])
    corr_matrix = np.corrcoef(preds_matrix)
    np.fill_diagonal(corr_matrix, 0.0)
    w_outer = np.outer(weights, weights)
    corr_pen = np.sum(np.abs(corr_matrix) * w_outer) / (np.sum(w_outer) + 1e-12)
    l2_pen = np.sum(weights**2)
    return -(score - (beta_corr * corr_pen + alpha_l2 * l2_pen))

def optimize_bucket(mask_idx, label):
    if len(mask_idx) == 0:
        # Fallback to equal weights if bucket is empty (shouldn't happen with 12-month folds)
        return np.ones(n_models) / n_models, 0.0
    y_bucket = true_values_full.loc[mask_idx]
    preds_bucket = [p.loc[mask_idx] for p in oof_predictions]

    bounds = [(0.0, 1.0)] * n_models
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    init = np.ones(n_models) / n_models

    res = minimize(
        lambda w: objective_linear(w, preds_bucket, y_bucket, alpha_l2=0.02, beta_corr=0),
        init,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 400, 'ftol': 1e-9}
    )
    best_w = res.x
    best_score = -objective_linear(best_w, preds_bucket, y_bucket, alpha_l2=0.02, beta_corr=0)
    print(f"{label} bucket OOF score: {best_score:.6f}")
    return best_w, best_score

# Correctly define the horizon buckets
horizon_values = pd.Series(np.tile(np.arange(1, 13), 4), index=true_values_full.index)

idx_short = horizon_values.index[horizon_values <= 4]        # Months 1-4
idx_mid   = horizon_values.index[horizon_values.between(5, 8)] # Months 5-8
idx_long  = horizon_values.index[horizon_values > 8]         # Months 9-12

# Optimize each bucket with the correct data
w_short, _ = optimize_bucket(idx_short, "Short (H 1-4)")
w_mid,   _ = optimize_bucket(idx_mid,   "Mid (H 5-8)")
w_long,  _ = optimize_bucket(idx_long,  "Long (H 9-12)")

# --------------------------
# Final prediction (row-wise weights by horizon)
# --------------------------
# Build each model's test predictions once
horizon = np.arange(67, 79)  # 12 months
test_predictions = []
for model in diverse_pool:
    params = model['params']
    t1, t2, weights_model, csi_scale = params['t1'], params['t2'], params['weights'], params['csi_scale']
    a_tr = amount_new_house_transactions
    a_tr_safe = a_tr.tail(t1).where(a_tr.tail(t1) > 0, 1.0)
    pred_values = np.exp(np.average(np.log(a_tr_safe), weights=weights_model, axis=0))
    pred_values = np.clip(pred_values, 0, 1e6)
    a_pred = pd.DataFrame(
        np.array([pred_values * seasonal[int(t % 12 + 1)] * (csi_trend * csi_scale) for t in horizon]),
        index=horizon, columns=a_tr.columns
    ).clip(0, 1e6)
    a_pred.loc[:, a_tr.columns[a_tr.tail(t2).min(axis=0) == 0]] = 0
    a_pred.index.rename('time', inplace=True)
    test_predictions.append(a_pred)

# Combine per horizon with the corresponding weights
rows = []
for i, t in enumerate(horizon):
    h = i + 1  # 1..12
    if h <= 4:
        w = w_short
    elif h <= 8:
        w = w_mid
    else:
        w = w_long
    row = sum(w_k * pred.loc[t] for w_k, pred in zip(w, test_predictions))
    rows.append(row)

final_prediction_df = pd.DataFrame(rows, index=horizon, columns=amount_new_house_transactions.columns)

# Save submission
test['new_house_transaction_amount'] = final_prediction_df.T.unstack().values
test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)

print("\nFirst few lines of submission.csv:")
with open('submission.csv', 'r') as f:
    for _ in range(5):
        print(f.readline().strip())

