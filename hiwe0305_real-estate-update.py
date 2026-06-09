!pip -q install "autogluon==1.1.1" --no-cache-dir


# =========================
# Cell 0: Setup & Imports
# =========================
import numpy as np
import pandas as pd

# reproducibility
RANDOM_STATE = 1337
np.random.seed(RANDOM_STATE)

# Kaggle paths
TRAIN_DIR = "/kaggle/input/china-real-estate-demand-prediction/train"
TEST_PATH = "/kaggle/input/china-real-estate-demand-prediction/test.csv"

# Month mapping
MONTH = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
         'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

def add_time_sector(df):
    """Add sector_id, year, month(1..12), time (0..66)"""
    df = df.copy()
    df['sector_id'] = df['sector'].str.extract(r'(\d+)').astype(int)
    df['year']  = df['month'].str.slice(0,4).astype(int)
    df['month'] = df['month'].str.slice(5).map(MONTH)
    df['time']  = (df['year'] - 2019) * 12 + df['month'] - 1
    return df

def parse_test_ids(test_df):
    parts = test_df['id'].str.split('_', expand=True)
    test_df = test_df.copy()
    test_df['year'] = parts[0].str[:4].astype(int)
    test_df['month'] = parts[0].str[5:].map(MONTH)
    test_df['time']  = (test_df['year'] - 2019) * 12 + test_df['month'] - 1
    test_df['sector_id'] = parts[1].str.extract(r'(\d+)').astype(int)
    return test_df

def comp_score(y_true, y_pred, eps=1e-12):
    """Competition metric."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))
    good = ape <= 1.0
    good_rate = good.mean()
    if good_rate < 0.7:
        return 0.0
    mape = ape[good].mean()
    return 1.0 - mape / max(good_rate, 1e-12)


# =========================
# Cell 1: Load & amount matrix
# =========================
# Load training target
nht = pd.read_csv(f"{TRAIN_DIR}/new_house_transactions.csv")
nht = add_time_sector(nht)

# Pivot thành ma trận A[time × sector_id], fill 0
A = nht.pivot_table(index='time', columns='sector_id',
                    values='amount_new_house_transactions', fill_value=0.0)

# đảm bảo đủ 1..96 (nhiều sector vắng mặt trong train)
for s in range(1, 97):
    if s not in A.columns:
        A[s] = 0.0
A = A[sorted(A.columns)]
A.head()


# =========================
# Cell 2: EWGM + December bump + backtest
# =========================
def dec_multipliers(A_hist, eps=1e-9, min_obs=1, clip_low=0.85, clip_high=1.40):
    """Sector-wise Dec vs Non-Dec ratio with clipping & simple shrink."""
    is_dec = (A_hist.index.values % 12) == 11
    dec_mean = A_hist[is_dec].mean(axis=0)
    non_mean = A_hist[~is_dec].mean(axis=0)
    raw = dec_mean / (non_mean + eps)
    overall = float(dec_mean.mean() / (non_mean.mean() + eps))
    dec_cnt = A_hist[is_dec].notna().sum(axis=0)
    raw = raw.where(dec_cnt >= min_obs, overall)
    raw = raw.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return raw.clip(lower=clip_low, upper=clip_high)

def ewgm_one_step(A_hist, n_lags=6, weight='exp', alpha=0.5, t2=6):
    """Predict next step for all sectors via weighted geometric mean, with rule-based zeroing."""
    cols = A_hist.columns
    pred = pd.Series(index=cols, dtype=float)

    # weights
    if weight == 'exp':
        w = np.array([alpha**(n_lags-1-i) for i in range(n_lags)], float)
    elif weight == 'linear':
        w = np.arange(1, n_lags+1, dtype=float)
    elif weight == 'square':
        w = (np.arange(1, n_lags+1, dtype=float)**2)
    else:
        raise ValueError("weight must be 'exp' | 'linear' | 'square'")
    w /= w.sum()

    tail = A_hist.tail(n_lags)
    base_tail_t2 = A_hist.tail(t2)

    for s in cols:
        # baseline zero rule
        if (base_tail_t2[s].min() == 0) or (A_hist[s].sum() == 0):
            pred[s] = 0.0; continue
        v = tail[s].to_numpy()
        if (v > 0).any():
            mask = v > 0
            ww = w[mask]; vv = v[mask]
            ww = ww / ww.sum()
            pred[s] = float(np.exp((ww * np.log(vv)).sum()))
        else:
            pred[s] = 0.0
    return pred

def backtest_6m(A, n_lags=6, weight='exp', alpha=0.5, t2=6,
                clip_low=0.85, clip_high=1.40, val_len=6):
    """1-step rolling backtest trên 6 tháng cuối theo comp_score và RMSE."""
    times = A.index.values
    val_times = times[-val_len:]
    y_all, p_all, rmses = [], [], []
    for t in val_times:
        hist = A.loc[A.index < t]
        y = A.loc[t]
        p = ewgm_one_step(hist, n_lags=n_lags, weight=weight, alpha=alpha, t2=t2)
        if (t % 12) == 11:
            bump = dec_multipliers(hist, clip_low=clip_low, clip_high=clip_high)
            p = p * bump.reindex(p.index).fillna(1.0)
        rmses.append(float(np.sqrt(np.mean((p.values - y.values)**2))))
        y_all.append(y.values); p_all.append(p.values)
    y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
    return float(np.mean(rmses)), comp_score(y_all, p_all)

# Tham số mặc định an toàn
PARAMS = dict(n_lags=6, weight='exp', alpha=0.5, t2=6, clip_low=0.85, clip_high=1.40)
rmse_bt, score_bt = backtest_6m(A, **PARAMS)
print(f"[BACKTEST] rmse={rmse_bt:.2f} | comp_score={score_bt:.4f} | params={PARAMS}")


# =========================
# Cell 3: Predict Test + Submission
# =========================
test = pd.read_csv(TEST_PATH)
test = parse_test_ids(test)

# Base one-step prediction từ lịch sử train (time <= 66)
pred_base = ewgm_one_step(A, n_lags=PARAMS['n_lags'],
                          weight=PARAMS['weight'],
                          alpha=PARAMS['alpha'],
                          t2=PARAMS['t2'])

# December bump vector (tính từ lịch sử)
bump = dec_multipliers(A, clip_low=PARAMS['clip_low'], clip_high=PARAMS['clip_high'])

# Áp cho từng dòng test
is_dec = (test['time'] % 12) == 11
vals = []
for t, s, dec_flag in zip(test['time'].to_numpy(), test['sector_id'].to_numpy(), is_dec.to_numpy()):
    v = float(pred_base.get(s, 0.0))
    if dec_flag:
        v *= float(bump.get(s, 1.0))
    vals.append(v)

sub = test[['id']].copy()
sub['new_house_transaction_amount'] = np.asarray(vals, float)

# Thông tin nhanh
print("Submission shape:", sub.shape,
      "| mean:", float(sub['new_house_transaction_amount'].mean()),
      "| zeros %:", float((sub['new_house_transaction_amount']==0).mean()))

# Save
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved -> /kaggle/working/submission.csv")

