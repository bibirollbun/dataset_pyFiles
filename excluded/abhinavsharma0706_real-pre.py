#  Imports Libraries

import numpy as np
import pandas as pd
import optuna
from functools import partial


#  Month mapping
def build_month_codes():
    return {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

#  Parse id into month text and sector string
def split_test_id_column(df):
    parts = df.id.str.split('_', expand=True)
    df['month_text'] = parts[0]
    df['sector'] = parts[1]
    return df


#  Add parsed time fields to a dataframe
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


#  Load competition tables used for submission
def load_competition_data():
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
    return train_nht, test


#  Build training matrix: amount_new_house_transactions [time x sector_id]
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

#  Compute sector-level December multipliers from training
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


#  Apply December bump on a specific row vector
def apply_december_bump_row(pred_row, sector_to_mult):
    bumped = pred_row.copy()
    for sector in bumped.index:
        m = sector_to_mult.get(sector, 1.0)
        bumped.loc[sector] = bumped.loc[sector] * m
    return bumped

#  Exponential weighted geometric mean per sector
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


#  Predict a single next-step vector for all sectors using current history
def predict_one_step(a_hist, n_lags, alpha):
    cols = a_hist.columns
    pred = pd.Series(index=cols, dtype=float)
    for sector in cols:
        if (a_hist.tail(n_lags)[sector].min() == 0) or (a_hist[sector].sum() == 0):
            pred.loc[sector] = 0.0
            continue
        base = ewgm_per_sector(a_tr=a_hist, sector=sector, n_lags=n_lags, alpha=alpha)
        pred.loc[sector] = base
    return pred


#  Apply December bump on a specific row vector
def apply_december_bump_row(pred_row, sector_to_mult):
    bumped = pred_row.copy()
    for sector in bumped.index:
        m = sector_to_mult.get(sector, 1.0)
        bumped.loc[sector] = bumped.loc[sector] * m
    return bumped

#  Exponential weighted geometric mean per sector
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

#  Predict a single next-step vector for all sectors using current history
def predict_one_step(a_hist, n_lags, alpha):
    cols = a_hist.columns
    pred = pd.Series(index=cols, dtype=float)
    for sector in cols:
        if (a_hist.tail(n_lags)[sector].min() == 0) or (a_hist[sector].sum() == 0):
            pred.loc[sector] = 0.0
            continue
        base = ewgm_per_sector(a_tr=a_hist, sector=sector, n_lags=n_lags, alpha=alpha)
        pred.loc[sector] = base
    return pred

#  Backtest loss over the last val_len months using rolling origin
def evaluate_params(a_tr_full, n_lags, alpha, t2, clip_low, clip_high, val_len=6):
    times = a_tr_full.index.values
    if len(times) < max(n_lags + 1, t2 + 1) + val_len:
        return 1e12
    val_times = times[-val_len:]
    rmse_list = []
    for t in val_times:
        a_hist = a_tr_full.loc[a_tr_full.index < t]
        if a_hist.shape[0] < max(n_lags, t2):
            continue
        y_true = a_tr_full.loc[t]
        y_pred = predict_one_step(a_hist=a_hist, n_lags=n_lags, alpha=alpha)
        if (t % 12) == 11:
            sector_to_mult = compute_december_multipliers(
                a_tr=a_hist,
                eps=1e-9,
                min_dec_obs=1,
                clip_low=clip_low,
                clip_high=clip_high
            )
            y_pred = apply_december_bump_row(pred_row=y_pred, sector_to_mult=sector_to_mult)
        diff = y_pred.values - y_true.values
        rmse = float(np.sqrt(np.mean(diff * diff)))
        rmse_list.append(rmse)
    if len(rmse_list) == 0:
        return 1e12
    return float(np.mean(rmse_list))




#  Build horizon predictions [time=67..78 x sectors]
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

#  Convert wide predictions into submission aligned with test ids
def build_submission_df(a_pred, test_raw, month_codes):
    test = split_test_id_column(test_raw.copy())
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    out = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return out

#  End-to-end generation with December bump
def generate_submission_with_december_bump(n_lags=6, alpha=0.5, t2=6, clip_low=0.85, clip_high=1.4):
    month_codes = build_month_codes()
    train_nht, test = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    a_pred = predict_horizon(a_tr=a_tr, n_lags=n_lags, alpha=alpha, t2=t2)
    sector_to_mult = compute_december_multipliers(
        a_tr=a_tr,
        eps=1e-9,
        min_dec_obs=1,
        clip_low=clip_low,
        clip_high=clip_high
    )
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    if len(dec_rows) > 0:
        for r in dec_rows:
            a_pred.loc[r] = apply_december_bump_row(pred_row=a_pred.loc[r], sector_to_mult=sector_to_mult)
    submission = build_submission_df(a_pred=a_pred, test_raw=test, month_codes=month_codes)
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    return a_tr, a_pred, submission

#  Optuna objective to minimize validation RMSE
def optuna_objective(trial, a_tr):
    n_lags = trial.suggest_int('n_lags', 3, 12)
    alpha = trial.suggest_float('alpha', 0.20, 0.95)
    t2 = trial.suggest_int('t2', 3, 9)
    clip_low = trial.suggest_float('clip_low', 0.70, 0.95)
    clip_high = trial.suggest_float('clip_high', 1.10, 1.80)
    if clip_low >= clip_high:
        clip_low = max(0.70, clip_high - 0.05)
    loss = evaluate_params(
        a_tr_full=a_tr,
        n_lags=n_lags,
        alpha=alpha,
        t2=t2,
        clip_low=clip_low,
        clip_high=clip_high,
        val_len=6
    )
    return loss

#  Run Optuna to search best parameters
def run_optuna_search(a_tr, n_trials, seed):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    objective_fn = partial(optuna_objective, a_tr=a_tr)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study

#  Main
def main():
    month_codes = build_month_codes()
    train_nht, _ = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    study = run_optuna_search(a_tr=a_tr, n_trials=4096, seed=1337)
    best = study.best_trial.params
    n_lags = int(best.get('n_lags', 7))
    alpha = float(best.get('alpha', 0.35))
    t2 = int(best.get('t2', 4))
    clip_low = float(best.get('clip_low', 0.80))
    clip_high = float(best.get('clip_high', 1.50))
    a_tr, a_pred, submission = generate_submission_with_december_bump(
        n_lags=n_lags,
        alpha=alpha,
        t2=t2,
        clip_low=clip_low,
        clip_high=clip_high
    )
    print('Best params:', best)
    print('Submission with tuned parameters saved to /kaggle/working/submission.csv')

#  Entry point
if __name__ == '__main__':
    main()

