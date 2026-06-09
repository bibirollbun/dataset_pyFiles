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


# ---------------------------
#  Imports
# ---------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor


# ---------------------------
#  Month mapping
# ---------------------------
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


# ---------------------------
#  Parse id into month text and sector string
# ---------------------------
def split_test_id_column(df):
    parts = df.id.str.split('_', expand=True)
    df['month_text'] = parts[0]
    df['sector'] = parts[1]
    return df


# ---------------------------
#  Add parsed time fields to a dataframe
# ---------------------------
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


# ---------------------------
#  Load competition tables used for submission
# ---------------------------
def load_competition_data():
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
    return train_nht, test


# ---------------------------
#  Build training matrix: amount_new_house_transactions [time x sector_id]
# ---------------------------
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


# ---------------------------
#  Compute sector-level December multipliers from training
# ---------------------------
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


# ---------------------------
#  Apply December bump on the forecast horizon
# ---------------------------
def apply_december_bump(a_pred, sector_to_mult):
    dec_rows = [t for t in a_pred.index.values if (t % 12) == 11]
    if len(dec_rows) == 0:
        return a_pred
    for sector in a_pred.columns:
        m = sector_to_mult.get(sector, 1.0)
        a_pred.loc[dec_rows, sector] = a_pred.loc[dec_rows, sector] * m
    return a_pred


# ---------------------------
#  Exponential weighted geometric mean per sector
# ---------------------------
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


# ---------------------------
#  Create features for SVM training
# ---------------------------
def create_svm_features(a_tr, n_lags=12):
    """Create features for SVM model using lagged values"""
    X, y = [], []
    
    for i in range(n_lags, len(a_tr)):
        # Features: lagged values for all sectors
        features = []
        for lag in range(1, n_lags + 1):
            features.extend(a_tr.iloc[i - lag].values)
        
        # Target: current values for all sectors
        targets = a_tr.iloc[i].values
        
        X.append(features)
        y.append(targets)
    
    return np.array(X), np.array(y)


# ---------------------------
#  SVM (RBF) prediction for horizon
# ---------------------------
def predict_with_svm_rbf(a_tr, n_lags=12, C=1.0, gamma='scale'):
    """
    Use SVM with RBF kernel for multi-output regression
    """
    # Prepare features and targets
    X, y = create_svm_features(a_tr, n_lags)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train multi-output SVM with RBF kernel
    svm_model = MultiOutputRegressor(
        SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.1),
        n_jobs=-1
    )
    svm_model.fit(X_scaled, y)
    
    # Generate predictions for horizon (67-78)
    predictions = []
    current_features = a_tr.iloc[-n_lags:].values.flatten().reshape(1, -1)
    
    for step in range(12):  # 12 months forecast
        # Scale current features
        current_scaled = scaler.transform(current_features)
        
        # Predict next step
        pred_step = svm_model.predict(current_scaled)[0]
        predictions.append(pred_step)
        
        # Update features for next prediction (shift window)
        if step < 11:
            current_features = np.roll(current_features, -96)  # 96 sectors
            current_features[0, -96:] = pred_step
    
    # Create prediction dataframe
    idx = np.arange(67, 79)
    a_pred = pd.DataFrame(
        predictions, 
        index=idx, 
        columns=a_tr.columns
    )
    a_pred.index.rename('time', inplace=True)
    
    return a_pred


# ---------------------------
#  Ensemble predictions (SVM + baseline)
# ---------------------------
def ensemble_predictions(svm_pred, baseline_pred, svm_weight=0.7):
    """Combine SVM and baseline predictions"""
    return svm_weight * svm_pred + (1 - svm_weight) * baseline_pred


# ---------------------------
#  Build horizon predictions [time=67..78 x sectors]
# ---------------------------
def predict_horizon(a_tr, n_lags, alpha, t2, use_svm=True, svm_weight=0.7):
    idx = np.arange(67, 79)
    cols = a_tr.columns
    
    if use_svm and len(a_tr) >= 24:  # Only use SVM if we have enough data
        # SVM prediction
        svm_pred = predict_with_svm_rbf(
            a_tr, 
            n_lags=min(12, len(a_tr) - 13),  # Adjust n_lags based on available data
            C=1.0, 
            gamma='scale'
        )
        
        # Baseline prediction
        baseline_pred = pd.DataFrame(index=idx, columns=cols, dtype=float)
        for sector in cols:
            if (a_tr.tail(t2)[sector].min() == 0) or (a_tr[sector].sum() == 0):
                baseline_pred[sector] = 0.0
                continue
            base_last_value = a_tr[sector].to_numpy()[-1]
            base_ewgm = ewgm_per_sector(a_tr=a_tr, sector=sector, n_lags=n_lags, alpha=alpha)
            baseline_pred[sector] = (base_last_value + base_ewgm) / 2
        
        # Ensemble
        a_pred = ensemble_predictions(svm_pred, baseline_pred, svm_weight)
    else:
        # Fall back to baseline only
        a_pred = pd.DataFrame(index=idx, columns=cols, dtype=float)
        for sector in cols:
            if (a_tr.tail(t2)[sector].min() == 0) or (a_tr[sector].sum() == 0):
                a_pred[sector] = 0.0
                continue
            base_last_value = a_tr[sector].to_numpy()[-1]
            base_ewgm = ewgm_per_sector(a_tr=a_tr, sector=sector, n_lags=n_lags, alpha=alpha)
            a_pred[sector] = (base_last_value + base_ewgm) / 2
    
    a_pred.index.rename('time', inplace=True)
    return a_pred


# ---------------------------
#  Convert wide predictions into submission aligned with test ids
# ---------------------------
def build_submission_df(a_pred, test_raw, month_codes):
    test = split_test_id_column(test_raw.copy())
    test = add_time_and_sector_fields(test, month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, how='left', on=['time', 'sector_id'])
    merged['pred'] = merged['pred'].fillna(0.0)
    out = merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})
    return out


# ---------------------------
#  End-to-end generation with December bump and SVM
# ---------------------------
def generate_submission_with_svm_december_bump(n_lags=6, alpha=0.5, t2=6, clip_low=0.85, clip_high=1.4, 
                                             use_svm=True, svm_weight=0.7, svm_C=1.0, svm_gamma='scale'):
    month_codes = build_month_codes()
    train_nht, test = load_competition_data()
    a_tr = build_amount_matrix(train_nht, month_codes)
    
    a_pred = predict_horizon(
        a_tr=a_tr, 
        n_lags=n_lags, 
        alpha=alpha, 
        t2=t2, 
        use_svm=use_svm, 
        svm_weight=svm_weight
    )
    
    sector_to_mult = compute_december_multipliers(
        a_tr=a_tr, 
        eps=1e-9, 
        min_dec_obs=1, 
        clip_low=clip_low, 
        clip_high=clip_high
    )
    
    a_pred = apply_december_bump(a_pred=a_pred, sector_to_mult=sector_to_mult)
    submission = build_submission_df(a_pred=a_pred, test_raw=test, month_codes=month_codes)
    
    return a_tr, a_pred, submission


# ---------------------------
#  Generate final submission with SVM
# ---------------------------
a_tr, a_pred, submission = generate_submission_with_svm_december_bump(
    n_lags=7,
    alpha=0.5,
    t2=6,
    clip_low=0.85,
    clip_high=1.4,
    use_svm=True,
    svm_weight=0.7,  # 70% weight to SVM, 30% to baseline
    svm_C=1.0,
    svm_gamma='scale'
)

print('Submission with SVM (RBF) and December seasonality saved to /kaggle/working/submission.csv')
print(f'Final submission shape: {submission.shape}')
print(f'Prediction range: [{submission["new_house_transaction_amount"].min():.2f}, {submission["new_house_transaction_amount"].max():.2f}]')

submission.to_csv('/kaggle/working/submission.csv', index=False)

