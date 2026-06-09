import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

from sklearn.model_selection import TimeSeriesSplit

def custom_score(y_true, y_pred, eps=1e-12):
    """Scoring function of the competition as defined on the competition overview page.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    eps : float, optional (exact value doesn't matter)

    Return value:
    -------------
    dict with keys 'score', 'good_rate' and 'str'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
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
    # score = max(0.0, score)
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}


df = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
df.head()


df_test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
df_test['id'][1151]


# We read all the data although this baseline notebook ignores most of it
# We convert the string-encoded months to integer values (time is 0..66 for train and 67..78 for test)

ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = {
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

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        # print(df.sector_id.min(), df.sector_id.max(), len(np.unique(df.sector_id)), len(df))
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1 # min=0, max=66
        print(df['time'].min(), df['time'].max())


amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
# Missing values must be filled with zero:
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
# We add sector 95, which has no transactions during the training period:
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions.astype(int)


t1 = 6 # months for geometric mean
t2 = 6 # months which must be nonzero
cv = TimeSeriesSplit(n_splits=4, test_size=12)
true, oof = [], []
for fold, (idx_tr, idx_va) in enumerate(cv.split(amount_new_house_transactions)):
    print(f"# Fold {fold}: train on months {idx_tr.min()}..{idx_tr.max()}, validate on months {idx_va.min()}..{idx_va.max()}")
    a_tr = amount_new_house_transactions.iloc[idx_tr]
    a_va = amount_new_house_transactions.iloc[idx_va]

    a_pred = pd.DataFrame(
        {time: np.exp(np.log(a_tr.tail(t1)).mean(axis=0)) for time in idx_va}
    ).T
    a_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0
    a_pred.index.rename('time', inplace=True)
    # display(a_pred.astype(int))
    print(f"# Fold {fold}: {custom_score(a_va, a_pred)['str']}\n")
    true.append(a_va)
    oof.append(a_pred)

print(f"# Overall {custom_score(pd.concat(true), pd.concat(oof))['str']} {t1=} {t2=}\n")
# Fold 0: train on months 0..18, validate on months 19..30
# Fold 0: score=0.391 good_rate=0.941

# Fold 1: train on months 0..30, validate on months 31..42
# Fold 1: score=0.440 good_rate=0.759

# Fold 2: train on months 0..42, validate on months 43..54
# Fold 2: score=0.479 good_rate=0.840

# Fold 3: train on months 0..54, validate on months 55..66
# Fold 3: score=0.511 good_rate=0.803

# Overall score=0.447 good_rate=0.836 t1=6 t2=6


a_tr = amount_new_house_transactions
a_pred = pd.DataFrame(
    {time: a_tr.tail(t1).mean(axis=0) for time in np.arange(67, 79)}
).T
a_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0
a_pred.index.rename('time', inplace=True)
display(a_pred.astype(int))


test['new_house_transaction_amount'] = a_pred.T.unstack().values

test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)
!head submission.csv




