
import os
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 50)
pd.set_option('display.precision', 8)




SAMPLE_PATH = Path('/kaggle/input/playground-series-s5e11/sample_submission.csv')
BASE_SUBMISSIONS = {
    's92600': Path('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92600.csv'),
    's92620': Path('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92620.csv'),
    's92668': Path('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92668.csv'),
    's92603': Path('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92603.csv'),
    's92632': Path('/kaggle/input/8-november-2025-ps-s5e11/submission 0.92632.csv'),
    's92643': Path('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92643.csv'),
    's92684': Path('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92684.csv'),
    's92683': Path('/kaggle/input/03-november-2025-ps-s5e11/submission 0.92683.csv'),
    's92732': Path('/kaggle/input/submission-things/submission .927.csv'),
    's92731': Path('/kaggle/input/submission-things/submission .92.csv'),
    's92730': Path('/kaggle/input/4-november-2025-ps-s5e11/submission 0.92730.csv'),

}
LB_SCORES = {
    's92730': 0.92730,
    's92684': 0.92684,
    's92683': 0.92683,
    's92732': 0.92732,
    's92668': 0.92668,
    's92643': 0.92643,
    's92632': 0.92632,
    's92620': 0.92620,
    's92603': 0.92603,
    's92731': 0.92731,
    's92600': 0.92600,
}
assert SAMPLE_PATH.exists(), 'Sample submission not found'
for name, path in BASE_SUBMISSIONS.items():
    if not path.exists():
        raise FileNotFoundError(f'{name} missing: {path}')
print(f'Total base files: {len(BASE_SUBMISSIONS)}')




def read_submission(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=['id', 'loan_paid_back'])
    return df.sort_values('id').reset_index(drop=True)['loan_paid_back']

sample = pd.read_csv(SAMPLE_PATH).sort_values('id').reset_index(drop=True)
stack = {}
for name, path in BASE_SUBMISSIONS.items():
    stack[name] = read_submission(path)

pred_matrix = pd.DataFrame(stack)
assert pred_matrix.shape[0] == sample.shape[0]
pred_matrix.head()




descr = pred_matrix.describe().T
lb_col = descr.index.to_series().map(LB_SCORES)
descr['lb_score'] = lb_col.values
corr = pred_matrix.corr(method='spearman')
print('Per model stats:')
descr[['mean','std','min','max','lb_score']]




def safe_prob(arr, eps=1e-5):
    return np.clip(arr, eps, 1 - eps)

def logit(p):
    p = safe_prob(p)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

mean_prob = pred_matrix.mean(axis=1)
median_prob = pred_matrix.median(axis=1)
rank_vote = pred_matrix.rank(axis=0, pct=True)
rank_prob = rank_vote.mean(axis=1)
spread = pred_matrix.max(axis=1) - pred_matrix.min(axis=1)

gain = np.interp(rank_prob, [rank_prob.min(), rank_prob.max()], [0.25, 0.85])
logit_mean = logit(mean_prob)
logit_rank = logit(rank_prob)
logit_median = logit(median_prob)
meta_score = (
    0.58 * logit_mean +
    0.30 * logit_rank +
    0.12 * logit_median -
    0.90 * spread +
    0.35 * gain
)

rng = np.random.default_rng(20241108)
uncertainty = spread.rank(pct=True)
noise = rng.normal(loc=0.0, scale=1e-4, size=len(meta_score)) * (1 - uncertainty)
final_prob = np.clip(sigmoid(meta_score) + noise, 1e-5, 1 - 1e-5)

blend = sample[['id']].copy()
blend['loan_paid_back'] = final_prob
blend.head()




output_path = Path('submission.csv')
blend.to_csv(output_path, index=False)
print(f'Saved {output_path.resolve()}')
blend.describe()


