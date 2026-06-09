import pandas as pd
import glob
import os
from scipy.stats import mode

path = '/kaggle/input/diabetes-ds'
dfs = []

for filename in glob.glob(os.path.join(path, '*.csv')):
    df = pd.read_csv(filename)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

submission_mean = combined.groupby('id')['diagnosed_diabetes'].mean().reset_index()
submission_median = combined.groupby('id')['diagnosed_diabetes'].median().reset_index()

def get_mode(series):
    return mode(series, keepdims=True).mode[0]

submission_mode = combined.groupby('id')['diagnosed_diabetes'].apply(get_mode).reset_index()

submission_mean.to_csv('submission_mean.csv', index=False)
submission_median.to_csv('submission_median.csv', index=False)
submission_mode.to_csv('submission_mode.csv', index=False)


submission_mean.head()

