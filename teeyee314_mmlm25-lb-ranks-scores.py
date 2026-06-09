import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


fps = sorted(glob('/kaggle/input/march-madness-2025-leaderboard/*.csv'))

dfs = []

for fp in fps:
    dfs.append(pd.read_csv(fp))


c = pd.DataFrame()

for i, df in enumerate(dfs):
    if c.shape[0] == 0:
        c = df
    else: 
        c = c.merge(df[['TeamId','Rank','Score']].rename(columns={'Rank':f'Rank{i}','Score':f'Score{i}'}), on=['TeamId'], how='inner')


rank_cols = [c for c in c.columns if 'Rank' in c]
score_cols = [s for s in c.columns if 'Score' in s]

c['rank_avg'] = c[rank_cols].mean(axis=1)
c['brier_avg'] = c[score_cols].mean(axis=1)


c


c.sort_values('brier_avg')[['TeamName','rank_avg','brier_avg']].head(25).reset_index(drop=True)


c.sort_values('rank_avg')[['TeamName','rank_avg','brier_avg']].head(25).reset_index(drop=True)


c[rank_cols].corr()


pd.set_option('display.max_rows', 2000)
display(c.sort_values('rank_avg')[['TeamName','rank_avg','brier_avg']].reset_index(drop=True))




