import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path

pd.options.mode.chained_assignment = None


################################################################################
# Load data
################################################################################
data_path = '/kaggle/input/neurips-open-polymer-prediction-2025/'
supl_data_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/'
train_df = pd.read_csv(Path(data_path, 'train.csv'))
dataset1_df = pd.read_csv(Path(supl_data_path, 'dataset1.csv')) # Tc
dataset3_df = pd.read_csv(Path(supl_data_path, 'dataset3.csv')) # Tg
dataset4_df = pd.read_csv(Path(supl_data_path, 'dataset4.csv')) # FFV
test_df = pd.read_csv(Path(data_path, 'test.csv'), dtype=str)

tasks = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


################################################################################
# Collate data
################################################################################
dataset1_df = dataset1_df.rename(columns={'TC_mean': 'Tc'})
train_df = pd.concat([train_df,
                      dataset1_df,
                      dataset3_df,
                      dataset4_df], axis=0).reset_index(drop=True)


################################################################################
# Data distributions
################################################################################
kde_dists = {}
fig, ax = plt.subplots(5, 1, figsize=[8, 20])
for task_id, task in enumerate(tasks):
    kde_dists[task] = stats.gaussian_kde(train_df[~train_df[task].isna()][task].values)
    temp_df = train_df[~train_df[task].isna()]
    temp_df[f'{task}_sym'] = kde_dists[task].resample(size=len(temp_df))[0]
    temp_df[task].hist(bins = 50, color='blue', ax=ax[task_id])
    temp_df[f'{task}_sym'].hist(bins = 50, color='red', ax=ax[task_id])
    ax[task_id].set_title(task)


################################################################################
# Make random predictions
################################################################################
for task in tasks:
    test_df[task] = kde_dists[task].resample(size=len(test_df))[0]
test_df[['id'] + tasks].to_csv('submission.csv', index=False)




