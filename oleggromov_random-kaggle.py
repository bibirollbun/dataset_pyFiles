import numpy as np
import pandas as pd

from pathlib import Path

pd.options.mode.chained_assignment = None


################################################################################
# Load data
################################################################################
data_path = '/kaggle/input/neurips-open-polymer-prediction-2025/'
train_df = pd.read_csv(Path(data_path, 'train.csv'))
test_df = pd.read_csv(Path(data_path, 'test.csv'), dtype=str)


################################################################################
# Make static predictions
################################################################################
tasks = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df['Rg'] = 1 / train_df['Rg']
for task in tasks:
    test_df[task] = train_df[task].mean()
test_df['Rg'] = 1 / test_df['Rg']
test_df[['id'] + tasks].to_csv('submission.csv', index=False)

