import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
from lifelines import WeibullFitter,LogNormalFitter, LogLogisticFitter, ExponentialFitter
from lifelines.plotting import qq_plot
import matplotlib.pyplot as plt

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
path = Path('/kaggle/input/equity-post-HCT-survival-predictions' if iskaggle else 'data')
df = pd.read_csv(path/'train.csv', index_col='ID')


fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.reshape(4,)
T = df.efs_time.values
E = df.efs.values >.5
for i, model in enumerate([WeibullFitter(), LogNormalFitter(), LogLogisticFitter(), ExponentialFitter()]):
    model.fit(T, E)
    qq_plot(model, ax=axes[i])

