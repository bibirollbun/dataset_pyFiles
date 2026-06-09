"""
Goal: Automatic EDA helps to understand the data points faster.

Author: Rudra Prasad Bhuyan
V1: 20-10-2025 16:06 IST
"""
print("")


!pip install autoviz


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

from autoviz.AutoViz_Class import AutoViz_Class

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.shape


train_df.columns


AV = AutoViz_Class()
%matplotlib inline
    
dfte = AV.AutoViz(
    filename, 
    sep=',',
    depVar='accident_risk',
    dfte=train_df,
    header=0,
    verbose=1,
    lowess=False,
    chart_format='svg',
    max_rows_analyzed=train_df.shape[0],
    max_cols_analyzed=train_df.shape[1],
    save_plot_dir=None
)

