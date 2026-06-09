"""
Goal: Understand how this data help to predict the accident.
      How we approach this data.

Author: Rudra Prasad Bhuyan
V1: 20-10-2025 15:48 IST
"""
print("")


# !pip install missingno


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msgo

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df.sample(3)


train_df.info()


train_df.columns


test_df.sample(3)


sub_df.sample(3)


train_df.isnull().sum()


msgo.matrix(train_df)


msgo.matrix(test_df)


msgo.matrix(sub_df)

