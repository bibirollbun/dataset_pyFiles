"""
Goal: Add more data to make the prediction more accurate.

Author: Rudra Prasad Bhuyan
V1: 21-10-2025 12:52 IST
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

extra_2k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv'
extra_10k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv'
extra_100k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

extra_2k_df = pd.read_csv(extra_2k_path)
extra_10k_df = pd.read_csv(extra_10k_path)
extra_100k_df = pd.read_csv(extra_100k_path)


train_df.info()


extra_2k_df.info()


extra_2k_df.size


extra_10k_df.size


extra_100k_df.size













