""" 
Goal: 
    - Understand the Problem ?
    - What we need to solve?
    - What is the requirement 
    - Before starting the any competition make sure
    - We Understood the Problem 

Author: Rudra Prasad Bhuyan
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')


sub_path    = r"/kaggle/input/playground-series-s5e11/sample_submission.csv" 
train_path  = r"/kaggle/input/playground-series-s5e11/train.csv"
test_path   = r"/kaggle/input/playground-series-s5e11/test.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
sub_df = pd.read_csv(sub_path)


train_df.info()


test_df.info()


sub_df.sample(3)




