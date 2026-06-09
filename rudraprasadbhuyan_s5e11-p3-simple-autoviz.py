""" 
Goal: Quick Auto EDA with Autoviz

Author: Rudra Prasad Bhuyan
"""
print("")


!pip install autoviz -q


import pandas as pd

from autoviz.AutoViz_Class import AutoViz_Class

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


AV = AutoViz_Class()
%matplotlib inline 

dfte = AV.AutoViz(
    filename='train_df',
    sep=',',
    depVar='loan_paid_back',
    dfte=train_df,
    header=0,
    verbose=1,
    save_plot_dir=None
)

