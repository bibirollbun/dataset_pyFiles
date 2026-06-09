""" 
Goal: Adding more Data

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
more_data_path = r"/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
sub_df = pd.read_csv(sub_path)
orig_df = pd.read_csv(more_data_path)


train_df.shape


orig_df.shape


# downcast numerical columns

def downcasting(data: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:

    mem_before = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
            
    for col in data.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="float")

    mem_after = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%\n")

    
    return data

train = downcasting(train_df)
test = downcasting(test_df)
orig = downcasting(orig_df)


orig_df.info()


train_df.info()

