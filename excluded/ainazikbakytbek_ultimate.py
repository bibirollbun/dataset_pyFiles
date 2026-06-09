import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 
%matplotlib inline


bankchurners = pd.read_csv("/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv")
bankchurners.head(10)


bankchurners.columns


bankchurners.isnull().sum()

