import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


train


print(train.shape)
print(train.info())


train['num_sold'].describe()


train['num_sold'].isnull().sum()


train[train['num_sold'].isnull()]


train['date'] = pd.to_datetime(train['date'])
#train.info()


train['date'].dt.year

