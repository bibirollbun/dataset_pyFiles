import numpy as np
import pandas as pd 

from pandas_profiling import ProfileReport


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


train.head()


train.shape


train.describe().T


train.info()


report = ProfileReport(train)


report




