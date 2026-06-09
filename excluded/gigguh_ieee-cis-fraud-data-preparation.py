import pandas as pd
import matplotlib as mp
import numpy as np


data = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv').merge(pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv'), on='TransactionID', how='left')


print(data.columns.tolist())


print(data.dtypes.tolist())


print(len(data.columns[data.isna().any()].to_list()))




