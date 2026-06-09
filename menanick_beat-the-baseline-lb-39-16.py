import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt


os.chdir('/kaggle/input/playground-series-s5e2')
df = pd.read_csv('train.csv', index_col='id')
test_df = pd.read_csv('test.csv', index_col='id')

target = 'Price'
df.head()


# My secret weapon
brand_mean = df.groupby('Brand', observed=False, as_index=False)[target].mean()
brand_mean


naive_df = test_df
naive_df['id'] = naive_df.index
naive_df = naive_df.merge(brand_mean, on='Brand', how='left')
naive_df[target] = naive_df[target].fillna(df[target].mean())
naive_df

naive_df[['id', 'Price']].set_index('id').to_csv('/kaggle/working/Submission.csv')

