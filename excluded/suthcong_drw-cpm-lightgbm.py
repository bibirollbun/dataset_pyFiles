# import packages
import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed = 42
np.random.seed(seed)
random.seed(seed)


train=pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
train=train.to_pandas()
test=pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
test=test.to_pandas()
print(f"train.shape:{train.shape},test.shape:{test.shape}")
train.head()



features = [c for c in train.columns if c!='label']
label_col = 'label'

# 1. BASIC DATA OVERVIEW
print("=== CRYPTO TRADING DATA EDA ===")
print(f"Dataset shape: {train.shape}")
print(f"Number of features: {len(features)}")
print(f"Time range: {train.index.min()} to {train.index.max()}")

print("\n=== 1. BASIC DATA OVERVIEW ===")
print("\nDataset Info:")
print(train.info())

print("\nMissing Values:")
missing_data = train[features + [label_col]].isnull().sum()
missing_pct = (missing_data / len(train)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing_data, 'Missing_Percentage': missing_pct})
missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


print("\nBasic Statistics:")
train[features + [label_col]].describe()


# Drop columns have exactly 1 value
NUNIQUE1=[c for c in train.columns if train[c].nunique()==1]
train.drop(NUNIQUE1+['timestamp'],axis=1,inplace=True)
test.drop(NUNIQUE1+['label'],axis=1,inplace=True)
train.head()


print(NUNIQUE1)


from lightgbm import LGBMRegressor


features = [c for c in train.columns if c!='label']  # features after drop
# Init and fit
model=LGBMRegressor()
model.fit(train[features].values,train['label'].values)

# predict 
test_preds=model.predict(test[features].values)


sub=pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sub['prediction']=test_preds
sub.to_csv("base.csv",index=None)

