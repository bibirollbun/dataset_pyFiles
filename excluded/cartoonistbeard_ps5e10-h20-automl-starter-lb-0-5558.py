!pip install h2o


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import seaborn as sns

import h2o
from h2o.automl import H2OAutoML

import warnings 
warnings.filterwarnings('ignore')
h2o.init()


class config:
    train = '/kaggle/input/playground-series-s5e10/train.csv'
    test = '/kaggle/input/playground-series-s5e10/test.csv'
    sub = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
    org_dir = '/kaggle/input/simulated-roads-accident-data'
    target = 'accident_risk'
    V = 5

cfg = config


import os

train = pd.read_csv(cfg.train,index_col = 'id')
test = pd.read_csv(cfg.test,index_col='id')
files = [f for f in os.listdir(cfg.org_dir)]
org = []

for file in files:
    file_path = os.path.join(cfg.org_dir,file)
    df = pd.read_csv(file_path)
    org.append(df)
    
org = pd.concat(org,axis=0,ignore_index=True)
display(train.head())
display(test.head())
display(org.head())


for col in test.columns:
    tmp = org.groupby(col)[cfg.target].mean()
    new_col = f"org_{col}"
    print(f"Processing {new_col}",end=' ')
    tmp.name = new_col
    train = train.merge(tmp,on=col,how='left')
    test = test.merge(tmp,on=col,how='left')

print(train.shape)


# CATS = [col for col in train.columns if train[col].dtype == 'O']
# PAIRS = [2, 3]

# for r in tqdm(PAIRS):
#     for cols in combinations(CATS, r):
#         new_col = '_'.join(cols)
        
#         # Combine categorical columns into one string column
#         train[new_col] = train[cols[0]].astype(str)
#         test[new_col] = test[cols[0]].astype(str)
#         org[new_col] = org[cols[0]].astype(str)
        
#         for col in cols[1:]:
#             train[new_col] += '_' + train[col].astype(str)
#             test[new_col] += '_' + test[col].astype(str)
#             org[new_col] += '_' + org[col].astype(str)
        
#         # Target encoding
#         tmp = org.groupby(new_col)[cfg.target].mean().rename(f'TE_{new_col}')
        
#         # Merge encoded values
#         train = train.merge(tmp, on=new_col, how='left')
#         test = test.merge(tmp, on=new_col, how='left')


# print(f"We have now {len(test.columns)} columns")


# https://www.kaggle.com/competitions/playground-series-s5e10/discussion/609994#3296622

from scipy.stats import norm

def f(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == "night").astype(int) +
        0.1 * (X["weather"] != "clear").astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

def clip(f_to_clip):
    def clip_f(X):
        sigma = 0.05
        mu = f_to_clip(X)
        a = -mu / sigma
        b = (1 - mu) / sigma
        
        Phi_a = norm.cdf(a)
        Phi_b = norm.cdf(b)
        phi_a = norm.pdf(a)
        phi_b = norm.pdf(b)
        
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
        
    return clip_f

# META = []

clipped_f = clip(f)
for df in [train, test, org]:
    df['Meta'] = clipped_f(df)


display(train.head())


train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

features = train_h2o.columns
features.remove(cfg.target)
# train,valid = train_h2o.split_frame(ratios=[0.8],seed=42)
model = H2OAutoML(
    max_runtime_secs=1000,seed=42
)
model.train(
    x=features, y=cfg.target,
    training_frame=train_h2o,
)


preds = model.predict(test_h2o)
preds = preds.as_data_frame()
display(preds.head())


sub = pd.read_csv(cfg.sub)
sub[cfg.target] = preds['predict']
display(sub.head())
sub.to_csv(f"PS5E10_{cfg.V}.csv",index=False)




