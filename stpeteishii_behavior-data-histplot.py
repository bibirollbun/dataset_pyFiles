import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


train=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
display(train[0:3].T)


print(len(train))
M=list(range(len(train)))
random.shuffle(M)
train=train.iloc[M[0:4000]]
train=train.reset_index(drop=True)


cols=train.columns.tolist()
print(cols)
print(len(cols))
cols=cols[0:60]


fig, ax = plt.subplots(30,2,figsize=(12,90))
plt.subplots_adjust(hspace=3)

for i in tqdm(range(len(cols))):
    r=i//2
    c=i%2
    sns.histplot(train[cols[i]], label='train '+cols[i], ax=ax[r,c], color='C1',bins=40, alpha=0.5)

    ax[r,c].legend()
    ax[r,c].grid()
    ax[r,c].tick_params(axis='x', labelrotation=80)

plt.show()


train2=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
display(train2[0:3].T)


print(len(train2))
M2=list(range(len(train2)))
random.shuffle(M2)
train2=train2.iloc[M2[0:4000]]
train2=train2.reset_index(drop=True)


cols2=train2.columns.tolist()
print(cols2)
print(len(cols2))



fig, ax = plt.subplots(4,2,figsize=(12,12))
plt.subplots_adjust(hspace=3)

for i in tqdm(range(len(cols2))):
    r=i//2
    c=i%2
    sns.histplot(train2[cols2[i]], label='train '+cols2[i], ax=ax[r,c], color='C1',bins=40, alpha=0.5)

    ax[r,c].legend()
    ax[r,c].grid()
    ax[r,c].tick_params(axis='x', labelrotation=80)

plt.show()




