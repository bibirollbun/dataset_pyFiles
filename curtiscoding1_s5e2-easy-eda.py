import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


train.info()


train.describe()


train.isna().sum()


conts = train.select_dtypes(include=np.number).columns.tolist()
cats = [col for col in train.columns if col not in conts]


sns.histplot(train['Price'])


train['Price'].value_counts()


print(f"Min price: {train['Price'].min()}, Max price: {train['Price'].max()}")


train[conts].corr()


def p_95(x):
    return np.percentile(x, 95)

def p_05(x):
    return np.percentile(x, 5)

for cat in cats:
    print(train.groupby(cat)['Price'].agg(avg= 'mean', count= 'count', std = 'std', perc_95 = p_95, perc_05 = p_05))
    print()


def plot_percentiles(df, percentile):
    fix, axs = plt.subplots(3, 3, figsize = (12.5, 12.5))

    for i, ax in enumerate(axs.flatten()):
    
        try:
            col = cats[i]
            sns.countplot(df, x=col, ax=ax)
        
            ax.set_title(f'{percentile}th Percentile {col} Distribution')
        except:
            pass
    
    fix.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.24)
    plt.show()


train_01 = train[train.Price < train.Price.quantile(.01)].copy()


plot_percentiles(train_01, 1)


train_05 = train[train.Price < train.Price.quantile(.05)].copy()


plot_percentiles(train_05, 5)


train_95 = train[train.Price > train.Price.quantile(.95)].copy()


plot_percentiles(train_95, 95)


train_99 = train[train.Price > train.Price.quantile(.98)].copy()


plot_percentiles(train_99, 99)


sns.histplot(train['Weight Capacity (kg)'])


train['WeightCap Binned'] = pd.cut(train['Weight Capacity (kg)'], bins=24)
train['WeightCap Binned mean'] = train.groupby('WeightCap Binned')['Price'].transform('mean')


sns.barplot(data = train, x = 'WeightCap Binned',  y = 'WeightCap Binned mean')
plt.xticks(rotation=90)
plt.show()


train.groupby('WeightCap Binned')['Price'].mean()


pd.pivot_table(train, index=['Compartments'], columns=['Size'], values = ['Price'], aggfunc= np.mean, fill_value=0)


multi_groupby = train.groupby(['Brand',
 'Material',
 'Color'])['Price'].mean().sort_values()


multi_groupby.head(10)


multi_groupby.tail(10)


bin_multi_groupby = train.groupby(['Compartments',
 'WeightCap Binned',
 'Color'])['Price'].mean().sort_values()


bin_multi_groupby.head(10)


bin_multi_groupby.tail(10)


bin_multi_groupby = train.groupby(['Compartments',
 'WeightCap Binned',
 'Color'])['Price'].agg(mean = np.mean, p_95 = p_95, p_05 = p_05).sort_values('mean')


bin_multi_groupby.head(10)


bin_multi_groupby.tail(10)


for col in cats:
    train[col] = train[col].astype('category')

for col in conts:
    train[col] = train[col].astype('float32')


model = lgb.LGBMRegressor()


X = train.drop(['Price',  'WeightCap Binned', 'WeightCap Binned mean', 'id'], axis = 1).copy()
y = train['Price'].copy()


model.fit(X, y)


lgb.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance (Gain)")




