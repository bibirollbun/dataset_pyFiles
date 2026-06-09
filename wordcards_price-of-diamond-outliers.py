# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.simplefilter('ignore')

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


train = pd.read_csv('../input/predicting-the-price-of-diamond/train.csv')
test = pd.read_csv('../input/predicting-the-price-of-diamond/test.csv')
# some duplicated rows in original data
orig = pd.read_csv('../input/diamonds/diamonds.csv').drop_duplicates()

train.shape, test.shape, orig.shape


# split test data into public and private
public = test.head(900)
private = test[~test['id'].isin(public['id'].unique())]


_train = train.drop(['id', 'price'], axis=1)
_test = test.drop('id', axis=1)
_orig = orig.drop('price', axis=1)

pd.concat([
    pd.DataFrame({
        'dtype': _train.dtypes,
        'uniques': _train.nunique(),
        'NA_train': _train.isna().sum(),
        'NA_test': _test.isna().sum(),
        'NA_orig': _orig.isna().sum(),
    }),
    _train.sample(n=3).T
], axis=1)


fig, ax = plt.subplots(figsize=(6,3))
sns.distplot(x=pd.concat([train.drop('id', axis=1), test])['price'], ax=ax)
ax.set_title('price')
plt.show()


num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
num_df = pd.concat([_train, _test, _orig])[num_cols].reset_index(drop=True)


fig, ax = plt.subplots(3,2,figsize=(12, 9))
for i, c in enumerate(num_cols):
    axy, axx = divmod(i,2)
    sns.distplot(x=num_df[c], ax=ax[axy,axx])
    ax[axy,axx].set_ylabel('')
    ax[axy,axx].set_yticklabels([])
    ax[axy,axx].set_title(c)
plt.tight_layout()


pd.concat([num_df.describe().loc[['min', 'max', 'mean']],
           num_df.quantile([0.01, 0.1, 0.9, 0.99])]
         ).loc[['min', 0.01, 0.1, 'mean', 0.9, 0.99, 'max']]


num_df.sort_values('carat', ascending=False)


pd.DataFrame({
    'data': ['train', 'orig', 'public', 'private'],
    'carat==0': [len(df[df['carat']==0]) for df in [train, orig, public, private]],
    'carat>100': [len(df[df['carat']>100]) for df in [train, orig, public, private]],
    'x==0': [len(df[df['x']==0]) for df in [train, orig, public, private]],
    'y==0': [len(df[df['y']==0]) for df in [train, orig, public, private]],
    'z==0': [len(df[df['z']==0]) for df in [train, orig, public, private]]
}).set_index('data')

