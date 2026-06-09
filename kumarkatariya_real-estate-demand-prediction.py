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


true = pd.DataFrame({
    'id': [1,2,3,4,5],
    'true': [100000,200000,300000,400000,500000]
})


true


y_pred = pd.DataFrame({
    'id': [1,2,3,4,5],
    'y_pred': [120000,220000,700000,390000,600000]
})


y_pred.head()


def custom_score(y_true, y_pred, eps=1e-12):
    """Scoring function of the competition as defined on the competition overview page.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    eps : float, optional (exact value doesn't matter)

    Return value:
    -------------
    dict with keys 'score', 'good_rate' and 'str'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')

    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))

    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)

    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    # score = max(0.0, score)
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}


custom_score(true,y_pred,eps = 1e-12)


ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')



test.head()
train_nht.head()



pd.set_option("display.max_rows",None)
train_nht['month'].value_counts() 


month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul':7,
    'Aug':8,
    'Sep':9,
    'Oct':10,
    'Nov':11,
    'Dec':12
}


test.head()


csi.head()


test['month'] = test['id'].str.split('_').str[0]
test['sector'] = test['id'].str.split('_').str[1]


test.head()


dfs = [sp,train_lt,train_ltns,train_pht,train_phtns,train_nht,train_nhtns]

for df in dfs:
    df['sector_id'] =  df['sector'].str.split(' ').str[1].astype('int')




sp.head()


# checking whether any of the sector columns in any of these tables follows the same pattern or having 
# different pattern

import re
dfs = [sp,train_lt,train_ltns,train_pht,train_phtns,train_nht,train_nhtns]

pattern = r'^sector \d+$'

for df in dfs:
    mask = ~df['sector'].str.match(pattern)
    print(mask.sum())



train_nhtns.head()


pd.set_option('display.max_columns',None)


sp.head()





dfs = [train_lt,train_ltns,train_pht,train_phtns,train_nht,train_nhtns]





train_lt.head()


# lets see whether the month column style/pattern is same across the tables


pattern = r'^\d+-[A-Za-z]+$'

for df in dfs:
    mask = ~df['month'].str.match(pattern)
    print(mask.sum())

# so the style of writing the month in all the cols are the same so now we can create new cols from it



for df in dfs:
    df['year'] = df['month'].str.split('-').str[0].astype('int')
    df['month'] = df['month'].str.split('-').str[1].map(month_codes)
    min_year = df['year'].min()
    df['time'] = (df['year']- min_year)*12 + df['month']-1 



train_lt.head()


dfs = [train_lt,train_ltns,train_pht,train_phtns,train_nht,train_nhtns]


train_nht.head()


train_nht['year'].value_counts()


test.head()


test['month'].value_counts()


amount_new_house_transactions = train_nht.pivot(index='time',columns='sector',values='amount_new_house_transactions')


amount_new_house_transactions = amount_new_house_transactions.fillna(0)

# Add missing sector 95 as string
amount_new_house_transactions['sector 95'] = 0

# Create ordered list of columns
cols = [f'sector {i}' for i in range(1, 97)]
amount_new_house_transactions = amount_new_house_transactions[cols]

# Convert values to int
amount_new_house_transactions = amount_new_house_transactions.astype(int)



# amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
# amount_new_house_transactions.astype(int)


amount_new_house_transactions.head()


import matplotlib.pyplot as plt
plt.title('Extrapolating a time series')
plt.plot(amount_new_house_transactions.sum(axis=1),
         color='b',
         label='total amount'
        ) 

plt.scatter(np.arange(11, 67, 12),
            amount_new_house_transactions.sum(axis=1).iloc[np.arange(11, 67, 12)],
            color='b',
            label='year-end peak') 
plt.xticks(np.arange(0, 80, 12))
plt.xlim(-2, 80)
plt.xlabel('time (months)')
plt.ylabel('Total amount_new_house_transactions')
plt.legend()
plt.show()


t1=6
t2=6
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=4,test_size=12)
true,oof= [],[]

for fold,(train_idx,val_idx) in enumerate(cv.split(amount_new_house_transactions)): 
    train = amount_new_house_transactions.iloc[train_idx]
    valid = amount_new_house_transactions.iloc[val_idx] 

    pred =  pd.DataFrame({time:np.exp(np.log(train.tail(t1)).mean(axis=0)) for time in val_idx}).T
    pred.loc[:,train.tail(t2).min(axis=0)==0]=0  
    pred.index.rename('time', inplace=True) 
    print(f"# Fold {fold}: {custom_score(valid, pred)}\n")
    true.append(valid)
    oof.append(pred)

print(f"# Overall {custom_score(pd.concat(true), pd.concat(oof))} {t1=} {t2=}\n") 



train = amount_new_house_transactions
pred = pd.DataFrame(
    {time: train.tail(t1).mean(axis=0) for time in np.arange(67, 79)}
).T
pred.loc[:, train.tail(t2).min(axis=0) == 0] = 0
pred.index.rename('time', inplace=True)
display(pred.astype(int))


test['new_house_transaction_amount'] = pred.T.unstack().values

test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False) 





































