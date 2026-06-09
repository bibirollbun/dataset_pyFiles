# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


df_train = pd.read_csv(
    '/kaggle/input/playground-series-s5e1/train.csv', 
    parse_dates = ['date'], index_col = ['id']
).assign(
    days = lambda x: (x['date'] - x['date'].min()).dt.days # Create a new column 'days' as the number of days since the earliest date
)
target = 'num_sold'


df_train[target].isna().sum()


pd.crosstab(index = df_train['country'], columns = df_train[target].isna()).T


pd.crosstab(index = df_train['product'], columns = df_train[target].isna()).T


pd.crosstab(index = df_train['store'], columns = df_train[target].isna()).T


df_train.loc[(df_train['country'] == 'Canada') & (df_train[target].isna()), ['store', 'product']].value_counts()


df_train.loc[(df_train['country'] == 'Kenya') & (df_train[target].isna()), ['store', 'product']].value_counts()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Canada'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Stickers for Less")],
    x = 'days', y = target
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()  # 현재 x축 레이블 가져오기
# 일부 레이블만 표시
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Kenya'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Stickers for Less")],
    x = 'days', y = target
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Canada'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Premium Sticker Mart")],
    x = 'days', y = target
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Kenya'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Premium Sticker Mart")],
    x = 'days', y = target
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()  # 현재 x축 레이블 가져오기
# 일부 레이블만 표시
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


#I will save the missing value indicators before handling the missing data.
df_train['num_sold_isna'] = df_train['num_sold'].isna()


df_train.loc[(df_train['product'] == 'Kerneler Dark Mode') & df_train[target].isna()]


df_train.loc[
    (df_train['product'] == 'Kerneler Dark Mode') & (df_train['country'] == 'Kenya') & df_train['days'].isin([262, 264])
]


# Kerneler Dark Mode has one missing value, and based on the num_sold before and after it, filling it with 5 seems appropriate. 
df_train.loc[df_train['product'] == 'Kerneler Dark Mode', target] = \
    df_train.loc[df_train['product'] == 'Kerneler Dark Mode', target].fillna(5)
# For Kerneler, which has multiple missing values, the visualization suggests that linear interpolation by country and store would be suitable.
df_train.loc[(df_train['product'] == 'Kerneler'), target] = \
    df_train.loc[(df_train['product'] == 'Kerneler')].groupby(['country', 'store'], observed = False)[target].transform(
        lambda x: x.interpolate()
    )


df_train.loc[df_train['product'] != 'Holographic Goose', target].isna().sum()


df_train.loc[~df_train['country'].isin(['Canada', 'Kenya']), target].isna().sum()


df_ratio_product = df_train.loc[~df_train['country'].isin(['Canada', 'Kenya'])].pipe(
    lambda x: x.pivot_table(index = x['days'], columns = x['product'], values=target, aggfunc = 'sum', observed = True)
).pipe(
    lambda x: x.divide(x.sum(axis=1), axis = 0)
)
df_ratio_product.head()


df_ratio_country = df_train.loc[df_train['product'] != 'Holographic Goose'].pipe(
    lambda x: x.pivot_table(index = x['days'], columns = x['country'], values=target, aggfunc = 'sum', observed = True)
).pipe(
    lambda x: x.divide(x.sum(axis=1), axis = 0)
)
df_ratio_country.head()


df_ratio_store = df_train.loc[(df_train['product'] != 'Holographic Goose') & ~df_train['country'].isin(['Canada', 'Kenya'])].pipe(
    lambda x: x.pivot_table(index = x['days'], columns = x['store'], values=target, aggfunc = 'sum', observed = True)
).pipe(
    lambda x: x.divide(x.sum(axis=1), axis = 0)
)
df_ratio_store.head()


X_cat = ['country', 'store', 'product']
df_ratio = pd.DataFrame(
    0,
    columns = pd.MultiIndex.from_product(df_train[X_cat].apply(lambda x: x.unique().tolist())),
    index = df_train['days'].unique()
).apply(
    lambda x: df_ratio_country.loc[:, x.name[0]] * df_ratio_store.loc[:, x.name[1]] * df_ratio_product.loc[:, x.name[2]]
)
df_ratio.head()


# By setting the proportions of the missing values to 0 and summing them, we can approximate the daily proportion of num_sold for non-missing values.
df_ratio_notna = pd.DataFrame(
    df_ratio.values * df_train.assign(target_notna = lambda x: x[target].notna()).pivot(
        index = 'days', columns = X_cat, values = 'target_notna'
    ).values, index = df_ratio.index, columns = df_ratio.columns
)
df_ratio_notna.head()


# Multiplying the daily sum of num_sold by the reciprocal of the daily proportion of non-missing values gives an approximation of the total daily sum of num_sold. 
s_est_daily_sum = df_train.groupby('days')[target].sum() * 1 / df_ratio_notna.sum(axis = 1)
s_est_daily_sum.head()


# Now that we know the total daily num_sold, we can use the daily proportions and multiply them to approximate the num_sold for each case.
df_train.loc[df_train[target].isna(), target] = df_train.loc[df_train[target].isna()].apply(
    lambda x: df_ratio.loc[x['days'], tuple(x[X_cat])] * s_est_daily_sum.loc[x['days']], axis = 1
)


df_train[target].isna().sum()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Canada'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Discount Stickers")],
    x = 'days', y = target, hue = 'num_sold_isna'
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()  # 현재 x축 레이블 가져오기
# 일부 레이블만 표시
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Kenya'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Discount Stickers")],
    x = 'days', y = target, hue = 'num_sold_isna'
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()  # 현재 x축 레이블 가져오기
# 일부 레이블만 표시
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Canada'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Stickers for Less")],
    x = 'days', y = target, hue = 'num_sold_isna'
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()  # 현재 x축 레이블 가져오기
# 일부 레이블만 표시
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()


plt.figure(figsize = (16, 3))
sns.barplot(
    df_train.loc[(df_train['country'].isin(['Kenya'])) & (df_train['product'] == "Holographic Goose") & (df_train['store'] == "Stickers for Less")],
    x = 'days', y = target, hue = 'num_sold_isna'
)
xticks = plt.gca().get_xticks() 
xtick_labels = plt.gca().get_xticklabels()
plt.xticks(xticks[::50], [label.get_text() for label in xtick_labels][::50], rotation=45) 
plt.show()

