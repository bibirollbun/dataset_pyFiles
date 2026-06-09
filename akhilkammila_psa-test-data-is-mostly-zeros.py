import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DIRECTORY = '/kaggle/input/competitive-data-science-predict-future-sales/'
train = pd.read_csv(DIRECTORY + 'sales_train.csv')
test = pd.read_csv(DIRECTORY + 'test.csv')


test_items = test['item_id'].unique()
test_shops = test['shop_id'].unique()

print("# Items in Test", len(test_items))
print("# Shops in Test", len(test_shops))
print("# of Combinations:", len(test_items) * len(test_shops))
print("# of Rows in Test:", len(test))


# What shops are in test?
train['datetime'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

shops_last_month_train = train[train.date_block_num == 33]['shop_id'].unique()
shops_last_2w_train = train[train.datetime > (train.datetime.max() - pd.Timedelta('2W'))]['shop_id'].unique()
test_and_last_2w = np.intersect1d(test_shops, shops_last_2w_train)

print('# Shops last month train:', len(shops_last_month_train))
print('# Shops last 2w train:', len(shops_last_2w_train))
print('# Shops in test:', len(test_shops))

print("Shop Overlaps between Test & Last 2W Train:", len(test_and_last_2w))


# What items are in test?
test_new_items = np.setdiff1d(test_items, train['item_id'].unique())
test_old_items = np.intersect1d(test_items, train['item_id'].unique())
old_items_train_end = np.intersect1d(test_old_items, train[train.date_block_num == 33]['item_id'].unique())

print("# of Items in test:", len(test_items))
print("# of New items in test:", len(test_new_items))
print("# of Old items in test:", len(test_old_items))

print("# Old Items that were sold in last 1m of train:", len(old_items_train_end))
print('# Old Items that were NOT SOLD in last 1m of train:', len(test_old_items) - len(old_items_train_end))


train['month_year'] = train['datetime'].dt.to_period('M')
month_group = train.groupby(by=['item_id', 'month_year'])['item_cnt_day'].sum().to_frame('times_sold').reset_index()
month_group = month_group.sort_values(by=['item_id', 'month_year'])
month_group['last_purchase'] = month_group.groupby(by='item_id')['month_year'].shift(1)
month_group['time_since_last_purchase'] = (month_group['month_year'].astype('int64') - month_group['last_purchase'].astype('int64'))
month_group['time_since_last_purchase'] = np.where(month_group['time_since_last_purchase'] < 0, 100, month_group['time_since_last_purchase'])

month_group['bins'] = pd.cut(month_group['time_since_last_purchase'], bins=[0,1,45,np.inf], \
                                labels=['Sold Last Month', 'Last Sale Before Last Month', 'New'], include_lowest=True)
filtered_month = month_group[month_group['times_sold'] > 0]

bin_counts = filtered_month.groupby(['month_year', 'bins'], observed=True)['item_id'].size().unstack()

bin_counts.plot(kind='bar', stacked=True, figsize=(15,5))
plt.axhline(y=5100, color='red')
plt.title('Historic Distribution of Items with at Least 1 Sale')
plt.ylabel('Item Count');


# How many zeros should we expect?
zero_pct = train.groupby('date_block_num').apply(lambda group : 1 - len(group)/(len(group['item_id'].unique()) * len(group['shop_id'].unique())))
sns.barplot(x=zero_pct.index, y=zero_pct.values)
plt.xticks(rotation=90)
plt.title('Percentage of Values Zero - Previous Months');

