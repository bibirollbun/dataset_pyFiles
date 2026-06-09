import pandas as pd

data_path ="/kaggle/input/competitive-data-science-predict-future-sales/"

sales_train = pd.read_csv(data_path + 'sales_train.csv')        #sales data
shops       = pd.read_csv(data_path + 'shops.csv')              #shop information
items       = pd.read_csv(data_path + 'items.csv')             #products
item_cats   = pd.read_csv(data_path + 'item_categories.csv')   # product category
test        = pd.read_csv(data_path + 'test.csv')
submission  = pd.read_csv(data_path + 'sample_submission.csv')


sales_train.head()


sales_train.info()


sales_train.info(show_counts=True)   # non-missing value count display


shops.head()


shops.info()


items.head()


items.info()


item_cats.head(10)


item_cats.info()


test.head()


train = sales_train.merge(items, on='item_id', how='left')
train = train.merge(shops, on='shop_id', how='left')
train = train.merge(item_cats, on='item_category_id', how='left')

train.head()


def resumetable(df):
  print(f"Dataset shape : {df.shape}")
  summary = pd.DataFrame(df.dtypes, columns = ['Data Types'])
  summary = summary.reset_index()
  summary = summary.rename(columns = {'index' : 'Features'})
  summary['Missing Values Count'] = df.isnull().sum().values
  summary['Unique Values Count'] = df.nunique().values
  summary['First Value'] = df.loc[0].values
  summary['Second Value'] = df.loc[1].values

  return summary

resumetable(train)


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
sns.boxplot(y = 'item_cnt_day', data = train)


sns.boxplot(y='item_price', data = train)


group = train.groupby(['date_block_num']).agg({'item_cnt_day':'sum'})
group.index
group  = group.reset_index() # Index reset
group= group.rename(columns = {'item_cnt_day': 'monthly_sales_amt', 'date_block_num': 'month'})
group.head()


mpl.rc('font', size=13) # font config
figure, ax = plt.subplots()
figure.set_size_inches(11, 5)

group_month_sum = train.groupby(['date_block_num']).agg({'item_cnt_day': 'sum'})
group_month_sum = group_month_sum.reset_index()
group_month_sum = group_month_sum.rename(columns={'item_cnt_day':'monthly_sales_amt', 'date_block_num': 'month'})

sns.barplot(x='month', y='monthly_sales_amt', data=group_month_sum, palette=sns.color_palette("Set2", 10))
# ax.set_title('Monthly Sales Amount Distribution')
# ax.set_xlabel('Month')
# ax.set_ylabel("Monthly Sales Amount")
ax.set(title = 'Monthly Sales Amount Distribution', xlabel='Month', ylabel='Monthly Sales Amount')


figure, ax = plt.subplots()
figure.set_size_inches(11, 5)

group_cat_sum = train.groupby(['item_category_id']).agg({'item_cnt_day':'sum'})
group_cat_sum = group_cat_sum.reset_index()

#Extract the sales amount of item categories which is greater than 1000 - Monthly sales
#items where montly sales amount is greater than 10000
group_cat_sum = group_cat_sum[group_cat_sum['item_cnt_day'] > 10000]

sns.barplot(x='item_category_id', y='item_cnt_day', data = group_cat_sum, palette=sns.color_palette('Set2'))
ax.set(title = 'Distribution of total item counts by item category id', xlabel='item category id', ylabel='Total Item cnts')
ax.tick_params(axis = 'x', labelrotation = 90) #x-axis label rotation


figure, ax = plt.subplots()
figure.set_size_inches(11, 5)

group_shop_sum = train.groupby(['shop_id']).agg({'item_cnt_day':'sum'})
group_shop_sum = group_shop_sum.reset_index()
group_shop_sum = group_shop_sum[group_shop_sum['item_cnt_day'] > 10000]

sns.barplot(x='shop_id', y='item_cnt_day', data=group_shop_sum, palette=sns.color_palette('Set2'))
ax.set(title = 'Distribution of total Sales amount by Shop Id', xlabel='Shop ID', ylabel = 'Total Item Counts')
ax.tick_params(axis = 'x', labelrotation= 90)

