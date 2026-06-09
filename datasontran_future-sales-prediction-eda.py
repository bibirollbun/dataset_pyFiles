import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import datetime 

pd.set_option('display.max_colwidth',None)


items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test_dataset = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')


train.head()


train.shape


train.isnull().sum()


train_dataset = train.copy()


train_dataset[train_dataset['item_cnt_day'] == 2169.0]


monthly_sales=train_dataset.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})


monthly_sales


monthly_sales.columns


sales_by_month = train_dataset.groupby(['date_block_num'])['item_cnt_day'].sum()
sales_by_month.plot()


corr = train_dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(train_dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(train_dataset.columns)
ax.set_yticklabels(train_dataset.columns)
plt.show()


items.head()


plt.rcParams['figure.figsize'] = (24, 9)
sns.barplot(items['item_category_id'], items['item_id'], palette = 'colorblind')
plt.title('Number of Item Sold Per Category', fontsize = 30)
plt.xlabel('Item Categories', fontsize = 15)
plt.ylabel('Items', fontsize = 15)
plt.show()


plt.rcParams['figure.figsize'] = (24, 9)
sns.countplot(train_dataset['date_block_num'], palette = 'colorblind')
plt.title('Number of Item Sold Per Month Over 2013 - 2015', fontsize = 30)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Items Count', fontsize = 15)
plt.show()


# item_cat['item_category_name'].count()
print(item_cat['item_category_name'].nunique())
print(shops['shop_name'].nunique())


from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (15, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'pink',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(shops['shop_name']))


plt.title('Wordcloud for Shop Names', fontsize = 25)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')


plt.rcParams['figure.figsize'] = (15, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'lightyellow',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(item_cat['item_category_name']))


plt.title('Wordcloud for Item Category Names', fontsize = 24)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')


train_dataset['date'] = pd.to_datetime(train_dataset['date'], errors='coerce')


days = []
months = []
years = []

for day in train_dataset['date']:
    days.append(day.day)
for month in train_dataset['date']:
    months.append(month.month)    
for year in train_dataset['date']:
    years.append(year.year)


plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(days, palette= 'pastel')
plt.title('The busiest days for the shops', fontsize = 24)
plt.xlabel('Days', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)

plt.show()


# busy month
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(months, palette= 'rocket')
plt.title('The busiest months for the shops', fontsize = 24)
plt.xlabel('Months', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)

plt.show()

# busy year
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(years, palette= 'cubehelix')
plt.title('The busiest years for the shops', fontsize = 24)
plt.xlabel('Years', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)

plt.show()


train_dataset['day'] = days
train_dataset['month'] = months
train_dataset['year'] = years


train_dataset


sns.countplot(train_dataset[(train_dataset.month == 2) & (train_dataset.year == 2013)]['shop_id'], palette='pastel')


train_dataset.describe()


plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_price.min(), train_dataset.item_price.max()*1.1)
sns.boxplot(x=train_dataset.item_price)


plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_cnt_day.min(), train_dataset.item_cnt_day.max()*1.1)
sns.boxplot(x=train_dataset.item_cnt_day)


train_dataset = train_dataset[train_dataset['item_price'] < 100000]
train_dataset = train_dataset[train_dataset['item_cnt_day'] < 1200]


train_dataset.shape


train_dataset[train_dataset['item_price'] < 0]


median = train_dataset[(train_dataset.shop_id==32)&(train_dataset.item_id==2973)&(train_dataset.date_block_num==4)&(train_dataset.item_price>0)].item_price.median()
median


train_dataset["item_price"] = train_dataset["item_price"].map(lambda x: median if x<0 else x)


train_dataset[train_dataset['item_price'] < 0]


train_dataset[train_dataset['item_cnt_day'] < 0]


train_dataset["item_cnt_day"] = train_dataset["item_cnt_day"].map(lambda x: 0 if x<0 else x)


train_dataset[train_dataset['item_cnt_day'] < 0]


train_dataset.head(2)


print("total unique items: ", items['item_id'].nunique())
print("total unique items in train dataset: ", train_dataset['item_id'].nunique())
print("total unique items in test dataset: ", test_dataset['item_id'].nunique())

print("total unique shops: ", shops['shop_id'].nunique())
print("total unique shops in train dataset: ", train_dataset['shop_id'].nunique())
print("total unique shops in test dataset: ", test_dataset['shop_id'].nunique())


test_item_list = [x for x in (np.unique(test_dataset['item_id']))]
train_item_list = [x for x in (np.unique(train_dataset['item_id']))]

missing_item_ids_ = [element for element in test_item_list if element not in train_item_list]
len(missing_item_ids_)


shops


# getting rid of "!" before shop_names
shops['shop_name'] = shops['shop_name'].map(lambda x: x.split('!')[1] if x.startswith('!') else x)
shops['shop_name'] = shops["shop_name"].map(lambda x: 'Ğ¡ĞµÑ€Ğ³Ğ¸ĞµĞ²ĞŸĞ¾Ñ�Ğ°Ğ´ Ğ¢Ğ¦ "7Ğ¯"' if x == 'Ğ¡ĞµÑ€Ğ³Ğ¸ĞµĞ² ĞŸĞ¾Ñ�Ğ°Ğ´ Ğ¢Ğ¦ "7Ğ¯"' else x)


shops['city'] = shops['shop_name'].map(lambda x: x.split(" ")[0])
# lets assign code to these city names too
shops['city_code'] = shops['city'].factorize()[0]


shops.head(2)


for shop_id in shops['shop_id'].unique():
    shops.loc[shop_id, 'num_products'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_id'].nunique()
    shops.loc[shop_id, 'min_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].min()
    shops.loc[shop_id, 'max_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].max()
    shops.loc[shop_id, 'mean_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].mean()


shops.head(2)


item_cat


cat_list = []
for name in item_cat['item_category_name']:
    cat_list.append(name.split('-'))


item_cat['split'] = (cat_list)
item_cat['cat_type'] = item_cat['split'].map(lambda x: x[0])
item_cat['cat_type_code'] = item_cat['cat_type'].factorize()[0]
item_cat['sub_cat_type'] = item_cat['split'].map(lambda x: x[1] if len(x)>1 else x[0])
item_cat['sub_cat_type_code'] = item_cat['sub_cat_type'].factorize()[0]


item_cat.head(2)


item_cat.drop('split', axis = 1, inplace=True)
item_cat.head(2)


train_dataset = train_dataset[train_dataset["item_cnt_day"]>0]
train_dataset = train_dataset[["month", "date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"]].groupby(
    ["date_block_num", "shop_id", "item_id"]).agg(
    {"item_price": "mean","item_cnt_day": "sum", "month": "min"}).reset_index()
train_dataset.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
train_dataset = pd.merge(train_dataset, items, on="item_id", how="inner")
train_dataset = pd.merge(train_dataset, shops, on="shop_id", how="inner")
train_dataset = pd.merge(train_dataset, item_cat, on="item_category_id", how="inner")


train_dataset.head(2)


train_dataset.drop(['item_name', 'shop_name', 'city', 'item_category_name', 'cat_type', 'sub_cat_type'], axis = 1, inplace=True)


train_dataset.head(1)


test_dataset.head()


test_dataset.shape


train_dataset.shape


train_dataset = train_dataset[train_dataset['shop_id'].isin(test_dataset['shop_id'].unique())]
train_dataset = train_dataset[train_dataset['item_id'].isin(test_dataset['item_id'].unique())]


train_dataset.shape


# final_train_df = train_dataset[['date_block_num','item_id','shop_id','item_cnt_month']]
final_train_df = train_dataset.copy()
final_train_df = final_train_df.pivot_table(index=['item_id','shop_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value = 0).reset_index()

final_train_df = pd.merge(test_dataset,final_train_df,on = ['item_id','shop_id'],how = 'left')
final_train_df.fillna(0,inplace = True)
final_train_df


final_train_df.shape


final_train_df.columns


# test = test_dataset.copy()
# test['date_block_num'] = 34
# test['date_block_num'] = test['date_block_num']
# test = test[['date_block_num','shop_id','item_id']]
# test

# # Now lets add the corresponding `item_price` to our `test_dataset`
# item_price=dict(train_dataset.groupby('item_id')['item_price'].last().reset_index().values)
# item_cnt_month=dict(train_dataset.groupby('item_id')['item_cnt_month'].last().reset_index().values)
# test['item_price']=test.item_id.map(item_price)
# test['item_cnt_month']=test.item_id.map(item_cnt_month)

# # filling in the nulls with median value
# test['item_price'] = test['item_price'].fillna(test['item_price'].median())
# test['item_cnt_month'] = test['item_cnt_month'].fillna(0)


# test.head()


# adding `date_block_num` 34 since we want to predict Nov 2015 sales

final_train_df[34] = 0
final_train_df


final_train_df = final_train_df.drop(34, axis = 1)
final_train_df


x_train = final_train_df.drop(33, axis=1)
y_train = final_train_df[33]
# x_valid = train_dataset[train_dataset.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# y_valid = train_dataset[train_dataset.date_block_num == 33]['item_cnt_month']
# deleting the column so that it can predict the future sales data
x_test = final_train_df.drop(0, axis=1)


print(x_train.shape, y_train.shape, x_test.shape)


x_test




