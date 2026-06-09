import pandas as pd
import numpy as np
from itertools import product
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import category_encoders as ce
import warnings

# pd.set_option('display.max_rows', 400)
# pd.set_option('display.max_columns', 160)
# pd.set_option('display.max_colwidth', 40)
# warnings.filterwarnings("ignore")


test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test.head()


categories = pd.read_csv('../input/predict-future-sales-eng-translation/categories.csv')
pd.DataFrame(categories.category_name.values.reshape(-1, 4))


#create broader category groupings
categories['group_name'] = categories['category_name'].str.extract(r'(^[\w\s]*)')
categories['group_name'] = categories['group_name'].str.strip()
#label encode group names
categories['group_id']  = le.fit_transform(categories.group_name.values)
categories.sample(5)


#load items
items = pd.read_csv('../input/predict-future-sales-eng-translation/items.csv')

#clean item_name
items['item_name'] = items['item_name'].str.lower()
items['item_name'] = items['item_name'].str.replace('.', '')
for i in [r'[^\w\d\s\.]', r'\bthe\b', r'\bin\b', r'\bis\b',
          r'\bfor\b', r'\bof\b', r'\bon\b', r'\band\b',  
          r'\bto\b', r'\bwith\b' , r'\byo\b']:
    items['item_name'] = items['item_name'].str.replace(i, ' ')
items['item_name'] = items['item_name'].str.replace(r'\b.\b', ' ')

#extract first n characters of name
items['item_name_no_space'] = items['item_name'].str.replace(' ', '')
items['item_name_first4'] = [x[:4] for x in items['item_name_no_space']]
items['item_name_first6'] = [x[:6] for x in items['item_name_no_space']]
items['item_name_first11'] = [x[:11] for x in items['item_name_no_space']]
del items['item_name_no_space']
                              
#label encode these columns
items.item_name_first4 = le.fit_transform(items.item_name_first4.values)
items.item_name_first6 = le.fit_transform(items.item_name_first6.values)
items.item_name_first11 = le.fit_transform(items.item_name_first11.values)

#join category_name, group_name and group_id to items
items = items.join(categories.set_index('category_id'), on='category_id')
items.sample(10)


dupes = items[(items.duplicated(subset=['item_name','category_id'],keep=False))]
dupes['in_test'] = dupes.item_id.isin(test.item_id.unique())
dupes = dupes.groupby('item_name').agg({'item_id':['first','last'],'in_test':['first','last']})

#if both item id's are in the test set do nothing
dupes = dupes[(dupes[('in_test', 'first')]==False) | (dupes[('in_test', 'last')]==False)]
#if only the first id is in the test set assign this id to both
temp = dupes[dupes[('in_test', 'first')]==True]
keep_first = dict(zip(temp[('item_id', 'last')], temp[('item_id',  'first')]))
#if neither id or only the second id is in the test set, assign the second id to both
temp = dupes[dupes[('in_test', 'first')]==False]
keep_second = dict(zip(temp[('item_id', 'first')], temp[('item_id',  'last')]))
item_map = {**keep_first, **keep_second}


#loading sales data
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sales = (sales
    .query('0 < item_price < 50000 and 0 < item_cnt_day < 1001') #removing outliers
    .replace({
        'shop_id':{0:57, 1:58, 11:10}, #replacing obsolete shop id's
        'item_id':item_map #fixing duplicate item id's  
    })    
)

#removing shops which don't appear in the test set
sales = sales[sales['shop_id'].isin(test.shop_id.unique())]

sales['date'] = pd.to_datetime(sales.date,format='%d.%m.%Y')
sales['weekday'] = sales.date.dt.dayofweek

#first day the item was sold, day 0 is the first day of the training set period
sales['first_sale_day'] = sales.date.dt.dayofyear 
sales['first_sale_day'] += 365 * (sales.date.dt.year-2013)
sales['first_sale_day'] = sales.groupby('item_id')['first_sale_day'].transform('min').astype('int16')

#revenue is needed to accurately calculate prices after grouping
sales['revenue'] = sales['item_cnt_day']*sales['item_price']


temp = sales.groupby(['shop_id','weekday']).agg({'item_cnt_day':'sum'}).reset_index()
temp = pd.merge(temp, sales.groupby(['shop_id']).agg({'item_cnt_day':'sum'}).reset_index(), on='shop_id', how='left')
temp.columns = ['shop_id','weekday', 'shop_day_sales', 'shop_total_sales']
temp['day_quality'] = temp['shop_day_sales']/temp['shop_total_sales']
temp = temp[['shop_id','weekday','day_quality']]

dates = pd.DataFrame(data={'date':pd.date_range(start='2013-01-01',end='2015-11-30')})
dates['weekday'] = dates.date.dt.dayofweek
dates['month'] = dates.date.dt.month
dates['year'] = dates.date.dt.year - 2013
dates['date_block_num'] = dates['year']*12 + dates['month'] - 1
dates['first_day_of_month'] = dates.date.dt.dayofyear
dates['first_day_of_month'] += 365 * dates['year']
dates = dates.join(temp.set_index('weekday'), on='weekday')
dates = dates.groupby(['date_block_num','shop_id','month','year']).agg({'day_quality':'sum','first_day_of_month':'min'}).reset_index()

dates.query('shop_id == 28').head(15)


sales = (sales
     .groupby(['date_block_num', 'shop_id', 'item_id'])
     .agg({
         'item_cnt_day':'sum', 
         'revenue':'sum',
         'first_sale_day':'first'
     })
     .reset_index()
     .rename(columns={'item_cnt_day':'item_cnt'})
)
sales.sample(5)


df = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]]))))

df = pd.DataFrame(np.vstack(df), columns=['shop_id', 'item_id', 'date_block_num'])
df.head()


#add the appropriate date_block_num value to the test set
test['date_block_num'] = 34
del test['ID']


#append test set to training dataframe
df = pd.concat([df,test]).fillna(0)
df = df.reset_index()
del df['index']


#join sales and item inforamtion to the training dataframe
df = pd.merge(df, sales, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
df = pd.merge(df, dates, on=['date_block_num','shop_id'], how='left')
df = pd.merge(df, items.drop(columns=['item_name','group_name','category_name']), on='item_id', how='left')


# shops = pd.read_csv('../input/predict-future-sales-eng-translation/shops.csv')

# #clustering shops
# shops_cats = pd.DataFrame(
#     np.array(list(product(*[df['shop_id'].unique(), df['category_id'].unique()]))),
#     columns =['shop_id', 'category_id']
# )
# temp = df.groupby(['category_id', 'shop_id']).agg({'item_cnt':'sum'}).reset_index()
# temp2 = temp.groupby('shop_id').agg({'item_cnt':'sum'}).rename(columns={'item_cnt':'shop_total'})
# temp = temp.join(temp2, on='shop_id')
# temp['category_proportion'] = temp['item_cnt']/temp['shop_total']
# temp = temp[['shop_id', 'category_id', 'category_proportion']]
# shops_cats = pd.merge(shops_cats, temp, on=['shop_id','category_id'], how='left')
# shops_cats = shops_cats.fillna(0)

# shops_cats = shops_cats.pivot(index='shop_id', columns=['category_id'])
# kmeans = KMeans(n_clusters=7, random_state=0).fit(shops_cats)
# shops_cats['shop_cluster'] = kmeans.labels_.astype('int8')

# #adding these clusters to the shops dataframe
# shops = shops.join(shops_cats['shop_cluster'], on='shop_id')





from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Loading shops.csv
shops = pd.read_csv('../input/predict-future-sales-eng-translation/shops.csv')

# Clustering shops
shops_cats = pd.DataFrame(
    np.array(list(product(*[df['shop_id'].unique(), df['category_id'].unique()]))),
    columns=['shop_id', 'category_id']
)

temp = df.groupby(['category_id', 'shop_id']).agg({'item_cnt': 'sum'}).reset_index()
temp2 = temp.groupby('shop_id').agg({'item_cnt': 'sum'}).rename(columns={'item_cnt': 'shop_total'})
temp = temp.join(temp2, on='shop_id')
temp['category_proportion'] = temp['item_cnt'] / temp['shop_total']
temp = temp[['shop_id', 'category_id', 'category_proportion']]

shops_cats = pd.merge(shops_cats, temp, on=['shop_id', 'category_id'], how='left')
shops_cats = shops_cats.fillna(0)
shops_cats = shops_cats.pivot(index='shop_id', columns='category_id', values='category_proportion')
shops_cats = shops_cats.fillna(0)  # in case some categories were missing

# Normalize the data
scaler = StandardScaler()
shops_scaled = scaler.fit_transform(shops_cats)

# Fit DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=2)  # You may need to tune these
shops_cats['shop_cluster'] = dbscan.fit_predict(shops_scaled).astype('int8')

# Adding these clusters to the shops dataframe
shops = shops.join(shops_cats['shop_cluster'], on='shop_id')






shops['shop_cluster'].value_counts()


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”
# ASSUMPTION: you already have `shops_scaled` (StandardScaler-normalized array)
# and `shops_cats` (the pivoted DataFrame of category proportions by shop_id).
# If not, run your preprocessing cell first.
# â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”

# 1) K-distance graph (k = min_samples = 2)
nbrs = NearestNeighbors(n_neighbors=2).fit(shops_scaled)
distances, _ = nbrs.kneighbors(shops_scaled)
k_dist = np.sort(distances[:, 1])

plt.figure()
plt.plot(k_dist)
plt.xlabel("Points sorted by 2nd-nearest-neighbor distance")
plt.ylabel("2nd NN distance")
plt.title("DBSCAN k-distance plot (min_samples=2)")
plt.show()


# 2) Grid-search eps â†’ #clusters & #noise
eps_vals = np.linspace(0.1, 10.0, 50)
records  = []
for eps in eps_vals:
    lbls = DBSCAN(eps=eps, min_samples=2).fit_predict(shops_scaled)
    n_clusters = len(set(lbls)) - (1 if -1 in lbls else 0)
    n_noise    = np.sum(lbls == -1)
    records.append((eps, n_clusters, n_noise))

eps_df = pd.DataFrame(records, columns=['eps','n_clusters','n_noise'])
print(eps_df)   # or inspect eps_df.head() / tail()


# 3) Pick an eps (e.g. ~1.5 from the elbow), run final DBSCANâ€¦
chosen_eps = 5.757143
final_lbls = DBSCAN(eps=chosen_eps, min_samples=2).fit_predict(shops_scaled)
shops_cats['shop_cluster'] = final_lbls.astype('int8')


# 4) 2D PCA visualization
pca = PCA(n_components=2)
coords2d = pca.fit_transform(shops_scaled)

plt.figure()
plt.scatter(coords2d[:,0], coords2d[:,1], c=final_lbls)
plt.title(f"DBSCAN Clusters (eps={chosen_eps})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


# 5) Join back to your shops DataFrame
shops['shop_cluster'] = shops_cats['shop_cluster']



#removing unused shop ids
shops.dropna(inplace=True)

#cleaning the name column
shops['shop_name'] = shops['shop_name'].str.lower()
shops['shop_name'] = shops['shop_name'].str.replace(r'[^\w\d\s]', ' ')

#creating a column for the type of shop
shops['shop_type'] = 'regular'

#there is some overlap in tc and mall, mall is given precedence
shops.loc[shops['shop_name'].str.contains(r'tc'), 'shop_type'] = 'tc'
shops.loc[shops['shop_name'].str.contains(r'mall|center|mega'), 'shop_type'] = 'mall'
shops.loc[shops['shop_id'].isin([9,20]), 'shop_type'] = 'special'
shops.loc[shops['shop_id'].isin([12,55]), 'shop_type'] = 'online'

#the first word of shop name is largely sufficient as a city feature
shops['shop_city'] = shops['shop_name'].str.split().str[0]
shops.loc[shops['shop_id'].isin([12,55]), 'shop_city'] = 'online'
shops.shop_city = le.fit_transform(shops.shop_city.values)
shops.shop_type = le.fit_transform(shops.shop_type.values)
shops.head()


#add shop information to the training dataframe
df = pd.merge(df, shops.drop(columns='shop_name'), on='shop_id', how='left')
df.head()


df['first_sale_day'] = df.groupby('item_id')['first_sale_day'].transform('max').astype('int16')
df.loc[df['first_sale_day']==0, 'first_sale_day'] = 1035
df['prev_days_on_sale'] = [max(idx) for idx in zip(df['first_day_of_month']-df['first_sale_day'],[0]*len(df))]
del df['first_day_of_month']


#freeing RAM, removing unneeded columns and encoding object columns
del sales, categories, shops, shops_cats, temp, temp2, test, dupes, item_map, 
df.head()


df['item_cnt_unclipped'] = df['item_cnt']
df['item_cnt'] = df['item_cnt'].clip(0, 20)


def downcast(df):
    #reduce size of the dataframe
    float_cols = [c for c in df if df[c].dtype in ["float64"]]
    int_cols = [c for c in df if df[c].dtype in ['int64']]
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int16')
    return df
df = downcast(df)


df['item_age'] = (df['date_block_num'] - df.groupby('item_id')['date_block_num'].transform('min')).astype('int8')
df['item_name_first4_age'] = (df['date_block_num'] - df.groupby('item_name_first4')['date_block_num'].transform('min')).astype('int8')
df['item_name_first6_age'] = (df['date_block_num'] - df.groupby('item_name_first6')['date_block_num'].transform('min')).astype('int8')
df['item_name_first11_age'] = (df['date_block_num'] - df.groupby('item_name_first11')['date_block_num'].transform('min')).astype('int8')
df['category_age'] = (df['date_block_num'] - df.groupby('category_id')['date_block_num'].transform('min')).astype('int8')
df['group_age'] = (df['date_block_num'] - df.groupby('group_id')['date_block_num'].transform('min')).astype('int8')
df['shop_age'] = (df['date_block_num'] - df.groupby('shop_id')['date_block_num'].transform('min')).astype('int8')


#indicates whether shops have previously sold the item
temp = df.query('item_cnt > 0').groupby(['item_id','shop_id']).agg({'date_block_num':'min'}).reset_index()
temp.columns = ['item_id', 'shop_id', 'item_shop_first_sale']
df = pd.merge(df, temp, on=['item_id','shop_id'], how='left')
df['item_shop_first_sale'] = df['item_shop_first_sale'].fillna(50)
#item age that stays at 0 if a shop hasn't sold the item
df['item_age_if_shop_sale'] = (df['date_block_num'] > df['item_shop_first_sale']) * df['item_age']
#the length of time an item has been for sale without being sold at individual shops
df['item_age_without_shop_sale'] = (df['date_block_num'] <= df['item_shop_first_sale']) * df['item_age']
del df['item_shop_first_sale']


def agg_cnt_col(df, merging_cols, new_col,aggregation):
    temp = df.groupby(merging_cols).agg(aggregation).reset_index()
    temp.columns = merging_cols + [new_col]
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

#individual items across all shops
df = agg_cnt_col(df, ['date_block_num','item_id'],'item_cnt_all_shops',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','category_id','shop_id'],'item_cnt_all_shops_median',{'item_cnt':'median'}) 
#all items in category at individual shops
df = agg_cnt_col(df, ['date_block_num','category_id','shop_id'],'category_cnt',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','category_id','shop_id'],'category_cnt_median',{'item_cnt':'median'}) 
#all items in category across all shops
df = agg_cnt_col(df, ['date_block_num','category_id'],'category_cnt_all_shops',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','category_id'],'category_cnt_all_shops_median',{'item_cnt':'median'})
#all items in group
df = agg_cnt_col(df, ['date_block_num','group_id','shop_id'],'group_cnt',{'item_cnt':'mean'})
#all items in group across all shops
df = agg_cnt_col(df, ['date_block_num','group_id'],'group_cnt_all_shops',{'item_cnt':'mean'})
#all items at individual shops
df = agg_cnt_col(df, ['date_block_num','shop_id'],'shop_cnt',{'item_cnt':'mean'})
#all items at all shops within the city
df = agg_cnt_col(df, ['date_block_num','shop_city'],'city_cnt',{'item_cnt':'mean'})


def new_item_sales(df, merging_cols, new_col):
    temp = (
        df
        .query('item_age==0')
        .groupby(merging_cols)['item_cnt']
        .mean()
        .reset_index()
        .rename(columns={'item_cnt': new_col})
    )
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

#mean units sold of new item in category at individual shop
df = new_item_sales(df, ['date_block_num','category_id','shop_id'], 'new_items_in_cat')
#mean units sold of new item in category across all shops
df = new_item_sales(df, ['date_block_num','category_id'], 'new_items_in_cat_all_shops')


def agg_price_col(df, merging_cols, new_col):
    temp = df.groupby(merging_cols).agg({'revenue':'sum','item_cnt_unclipped':'sum'}).reset_index()
    temp[new_col] = temp['revenue']/temp['item_cnt_unclipped']
    temp = temp[merging_cols + [new_col]]
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

#average item price
df = agg_price_col(df,['date_block_num','item_id'],'item_price')
#average price of items in category
df = agg_price_col(df,['date_block_num','category_id'],'category_price')
#average price of all items
df = agg_price_col(df,['date_block_num'],'block_price')


df = downcast(df)


def lag_feature(df, lag, col, merge_cols):        
    temp = df[merge_cols + [col]]
    temp = temp.groupby(merge_cols).agg({f'{col}':'first'}).reset_index()
    temp.columns = merge_cols + [f'{col}_lag{lag}']
    temp['date_block_num'] += lag
    df = pd.merge(df, temp, on=merge_cols, how='left')
    df[f'{col}_lag{lag}'] = df[f'{col}_lag{lag}'].fillna(0).astype('float32')
    return df


lag12_cols = {
    'item_cnt':['date_block_num', 'shop_id', 'item_id'],
    'item_cnt_all_shops':['date_block_num', 'item_id'],
    'category_cnt':['date_block_num', 'shop_id', 'category_id'],
    'category_cnt_all_shops':['date_block_num', 'category_id'],
    'group_cnt':['date_block_num', 'shop_id', 'group_id'],
    'group_cnt_all_shops':['date_block_num', 'group_id'],
    'shop_cnt':['date_block_num', 'shop_id'],
    'city_cnt':['date_block_num', 'shop_city'],
    'new_items_in_cat':['date_block_num', 'shop_id', 'category_id'],
    'new_items_in_cat_all_shops':['date_block_num', 'category_id']
}
for col,merge_cols in lag12_cols.items():
    df[f'{col}_lag1to12'] = 0
    for i in range(1,13):
        df = lag_feature(df, i, col, merge_cols)
        df[f'{col}_lag1to12'] += df[f'{col}_lag{i}']
        if i > 2:
            del df[f'{col}_lag{i}']
    if col == 'item_cnt':
        del df[f'{col}_lag1']
        del df[f'{col}_lag2']        
    else:
        del df[col]


lag2_cols = {
    'item_cnt_unclipped':['date_block_num', 'shop_id', 'item_id'],
    'item_cnt_all_shops_median':['date_block_num', 'item_id'],
    'category_cnt_median':['date_block_num', 'shop_id', 'category_id'],
    'category_cnt_all_shops_median':['date_block_num', 'category_id']
}
for col in lag2_cols:
    df = lag_feature(df, 1, col, merge_cols)
    df = lag_feature(df, 2, col, merge_cols)
    if col!='item_cnt_unclipped':
        del df[col]


df['item_cnt_diff'] = df['item_cnt_unclipped_lag1']/df['item_cnt_lag1to12']
df['item_cnt_all_shops_diff'] = df['item_cnt_all_shops_lag1']/df['item_cnt_all_shops_lag1to12']
df['category_cnt_diff'] = df['category_cnt_lag1']/df['category_cnt_lag1to12']
df['category_cnt_all_shops_diff'] = df['category_cnt_all_shops_lag1']/df['category_cnt_all_shops_lag1to12']


df = lag_feature(df, 1, 'category_price',['date_block_num', 'category_id'])
df = lag_feature(df, 1, 'block_price',['date_block_num'])
del df['category_price'], df['block_price']


df.loc[(df['item_age']>0) & (df['item_cnt_lag1to12'].isna()), 'item_cnt_lag1to12'] = 0
df.loc[(df['category_age']>0) & (df['category_cnt_lag1to12'].isna()), 'category_cnt_lag1to12'] = 0
df.loc[(df['group_age']>0) & (df['group_cnt_lag1to12'].isna()), 'group_cnt_lag1to12'] = 0


df['item_cnt_lag1to12'] /= [min(idx) for idx in zip(df['item_age'],df['shop_age'],[12]*len(df))]
df['item_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['item_age'],[12]*len(df))]
df['category_cnt_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],df['shop_age'],[12]*len(df))]
df['category_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],[12]*len(df))]
df['group_cnt_lag1to12'] /= [min(idx) for idx in zip(df['group_age'],df['shop_age'],[12]*len(df))]
df['group_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['group_age'],[12]*len(df))]
df['city_cnt_lag1to12'] /= [min(idx) for idx in zip(df['date_block_num'],[12]*len(df))]
df['shop_cnt_lag1to12'] /= [min(idx) for idx in zip(df['shop_age'],[12]*len(df))]
df['new_items_in_cat_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],df['shop_age'],[12]*len(df))]
df['new_items_in_cat_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],[12]*len(df))]


df = downcast(df)


def past_information(df, merging_cols, new_col, aggregation):
    temp = []
    for i in range(1,35):
        block = df.query(f'date_block_num < {i}').groupby(merging_cols).agg(aggregation).reset_index()
        block.columns = merging_cols + [new_col]
        block['date_block_num'] = i
        block = block[block[new_col]>0]
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=['date_block_num']+merging_cols, how='left')
    return df

#average item price in latest block item was sold
df = past_information(df, ['item_id'],'last_item_price',{'item_price':'last'})
#total units of item sold at individual shop
df = past_information(df, ['shop_id','item_id'],'item_cnt_sum_alltime',{'item_cnt':'sum'})
#total units of item sold at all shops
df = past_information(df, ['item_id'],'item_cnt_sum_alltime_allshops',{'item_cnt':'sum'})

#these columns are no longer needed, and would cause data leakage if retained
del df['revenue'], df['item_cnt_unclipped'], df['item_price']


df['relative_price_item_block_lag1'] = df['last_item_price']/df['block_price_lag1']


df['item_cnt_per_day_alltime'] = (df['item_cnt_sum_alltime']/df['prev_days_on_sale']).fillna(0)
df['item_cnt_per_day_alltime_allshops'] = (df['item_cnt_sum_alltime_allshops']/df['prev_days_on_sale']).fillna(0)


import gc
gc.collect()
df = downcast(df)


def matching_name_cat_age(df,n,all_shops):
    temp_cols = [f'same_name{n}catage_cnt','date_block_num', f'item_name_first{n}','item_age','category_id']
    if all_shops:
        temp_cols[0] += '_all_shops'
    else:
        temp_cols += ['shop_id']
    temp = []
    for i in range(1,35):
        block = (
            df
            .query(f'date_block_num < {i}')
            .groupby(temp_cols[2:])
            .agg({'item_cnt':'mean'})
            .reset_index()
            .rename(columns={'item_cnt':temp_cols[0]})
        )
        block = block[block[temp_cols[0]]>0]
        block['date_block_num'] = i
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=temp_cols[1:], how='left')
    return df

for n in [4,6,11]:
    for all_shops in [True,False]:
        df = matching_name_cat_age(df,n,all_shops)


#assign appropriate datatypes
df = downcast(df)
int8_cols = [
    'item_cnt','month','group_id','shop_type',
    'shop_city','shop_id','date_block_num','category_id',
    'item_age',
]
int16_cols = [
    'item_id','item_name_first4',
    'item_name_first6','item_name_first11'
]
for col in int8_cols:
    df[col] = df[col].astype('int8')
for col in int16_cols:
    df[col] = df[col].astype('int16')


def nearby_item_data(df,col):
    if col in ['item_cnt_unclipped_lag1','item_cnt_lag1to12']:
        cols = ['date_block_num', 'shop_id', 'item_id']
        temp = df[cols + [col]] 
    else:
        cols = ['date_block_num', 'item_id']
        temp = df.groupby(cols).agg({col:'first'}).reset_index()[cols + [col]]   
    
    temp.columns = cols + [f'below_{col}']
    temp['item_id'] += 1
    df = pd.merge(df, temp, on=cols, how='left')
    
    temp.columns = cols + [f'above_{col}']
    temp['item_id'] -= 2
    df = pd.merge(df, temp, on=cols, how='left')
    
    return df

item_cols = ['item_cnt_unclipped_lag1','item_cnt_lag1to12',
             'item_cnt_all_shops_lag1','item_cnt_all_shops_lag1to12']
for col in item_cols:
    df = nearby_item_data(df,col)
    
del temp


results = Counter()
items['item_name'].str.split().apply(results.update)

words = []
cnts = []
for key, value in results.items():
    words.append(key)
    cnts.append(value)
    
counts = pd.DataFrame({'word':words,'count':cnts})
common_words = counts.query('count>200').word.to_list()
for word in common_words:
    items[f'{word}_in_name'] = items['item_name'].str.contains(word).astype('int8')
drop_cols = [
    'item_id','category_id','item_name','item_name_first4',
    'item_name_first6','item_name_first11',
    'category_name','group_name','group_id'
]
items = items.drop(columns=drop_cols)


#join these word vectors to the training dataframe
df = df.join(items, on='item_id')


def binary_encode(df, letters, cols):
    encoder = ce.BinaryEncoder(cols=[f'item_name_first{letters}'], return_df=True)
    temp = encoder.fit_transform(df[f'item_name_first{letters}'])
    df = pd.concat([df,temp], axis=1)
    del df[f'item_name_first{letters}_0']
    name_cols = [f'item_name_first{letters}_{x}' for x in range(1,cols)]
    df[name_cols] = df[name_cols].astype('int8')
    return df

df = binary_encode(df, 11, 15)
    
del df['item_name_first4'], df['item_name_first6']


#save dataframe for later use
df.to_pickle('df_complete.pkl')


#Reset the kernel to clear memory.
%reset -f


import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pd.set_option('display.max_rows', 160)
pd.set_option('display.max_columns', 160)
pd.set_option('display.max_colwidth', 30)

import warnings
warnings.filterwarnings("ignore")


#load the saved training dataframe
df = pd.read_pickle('../input/files-top-scoring-notebook-output-exploration/df_complete.pkl')

X_train = df[~df.date_block_num.isin([0,1,33,34])]
y_train = X_train['item_cnt']
del X_train['item_cnt']

X_val = df[df['date_block_num']==33]
y_val = X_val['item_cnt']
del X_val['item_cnt']

X_test = df[df['date_block_num']==34].drop(columns='item_cnt')
X_test = X_test.reset_index()
del X_test['index']

#free memory
del df


def build_lgb_model(params, X_train, X_val, y_train, y_val, cat_features):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)
    model = lgb.train(params=params, train_set=lgb_train, valid_sets=(lgb_train, lgb_val), verbose_eval=50,
                     categorical_feature=cat_features)
    return model


#skip this cell if directly loading saved model 
params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'num_leaves': 1023,
    'min_data_in_leaf':10,
    'feature_fraction':0.7,
    'learning_rate': 0.01,
    'num_rounds': 1000,
    'early_stopping_rounds': 30,
    'seed': 1
}
#designating the categorical features which should be focused on
cat_features = ['category_id','month','shop_id','shop_city']

lgb_model = build_lgb_model(params, X_train, X_val, y_train, y_val, cat_features)

#save model for later use
lgb_model.save_model('initial_lgb_model.txt')


#lgb_model = lgb.Booster(model_file='../input/files-top-scoring-notebook-output-exploration/initial_lgb_model.txt')#
#lgb_model.params['objective'] = 'rmse'


submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission['item_cnt_month'] = lgb_model.predict(X_test).clip(0,20)
submission[['ID', 'item_cnt_month']].to_csv('initial_lgb_submission.csv', index=False)
submission[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)


#load item/category/shop information to make analysing the results easier
categories = pd.read_csv('../input/predict-future-sales-eng-translation/categories.csv')
categories['group_name'] = categories['category_name'].str.extract(r'(^[\w\s]*)')
categories['group_name'] = categories['group_name'].str.strip()

items = pd.read_csv('../input/predict-future-sales-eng-translation/items.csv')
items['item_name'] = items['item_name'].str.lower()
for i in [r'[^\w\d\s]', r'\bthe\b', r'\bin\b', r'\bfor\b', r'\bof\b', r'\bd\b', r'\bis\b', r'\bon\b']:
    items['item_name'] = items['item_name'].str.replace(i, ' ')
items['item_name'] = items['item_name'].str.replace(' ', '')
items = items.join(categories.set_index('category_id'), on='category_id')

shops = pd.read_csv('../input/predict-future-sales-eng-translation/shops.csv')
shops['shop_name'] = shops['shop_name'].str.lower()
shops['shop_name'] = shops['shop_name'].str.replace(r'[^\w\d\s]', ' ')
shops['shop_city'] = shops['shop_name'].str.split().str[0]
shops.loc[shops['shop_id'].isin([12,55]), 'shop_city'] = 'online'


X_train['lgb_pred'] = lgb_model.predict(X_train).clip(0,20)
X_train['target'] = y_train
X_train['sq_err'] = (X_train['lgb_pred']-X_train['target'])**2

X_val['lgb_pred'] = lgb_model.predict(X_val).clip(0,20)
X_val['target'] = y_val
X_val['sq_err'] = (X_val['lgb_pred']-X_val['target'])**2

X_test['lgb_pred'] = lgb_model.predict(X_test).clip(0,20)


data = X_train.groupby('date_block_num').agg({'lgb_pred':'mean','target':'mean','sq_err':'mean'}).reset_index()
data['new_item_rmse'] = np.sqrt(X_train.query('item_age<=1').groupby('date_block_num').agg({'sq_err':'mean'}).sq_err)
data['old_item_rmse'] = np.sqrt(X_train.query('item_age>1').groupby('date_block_num').agg({'sq_err':'mean'}).sq_err)
data = data.append([
    {'date_block_num':33,
     'target':X_val.target.mean(),
     'lgb_pred':X_val.lgb_pred.mean(),
     'sq_err':np.sqrt(X_val.sq_err.mean()),
     'old_item_rmse':np.sqrt(X_val.query('item_age>1').sq_err.mean()),
     'new_item_rmse':np.sqrt(X_val.query('item_age<=1').sq_err.mean())},
    {'date_block_num':34,
     'target':0,
     'lgb_pred':X_test.lgb_pred.mean(),
     'sq_err':0,
     'old_item_rmse':0,
     'new_item_rmse':0}
    ],
    ignore_index=True
)
data['date'] = [x[:7] for x in pd.date_range(start='2013-03',end='2015-09',freq='MS').astype('str')]+['Validation','Test']


temp = X_test.drop(columns='lgb_pred').sample(10000)
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(temp)
shap.summary_plot(shap_values, temp, max_display=30)


fig = go.Figure(data=[
    go.Bar(name='Prediction',x=data.date,y=data.lgb_pred),
    go.Bar(name='Target',x=data.date,y=data.target)
])
fig.update_layout(
    title='Mean Prediction and Target Values by Month',
    xaxis={'title':'Month','type':'category'},
    yaxis={'title':'Mean Value'},
    legend={'yanchor':'top','y':1.05,'xanchor':'left','x':0.01},
    template='plotly_dark'
)

text = '''
    The model may not be adequately accounting for<br>
    the December sales spike. There is no systemic under or<br>
    over-prediction visible for the test month, November.
'''
fig.add_annotation(
    yref='paper', y=1.1,
    xref='paper', x=0.7,
    text=text,
    font={'size':11},
    showarrow=False)

text = '''
    Prediction mean is very close to the  <br> 
    target mean in our validation set.
'''
fig.add_annotation(
    xref='paper', x=0.95,
    yref='paper', y=0.58,
    text=text,
    font={'size':11},
    showarrow=True, arrowhead=1)
fig.show()


fig = go.Figure(data=[go.Bar(
    x=data.date[:-1],
    y=data.sq_err[:-1],
    marker_color=[x%12 for x in range(2,34)]
)])
fig.update_layout(
    title='RMSE by Month',
    xaxis={'title':'Month','type':'category'},
    yaxis={'title':'RMSE'},
    template='plotly_dark'
)
text = '''
    In the training set, RMSE is highest in December. <br>
    
'''
fig.add_annotation(
    yref='paper', y=0.9,
    xref='paper', x=0.1,
    text=text,
    font={'size':11},
    showarrow=False)
text = '''
    Validation RMSE is higher than in any month of <br>
    the training set. This is expected given the <br>
    gap in training set and validation set RMSE we <br>
    saw when training the model.
'''
fig.add_annotation(
    yref='paper', y=1.1,
    xref='paper', x=1,
    text=text,
    font={'size':11},
    showarrow=False)
fig.show()


temp = X_val.groupby('item_age').agg({'sq_err':'mean','target':'mean'}).reset_index()
temp['sq_err'] = np.sqrt(temp['sq_err'])
fig = go.Figure(data=[
    go.Bar(name='RMSE',x=temp.item_age,y=temp.sq_err)
])
fig.update_layout(
    title='Validation set RMSE by Item Age',
    xaxis={'title':'Item Age','type':'category'},
    yaxis={'title':'RMSE'},
    template='plotly_dark'
)
text = '''
    RMSE is much higher for items with age 0 or 1. There is a <br>
    sharp decrease in RMSE between ages 0 and 2, then a much <br>
    flatter downwards trend as age increases beyond 2.
'''
fig.add_annotation(
    yref='paper', y=0.8,
    xref='paper', x=0.04,
    text=text,
    font={'size':11},
    showarrow=False)
fig.show()


fig = go.Figure(data=[
    go.Bar(name='Item Age <=1 RMSE',x=data.date[:-1],y=data.new_item_rmse[:-1]),
    go.Bar(name='Item Age > 1 RMSE',x=data.date[:-1],y=data.old_item_rmse[:-1])
])
fig.update_layout(
    title='RMSE by Month for New and Old Items',
    xaxis={'title':'Month','type':'category'},
    yaxis={'title':'Mean Value'},
    legend={'yanchor':'top','y':1.05,'xanchor':'left','x':0.01},
    template='plotly_dark'
)
text = '''
    For older items, performance on the validation set is no worse than <br>
    performance on the training set. The higher overall validation set <br>
    RMSE is caused by a significantly higher RMSE for new items.
'''
fig.add_annotation(yref='paper', y=1.1,
                   xref='paper', x=0.93,
                   text=text,
                   font={'size':11},
                   showarrow=False)
fig.show()


df = pd.read_pickle('../input/files-top-scoring-notebook-output-exploration/df_complete.pkl')
(
df
[df['category_id'].isin(X_test.category_id.unique())]
.query('item_cnt>0')
.groupby('category_id')
.agg({
    'category_age':'max',
    'shop_id':['nunique','unique'],
    'item_cnt':'sum'
    })
.join(categories['category_name'])
.join(
    X_test       
    .groupby('category_id')
    .agg({'item_id':'nunique'})
    .rename(columns={'item_id':'test_set_items'})
)
.sort_values(('shop_id', 'nunique'))
.head(20)
)


(
X_val
[X_val['category_id'].isin(X_test.category_id.unique())]
.groupby('shop_id')
.agg({
    'sq_err':'mean',
    'target':'mean',
    'lgb_pred':'mean'
})
.join(
    X_test
    .rename(columns={'lgb_pred':'test_pred'})
    .groupby('shop_id')
    .agg({'test_pred':'mean'})
)
.join(shops['shop_name'])
.sort_values('sq_err', ascending=False)
.head(20)
)


(
X_val
[X_val['category_id'].isin(X_test.category_id.unique())]
.groupby('category_id')
.agg({
    'sq_err':['sum','mean'],
    'target':'mean',
    'lgb_pred':['sum','mean'],
    'item_id':'nunique'
})
.join(
    X_test
    .rename(columns={'lgb_pred':'test_pred','item_id':'test_items'})
    .groupby('category_id')        
    .agg({
        'test_pred':['sum','mean'],
        'test_items':'nunique'
    }),
    on='category_id'
)
.join(categories)
.sort_values(('sq_err', 'mean'), ascending=False)
.head(20)
)


#edit this value to look through the biggest problem items in each category
CATEGORY = 20
(
items[
    items['item_id'].isin(X_val.item_id.unique()) & 
    items['item_id'].isin(X_test.item_id.unique())
]
[['category_id','category_name','item_id','item_name']]
.join(
    X_test
    .rename(columns={'lgb_pred':'test_pred'})
    .groupby('item_id')        
    .agg({'test_pred':'mean'}),
    on='item_id'
)
.join(
    X_val
    .groupby('item_id')
    .agg({
        'lgb_pred':'mean',
        'target':'mean',
        'sq_err':'mean',
        'same_name4catage_cnt_all_shops':'first',
        'new_items_in_cat_all_shops_lag1to12':'first',
        'item_cnt_all_shops_lag1':'first',
        'category_cnt_all_shops_lag1':'first',
        'item_cnt_sum_alltime_allshops':'first',
        'prev_days_on_sale':'first'
    })
    .rename(columns={
        'lgb_pred':'val_pred',
        'target':'val_target',  
    }),
    on='item_id'
)
.query(f'category_id=={CATEGORY}')
.sort_values('sq_err',ascending=False)
.rename(columns={
    'same_name4catage_cnt_all_shops':'name4mean',
    'new_items_in_cat_all_shops_lag1to12':'new_in_cat_mean',
    'item_cnt_all_shops_lag1':'item_cnt_lag1',
    'category_cnt_all_shops_lag1':'cat_cnt_lag1',
    'category_id':'cat',
    'item_cnt_sum_alltime_allshops':'item_cnt_alltime'
})
.head(20)
)


CATEGORY = 20  
(
items[items['item_id'].isin(X_test.item_id.unique())]
[['category_id','category_name','item_id','item_name']]
.join(
    X_test
    .groupby('item_id')
    .agg({
        'lgb_pred':'mean',
        'same_name4catage_cnt_all_shops':'first',
        'new_items_in_cat_all_shops_lag1to12':'first',
        'item_cnt_all_shops_lag1':'first',
        'category_cnt_all_shops_lag1':'first',
        'item_cnt_sum_alltime_allshops':'first',
        'prev_days_on_sale':'first'
    })
    .rename(columns={'lgb_pred':'test_pred'}),
    on='item_id'
)
.query(f'category_id=={CATEGORY}')
.sort_values('test_pred',ascending=False)
.rename(columns={
     'same_name4catage_cnt_all_shops':'name4mean',
     'new_items_in_cat_all_shops_lag1to12':'new_in_cat_mean',
     'item_cnt_all_shops_lag1':'item_cnt_lag1',
     'category_cnt_all_shops_lag1':'cat_cnt_lag1',
     'item_category_id':'cat',
     'item_cnt_sum_alltime_allshops':'item_cnt_alltime'
})
.head(20)
)

