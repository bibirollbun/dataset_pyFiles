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


sales_train_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_categories_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')


sales_train_df.head()


sales_train_df.info(show_counts=True)


shops_df.head()


shops_df.info()


items_df.head()


items_df.info()


item_categories_df.head()


item_categories_df.info()


test_df.head()


train_df = sales_train_df.merge(shops_df, on='shop_id', how='left')
train_df = train_df.merge(items_df, on='item_id', how='left')
train_df = train_df.merge(item_categories_df, on='item_category_id', how='left')


train_df.head()


def resumetable(df):
    print(f'shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['data_type'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'feature'})
    summary['missing'] = df.isnull().sum().values
    summary['unique'] = df.nunique().values
    summary['first'] = df.iloc[0].values
    summary['last'] = df.iloc[-1].values
    return summary


resumetable(train_df)


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


sns.boxplot(y='item_cnt_day', data=train_df)


sns.boxplot(y='item_price', data=train_df)





figure, ax = plt.subplots()
figure.set_size_inches(12, 6)

group_month_sum_df = train_df.groupby('date_block_num').agg({'item_cnt_day': 'sum'})
group_month_sum_df = group_month_sum_df.reset_index()

sns.barplot(x='date_block_num', y='item_cnt_day', data=group_month_sum_df)


train_df['item_category_id'].nunique()


figure, ax = plt.subplots()
figure.set_size_inches(12, 6)

group_cat_sum_df = train_df.groupby('item_category_id').agg({'item_cnt_day': 'sum'})
group_cat_sum_df = group_cat_sum_df.reset_index()

group_cat_sum_df = group_cat_sum_df[group_cat_sum_df['item_cnt_day'] > 1000]

sns.barplot(x='item_category_id', y='item_cnt_day', data=group_cat_sum_df)
ax.tick_params(axis='x', labelrotation=90)


figure, ax = plt.subplots()
figure.set_size_inches(12, 6)

group_shop_sum_df = train_df.groupby('shop_id').agg({'item_cnt_day': 'sum'})
group_shop_sum_df = group_shop_sum_df.reset_index()

group_shop_sum_df = group_shop_sum_df[group_shop_sum_df['item_cnt_day'] > 10000]

sns.barplot(x='shop_id', y='item_cnt_day', data=group_shop_sum_df)
ax.tick_params(axis='x', labelrotation=90)


sales_train_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_categories_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')


def downcast(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        dtype_name = df[col].dtype.name
        if dtype_name == 'object':
            pass
        elif dtype_name == 'bool':
            df[col] = df[col].astype('int8')
        elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('{:.1f}% compressed'.format(100 * (start_mem - end_mem) / start_mem))
    return df


all_df = [sales_train_df, shops_df, items_df, item_categories_df, test_df]
for df in all_df:
    df = downcast(df)


sales_train_df = sales_train_df[sales_train_df['item_price'] > 0]
sales_train_df = sales_train_df[sales_train_df['item_price'] < 50000]

sales_train_df = sales_train_df[sales_train_df['item_cnt_day'] < 700]


print(shops_df['shop_name'][0] + "\n" + shops_df['shop_name'][57])
print("====================")
print(shops_df['shop_name'][1] + "\n" + shops_df['shop_name'][58])
print("====================")
print(shops_df['shop_name'][10] + "\n" + shops_df['shop_name'][11])
print("====================")
print(shops_df['shop_name'][39] + "\n" + shops_df['shop_name'][40])
print("====================")



sales_train_df.loc[sales_train_df['shop_id'] == 0, 'shop_id'] = 57
sales_train_df.loc[sales_train_df['shop_id'] == 1, 'shop_id'] = 58
sales_train_df.loc[sales_train_df['shop_id'] == 11, 'shop_id'] = 10
sales_train_df.loc[sales_train_df['shop_id'] == 40, 'shop_id'] = 39


shops_df['city'] = shops_df['shop_name'].apply(lambda x: x.split()[0])


unique_cities = pd.DataFrame(shops_df['city'].unique(), columns=['city'])

# Save to CSV
unique_cities.to_csv("cities.csv", index=False)


df = pd.read_csv('/kaggle/input/translate/translated_cities.csv')
df


# Merge shops with df on 'city'
shops_df = shops_df.merge(df[['city', 'cities_en']], on='city', how='left')

# Replace old 'city' with 'cities_en'
shops_df['city'] = shops_df['cities_en']
shops_df.drop(columns=['cities_en'], inplace=True)



shops_df['city'].unique()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
shops_df['city'] = label_encoder.fit_transform(shops_df['city'])


shops_df = shops_df.drop('shop_name', axis=1)
shops_df.head()


items_df = items_df.drop(['item_name'], axis=1)


items_df['first_sale_month'] = sales_train_df.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
items_df.head()


items_df[items_df['first_sale_month'].isna()]


items_df['first_sale_month'] = items_df['first_sale_month'].fillna(34)


item_categories_df['large_category'] = item_categories_df['item_category_name'].apply(lambda x: x.split()[0])


item_categories_df['large_category'].value_counts()


def make_etc(x):
    if len(item_categories_df[item_categories_df['large_category'] == x]) >= 5:
        return x
    else:
        return 'etc'
    
item_categories_df['large_category'] = item_categories_df['large_category'].apply(make_etc)


item_categories_df.head()


label_encoder = LabelEncoder()

item_categories_df['large_category'] = label_encoder.fit_transform(item_categories_df['large_category'])


item_categories_df = item_categories_df.drop('item_category_name', axis=1)


item_categories_df.head()


from itertools import product

train_df = []
for i in sales_train_df['date_block_num'].unique():
    all_shop_df = sales_train_df.loc[sales_train_df['date_block_num'] == i, 'shop_id'].unique()
    all_item_df = sales_train_df.loc[sales_train_df['date_block_num'] == i, 'item_id'].unique()
    train_df.append(np.array(list(product([i], all_shop_df, all_item_df))))
    
idx_features = ['date_block_num', 'shop_id', 'item_id']
train_df = pd.DataFrame(np.vstack(train_df), columns=idx_features)


sales_train_df


group_df = sales_train_df.groupby(idx_features).agg({'item_cnt_day': 'sum', 'item_price': 'mean'})

group_df = group_df.reset_index()
group_df = group_df.rename(columns={'item_cnt_day': 'item_cnt_month', 'item_price': 'item_price_avg'})

train_df = train_df.merge(group_df, on=idx_features, how='left')
train_df.head()


import gc

del group_df
gc.collect()


group_df = sales_train_df.groupby(idx_features).agg({'item_cnt_day': 'count'})
group_df = group_df.reset_index()
group_df = group_df.rename(columns={'item_cnt_day': 'sales_days_per_month'})

train_df = train_df.merge(group_df, on=idx_features, how='left')
train_df.head()


import gc

del group_df
gc.collect()


test_df['date_block_num'] = 34

all_df = pd.concat([train_df, test_df.drop('ID', axis=1)],
                  ignore_index=True,
                  keys=idx_features)

all_df = all_df.fillna(0)
all_df.head()


all_df = all_df.merge(shops_df, on='shop_id', how='left')
all_df = all_df.merge(items_df, on='item_id', how='left')
all_df = all_df.merge(item_categories_df, on='item_category_id', how='left')

all_df = downcast(all_df)


del shops_df, items_df, item_categories_df
gc.collect()


def add_mean_features(df, mean_features, idx_features):
    assert(idx_features[0] == 'date_block_num') and len(idx_features) in [2, 3]
    
    if len(idx_features) == 2:
        feature_name = 'avg_sales_per_' + idx_features[1]
    else:
        feature_name = 'avg_sales_per_' + idx_features[1] + '_and_' + idx_features[2]
        
    group_df = df.groupby(idx_features).agg({'item_cnt_month': 'mean'})
    group_df = group_df.reset_index()
    group_df = group_df.rename(columns={'item_cnt_month': feature_name})
    
    df = df.merge(group_df, on=idx_features, how='left')
    df = downcast(df, verbose=False)
    
    mean_features.append(feature_name)
    
    del group_df
    gc.collect()
    
    return df, mean_features


item_mean_features = []

all_df, item_mean_features = add_mean_features(df=all_df, 
                                              mean_features=item_mean_features,
                                              idx_features=['date_block_num', 'item_id'])

all_df, item_mean_features = add_mean_features(df=all_df, 
                                              mean_features=item_mean_features,
                                              idx_features=['date_block_num', 'item_id', 'city'])


item_mean_features


shop_mean_features = []

all_df, shop_mean_features = add_mean_features(df=all_df, 
                                               mean_features=shop_mean_features,
                                               idx_features=['date_block_num', 'shop_id', 'item_category_id'])


shop_mean_features


def add_lag_features(df, lag_features_to_clip, idx_features, lag_feature, nlags=3, clip=False):
    df_temp = df[idx_features + [lag_feature]].copy()
    
    for i in range(1, nlags + 1):
        lag_feature_name = lag_feature + '_time_lag_' + str(i)
        df_temp.columns = idx_features + [lag_feature_name]
        df_temp['date_block_num'] += i
        df = df.merge(df_temp.drop_duplicates(),
                      on=idx_features,
                      how='left')
        df[lag_feature_name] = df[lag_feature_name].fillna(0)
        if clip:
            lag_features_to_clip.append(lag_feature_name)
            
    df = downcast(df, False)
    
    del df_temp
    gc.collect()
    
    return df, lag_features_to_clip


lag_features_to_clip = []
idx_features = ['date_block_num', 'shop_id', 'item_id']

all_df, lag_features_to_clip = add_lag_features(df=all_df,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='item_cnt_month',
                                                nlags=3,
                                                clip=True)


all_df.head().T


lag_features_to_clip


all_df, lag_features_to_clip = add_lag_features(df=all_df,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='sales_days_per_month',
                                                nlags=3)



all_df, lag_features_to_clip = add_lag_features(df=all_df,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='item_price_avg',
                                                nlags=3)


for item_mean_feature in item_mean_features:
    all_df, lag_features_to_clip = add_lag_features(df=all_df,
                                                   lag_features_to_clip=lag_features_to_clip,
                                                   idx_features=idx_features,
                                                   lag_feature=item_mean_feature,
                                                   nlags=3,
                                                   clip=True)
all_df = all_df.drop(item_mean_features, axis=1)


for shop_mean_feature in shop_mean_features:
    all_df, lag_features_to_clip = add_lag_features(df=all_df,
                                                   lag_features_to_clip=lag_features_to_clip,
                                                   idx_features=['date_block_num', 'shop_id', 'item_category_id'],
                                                   lag_feature=shop_mean_feature,
                                                   nlags=3,
                                                   clip=True)
all_df = all_df.drop(shop_mean_features, axis=1)


all_df = all_df.drop(all_df[all_df['date_block_num'] < 3].index)


all_df.head().T


all_df['avg_monthly_sales_time_lag'] = all_df[['item_cnt_month_time_lag_1',
                                               'item_cnt_month_time_lag_2',
                                               'item_cnt_month_time_lag_3']].mean(axis=1)


all_df[lag_features_to_clip + ['item_cnt_month', 'avg_monthly_sales_time_lag']] = all_df[lag_features_to_clip + ['item_cnt_month', 'avg_monthly_sales_time_lag']].clip(0, 20)


all_df.head().T


all_df['time_lag_change_1'] = all_df['item_cnt_month_time_lag_1'] / all_df['item_cnt_month_time_lag_2']
all_df['time_lag_change_1'] = all_df['time_lag_change_1'].replace([np.inf, -np.inf], np.nan).fillna(0)

all_df['time_lag_change_2'] = all_df['item_cnt_month_time_lag_2'] / all_df['item_cnt_month_time_lag_3']
all_df['time_lag_change_2'] = all_df['time_lag_change_2'].replace([np.inf, -np.inf], np.nan).fillna(0)


all_df['new_arrival'] = all_df['first_sale_month'] == all_df['date_block_num']


all_df['period_after_first_sale'] = all_df['date_block_num'] - all_df['new_arrival']


all_df['month'] = all_df['date_block_num'] % 12


all_df = all_df.drop(['first_sale_month', 'item_price_avg', 'sales_days_per_month'], axis=1)


all_df = downcast(all_df, False)


all_df.info()


all_df.info()


X_train = all_df[all_df['date_block_num'] <= 33]
X_train = X_train.drop(['item_cnt_month'], axis=1)

X_test = all_df[all_df['date_block_num'] == 34]
X_test = X_test.drop(['item_cnt_month'], axis=1)

y_train = all_df[all_df['date_block_num'] <= 33]['item_cnt_month']


del all_df
gc.collect()


from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
test_pred_cat = np.zeros(len(X_test))
test_pred_lgb = np.zeros(len(X_test))

cat_features = ['shop_id', 'city', 'item_category_id', 'large_category', 'month']

# LightGBM requires categorical columns to be category dtype
for col in cat_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 1️⃣ Cross-validation for base models
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    print(f'Fold {fold + 1}/{N_SPLITS}')
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    # --- CatBoost with GPU
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        eval_metric='RMSE',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=0
    )
    cat_model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_cat[valid_idx] = cat_model.predict(X_val)
    test_pred_cat += cat_model.predict(X_test) / N_SPLITS

    # --- LightGBM with GPU
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_valid = lgb.Dataset(X_val, y_val)
    lgb_model = lgb.train(
        {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'verbosity': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        },
        train_set=lgb_train,
        valid_sets=[lgb_valid],
        num_boost_round=2000,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    oof_lgb[valid_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_SPLITS

# 2️⃣ Meta model
X_meta_train = np.vstack([oof_cat, oof_lgb]).T
X_meta_test = np.vstack([test_pred_cat, test_pred_lgb]).T

meta_model = ElasticNetCV(cv=5, random_state=42)
meta_model.fit(X_meta_train, y_train)
final_predictions = meta_model.predict(X_meta_test)

# Evaluation
train_rmse = mean_squared_error(y_train, meta_model.predict(X_meta_train), squared=False)
print(f"Meta-model RMSE on training set: {train_rmse:.4f}")


