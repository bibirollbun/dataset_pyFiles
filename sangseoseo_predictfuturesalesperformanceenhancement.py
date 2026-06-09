import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(data_path + 'sales_train.csv')  # sales record
shops       = pd.read_csv(data_path + 'shops.csv')   #Store information
items       = pd.read_csv(data_path + 'items.csv')  # Products
item_categories = pd.read_csv(data_path + 'item_categories.csv') #Product categories
test        = pd.read_csv(data_path + 'test.csv')
submission  = pd.read_csv(data_path + 'sample_submission.csv')


test.head()


sales_train = sales_train.rename(columns = {
    'date': '날짜',
    'date_block_num': '월ID',
    'shop_id'       : '상점ID',
    'item_id'       : '상품ID',
    'item_price'    : '판매가',
    'item_cnt_day'  : '판매량'
})

shops = shops.rename(columns = {
    'shop_name': '상점명',
    'shop_id'  : '상점ID'
})
items = items.rename(columns = {
    'item_name': '상품명',
    'item_id'  : '상품ID',
    'item_category_id': '상품분류ID'
 })

item_categories = item_categories.rename(columns = {
  'item_category_id': '상품분류ID',
  'item_category_name': '상품분류명'
 })

test = test.rename(columns = {
    'shop_id': '상점ID',
    'item_id': '상품ID'
})



def downcast(df, verbose=True):
  start_mem = df.memory_usage().sum() / 1024**2
  for col in df.columns:
    dtype_name= df[col].dtype.name
    if dtype_name == 'object':
      pass
    elif dtype_name =='bool':
      df[col] = df[col].astype('int8')
    elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
      df[col] = pd.to_numeric(df[col], downcast='integer')
    else:
      df[col] = pd.to_numeric(df[col],downcast='float')
  end_mem = df.memory_usage().sum() / 1024**2
  if verbose:
    print(f'{100 * (start_mem - end_mem) / start_mem:.1f}% compressed')

  return df

all_df = [sales_train, shops, items, item_categories,   test]
for df in all_df:
  df = downcast(df)


sales_train = sales_train[sales_train['판매가'] > 0]
sales_train = sales_train[sales_train['판매가'] < 50000]
sales_train = sales_train[sales_train['판매량'] > 0]
sales_train = sales_train[sales_train['판매량'] < 1000]


print(shops['상점명'][0], '||', shops['상점명'][57])
print(shops['상점명'][1], '||', shops['상점명'][58])
print(shops['상점명'][10], '||', shops['상점명'][11])
print(shops['상점명'][39], '||', shops['상점명'][40])



sales_train.loc[sales_train['상점ID']==0, '상점ID'] = 57
sales_train.loc[sales_train['상점ID']==1, '상점ID'] = 58
sales_train.loc[sales_train['상점ID']==10, '상점ID'] = 11
sales_train.loc[sales_train['상점ID']==39, '상점ID'] = 40


#test dataset
test.loc[test['상점ID']==0, '상점ID'] = 57
test.loc[test['상점ID']==1, '상점ID'] = 58
test.loc[test['상점ID']==10, '상점ID'] = 11
test.loc[test['상점ID']==39, '상점ID'] = 40


shops['도시'] = shops['상점명'].apply(lambda x : x.split()[0])


shops['도시'].unique()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
shops['도시'] = label_encoder.fit_transform(shops['도시'])

#Drop the feature
shops = shops.drop(['상점명'], axis = 1)
shops.head()


items = items.drop('상품명', axis = 1)


items['첫판매월'] = sales_train.groupby(['상품ID']).agg({'월ID': 'min'})['월ID']


items.head()


# The number of missing values : 368
items[items['첫판매월'].isna()]


sales_train['월ID'].unique()


items['첫판매월'] = items['첫판매월'].fillna(34)


item_categories['대분류'] = item_categories['상품분류명'].apply(lambda  x : x.split()[0])


item_categories['대분류'].value_counts()


def make_etc(x):
  if len(item_categories[item_categories['대분류']==x]) >= 5:
    return x
  else:
    return 'etc'

item_categories['대분류'] = item_categories['대분류'].apply(make_etc)


item_categories.head()


label_encoder = LabelEncoder()

item_categories['대분류'] = label_encoder.fit_transform(item_categories['대분류'])
item_categories = item_categories.drop('상품분류명', axis = 1)


item_categories.head()


from itertools import product

train = []
for i in sales_train['월ID'].unique():
  all_shops = sales_train.loc[sales_train['월ID']==i,'상점ID'].unique()
  all_items = sales_train.loc[sales_train['월ID']==i,'상품ID'].unique()

  train.append(np.array(list(product([i], all_shops, all_items))))
idx_features = ['월ID','상점ID', '상품ID']
train = pd.DataFrame(np.vstack(train), columns=idx_features)
train.head()


group = sales_train.groupby(idx_features).agg({'판매량': 'sum', '판매가': 'mean'})
group = group.reset_index()
group = group.rename(columns = {
    '판매량': '월간 판매량',
    '판매가': '평균 판매가'
})

train = train.merge(group, on = idx_features, how='left')
train.head()


import gc

del group
gc.collect()


group  = sales_train.groupby(idx_features).agg({'판매량': 'count'})
grop   = group.reset_index()
group  = group.rename(columns={'판매량': '판매건수'})

train = train.merge(group, on = idx_features, how='left')
train.head()

del group, sales_train
gc.collect()

train.head()


test['월ID'] = 34

all_data = pd.concat([train, test.drop('ID', axis = 1)], ignore_index=True, keys=idx_features)

#Replace missing values into 0
all_data = all_data.fillna(0)
all_data.head()


all_data = all_data.merge(shops, on='상점ID', how='left')
all_data = all_data.merge(items, on='상품ID', how='left')
all_data = all_data.merge(item_categories, on='상품분류ID', how='left')

all_data = downcast(all_data)


del shops, items, item_categories
gc.collect()


def add_mean_feature(df,mean_features, idx_features):
  """
  Derive features based on monthly average sales per base feature
  """
  assert (idx_features[0] == '월ID') and len(idx_features) in  [2, 3]

  feature_name = ''  #Assign default value
  if len(idx_features) == 2:
    feature_name = idx_features[1] + '별 평균 판매량'
  else:
    feature_name = idx_features[1] + ' ' + idx_features[2] + '별 평균 판매량'

  group = df.groupby(idx_features).agg({'월간 판매량': 'mean'})
  group = group.reset_index()
  group = group.rename(columns = {'월간 판매량': feature_name})

  df = df.merge(group, on = idx_features, how='left')
  df = downcast(df, verbose=False)

  mean_features.append(feature_name)

  del group
  gc.collect()

  return df, mean_features


item_mean_features = []
#Monthly average sales volume by product
all_data, item_mean_features = add_mean_feature(all_data, item_mean_features, idx_features=['월ID', '상품ID'])

#Monthly average sales volumn by city + product
all_data , item_mean_features = add_mean_feature(all_data, item_mean_features, idx_features=['월ID', '상품ID', '도시'])


item_mean_features


shop_mean_features = []

all_data, shop_mean_features = add_mean_feature(all_data, shop_mean_features, idx_features=['월ID', '상점ID', '상품분류ID'])


shop_mean_features


all_data.head()


def add_lag_features(df, lag_features_to_clip, idx_features, lag_feature, nlags=3, clip=False):
  """
  time lag feature creation
  """
  #Copy dataframe needed to create time lag features
  df_temp = df[idx_features + [lag_feature]].copy()

  #Create time lag features
  for i in range(1, nlags + 1):
    lag_feature_name = lag_feature + '_시차' + str(i)
    df_temp.columns = idx_features + [lag_feature_name]

    df_temp['월ID'] += 1

    #After removal of the duplicated rows and merge
    df = df.merge(df_temp.drop_duplicates(), on = idx_features, how = 'left')
    df[lag_feature_name] = df[lag_feature_name].fillna(0)

    if clip:
      lag_features_to_clip.append(lag_feature_name)

  # data downcast
  df = downcast(df, False)

  #garbage collection
  del df_temp
  gc.collect()

  return df, lag_features_to_clip


lag_features_to_clip = []
idx_features = ['월ID', '상점ID', '상품ID'] #base features

all_data, lag_features_to_clip = add_lag_features(all_data
                                                  , lag_features_to_clip=lag_features_to_clip
                                                  ,idx_features=idx_features
                                                  , lag_feature='월간 판매량'
                                                  , nlags=3
                                                  , clip=True)


lag_features_to_clip


all_data.head().T


# Create a three-month lag feature for the sales count feature based on idx_features
all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='판매건수',
                                                  nlags=3)

#Create a three-month lag feature for the average selling price feature based on idx_features.
all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='평균 판매가',
                                                  nlags=3)


all_data.head().T


item_mean_features


for item_mean_feature in item_mean_features:
  all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                    lag_features_to_clip=lag_features_to_clip,
                                                    idx_features=idx_features,
                                                    lag_feature = item_mean_feature,
                                                    nlags=3,
                                                    clip=True)

all_data = all_data.drop(item_mean_features, axis = 1)


all_data.head().T


shop_mean_features


for shop_mean_feature in shop_mean_features:
  all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                    lag_features_to_clip=lag_features_to_clip,
                                                    idx_features = ['월ID', '상점ID', '상품분류ID'],
                                                    lag_feature = shop_mean_feature,
                                                    nlags=3,
                                                    clip=True)

all_data = all_data.drop(shop_mean_features, axis = 1)


all_data.head().T


#There are no missing values
all_data[all_data['월ID'] < 3].isna().sum()


# all_data = all_data.drop(all_data[all_data['월ID] < 3].index)


all_data['월간 판매량 시차평균'] = all_data[['월간 판매량_시차1','월간 판매량_시차2', '월간 판매량_시차3']].mean(axis = 1)


all_data.head().T


all_data[lag_features_to_clip + ['월간 판매량', '월간 판매량 시차평균']] = \
all_data[lag_features_to_clip + ['월간 판매량', '월간 판매량 시차평균']].clip(0, 20)


all_data['시차변화량1'] = all_data['월간 판매량_시차1'] / all_data['월간 판매량_시차2']
all_data['시차변화량1'] = all_data['시차변화량1'].replace([np.inf, -np.inf], np.nan).fillna(0)

all_data['시차변화량2'] = all_data['월간 판매량_시차2'] / all_data['월간 판매량_시차3']
all_data['시차변화량2'] = all_data['시차변화량2'].replace([np.inf, -np.inf], np.nan).fillna(0)


all_data['신상여부'] = all_data['첫판매월'] ==all_data['월ID']


all_data['첫 판매 후 기간'] = all_data['월ID'] - all_data['첫판매월']


all_data['월'] = all_data['월ID'] % 12


all_data.head().T


all_data = all_data.drop(['첫판매월', '평균 판매가', '판매건수'], axis = 1)


all_data = downcast(all_data, verbose=False)


all_data.info()


#Train data(features)
X_train = all_data[all_data['월ID'] < 33]
X_train = X_train.drop(['월간 판매량'],axis = 1)

# Validatoin data(features)
X_valid = all_data[all_data['월ID'] == 33]
X_valid = X_valid.drop(['월간 판매량'], axis = 1)

# Test dataset(features)
X_test = all_data[all_data['월ID']== 34]
X_test = X_test.drop(['월간 판매량'], axis = 1)

# train data(target)
y_train = all_data[all_data['월ID'] < 33]['월간 판매량']
y_valid = all_data[all_data['월ID'] ==33]['월간 판매량']

#garbage collection
del all_data
gc.collect()


!pip install lightgbm


import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation

#LightGBM Hyperparameter
params = {
      'metric': 'rmse',
      'num_leaves': 255,
      'learning_rate': 0.005,
      'feature_fraction': 0.75,
      'bagging_fraction': 0.75,
      'bagging_freq'    : 5,
      'force_col_wise'  : True,
      'random_state'    : 10
}

cat_features = ['상점ID', '도시', '상품분류ID', '대분류', '월']

dtrain   =  lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
dvalid   =  lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)

lgb_model = lgb.train(params = params,
          train_set = dtrain,
          num_boost_round=1500,
          valid_sets = (dtrain, dvalid),
          callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=100)]
          )


preds = lgb_model.predict(X_test).clip(0, 20)
submission['item_cnt_month'] = preds
submission.to_csv("submission.csv", index=False)


del X_train, y_train, X_valid, y_valid, X_test, lgb_model, dtrain, dvalid
gc.collect()

