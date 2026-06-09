from typing import ItemsView
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')
data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops       = pd.read_csv(data_path + 'shops.csv')
items       = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test        = pd.read_csv(data_path + 'test.csv')
submission  = pd.read_csv(data_path + 'sample_submission.csv')


sales_train = sales_train.rename(columns = {
    'date' : '날짜',
    'date_block_num': '월ID',
    'shop_id'       : '상점ID',
    'item_id'       : '상품ID',
    'item_price'    : '판매가',
    'item_cnt_day'  : '판매량'
})

shops = shops.rename(columns = {
    'shop_name' : '상점명',
    'shop_id'   : '상점ID'
})

items =  items.rename(columns = {
    'item_name' : '상품명',
    'item_id'   : '상품ID',
    'item_category_id' : '상품분류ID'
})

item_categories = item_categories.rename(columns = {
    'item_category_name' : '상품분류명',
    'item_category_id'   : '상품분류ID'
})

test = test.rename(columns = {
    'shop_id' : '상점ID',
    'item_id' : '상품ID'
})


def downcast(df, verbose=True):
  '''
  Data downcast
  '''
  start_mem = df.memory_usage().sum() / 1024 **2
  for col in df.columns:
    dtype_name = df[col].dtype.name

    if dtype_name == 'object':
      pass
    elif dtype_name == 'bool':
      df[col] = df[col].astype('int8')
    elif (dtype_name.startswith('int')) or (df[col].round() == df[col]).all():
      df[col] = pd.to_numeric(df[col], downcast='integer')
    else:
      df[col] = pd.to_numeric(df[col], downcast = 'float')
  end_mem = df.memory_usage().sum() / 1024 ** 2

  if verbose:
    print('{:.1f}% compressed.'.format(100 * (start_mem - end_mem) / start_mem))

  return df

all_df = [sales_train, shops, items, item_categories, test]
for df in all_df:
  df = downcast(df)


# Extract dataset where the store information is exists in test dataset
unique_test_shop_id = test['상점ID'].unique()
sales_train = sales_train[sales_train['상점ID'].isin(unique_test_shop_id)]

#Sales Price greater than 0 and less than 5000 is the target
sales_train = sales_train.loc[(sales_train['판매가']> 0) & (sales_train['판매가'] < 50000)]

#sales volumn greater than 0 and less than 1000 is the target
sales_train = sales_train.loc[(sales_train['판매량'] > 0)  & (sales_train['판매량'] < 1000)]


sales_train


print(shops['상점명'][0], '||', shops['상점명'][57])
print(shops['상점명'][1], '||', shops['상점명'][58])
print(shops['상점명'][10], '||',shops['상점명'][11])
print(shops['상점명'][39],  '||', shops['상점명'][40])


print(shops.iloc[[0,57]])
print(shops.iloc[[1, 58]])
print(shops.iloc[[10, 11]])
print(shops.iloc[[39, 40]])


#Modify 상점ID in sales_train dataset
sales_train.loc[sales_train['상점ID']== 0, '상점ID'] = 57
sales_train.loc[sales_train['상점ID']== 1, '상점ID'] = 58
sales_train.loc[sales_train['상점ID']== 10, '상점ID'] = 11
sales_train.loc[sales_train['상점ID']== 39, '상점ID'] = 40


#Modify 상점ID in test data
test.loc[test['상점ID']== 0, '상점ID'] = 57
test.loc[test['상점ID']== 1, '상점ID'] = 58
test.loc[test['상점ID']== 10, '상점ID'] = 11
test.loc[test['상점ID']== 39, '상점ID'] = 40


shops['도시'] = shops['상점명'].apply(lambda x : x.split()[0])


shops['도시'].unique()


shops.loc[shops['도시']== '!Якутск', '도시'] = 'Якутск'


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

shops['도시'] = label_encoder.fit_transform(shops['도시'])


# Removal of the unnecessary feature after encoding
shops = shops.drop('상점명', axis = 1)
shops.head()


items


items = items.drop('상품명', axis = 1)


items['첫 판매월'] = sales_train.groupby('상품ID').agg({'월ID': 'min'})['월ID']
items.head()


# The number of missing values : 368
items[items['첫 판매월'].isna()]


items['첫 판매월'] = items['첫 판매월'].fillna(34)


item_categories['대분류'] = item_categories['상품분류명'].apply(lambda x : x.split()[0])


item_categories['대분류'].value_counts()


def make_etc(x):
  """
  return item categories
  """
  if len(item_categories[item_categories['대분류']==x]) >= 5:
    return x
  else:
    return 'etc'

item_categories['대분류'] = item_categories['대분류'].apply(make_etc)


label_encoder = LabelEncoder()

item_categories['대분류'] = \
label_encoder.fit_transform(item_categories['대분류'])


item_categories = item_categories.drop('상품분류명', axis = 1)


from itertools import product

train = []

for i in sales_train['월ID'].unique():
  all_shops = sales_train.loc[sales_train['월ID']== i, '상점ID'].unique()
  all_items = sales_train.loc[sales_train['월ID']== i, '상품ID'].unique()

  train.append(np.array(list(product([i], all_shops, all_items))))

idx_features = ['월ID', '상점ID', '상품ID']  #base features
train        = pd.DataFrame(np.vstack(train), columns = idx_features)


train


group = sales_train.groupby(idx_features).agg({'판매량': 'sum', '판매가': 'mean'})
group = group.reset_index()
group = group.rename(columns = {'판매량': '월간 판매량', '판매가': '평균 판매가'})

train = train.merge(group, on = idx_features, how= 'left')
train.head()


import gc

del group
gc.collect()


group = sales_train.groupby(idx_features).agg({'판매량': 'count'})
group = group.reset_index()
group = group.rename(columns = {'판매량': '판매건수'})

train = train.merge(group, on = idx_features, how='left')

del group, sales_train
gc.collect()

train.head()


test['월ID'] = 34

#concatenate train and test
all_data = pd.concat([train, test.drop('ID', axis = 1)], ignore_index=True, keys=idx_features)

all_data = all_data.fillna(0)
all_data.head()


all_data


all_data = all_data.merge(shops, on ='상점ID', how='left')
all_data = all_data.merge(items, on = '상품ID', how='left')
all_data = all_data.merge(item_categories, on ='상품분류ID', how='left')

# Data downcast
all_data = downcast(all_data)


del shops, items, item_categories
gc.collect()


def add_mean_features(df, mean_features, idx_features):
  """
  df - DataFrame target
  mean_features : List for storing the newly generated features
  idx_features  : standard features
  """
  assert (idx_features[0]=='월ID') and \
  len(idx_features) in  [2,3]

  if len(idx_features) == 2:
    feature_name = idx_features[1] +'별 평균 판매량'
  else:
    feature_name = idx_features[1] + ' ' + idx_features[2] + '별 평균 판매량'

  #aggragation based on standard features
  group = df.groupby(idx_features).agg({'월간 판매량': 'mean'})
  group = group.reset_index()
  group = group.rename(columns={'월간 판매량': feature_name})

  #left outer join with target dataset
  df    = df.merge(group, on = idx_features, how='left')
  df    = downcast(df)

  #list for storing the newly generated features
  mean_features.append(feature_name)

  del group
  gc.collect()

  return df, mean_features


item_mean_feautures = []

all_data , item_mean_feautures = add_mean_features(df = all_data,
                                                   mean_features=item_mean_feautures,
                                                   idx_features=['월ID', '상품ID'])
all_data , item_mean_feautures = add_mean_features(df = all_data,
                                                   mean_features=item_mean_feautures,
                                                   idx_features=['월ID', '상품ID', '도시'])

item_mean_feautures


shop_mean_features = []
all_data, shop_mean_features = add_mean_features(df=all_data,
                                                 mean_features=shop_mean_features,
                                                 idx_features=['월ID', '상점ID', '상품분류ID'])
shop_mean_features


def add_lag_features(df, lag_features_to_clip, idx_features, lag_feature, nlags=3, clip=False):
  """
  df : original dataset
  lag_features_to_clip : list for storing
  idx_features : standard features
  lag_feature  : feature for time lag features
  nlag  : time lag
     1 - Generate features with a one-month time lag
     2 - Generate features with a two-month time lag
     3 - Generate features with a three-month time lag
  clip  : Whether to save the newly created time lag features to the `lag_features_to_clip` list
  """
  df_temp = df[idx_features + [lag_feature]].copy()

  for i in range(1,nlags + 1):
    lag_feature_name = lag_feature + '_시차' + str(i)

    df_temp.columns = idx_features + [lag_feature_name]
    df_temp['월ID'] += 1
    df = df.merge(df_temp.drop_duplicates(), on = idx_features, how='left')
    df[lag_feature_name] = df[lag_feature_name].fillna(0)

    if clip:
      lag_features_to_clip.append(lag_feature_name)

  df = downcast(df, False)

  del df_temp
  gc.collect()

  return df, lag_features_to_clip


lag_features_to_clip = []
idx_features = ['월ID', '상점ID', '상품ID']

#Generate 3 months' worth of time-lag features (monthly sales volumn)
all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='월간 판매량',
                                                  nlags= 3,
                                                  clip=True)


all_data.head().T


all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='판매건수',
                                                  nlags=3
                                                  )

all_data, lag_features_to_clip = add_lag_features(df = all_data,
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='평균 판매가',
                                                  nlags=3
                                                  )


item_mean_feautures


for item_mean_feature in item_mean_feautures:
  all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                    lag_features_to_clip=lag_features_to_clip,
                                                    idx_features=idx_features,
                                                    lag_feature=item_mean_feature,
                                                    nlags=3,
                                                    clip=True)

# item_mean_features is unnecessary for modeling after the time lage feature generation
all_data = all_data.drop(item_mean_feautures, axis = 1)


shop_mean_features


for shop_mean_feature in shop_mean_features:
  all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                    lag_features_to_clip=lag_features_to_clip,
                                                    idx_features=['월ID', '상점ID', '상품분류ID'],
                                                    lag_feature = shop_mean_feature,
                                                    nlags=3,
                                                    clip=True)
all_data = all_data.drop(shop_mean_features, axis = 1)


all_data.head().T


all_data[all_data['월ID'] < 3].isna().sum()  # Missing values isn't exists in `all_data.`


all_data['월간 판매량 시차평균'] = all_data[['월간 판매량_시차1', '월간 판매량_시차2', '월간 판매량_시차3']].mean(axis = 1)


# Adjust sales volumn between 0 and 20
all_data[lag_features_to_clip + ['월간 판매량', '월간 판매량 시차평균']] = \
all_data[lag_features_to_clip + ['월간 판매량', '월간 판매량 시차평균']].clip(0,20)


all_data['판매건수 시차평균'] = all_data[['판매건수_시차1', '판매건수_시차2', '판매건수_시차3']].mean(axis = 1)


all_data['시차변화량1'] = all_data['월간 판매량_시차1'] / all_data['월간 판매량_시차2']
all_data['시차변화량1'] = all_data['시차변화량1'].replace([np.inf, -np.inf], np.nan).fillna(0)

all_data['시차변화량2'] = all_data['월간 판매량_시차2'] / all_data['월간 판매량_시차3']
all_data['시차변화량2'] = all_data['시차변화량2'].replace([np.inf, -np.inf], np.nan).fillna(0)


all_data['신상여부'] = all_data['첫 판매월'] == all_data['월ID']


all_data['첫 판매후 경과 기간'] = all_data['월ID'] - all_data['첫 판매월']


all_data['월'] = all_data['월ID'] % 12


all_data = all_data.drop(['첫 판매월', '평균 판매가', '판매건수'], axis = 1)


# downcast
all_data = downcast(all_data)


all_data.info()


#train data(feature)
X_train = all_data[all_data['월ID'] < 33]
X_train = X_train.drop(['월간 판매량'],axis = 1)

# Validatoin data(feature)
X_valid = all_data[all_data['월ID'] == 33]
X_valid = X_valid.drop(['월간 판매량'], axis = 1)

#Test data(feature)
X_test  = all_data[all_data['월ID']== 34]
X_test  = X_test.drop(['월간 판매량'], axis = 1)

#trainning data(target values)
y_train = all_data[all_data['월ID'] < 33]['월간 판매량']

#validation data(target values)
y_valid = all_data[all_data['월ID'] == 33]['월간 판매량']

#garbage collection
del all_data
gc.collect()


!pip install lightgbm


import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
#LightGBM  Hyperparameter
# bagging_freq : bagging frequency - Decide how many iterations to perform bagging. 0 - no bagging, 1 - On each iterations, trains the trees with the new sampling data
# learning_rate : Step size shrinkage used in update to prevent overfitting.
params = {
    'metric': 'rmse',
    'num_leaves' : 255,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,  # Feature sampling ratio to use for trainning the individual trees.
    'bagging_fraction': 0.75,  # Data sampling ratio to use for trainning the individual trees.To enable bagging, set the bagging_fraction parameter to a value other than 0
    'bagging_freq'    : 5,
    'force_col_wise'  : True,
    'random_state'    : 10
}

cat_features = ['상점ID', '도시', '상품분류ID', '대분류', '월']

#training and validation data for LightGBM
dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
dvalid = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)

#model trainning
lgb_model = lgb.train(params = params,
                      train_set = dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=100)]
                      )



preds = lgb_model.predict(X_test).clip(0, 20)   # Adjust the predicted values between 0 and 20
submission['item_cnt_month'] = preds
submission.to_csv('submission.csv' ,index=False)


del X_train, y_train, X_valid, y_valid, X_test, lgb_model, dtrain, dvalid
gc.collect()

