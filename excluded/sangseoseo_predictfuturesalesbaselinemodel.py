import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')  # disregard the unnecessary warning messages.

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'
sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops       = pd.read_csv(data_path + 'shops.csv')
items       = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path +'item_categories.csv')
test        = pd.read_csv(data_path + 'test.csv')
submission  = pd.read_csv(data_path + 'sample_submission.csv')


sales_train = sales_train.rename(
    columns = {'date': '날짜',
               'date_block_num': '월ID',
               'shop_id': '상점ID',
               'item_id': '상품ID',
               'item_price': '판매가',
               'item_cnt_day': '판매량'
               }
              )
sales_train.head()


shops = shops.rename(
    columns = {'shop_name': '상점명', 'shop_id': '상점ID'}
)
shops.head()


items = items.rename(columns = {'item_name':'상품명', 'item_id': '상품ID', 'item_category_id': '상품분류ID'})
items.head()


item_categories = item_categories.rename(columns = {'item_category_name': '상품분류명', 'item_category_id':'상품분류ID'})
item_categories.head()


test = test.rename(columns={'shop_id': '상점ID', 'item_id': '상품ID'})
test.head()


def downcast(df, verbose=True):
  """
  downcasting datat types
  """
  start_mem = df.memory_usage().sum() / 1024**2
  for col in df.columns:
    dtype_name = df[col].dtype.name
    if dtype_name == 'object':
      pass
    elif dtype_name =='bool':
      df[col] = df[col].astype('int8')
    elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
      df[col] = pd.to_numeric(df[col], downcast='integer')
    else:
      df[col] = pd.to_numeric(df[col], downcast='float')
  end_mem = df.memory_usage().sum() / 1024 **2
  if verbose:
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

  return df

all_df = [sales_train, shops, items, item_categories, test]
for df in all_df:
  df = downcast(df)


from itertools import product

train = []
for i in sales_train['월ID'].unique():
  all_shop = sales_train.loc[sales_train['월ID']== i, '상점ID'].unique()
  all_items = sales_train.loc[sales_train['월ID']== i, '상품ID'].unique()
  train.append(np.array(list(product([i], all_shop, all_items))))

idx_features = ['월ID', '상점ID', '상품ID'] # Reference feature
train = pd.DataFrame(np.vstack(train),  columns = idx_features)

train


groups = sales_train.groupby(idx_features).agg({'판매량': 'sum'})

groups = groups.reset_index()
groups = groups.rename(columns = {'판매량':'월간판매량'})
groups


train = train.merge(groups, on = idx_features, how='left')
train


import gc
del groups
gc.collect()


test['월ID'] = 34 # November, 2015


test.head()


all_data = pd.concat([train, test.drop('ID', axis = 1)],ignore_index = True, keys=idx_features)
# all_data.head()


all_data = all_data.fillna(0)
all_data


all_data = all_data.merge(shops, on='상점ID', how='left')
all_data = all_data.merge(items, on='상품ID', how='left')
all_data = all_data.merge(item_categories, on='상품분류ID', how='left')

#downcast
all_data = downcast(all_data)

del shops, items, item_categories
gc.collect()


all_data = all_data.drop(['상점명', '상품명', '상품분류명'], axis = 1)


#train data
X_train = all_data[all_data['월ID'] < 33]
X_train = X_train.drop(['월간판매량'], axis = 1)

#Validation data
X_valid = all_data[all_data['월ID']== 33]
X_valid = X_valid.drop(['월간판매량'], axis = 1)

#Test data
X_test = all_data[all_data['월ID']== 34]
X_test = X_test.drop(['월간판매량'], axis = 1)

#target value for train
y_train = all_data[all_data['월ID'] < 33]['월간판매량']
y_train = y_train.clip(0, 20)  # 0 - lower bound limit, 20 - Upper bound limit

# Validation data
y_valid = all_data[all_data['월ID']==33]['월간판매량']
y_valid = y_valid.clip(0, 20)


del all_data
gc.collect()


!pip install lightgbm


import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
params = {
    'metric'         : 'rmse'  # Root Mean Squared Error
    ,'num_leaves'    : 255
    ,'learning_rate' : 0.01
    ,'force_col_wise': True
    ,'random_state'  : 10
}

cat_features = ['상점ID', '상품분류ID']
dtrain = lgb.Dataset(X_train, y_train,categorical_feature=cat_features)
dvalid = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)


lgb_model = lgb.train(params = params,
                      train_set=dtrain,
                      num_boost_round=2500, # number of boosting iterations
                      valid_sets=(dtrain, dvalid),
                      callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=100)]
                      )


# cat_features = ['상점ID', '상품분류ID']
# for cat_feature in cat_features:
#   all_data[cat_feature] = all_data[cat_feature].astype('category')


preds = lgb_model.predict(X_test).clip(0, 20) # target values are between 0 and 20
submission['item_cnt_month'] = preds
submission.to_csv('submission.csv', index = False)


del X_train, y_train, X_valid, y_valid, X_test, lgb_model, dtrain, dvalid
gc.collect()

