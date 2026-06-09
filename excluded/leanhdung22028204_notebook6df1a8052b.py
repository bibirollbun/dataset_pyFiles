import pandas as pd

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops = pd.read_csv(data_path + 'shops.csv')
items = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


# sales_train = sales_train.rename(columns={'date': 'Date', 
#                                           'date_block_num': 'MonthID',
#                                           'shop_id': 'ShopID',
#                                           'item_id': 'ProductID',
#                                           'item_price': 'Price',
#                                           'item_cnt_day': 'Sales'})

# shops = shops.rename(columns={'shop_name': 'ShopName',
#                               'shop_id': 'ShopID'})

# items = items.rename(columns={'item_name': 'ProductName',
#                               'item_id': 'ProductID',
#                               'item_category_id': 'CategoryID'})

# item_categories = item_categories.rename(columns=
#                                          {'item_category_name': 'CategoryName',
#                                           'item_category_id': 'CategoryID'})

# test = test.rename(columns={'shop_id': 'ShopID',
#                             'item_id': 'ProductID'})


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
        print('{:.1f}% Compressed'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

all_df = [sales_train, shops, items, item_categories, test]
for df in all_df:
    df = downcast(df)


# Extract data where the price is greater than 0
sales_train = sales_train[sales_train['Price'] > 0]
# Extract data where the price is less than 50,000
sales_train = sales_train[sales_train['Price'] < 50000]

# Extract data where the sales are greater than 0
sales_train = sales_train[sales_train['Sales'] > 0]
# Extract data where the sales are less than 1,000
sales_train = sales_train[sales_train['Sales'] < 1000]


# Modify ShopID in the sales_train dataset
sales_train.loc[sales_train['ShopID'] == 0, 'ShopID'] = 57
sales_train.loc[sales_train['ShopID'] == 1, 'ShopID'] = 58
sales_train.loc[sales_train['ShopID'] == 10, 'ShopID'] = 11
sales_train.loc[sales_train['ShopID'] == 39, 'ShopID'] = 40

# Modify ShopID in the test dataset
test.loc[test['ShopID'] == 0, 'ShopID'] = 57
test.loc[test['ShopID'] == 1, 'ShopID'] = 58
test.loc[test['ShopID'] == 10, 'ShopID'] = 11
test.loc[test['ShopID'] == 39, 'ShopID'] = 40


shops['City'] = shops['ShopName'].apply(lambda x: x.split()[0])


shops.loc[shops['City'] == '!Якутск', 'City'] = 'Якутск'


from sklearn.preprocessing import LabelEncoder

# Create label encoder
label_encoder = LabelEncoder()
# Apply label encoding to the city feature
shops['City'] = label_encoder.fit_transform(shops['City'])


# Remove the ShopName feature
shops = shops.drop('ShopName', axis=1)

shops.head()


# Remove the ProductName feature
items = items.drop(['ProductName'], axis=1)


# Add a feature for the first month the product was sold
items['FirstSaleMonth'] = sales_train.groupby('ProductID').agg({'MonthID': 'min'})['MonthID']

items.head()


# Replace missing values in the FirstSaleMonth feature with 34
items['FirstSaleMonth'] = items['FirstSaleMonth'].fillna(34)


# Extract the first word of the category name as the major category
item_categories['MajorCategory'] = item_categories['CategoryName'].apply(lambda x: x.split()[0])


def make_etc(x):
    if len(item_categories[item_categories['MajorCategory'] == x]) >= 5:
        return x
    else:
        return 'etc'

# Change the major category to 'etc' if the number of unique values is less than 5
# 'Игры' (Games) is a rapidly growing seasonal category, so it's fine that it doesn't get changed to 'etc'
item_categories['MajorCategory'] = item_categories['MajorCategory'].apply(make_etc)


# Create label encoder
label_encoder = LabelEncoder()

# Apply label encoding to the major category feature
item_categories['MajorCategory'] = label_encoder.fit_transform(item_categories['MajorCategory'])

# Remove the CategoryName feature
item_categories = item_categories.drop('CategoryName', axis=1)


sales_train


# Convert datetime from object to datetime type (feat. 송석리 선생님)
sales_train['Date'] = pd.to_datetime(sales_train['Date'], format="%d.%m.%Y")
sales_train['Year'] = sales_train['Date'].dt.year


grouped = sales_train.groupby(['Year', 'MonthID'])
dicts = [dict(zip(('Year', 'MonthID'), key)) for key in grouped.groups.keys()]
dicts


from itertools import product
import numpy as np

train = []
# Generate combinations of MonthID, ShopID, and ProductID
for i in sales_train['MonthID'].unique():
    all_shop = sales_train.loc[sales_train['MonthID'] == i, 'ShopID'].unique()
    all_item = sales_train.loc[sales_train['MonthID'] == i, 'ProductID'].unique()
    train.append(np.array(list(product([i], all_shop, all_item))))

idx_features = ['MonthID', 'ShopID', 'ProductID']  # Key features
train = pd.DataFrame(np.vstack(train), columns=idx_features)
train


group = sales_train.groupby(idx_features).agg({'Sales': 'sum',
                                               'Price': 'mean'})
group = group.reset_index()
group = group.rename(columns={'Sales': 'MonthlySales', 'Price': 'AveragePrice'})

train = train.merge(group, on=idx_features, how='left')

train.head()


train = train.merge(pd.DataFrame(dicts), on='MonthID', how='left')
train.head()


import gc

# Garbage collect the group variable
del group
gc.collect();


# Add product sales count feature
group = sales_train.groupby(idx_features).agg({'Sales': 'count'})
group = group.reset_index()
group = group.rename(columns={'Sales': 'SalesCount'})

train = train.merge(group, on=idx_features, how='left')

# Garbage collection
del group, sales_train
gc.collect()

train.head()


# Set MonthID to 34 for the test data
test['MonthID'] = 34

# Concatenate train and test
all_data = pd.concat([train, test.drop('ID', axis=1)],
                     ignore_index=True,
                     keys=idx_features)
# Replace missing values with 0
all_data = all_data.fillna(0)

all_data.head()


# Merge the remaining data
all_data = all_data.merge(shops, on='ShopID', how='left')
all_data = all_data.merge(items, on='ProductID', how='left')
all_data = all_data.merge(item_categories, on='CategoryID', how='left')

# Data downcasting
all_data = downcast(all_data)


# Garbage collection
del shops, items, item_categories
gc.collect();


def add_mean_features(df, mean_features, idx_features):
    # Check key features
    assert (idx_features[0] == 'MonthID') and \
           len(idx_features) in [2, 3]
    
    # Set derived feature name
    if len(idx_features) == 2:
        feature_name = idx_features[1] + '_Monthly_Avg_Sales'
    else:
        feature_name = idx_features[1] + '_' + idx_features[2] + '_Monthly_Avg_Sales'
    
    # Group by key features and calculate monthly average sales
    group = df.groupby(idx_features).agg({'MonthlySales': 'mean'})
    group = group.reset_index()
    group = group.rename(columns={'MonthlySales': feature_name})
    
    # Merge df with group
    df = df.merge(group, on=idx_features, how='left')
    # Downcast data
    df = downcast(df, verbose=False)
    # Add new feature name to the mean_features list
    mean_features.append(feature_name)
    
    # Garbage collection
    del group
    gc.collect()
    
    return df, mean_features


# List to store derived feature names that include 'ProductID' among the grouping features
item_mean_features = []

# Create monthly average sales feature grouped by ['MonthID', 'ProductID']
all_data, item_mean_features = add_mean_features(df=all_data,
                                                 mean_features=item_mean_features,
                                                 idx_features=['MonthID', 'ProductID'])

# Create monthly average sales feature grouped by ['MonthID', 'ProductID', 'City']
all_data, item_mean_features = add_mean_features(df=all_data,
                                                 mean_features=item_mean_features,
                                                 idx_features=['MonthID', 'ProductID', 'City'])


# A list to store derived feature names that include 'Store ID' among the grouping features
shop_mean_features = []

# Create monthly average sales feature grouped by ['MonthID', 'ShopID', 'ItemCategoryID']
all_data, shop_mean_features = add_mean_features(df=all_data, 
                                                 mean_features=shop_mean_features,
                                                 idx_features=['MonthID', 'ShopID', 'CategoryID'])


def add_lag_features(df, lag_features_to_clip, idx_features, 
                     lag_feature, nlags=3, clip=False):
    # Copy only the necessary part of the DataFrame for time lag feature creation
    df_temp = df[idx_features + [lag_feature]].copy() 

    #Time Lag Feature Generation
    for i in range(1, nlags+1):
        #Time Lag Feature Names
        lag_feature_name = lag_feature +'_timelag' + str(i)
        #Set the column names of df_temp
        df_temp.columns = idx_features + [lag_feature_name]
        # Add 1 to the 'date_block_num' feature of df_temp
        df_temp['MonthID'] += 1
        # Merge df and df_temp using idx_feature as the key
        df = df.merge(df_temp.drop_duplicates(), 
                      on=idx_features, 
                      how='left')
        # Replace missing values with 0
        df[lag_feature_name] = df[lag_feature_name].fillna(0)
        # Add lag feature names to lag_features_to_clip that need to be limited between 0 and 20
        if clip: 
            lag_features_to_clip.append(lag_feature_name)
    
    # Downcasting the data
    df = downcast(df, False)
    # Perform garbage collection
    del df_temp
    gc.collect()
    
    return df, lag_features_to_clip


idx_features = ['MonthID', 'ShopID', 'ProductID']  # Index features
lag_features_to_clip = []
# Create 3-month lag features for monthly sales based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data, 
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='MonthlySales', 
                                                  nlags=3,
                                                  clip=True)  # Limit values between 0 and 20



# Generate 3-month lag features for sales quantity based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='SalesCount',  # Previously '판매건수'
                                                nlags=3)

# Generate 3-month lag features for average price based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='AveragePrice',  # Previously '평균 판매가'
                                                nlags=3)


item_mean_features


all_data


# Generate lag features for each element of item_mean_features based on idx_features
for item_mean_feature in item_mean_features:
    all_data, lag_features_to_clip = add_lag_features(df=all_data, 
                                                      lag_features_to_clip=lag_features_to_clip, 
                                                      idx_features=idx_features, 
                                                      lag_feature=item_mean_feature, 
                                                      nlags=3,
                                                      clip=True)

# Remove item_mean_features from the dataset
all_data = all_data.drop(item_mean_features, axis=1)


# Generate lag features for each element of shop_mean_features based on ['MonthID', 'StoreID', 'ProductCategoryID']
for shop_mean_feature in shop_mean_features:
    all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                      lag_features_to_clip=lag_features_to_clip, 
                                                      idx_features=['MonthID', 'ShopID', 'CategoryID'], 
                                                      lag_feature=shop_mean_feature, 
                                                      nlags=3,
                                                      clip=True)

# Remove shop_mean_features from the dataset
all_data = all_data.drop(shop_mean_features, axis=1)


# Remove data with Month_ID less than 3 (handle missing values)
all_data = all_data.drop(all_data[all_data['MonthID'] < 3].index)


# Calculate the average of the monthly sales lag features
all_data['MonthlySales_Timelag_Avg'] = all_data[['MonthlySales_timelag1',
                                              'MonthlySales_timelag2', 
                                              'MonthlySales_timelag3']].mean(axis=1)


# Clip the values of lag features, Monthly Sales, and Monthly Sales Lag Avg between 0 and 20
all_data[lag_features_to_clip + ['MonthlySales', 'MonthlySales_Timelag_Avg']] = all_data[lag_features_to_clip + ['MonthlySales', 'MonthlySales_Timelag_Avg']].clip(0, 20)


# Calculate the lag change for Lag1
all_data['Lag_Change1'] = all_data['MonthlySales_timelag1'] / all_data['MonthlySales_timelag2']
all_data['Lag_Change1'] = all_data['Lag_Change1'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate the lag change for Lag2
all_data['Lag_Change2'] = all_data['MonthlySales_timelag2'] / all_data['MonthlySales_timelag3']
all_data['Lag_Change2'] = all_data['Lag_Change2'].replace([np.inf, -np.inf], np.nan).fillna(0)


# Create a 'New_Product_Flag' feature where it's True if the first sales month equals the current Month_ID
all_data['New_Product_Flag'] = all_data['FirstSaleMonth'] == all_data['MonthID']


# Calculate the period since the first sale
all_data['TimeSinceFirstSale'] = all_data['MonthID'] - all_data['FirstSaleMonth']


# Extract the month from the Month_ID
all_data['Month'] = all_data['MonthID'] % 12



# Remove the features: First Sales Month, Average Price, and Sales Quantity
all_data = all_data.drop(['FirstSaleMonth', 'AveragePrice', 'SalesCount'], axis=1)


all_data = downcast(all_data, False)


all_data.info()


# Training data (feature)
X_train = all_data[all_data['MonthID'] < 33]
X_train = X_train.drop(['MonthlySales'], axis=1)
# Validation data (feature)
X_valid = all_data[all_data['MonthID'] == 33]
X_valid = X_valid.drop(['MonthlySales'], axis=1)
# Test data (feature)
X_test = all_data[all_data['MonthID'] == 34]
X_test = X_test.drop(['MonthlySales'], axis=1)

# Training data (target value)
y_train = all_data[all_data['MonthID'] < 33]['MonthlySales']
# Validation data (target value)
y_valid = all_data[all_data['MonthID'] == 33]['MonthlySales']

# Garbage Collection
del all_data
gc.collect();


# import lightgbm as lgb

# bayes_dtrain = lgb.Dataset(X_train, y_train)
# bayes_dvalid = lgb.Dataset(X_valid, y_valid)


# # Hyperparameter bounds for Bayesian optimization
# param_bounds = {'num_leaves': (250, 260),
# 'lambda_l1': (0.7, 0.9),
# 'lambda_l2': (0.9, 1),
# 'feature_fraction': (0.7, 0.8),
# 'bagging_fraction': (0.7, 0.8),
# 'min_child_samples': (19, 21),
# 'min_child_weight': (1e-3, 1)}

# # Hyperparameters with fixed values
# fixed_params = {'objective': 'regression',
#                 'learning_rate': 0.005,
#                 'bagging_freq': 5,
#                 'force_row_wise': True,
#                 'random_state': 10}


# def eval_function(num_leaves, lambda_l1, lambda_l2, feature_fraction,
#                   bagging_fraction, min_child_samples, min_child_weight):
#     '''Function to calculate the evaluation metric (Gini coefficient) for optimization'''
    
#     # Hyperparameters to be optimized by Bayesian optimization
#     params = {'num_leaves': int(round(num_leaves)),
#               'lambda_l1': lambda_l1,
#               'lambda_l2': lambda_l2,
#               'feature_fraction': feature_fraction,
#               'bagging_fraction': bagging_fraction,
#               'min_child_samples': int(round(min_child_samples)),
#               'min_child_weight': min_child_weight,
#               'feature_pre_filter': False}
#     # Also add fixed hyperparameters
#     params.update(fixed_params)
    
#     print('Hyperparameters:', params)    
    
#     cat_features = ['StoreID', 'City', 'ProductCategoryID', 'MainCategory', 'Month', 'Year']
    
#     # Train LightGBM model
#     lgb_model = lgb.train(params=params, 
#                           train_set=bayes_dtrain,
#                           num_boost_round=1500,
#                           valid_sets=bayes_dvalid,
#                           early_stopping_rounds=150,
#                           categorical_feature=cat_features,
#                           verbose_eval=False)
#     # Perform prediction on validation data
#     score = lgb_model.score(X_valid, y_valid) 

#     return 1-score


# from bayes_opt import BayesianOptimization

# # Create a Bayesian optimization object
# optimizer = BayesianOptimization(f=eval_function,      # Evaluation metric calculation function
#                                  pbounds=param_bounds, # Hyperparameter ranges
#                                  random_state=0)


# # Perform Bayesian optimization
# optimizer.maximize(init_points=3, n_iter=6)


import lightgbm as lgb

# LightGBM hyperparameters
params = {'metric': 'rmse', 
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise': True,
          'random_state': 10}

cat_features = ['ShopID', 'City', 'CategoryID', 'MajorCategory', 'Month', 'Year']

# LightGBM training and validation datasets
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)

# Train LightGBM model
lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      # callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=True)]
                      early_stopping_rounds=150,
                      categorical_feature=cat_features
                      verbose_eval=100
)


# import xgboost as xgb   

# # Create XGBoost-specific datasets
# dtrain = xgb.DMatrix(X_train, y_train)
# dvalid = xgb.DMatrix(X_valid, y_valid)
# dtest = xgb.DMatrix(X_test)

# params = {'objective': 'reg:squarederror', # Regression task
#           'eval_metric': 'rmse', 
#           'num_leaves': 255,
#           'learning_rate': 0.005,
#           'feature_fraction': 0.75,
#           "colsample_bytree": 0.75,
#          }

# cat_features = ['StoreID', 'City', 'ProductCategoryID', 'MainCategory', 'Month', 'Year']

# # Train XGBoost model
# xgb_model = xgb.train(params=params, 
#                       dtrain=dtrain,
#                       num_boost_round=2000,
#                       evals=[(dvalid, 'valid')],
#                       early_stopping_rounds=200)


# Prediction
preds = lgb_model.predict(X_test).clip(0, 20)

# Create submission file
submission['item_cnt_month'] = preds
submission.to_csv('submission.csv', index=False)


del X_train, y_train, X_valid, y_valid, X_test, lgb_model, dtrain, dvalid
gc.collect();

