import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops = pd.read_csv(data_path + 'shops.csv')
items = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


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
sales_train = sales_train[sales_train['item_price'] > 0]
# Extract data where the price is less than 45,000
sales_train = sales_train[sales_train['item_price'] < 45000]

# Extract data where the sales are greater than 0
sales_train = sales_train[sales_train['item_cnt_day'] > 0]
# Extract data where the sales are less than 1,000
sales_train = sales_train[sales_train['item_cnt_day'] < 1000]


# Modify shop_id in the sales_train dataset
sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11
sales_train.loc[sales_train['shop_id'] == 39, 'shop_id'] = 40

# Modify shop_id in the test dataset
test.loc[test['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 39, 'shop_id'] = 40


shops['shop_city'] = shops['shop_name'].apply(lambda x: x.split()[0])
shops.head()


shops.loc[shops['shop_city'] == '!Якутск', 'shop_city'] = 'Якутск'


sales_train.head()


shops.head()


items.head()


item_categories.head()


monthly = sales_train.copy()
monthly['item_cnt_month'] = monthly.groupby(['item_id', 'shop_id'])['item_cnt_day'].transform('sum')
monthly.head()


# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


monthly_sample = monthly.sample(n=10000, random_state=0)  # Use a smaller sample
mi_scores = make_mi_scores(monthly_sample, monthly_sample.pop('item_cnt_month'))


plot_mi_scores(mi_scores)


from sklearn.preprocessing import LabelEncoder

# Create label encoder
label_encoder = LabelEncoder()
# Apply label encoding to the city feature
shops['shop_city'] = label_encoder.fit_transform(shops['shop_city'])


# Remove the shop_name feature
shops = shops.drop('shop_name', axis=1)
shops.head()


# Add a feature for the first month the product was sold
items['first_sale_month'] = sales_train.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']

# Replace missing values in the first_sale_month feature with 34
items['first_sale_month'] = items['first_sale_month'].fillna(34)

# Remove the item_name feature
items = items.drop(['item_name'], axis=1)
items.head()


# Extract the first word of the category name as the major category
item_categories['major_category'] = item_categories['item_category_name'].apply(lambda x: x.split()[0])
item_categories.head()


def make_etc(x):
    if len(item_categories[item_categories['major_category'] == x]) >= 5:
        return x
    else:
        return 'Other'

# Change the major category to 'etc' if the number of unique values is less than 5
# 'Игры' (Games) is a rapidly growing seasonal category, so it's fine that it doesn't get changed to 'etc'
item_categories['major_category'] = item_categories['major_category'].apply(make_etc)
item_categories.head()


# Create label encoder
label_encoder = LabelEncoder()

# Apply label encoding to the major category feature
item_categories['major_category'] = label_encoder.fit_transform(item_categories['major_category'])

# Remove the item_category_name feature
item_categories = item_categories.drop('item_category_name', axis=1)


sales_train.head()


# Convert datetime from object to datetime type 
sales_train['date'] = pd.to_datetime(sales_train['date'], format="%d.%m.%Y")
sales_train['year'] = sales_train['date'].dt.year


sales_train.head()


# Step 1: Compute the mode price per item
normal_price = sales_train.groupby('item_id')['item_price'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.median())
normal_price = normal_price.rename('normal_price').reset_index()

# Step 2: Merge it with the main dataset
sales_train = sales_train.merge(normal_price, on='item_id', how='left')

# Step 3: Create the deviation feature
sales_train['price_deviation'] = sales_train['item_price'] - sales_train['normal_price']


grouped = sales_train.groupby(['year', 'date_block_num'])
dicts = [dict(zip(('year', 'date_block_num'), key)) for key in grouped.groups.keys()]
dicts


from itertools import product
import numpy as np

train = []
# Generate combinations of date_block_num, shop_id, and item_id 
for i in sales_train['date_block_num'].unique():
    # Shops that had sales
    all_shop = sales_train.loc[sales_train['date_block_num'] == i, 'shop_id'].unique()
    # Items that had sales
    all_item = sales_train.loc[sales_train['date_block_num'] == i, 'item_id'].unique()
    train.append(np.array(list(product([i], all_shop, all_item))))

idx_features = ['date_block_num', 'shop_id', 'item_id']  # Key features
train = pd.DataFrame(np.vstack(train), columns=idx_features)
train


group = sales_train.groupby(idx_features).agg({'item_cnt_day': 'sum',
                                               'item_price': 'mean'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_day': 'monthly_sales', 'item_price': 'avg_price'})

train = train.merge(group, on=idx_features, how='left')

train.head()


### FIll nan


train['monthly_sales'] = train['monthly_sales'].fillna(0)
train['avg_price'] = train['avg_price'].fillna(0)


print(train.isna().sum())  # Check for NaNs
print(np.isinf(train).sum())  # Check for infinite values


train = train.merge(pd.DataFrame(dicts), on='date_block_num', how='left')
train.head()


import gc

# Garbage collect the group variable
del group
gc.collect();


# Add product sales count feature
group = sales_train.groupby(idx_features).agg({'item_cnt_day': 'count'})
group = group.reset_index()
group = group.rename(columns={'item_cnt_day': 'sales_frequency'})

train = train.merge(group, on=idx_features, how='left')

# Garbage collection
del group, sales_train
gc.collect()

train.head()


test.head()


train.head()


# Set date_block_num to 34 for the test data
test['date_block_num'] = 34

# Concatenate train and test
all_data = pd.concat([train, test.drop('ID', axis=1)],
                     ignore_index=True,
                     keys=idx_features)
# Replace missing values with 0
all_data = all_data.fillna(0)

all_data.head()


# Merge the remaining data
all_data = all_data.merge(shops, on='shop_id', how='left')
all_data = all_data.merge(items, on='item_id', how='left')
all_data = all_data.merge(item_categories, on='item_category_id', how='left')

# Data downcasting
all_data = downcast(all_data)


all_data.head()


# Garbage collection
del shops, items, item_categories
gc.collect();


def add_mean_features(df, mean_features, idx_features):
    # Check key features
    assert (idx_features[0] == 'date_block_num') and \
           len(idx_features) in [2, 3]
    
    # Set derived feature name
    if len(idx_features) == 2:
        feature_name = idx_features[1] + '_monthly_avg_sales'
    else:
        feature_name = idx_features[1] + '_' + idx_features[2] + '_monthly_avg_sales'
    
    # Group by key features and calculate monthly average sales
    group = df.groupby(idx_features).agg({'monthly_sales': 'mean'})
    group = group.reset_index()
    group = group.rename(columns={'monthly_sales': feature_name})
    
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


# List to store derived feature names that include 'item_id' among the grouping features
item_mean_features = []

# Create monthly average sales feature grouped by ['date_block_num', 'item_id']
all_data, item_mean_features = add_mean_features(df=all_data,
                                                 mean_features=item_mean_features,
                                                 idx_features=['date_block_num', 'item_id'])

# Create monthly average sales feature grouped by ['date_block_num', 'item_id', 'shop_id']
all_data, item_mean_features = add_mean_features(df=all_data,
                                                 mean_features=item_mean_features,
                                                 idx_features=['date_block_num', 'item_id', 'shop_id'])


# A list to store derived feature names that include 'shop_id' among the grouping features
shop_mean_features = []

# Create monthly average sales feature grouped by ['date_block_num', 'shop_id', 'item_category_id']
all_data, shop_mean_features = add_mean_features(df=all_data, 
                                                 mean_features=shop_mean_features,
                                                 idx_features=['date_block_num', 'shop_id', 'item_category_id'])


all_data.head()


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
        df_temp['date_block_num'] += 1
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


idx_features = ['date_block_num', 'shop_id', 'item_id']  # Index features
lag_features_to_clip = []
# Create 3-month lag features for monthly sales based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data, 
                                                  lag_features_to_clip=lag_features_to_clip,
                                                  idx_features=idx_features,
                                                  lag_feature='monthly_sales', 
                                                  nlags=3,
                                                  clip=True)  # Limit values between 0 and 20


# Generate 3-month lag features for sales quantity based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='sales_frequency',  # Previously 'item_cnt_day'
                                                nlags=3)

# Generate 3-month lag features for average price based on idx_features
all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                lag_features_to_clip=lag_features_to_clip,
                                                idx_features=idx_features,
                                                lag_feature='avg_price',  # Previously 'item_price'
                                                nlags=3)


all_data.columns


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


# Generate lag features for each element of shop_mean_features based on ['date_block_num', 'shop_id', 'item_category_id']
for shop_mean_feature in shop_mean_features:
    all_data, lag_features_to_clip = add_lag_features(df=all_data,
                                                      lag_features_to_clip=lag_features_to_clip, 
                                                      idx_features=['date_block_num', 'shop_id', 'item_category_id'], 
                                                      lag_feature=shop_mean_feature, 
                                                      nlags=3,
                                                      clip=True)

# Remove shop_mean_features from the dataset
all_data = all_data.drop(shop_mean_features, axis=1)


# Remove data with date_block_num less than 3 (handle missing values)
all_data = all_data.drop(all_data[all_data['date_block_num'] < 3].index)


all_data.columns


# Calculate the average of the monthly sales lag features
all_data['monthly_sales_timelag_avg'] = all_data[['monthly_sales_timelag1',
                                              'monthly_sales_timelag2', 
                                              'monthly_sales_timelag3']].mean(axis=1)


# Clip the values of lag features, Monthly Sales, and Monthly Sales Lag Avg between 0 and 20
all_data[lag_features_to_clip + ['monthly_sales', 'monthly_sales_timelag_avg']] = all_data[lag_features_to_clip + ['monthly_sales', 'monthly_sales_timelag_avg']].clip(0, 20)


# Calculate the lag change for Lag1
all_data['lag_change1'] = all_data['monthly_sales_timelag1'] / all_data['monthly_sales_timelag2']
all_data['lag_change1'] = all_data['lag_change1'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Calculate the lag change for Lag2
all_data['lag_change2'] = all_data['monthly_sales_timelag2'] / all_data['monthly_sales_timelag3']
all_data['lag_change2'] = all_data['lag_change2'].replace([np.inf, -np.inf], np.nan).fillna(0)


# Create a 'New_Product_Flag' feature where it's True if the first sales month equals the current date_block_num
all_data['new_product_flag'] = all_data['first_sale_month'] == all_data['date_block_num']


# Calculate the period since the first sale
all_data['time_since_first_sale'] = all_data['date_block_num'] - all_data['first_sale_month']


# Extract the month from the date_block_num
all_data['month'] = all_data['date_block_num'] % 12


# Remove the features: First Sales Month, Average Price, and Sales Quantity
all_data = all_data.drop(['first_sale_month', 'avg_price', 'sales_frequency'], axis=1)


all_data = downcast(all_data, False)


all_data.info()


all_data.head()





# Training data (feature)
X_train = all_data[all_data['date_block_num'] < 33]
X_train = X_train.drop(['monthly_sales'], axis=1)
# Validation data (feature)
X_valid = all_data[all_data['date_block_num'] == 33]
X_valid = X_valid.drop(['monthly_sales'], axis=1)
# Test data (feature)
X_test = all_data[all_data['date_block_num'] == 34]
X_test = X_test.drop(['monthly_sales'], axis=1)

# Training data (target value)
y_train = all_data[all_data['date_block_num'] < 33]['monthly_sales']
# Validation data (target value)
y_valid = all_data[all_data['date_block_num'] == 33]['monthly_sales']

# Garbage Collection
del all_data
gc.collect()



import lightgbm as lgb

cat_features = ['shop_id', 'shop_city', 'item_category_id', 'major_category', 'month', 'year']

dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)  
dvalid = lgb.Dataset(X_valid, y_valid, categorical_feature=cat_features)  

# Remove 'categorical_feature' from params
params = {
    'metric': 'rmse',
    'num_leaves': 255,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'force_col_wise': True,
    'random_state': 10,
    'early_stopping_rounds': 150,  
    'verbose_eval': 100
}

lgb_model = lgb.train(
    params=params,
    train_set=dtrain,
    num_boost_round=1500,
    valid_sets=(dtrain, dvalid)
)


# Prediction
preds = lgb_model.predict(X_test).clip(0, 20)

# Create submission file
submission['item_cnt_month'] = preds
submission.to_csv('submission.csv', index=False)


del X_train, y_train, X_valid, y_valid, X_test, lgb_model, dtrain, dvalid
gc.collect();

