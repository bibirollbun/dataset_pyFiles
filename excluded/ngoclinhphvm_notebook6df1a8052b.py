import pandas as pd

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

sales_train = pd.read_csv(data_path + 'sales_train.csv')
shops = pd.read_csv(data_path + 'shops.csv')
items = pd.read_csv(data_path + 'items.csv')
item_categories = pd.read_csv(data_path + 'item_categories.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


sales_train = sales_train.rename(columns={'date': 'Date', 
                                          'date_block_num': 'MonthID',
                                          'shop_id': 'ShopID',
                                          'item_id': 'ProductID',
                                          'item_price': 'Price',
                                          'item_cnt_day': 'Sales'})

shops = shops.rename(columns={'shop_name': 'ShopName',
                              'shop_id': 'ShopID'})

items = items.rename(columns={'item_name': 'ProductName',
                              'item_id': 'ProductID',
                              'item_category_id': 'CategoryID'})

item_categories = item_categories.rename(columns=
                                         {'item_category_name': 'CategoryName',
                                          'item_category_id': 'CategoryID'})

test = test.rename(columns={'shop_id': 'ShopID',
                            'item_id': 'ProductID'})


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




