import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv'
test_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv'
inventory_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv'
calendar_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv'
test_weights_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv'


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
inventory = pd.read_csv(inventory_path)
calendar = pd.read_csv(calendar_path)
test_weights = pd.read_csv(test_weights_path)


print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')
print(f'inventory shape: {inventory.shape}')
print(f'calendar shape: {calendar.shape}')
print(f'test_weights shape: {test_weights.shape}')


def merge_data(sales_df):
  merged = sales_df.merge(calendar, on=['date', 'warehouse'], how='left')
  merged = merged.merge(inventory, on=['unique_id', 'warehouse'], how='left')
  return merged


# def data_cleaning(df):
#   df['date'] = pd.to_datetime(df['date'])
#   df['warehouse'] = df['warehouse'].astype('category')
#   df['holiday'] = df['holiday'].fillna(0).astype(int)
#   df['school_holidays'] = df['school_holidays'].fillna(0).astype(int)

#   for col in ['L1_category_name_en', 'L2_category_name_en']:
#     df[col] = df[col].fillna('Unknown')

#   return df


closed_days = calendar[calendar['shops_closed'] == 1]
closed_days.to_csv('closed_days.csv', index=False)


train_copy = train.copy()
merged_day = train_copy.merge(closed_days, on=['date', 'warehouse'], how='left')



merged_day['date'] = pd.to_datetime(merged_day['date'])
merged_day.set_index('date', inplace=True)



merged_day['month'] = merged_day.index.month
monthly_closed_days = merged_day.groupby('month')['shops_closed'].count()

plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_closed_days.index, y=monthly_closed_days.values, marker='o')
plt.title('Total days off by Month (Seasonality Analysis)')
plt.xlabel('Month')
plt.ylabel('Total days off')
plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()


datasets = [train, inventory, calendar,test]
names = ['sales_train','inventory', 'calendar','sales_test']

data_info = pd.DataFrame({})
data_info['dataset'] = names
data_info['n_rows'] = [df.shape[0] for df in datasets]
data_info['n_cols'] = [df.shape[1] for df in datasets]
data_info['null_amount'] = [df.isnull().sum().sum() for df in datasets]
data_info['total_null_columns'] = [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
data_info['null_columns'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]

data_info.style.background_gradient()



train = train.dropna()



full_train_data = merge_data(train)
full_train_data.isnull().sum()


test = merge_data(test)
test.isnull().sum()


full_train_data= pd.concat([full_train_data, test])


full_train_data.isnull().sum()


df_nulls = pd.DataFrame(full_train_data.isnull().sum().sort_values(ascending=False), columns=['Number of Missing Values'])
df_nulls['% Missing'] = full_train_data.isnull().sum().sort_values(ascending=False)/len(full_train_data)
df_nulls = df_nulls[df_nulls['Number of Missing Values']>0]
df_nulls


datasets = [train, inventory, calendar,test]
names = ['train data','inventory', 'calendar','sales_test']

data_info = pd.DataFrame({})
data_info['dataset'] = names
data_info['n_rows'] = [df.shape[0] for df in datasets]
data_info['n_cols'] = [df.shape[1] for df in datasets]
data_info['null_amount'] = [df.isnull().sum().sum() for df in datasets]
data_info['total_null_columns'] = [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
data_info['null_columns'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]

data_info.style.background_gradient()



full_train_data.shape


#log transform
full_train_data['log_orders'] = np.log1p(full_train_data['total_orders'])



plt.figure(figsize=(10, 6))
sns.histplot(full_train_data['log_orders'], kde=True, bins=50)
plt.title('Distribution of Total Orders')
plt.xlabel('Total Orders')
plt.ylabel('Frequency')
plt.show()


full_train_data['sell_price_main_log'] = np.log1p(full_train_data['sell_price_main'])



plt.figure(figsize=(10, 6))
sns.histplot(full_train_data['sell_price_main_log'], kde=True, bins=50)
plt.title('Distribution of Selling Price')
plt.xlabel('Sell Price')
plt.ylabel('Frequency')
plt.show()


# cắt ngưỡng
upper_clip = full_train_data['sell_price_main_log'].quantile(0.99)
full_train_data['sell_price_main_log'] = full_train_data['sell_price_main_log'].clip(upper=upper_clip)


plt.figure(figsize=(10, 6))
sns.histplot(full_train_data['sell_price_main_log'], kde=True, bins=50)
plt.title('Distribution of Selling Price')
plt.xlabel('Sell Price')
plt.ylabel('Frequency')
plt.show()


def fe_other(df):
    discount_cols = ['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']
    df[discount_cols] = df[discount_cols].clip(0)
    return df


full_train_data = fe_other(full_train_data)


discount_cols = [f'type_{i}_discount' for i in range(7)]

fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(28, 4))

for i, col in enumerate(discount_cols):
    sns.scatterplot(data=full_train_data, x=col, y='sales', ax=axes[i])
    axes[i].set_title(f'{col} vs Sales')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Sales')

plt.tight_layout()
plt.show()


closed_days = calendar[calendar['shops_closed'] == 1]
closed_days.to_csv('closed_days.csv', index=False)


plt.figure(figsize=(14, 8))
warehouse_sales = full_train_data.groupby('holiday_name', observed=False)['sales'].sum().sort_values(ascending=False)

sns.barplot(x=warehouse_sales.values, y=warehouse_sales.index, palette="Spectral")
plt.title('sales on holidays', fontsize=16)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('holiday_name', fontsize=12)
plt.xticks(fontsize=10)
plt.show()


full_train_data['date'] = pd.to_datetime(full_train_data['date'])
full_train_data['day_of_week'] = full_train_data['date'].dt.dayofweek
full_train_data['day_of_year'] = full_train_data['date'].dt.day_of_year
full_train_data['week_of_year'] = full_train_data['date'].dt.isocalendar().week
full_train_data['month'] = full_train_data['date'].dt.month
full_train_data['day'] = full_train_data['date'].dt.day
full_train_data['month'] = full_train_data['date'].dt.month
full_train_data['year'] = full_train_data['date'].dt.year
full_train_data['cos_day'] = np.cos(full_train_data['day_of_year']*2*np.pi/365)
full_train_data['sin_day'] = np.sin(full_train_data['day_of_year']*2*np.pi/365)
full_train_data['is_weekend'] = full_train_data['day_of_week'].isin([5,6]).astype('int8')
full_train_data['long_weekend'] = ((full_train_data['shops_closed'] == 1) & (full_train_data['shops_closed'].shift(1) == 1)).astype(int)


def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'


full_train_data['season'] = full_train_data['month'].apply(get_season)


full_train_data['max_discount'] = full_train_data[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount']].max(axis=1)



plt.figure(figsize=(10, 6))
sns.histplot(full_train_data['max_discount'], kde=True, bins=50)
plt.title('Distribution of max discount')
plt.xlabel('max discount')
plt.ylabel('Frequency')
plt.show()


full_train_data['common_name'] = full_train_data['name'].apply(lambda x: x[:x.find('_')])



cols = full_train_data.columns.tolist()
cols.remove('common_name')
target_col_index = cols.index('name')
cols.insert(target_col_index + 1, 'common_name')
full_train_data = full_train_data[cols]


top_products = full_train_data.groupby('common_name')['sales'].sum().nlargest(20)

plt.figure(figsize=(16, 10))
sns.barplot(x=top_products.values, y=top_products.index, palette="plasma")
plt.title('Best Selling categories', fontsize=16)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.tight_layout()
plt.show()


full_train_data['common_name'].unique()


len(full_train_data['common_name'].unique())


plt.figure(figsize=(14, 8))
df_iwd = full_train_data[full_train_data['holiday_name'] == 'International womens day']
iwd_sales = df_iwd.groupby('common_name', observed=False)['sales'].sum().sort_values(ascending=False).nlargest(20)

sns.barplot(x=iwd_sales.values, y=iwd_sales.index, palette="Spectral")
plt.title('sales on iwd', fontsize=16)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('product_name', fontsize=12)
plt.xticks(fontsize=10)
plt.show()


full_train_data['total_products'] = full_train_data.groupby(['date','warehouse','common_name'])['unique_id'].transform('nunique')
full_train_data['discount_avg'] = full_train_data.groupby(['date','warehouse','common_name'])['max_discount'].transform('mean')


grouped = full_train_data.groupby('common_name').agg({
    'discount_avg': 'mean',
    'sales': 'mean'
}).reset_index()

sns.scatterplot(data=grouped, x='discount_avg', y='sales')
plt.title("Mean Discount vs Mean Sales by Category")
plt.xlabel('avg_discount')
plt.ylabel('Sales')
plt.show()


def load_calendar(calendar):
    calendar = calendar.sort_values('date')
    calendar.reset_index(drop=True, inplace=True)

    calendar.loc[calendar['holiday_name'].isna(), 'holiday'] = 0 

    calendar['last_holiday_date'] = calendar['date']
    calendar['next_holiday_date'] = calendar['date']

    calendar.loc[calendar['holiday'] == 0, ['last_holiday_date','next_holiday_date']] = np.nan

    calendar['last_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['last_holiday_date'].ffill()
    calendar['next_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['next_holiday_date'].bfill()

    calendar['days_since_last_holiday'] = ((calendar['date'] - calendar['last_holiday_date']).dt.days)
    calendar['days_to_next_holiday'] = ((calendar['next_holiday_date'] - calendar['date']).dt.days)

    calendar['day_before_holiday'] = calendar['days_to_next_holiday'] == 1
    calendar['day_after_holiday'] = calendar['days_since_last_holiday'] == 1

    calendar.drop(['last_holiday_date','next_holiday_date'],axis=1,inplace=True)
    
    calendar.drop(['shops_closed','winter_school_holidays','school_holidays','holiday_name'],axis=1,inplace=True)
    return calendar


full_train_data = load_calendar(full_train_data)


PERIODS = [14, 16, 18, 21, 30, 60, 90, 120, 180]
def add_lagged_product_sales(df):
    df = df.sort_values(['warehouse', 'name', 'date'])
    for shift in PERIODS:
        df[f'product_sales_{shift}']=df.groupby(['warehouse','name'])['sales'].shift(periods=shift)
    return df


full_train_data = add_lagged_product_sales(full_train_data)


full_train_data['days_since_2020'] = (full_train_data['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')

mean_prices = full_train_data.groupby(full_train_data['unique_id'])['sell_price_main'].mean()
std_prices = full_train_data.groupby(full_train_data['unique_id'])['sell_price_main'].std()
full_train_data['price_scaled'] = np.where(
    full_train_data['unique_id'].map(std_prices) == 0, 0, 
    (full_train_data['sell_price_main'] - full_train_data['unique_id'].map(mean_prices))/full_train_data['unique_id'].map(std_prices))
full_train_data['price_detrended'] = full_train_data['price_scaled'] - full_train_data.groupby(['days_since_2020','warehouse'])['price_scaled'].transform('mean')
full_train_data.drop('price_scaled',axis=1,inplace=True)


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
le = LabelEncoder()


warehouse_encoded = oe.fit_transform(full_train_data[['warehouse']])

warehouse_df = pd.DataFrame(warehouse_encoded, columns=oe.get_feature_names_out(['warehouse']))
full_train_data = pd.concat([full_train_data.drop(columns=['warehouse']), warehouse_df], axis=1)



label_cols = ['name', 'common_name', 'L1_category_name_en',
              'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en','season']

for col in label_cols:
    full_train_data[col] = le.fit_transform(full_train_data[col])


full_train_data.columns


# Tính toán ma trận tương quan
correlation_matrix = full_train_data.corr()

# Vẽ heatmap để quan sát mối quan hệ giữa các đặc trưng
plt.figure(figsize=(18, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={'size': 6},
            cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Correlation'})
colorbar = plt.gca().collections[0].colorbar
colorbar.ax.tick_params(labelsize=10) 

plt.title('Correlation Matrix', fontsize=18)
plt.tight_layout() 
plt.show()


X_train_final = full_train_data.loc[full_train_data['date'] < '2024-06-03']
X_test_final = full_train_data.loc[full_train_data['date'] >= '2024-06-03']



X_train_final.shape


X_test_final.shape


print('File train:')
print("Ngày cũ nhất:", train['date'].min())
print("Ngày mới nhất:", train['date'].max())



print('File test:')
print("Ngày cũ nhất:", test['date'].min())
print("Ngày mới nhất:", test['date'].max())



import xgboost as xgb
from sklearn.model_selection import train_test_split


X = X_train_final.drop(columns=['sales', 'date'])  
y = X_train_final['sales']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)


import numpy as np

print(np.isnan(y_train).sum())       # số lượng NaN
print(np.isinf(y_train).sum())        # số lượng Inf
print(np.max(y_train))                # giá trị lớn nhất
print(np.min(y_train))                # giá trị nhỏ nhất



model.fit(X_train, y_train)


importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df)





features = feature_importance_df['feature']
importance = feature_importance_df['importance']

y_pos = np.arange(len(features)) * 2  

plt.figure(figsize=(12, 8))
bars = plt.barh(y_pos, importance, align='center', height=1.2) 
plt.yticks(y_pos, features) 
plt.gca().invert_yaxis()
plt.xlabel('Importance Score')
plt.title('Feature Importance from XGBoost')

plt.show()



def drop_discount_features(df):
    discount_cols = [col for col in df.columns if col.startswith('type_') and col.endswith('_discount')]
    df = df.drop(columns=discount_cols)
    return df


full_train_data = drop_discount_features(full_train_data)

full_train_data = full_train_data.drop(columns=['total_orders', 'sell_price_main'])


full_train_data.columns


import scipy.stats as stats

f_statistic_L1, p_value_L1 = stats.f_oneway(
    X_train_final[X_train_final['L1_category_name_en'] == 0]['sales'],
    X_train_final[X_train_final['L1_category_name_en'] == 1]['sales'],
    X_train_final[X_train_final['L1_category_name_en'] == 2]['sales']
)


print(p_value_L1)


if p_value_L1 < 0.05:
    print("Có sự khác biệt đáng kể giữa các nhóm của L1_category_name_en.")
else:
    print("Không có sự khác biệt đáng kể giữa các nhóm của L1_category_name_en.")




