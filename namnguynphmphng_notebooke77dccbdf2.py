import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


calendar_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
train_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
df5 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
weights_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')



print("Number of unique day:",train_df['date'].nunique())
train_df.head()


print(f"Shape")
print(f"train: {train_df.shape}")
print(f"test: {test_df.shape}")


train_df.isnull().sum()


train_df = train_df.dropna(subset=['sales'])
train_df.isnull().sum()


Q1 = np.log1p(train_df["sales"]).quantile(0.25)
Q3 = np.log1p(train_df["sales"]).quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

train_df = train_df[(np.log1p(train_df["sales"]) >= lower) & (np.log1p(train_df["sales"]) <= upper)]


plt.figure(figsize=(6, 5))

sns.histplot(x=np.log1p(train_df['sales']), bins=100, kde=True)

plt.title(f'Log histplot of sales')
plt.xlabel('sales')
plt.ylabel('Frequency')

plt.show()


train_df.loc[train_df['type_0_discount'] < 0, 'type_0_discount'] = 0
train_df.loc[train_df['type_4_discount'] < 0, 'type_4_discount'] = 0
train_df.loc[train_df['type_6_discount'] < 0, 'type_6_discount'] = 0


from sklearn.preprocessing import MinMaxScaler
cols_to_scale = ['total_orders', 'sell_price_main']

scaler = MinMaxScaler()
train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])


train_df.head()


train_df = pd.get_dummies(train_df, columns=['warehouse'], prefix='wh', drop_first=False)


train_df.head()


keep_columns =  list(train_df.columns)
keep_columns


# Merge sales_train with calendar_df
train_df = train_df.merge(calendar_df, on='date', how='left')
# Merge sales_test with calendar_df
test_df = test_df.merge(calendar_df, on='date', how='left')


# Convert 'date' column to datetime format in both DataFrames
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Merge sales_train with inventory_df
train_df = train_df.merge(inventory_df, on='unique_id', how='left')
# Merge sales_test with inventory_df
test_df = test_df.merge(inventory_df, on='unique_id', how='left')


# Merge sales_test with test_weights
test_df = test_df.merge(weights_df, on='unique_id', how='left')
#Merge sales_train with test_weights
train_df = train_df.merge(weights_df, on='unique_id', how='left')


train_df['holiday_name'] = train_df['holiday_name'].fillna("N/A")
test_df['holiday_name'] = test_df['holiday_name'].fillna("N/A")

# Ensure holiday names in train and test are from the same set
common_holidays = set(train_df['holiday_name'].unique()).intersection(set(test_df['holiday_name'].unique()))

# Replace unknown holidays with "N/A" in train and test
train_df['holiday_name'] = train_df['holiday_name'].apply(lambda x: x if x in common_holidays else "N/A")
test_df['holiday_name'] = test_df['holiday_name'].apply(lambda x: x if x in common_holidays else "N/A")


print(list(train_df.columns))
keep_columns


# Total and average sales per category
category_sales = train_df.groupby('L1_category_name_en')['sales'].agg(['sum', 'mean']).reset_index()
category_sales.rename(columns={'sum': 'category_sales_sum', 'mean': 'category_sales_avg'}, inplace=True)
# Merge into training and test datasets
train_df = train_df.merge(category_sales, on='L1_category_name_en', how='left')
test_df = test_df.merge(category_sales, on='L1_category_name_en', how='left')

# Convert date to datetime and extract components
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week

# 1.Total and average orders per category
cat_order_stats = train_df.groupby('L1_category_name_en')['total_orders'].agg(['sum','mean']).reset_index()
cat_order_stats.rename(columns={'sum': 'category_orders_sum', 'mean': 'category_orders_avg'}, inplace=True)

# 2. Merge các chỉ số này vào train_df
train_df = train_df.merge(cat_order_stats,
                          on='L1_category_name_en',
                          how='left')

# 3. Merge các chỉ số này vào test_df
test_df = test_df.merge(cat_order_stats,
                        on='L1_category_name_en',
                        how='left')




# Calculate the total discount applied in Sales_Train
discount_columns = [f'type_{i}_discount' for i in range(7)]
train_df['total_discount'] = train_df[discount_columns].sum(axis=1)
# Calculate the total discount applied in Sales_Test
discount_columns_test = [f'type_{i}_discount' for i in range(7)]
test_df['total_discount'] = train_df[discount_columns_test].sum(axis=1)


keep_columns.append('category_sales_sum')
keep_columns.append('category_sales_avg')
keep_columns.append('category_orders_sum')
keep_columns.append('category_orders_avg')
keep_columns.append('total_discount')
keep_columns.append(('L1_category_name_en'))
train_df = train_df[keep_columns]
train_df.columns


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
# 1. Tạo feature nhiễu
np.random.seed(42)
train_df['random_noise'] = np.random.normal(0, 1, size=len(train_df))

# 2. Mã hóa và chuyển đổi date
le = LabelEncoder()
train_df['L1_category_name_en'] = le.fit_transform(train_df['L1_category_name_en'])

train_df['date'] = pd.to_datetime(train_df['date'])
train_df['date'] = (train_df['date'] - train_df['date'].min()).dt.days

# 3. Tách X và y, loại NaN
X = train_df.drop(columns=['sales'])
y = train_df['sales']
X = X.dropna()
y = y.loc[X.index]

# 4. Lấy sample để tránh OOM
sample_size = min(500_000, len(X))
X = X.sample(n=sample_size, random_state=42)
y = y.loc[X.index]

# 5. Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_columns = X.columns

# 6. Huấn luyện Lasso
lasso = Lasso(alpha=0.01, random_state=42, max_iter=5000)
lasso.fit(X_scaled, y)

# 7. Lấy hệ số và chọn feature quan trọng hơn noise
coef = pd.Series(lasso.coef_, index=X_columns)
noise_coef = coef.get('random_noise', 0)
important_features = coef[coef.abs() > abs(noise_coef)].drop('random_noise', errors='ignore') \
                           .sort_values(ascending=False)

important_features


