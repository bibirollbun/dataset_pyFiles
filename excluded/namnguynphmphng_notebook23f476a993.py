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



print(train_df.columns)
print(test_df.columns)


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


# Tạo các cột mới để xác định xem ngày đó có phải là holiday hoặc cửa hàng đóng cửa
train_df['is_holiday'] = train_df['holiday'].apply(lambda x: 1 if x == 1 else 0)
train_df['is_shop_closed'] = train_df['shops_closed'].apply(lambda x: 1 if x == 1 else 0)
train_df['is_winter_school_holiday'] = train_df['winter_school_holidays'].apply(lambda x: 1 if x == 1 else 0)

# So sánh doanh thu trong các ngày lễ và không lễ
holiday_sales = train_df.groupby('is_holiday')['sales'].mean()

# So sánh doanh thu trong các ngày cửa hàng đóng và cửa hàng mở
shop_closed_sales = train_df.groupby('is_shop_closed')['sales'].mean()

# So sánh doanh thu trong kỳ nghỉ đông và không kỳ nghỉ đông
winter_holiday_sales = train_df.groupby('is_winter_school_holiday')['sales'].mean()

print(holiday_sales)
print(shop_closed_sales)
print(winter_holiday_sales)


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

# Add product-level sales or availability patterns
product_sales = train_df.groupby('product_unique_id')['sales'].sum().reset_index()
product_sales.rename(columns={'sales': 'product_sales_sum'}, inplace=True)
# Merge with train and test
train_df = train_df.merge(product_sales, on='product_unique_id', how='left')
test_df = test_df.merge(product_sales, on='product_unique_id', how='left')


# Calculate the total discount applied in Sales_Train
discount_columns = [f'type_{i}_discount' for i in range(7)]
train_df['total_discount'] = train_df[discount_columns].sum(axis=1)
# Calculate the total discount applied in Sales_Test
discount_columns_test = [f'type_{i}_discount' for i in range(7)]
test_df['total_discount'] = test_df[discount_columns_test].sum(axis=1)


keep_columns.append('category_sales_sum')
keep_columns.append('category_sales_avg')
keep_columns.append('product_sales_sum')
keep_columns.append('total_discount')
keep_columns.append('product_unique_id')
keep_columns.append(('L1_category_name_en'))
keep_columns.append(('holiday'))
keep_columns.append(('shops_closed'))
keep_columns.append(('winter_school_holidays'))
train_df = train_df[keep_columns]
train_df.columns


# train_df.head()
from sklearn.preprocessing import LabelEncoder

# Khởi tạo LabelEncoder
label_encoder = LabelEncoder()

train_df['random_noise'] = np.random.normal(0, 1, size=len(train_df))
X = train_df.drop(columns=['sales'])  # hoặc đổi thành tên cột mục tiêu thật sự
X['L1_category_name_en'] = label_encoder.fit_transform(X['L1_category_name_en'])
X['warehouse'] = label_encoder.fit_transform(X['warehouse'])
X['date'] = (X['date'] - X['date'].min()).dt.days  # Chuyển đổi ngày thành số ngày kể từ ngày sớm nhất trong dữ liệu
y = train_df['sales']
X = X.dropna()
y = y.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_columns = X.columns

lasso = Lasso(alpha=0.01)  # có thể thử LassoCV nếu muốn chọn alpha tốt nhất
lasso.fit(X_scaled, y)

coef = pd.Series(lasso.coef_, index=X_columns)

# Hệ số của cột nhiễu
noise_coef = coef['random_noise']

# Các đặc trưng có trọng số tuyệt đối lớn hơn nhiễu
important_features = coef[np.abs(coef) > np.abs(noise_coef)].drop('random_noise', errors='ignore').sort_values(ascending=False)
important_features

