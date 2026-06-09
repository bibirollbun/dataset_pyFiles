import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


train_data.info()


train_data.head()


#we have 8871 total entires of num_sold column as null, which is quite a lot, so we cannot drop
train_data.isnull().sum()


# Replace null values with mean of that column
mean_num_sold = train_data['num_sold'].mean()
train_data['num_sold'].fillna(mean_num_sold, inplace=True)
train_data.isnull().sum()


#Next, convert 'date' to datetime
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


# Check how sales vary by country, store, and product
import seaborn as sns
import matplotlib.pyplot as plt

# Sales distribution by country
sns.boxplot(x='country', y='num_sold', data=train_data)
plt.title('Sales by Country')
plt.show()

# Sales distribution by product
sns.boxplot(x='product', y='num_sold', data=train_data)
plt.xticks(rotation=45)
plt.title('Sales by Product')
plt.show()


# Extract day of week and month
train_data['day_of_week'] = train_data['date'].dt.dayofweek  # 0=Monday, 6=Sunday
train_data['month'] = train_data['date'].dt.month

# Average sales by day of week
train_data.groupby('day_of_week')['num_sold'].mean().plot(kind='bar', title='Sales by Day of Week')
plt.show()


# Create features to help the model capture patterns.
def add_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 5=Sat, 6=Sun
    return df

train_data = add_date_features(train_data)
test_data = add_date_features(test_data)


import holidays

def add_holidays(df):
    df['is_holiday'] = 0
    for country in df['country'].unique():
        country_holidays = holidays.CountryHoliday(country)
        df.loc[df['country'] == country, 'is_holiday'] = df['date'].isin(country_holidays).astype(int)
    return df

train_data = add_holidays(train_data)
test_data = add_holidays(test_data)



# Convert categorical columns to 'category' type
cat_cols = ['country', 'store', 'product']
train_data[cat_cols] = train_data[cat_cols].astype('category')
test_data[cat_cols] = test_data[cat_cols].astype('category')


features = ['year', 'month', 'day_of_week', 'is_weekend', 'country', 'store', 'product']

# Sort data by date
train_data = train_data.sort_values('date')


# Split by time (e.g., last 20% as validation)
split_idx = int(0.8 * len(train_data))
X_train = train_data.iloc[:split_idx][features]
y_train = train_data.iloc[:split_idx]['num_sold']
X_val = train_data.iloc[split_idx:][features]
y_val = train_data.iloc[split_idx:]['num_sold']


print("X_train columns:", X_train.columns.tolist())
print("X_val columns:", X_val.columns.tolist())


import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Predict
val_preds = model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, val_preds)}")


#To be improved

