import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train=  pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


train


test


train.isna().sum()


train.dropna(subset=['num_sold'], inplace=True)


train['date'].unique()


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])



for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    



train



import numpy as np
# Replace inf and -inf with NaN in both `df` and `train` if needed
df.replace([np.inf, -np.inf], np.nan, inplace=True)
train.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN in key columns to avoid issues during plotting
train.dropna(subset=['date', 'num_sold'], inplace=True)

# Convert date column to datetime if it's not already
train['date'] = pd.to_datetime(train['date'], errors='coerce')

# Group by date and sum the number of stickers sold
sales_over_time = train.groupby('date')['num_sold'].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_over_time, x='date', y='num_sold')
plt.title('Sticker Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Stickers Sold', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



sales_by_country = train.groupby('country')['num_sold'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=sales_by_country, x='num_sold', y='country', palette='viridis')
plt.title('Total Sticker Sales by Country')
plt.xlabel('Total Stickers Sold')
plt.ylabel('Country')
plt.show()



sales_by_product = train.groupby('product')['num_sold'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=sales_by_product, x='num_sold', y='product', palette='coolwarm')
plt.title('Total Sticker Sales by Product')
plt.xlabel('Total Stickers Sold')
plt.ylabel('Product')
plt.show()




train['day_of_week'] = train['date'].dt.dayofweek

sales_by_day = train.groupby('day_of_week')['num_sold'].sum().reset_index()

# Line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=sales_by_day, x='day_of_week', y='num_sold', marker='o')
plt.title('Daily Sales Patterns')
plt.xlabel('Days (0=Monday, 6=Sunday)')
plt.ylabel('Total Stickers Sold')
plt.show()



from sklearn.preprocessing import LabelEncoder

for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



train


X = train.drop(['id', 'date', 'num_sold'], axis=1)
y = train['num_sold']
X_test = test.drop(['id', 'date'], axis=1)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f'Mean Absolute Percentage Error(MAPE): {mape}')




test['num_sold'] = model.predict(X_test)

submission = test[['id', 'num_sold']]
submission.to_csv('submission.csv', index=False)
submission


