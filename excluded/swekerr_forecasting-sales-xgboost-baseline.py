# for eda & feature engineering
import numpy as np
import pandas as pd

# for heatmap visualization
import matplotlib.pyplot as plt
import seaborn as sns

# for xgboost modelling
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


df.head()


df.info()


df.describe()


df.isna().sum()


df = df.dropna()


df.isna().sum()


store_heatmap_data = df.pivot_table(index='store', columns='product', 
                                      values='num_sold', aggfunc='sum')

plt.figure(figsize=(5, 3))
sns.heatmap(store_heatmap_data)
plt.title('Sales Heatmap: Store vs Product')
plt.show()


country_heatmap_data = df.pivot_table(index='country', columns='product', 
                                        values='num_sold', aggfunc='sum')

plt.figure(figsize=(5, 3))
sns.heatmap(country_heatmap_data)
plt.title('Sales Heatmap: Country vs Product')
plt.show()


df['date'] = pd.to_datetime(df['date'])

df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['week'] = df['date'].dt.strftime('%W').astype(int)
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['is_weekend'] = df['day_of_week'] >= 5  # 0-4: weekdays, 5-6: weekends


import holidays

canada_holidays = holidays.Canada(years=df['year'].unique()) # why canada? -> coz saw in df.head()
df['is_holiday'] = df['date'].dt.date.isin(canada_holidays) 


df.head()


df.columns


X = df[['country', 'store', 'product', 'day_of_week', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']]
y = df['num_sold']


le = LabelEncoder()


for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col])


X = X.apply(pd.to_numeric)
X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=7000, 
                         learning_rate=0.001, max_depth=8, enable_categorical=True)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


df_test= pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

df_test['date'] = pd.to_datetime(df_test['date'])

df_test['day_of_week'] = df_test['date'].dt.dayofweek  # Monday=0, Sunday=6
df_test['week'] = df_test['date'].dt.strftime('%W').astype(int)
df_test['month'] = df_test['date'].dt.month
df_test['quarter'] = df_test['date'].dt.quarter
df_test['year'] = df_test['date'].dt.year
df_test['is_weekend'] = df_test['day_of_week'] >= 5  # 0-4: weekdays, 5-6: weekends

canada_holidays = holidays.Canada(years=df_test['year'].unique()) 
df_test['is_holiday'] = df_test['date'].dt.date.isin(canada_holidays)

X = df_test[['country', 'store', 'product', 'day_of_week', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']]

for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col])

X = X.apply(pd.to_numeric)


#X


prediction = model.predict(X)


output = pd.DataFrame({'id': df_test.id, 'num_sold': prediction})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

