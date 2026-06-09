import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
data.head(10)


print("Procentage of missing data: \n\n{}".format((data.isnull().sum() / data.count()) * 100 ))


data = data.dropna()

print("Procentage of missing data: \n\n{}".format((data.isnull().sum() / data.count()) * 100 ))


import matplotlib.pyplot as plt
import seaborn as sns


# Pivot data for heatmap
store_heatmap_data = data.pivot_table(index='store', columns='product', values='num_sold', aggfunc='sum')

plt.figure(figsize=(5, 3))
sns.heatmap(store_heatmap_data, cmap="coolwarm")
plt.title('Sales Heatmap: Store vs Product')
plt.show()


# Pivot data for heatmap
country_heatmap_data = data.pivot_table(index='country', columns='product', values='num_sold', aggfunc='sum')

plt.figure(figsize=(5, 3))
sns.heatmap(country_heatmap_data, cmap="coolwarm")
plt.title('Sales Heatmap: Country vs Product')
plt.show()


for country in data.country.unique():
    num_sold_per_country = data.loc[data['country'] == country].num_sold
    print("The average number sold in {} was {}".format(country, num_sold_per_country.mean()))


data['date'] = pd.to_datetime(data['date'])

data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
data['week'] = data['date'].dt.strftime('%W').astype(int)
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['year'] = data['date'].dt.year
data['is_weekend'] = data['day_of_week'] >= 5  # 0-4: weekdays, 5-6: weekends


import holidays

us_holidays = holidays.US(years=data['year'].unique())  # You can specify years for your data
data['is_holiday'] = data['date'].dt.date.isin(us_holidays)


data.head()


data.columns


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Prepare your features and target
X = data[['country', 'store', 'product', 'day_of_week', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']]
y = data['num_sold']


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Convert categorical columns to 'category' dtype

for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col])


X = X.apply(pd.to_numeric)


X


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the XGBoost regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5000, learning_rate=0.1, max_depth=10, enable_categorical=True)

# Fit the model
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()


# Plot the first tree of the model
#xgb.plot_tree(model, num_trees=0)
#plt.show()





data = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

data['date'] = pd.to_datetime(data['date'])

data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
data['week'] = data['date'].dt.strftime('%W').astype(int)
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['year'] = data['date'].dt.year
data['is_weekend'] = data['day_of_week'] >= 5  # 0-4: weekdays, 5-6: weekends

us_holidays = holidays.US(years=data['year'].unique())  # You can specify years for your data
data['is_holiday'] = data['date'].dt.date.isin(us_holidays)

X = data[['country', 'store', 'product', 'day_of_week', 'week', 'month', 'quarter', 'is_weekend', 'is_holiday']]

for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col])

X = X.apply(pd.to_numeric)


X


prediction = model.predict(X)


# Adjust the output DataFrame according to your dataset
output = pd.DataFrame({'id': data.id, 'num_sold': prediction})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

