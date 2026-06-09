import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor


df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df.head()


df.shape


df.info()


df.isnull().sum()


#sales over time
df['date'] = pd.to_datetime(df['date'])

time_series = df.groupby('date')['num_sold'].sum()
plt.figure(figsize=(10, 6))
plt.plot(time_series.index, time_series.values, marker='o', linestyle='-', color='blue')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Stickers Sold')
plt.grid(True)
plt.show()


#sales by product
product_sales = df.groupby('product')['num_sold'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
product_sales.plot(kind='bar', color='orange')
plt.title('Product Sales Performance')
plt.xlabel('Product')
plt.ylabel('Number of Stickers Sold')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


#monthly sales distribution
df['month'] = df['date'].dt.month
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='num_sold', data=df, palette='coolwarm')
plt.title('Monthly Sales Distribution')
plt.xlabel('Month')
plt.ylabel('Number of Stickers Sold')
plt.show()


#total sales by country
country_sales = df.groupby('country')['num_sold'].sum().reset_index()
pivot_table = country_sales.pivot_table(index='country', values='num_sold')
plt.figure(figsize=(8, 10))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu', linewidths=0.5)
plt.title('Sales by Country')
plt.xlabel('Country')
plt.ylabel('Number of Stickers Sold')
plt.show()


#converting 'date' to datetime and split into 'year', 'month', and 'day'
df[['year', 'month', 'day']] = df['date'].apply(lambda x: pd.Series([x.year, x.month, x.day]))

#dropping the date column
df.drop(columns='date', inplace=True)


#filling the missing values in the num_sold column using KNN imputer.
features = ['num_sold', 'year', 'month', 'day']
knn_data = df[features]

knn_imputer = KNNImputer(n_neighbors=5)
imputed_data = knn_imputer.fit_transform(knn_data)
df[features] = imputed_data


#bar plot for monthly sales
df['month_year'] = df['year'].astype(str) + '-' + df['month'].astype(str)  # Create a 'month-year' column
monthly_sales = df.groupby('month_year')['num_sold'].sum()

plt.figure(figsize=(20, 10))
monthly_sales.plot(kind='bar', color='green')
plt.title('Total Sticker Sales Per Month')
plt.xlabel('Month-Year')
plt.ylabel('Number of Stickers Sold')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()


#sales distribution by day of the week
df['day_of_week'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.day_name()

plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='num_sold', data=df, palette='coolwarm', order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Sales Distribution by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Stickers Sold')
plt.show()


#features and target
x = df[['country', 'store', 'product', 'month', 'year', 'day']]
y = df['num_sold']

x = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#initializing models
models = {"Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)}


#training and evaluating each model
results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    results.append({"Model": name, "R² Score": r2, "MAPE": mape})

results_df = pd.DataFrame(results).sort_values(by="MAPE")
print(results_df)


#predicting again using random forest, which has the best MAPE, for visualization
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(x_train, y_train)
y_pred_rf = random_forest_model.predict(x_test)

#actual vs. predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Actual vs. Predicted Values (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()


#feature importance
importance = random_forest_model.feature_importances_
features = x.columns

indices = np.argsort(importance)
sorted_features = [features[i] for i in indices]
plt.figure(figsize=(6, 8))
plt.barh(sorted_features, importance[indices], color='purple')  # Horizontal bars
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(axis='x')
plt.show()


#loading the test dataset
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

#preprocessing test data
submission = pd.DataFrame({'id': test['id']})
test = test.drop(columns=['id'])

# One-hot encoding the test data and aligning columns
test = pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=x.columns, fill_value=0)


#making predictions on test data using random forest
predictions = random_forest_model.predict(test)

#adding predicted values to submission DataFrame and creating submission file
submission['num_sold'] = predictions.flatten().astype(int)
submission.to_csv('submission.csv', index=False)

