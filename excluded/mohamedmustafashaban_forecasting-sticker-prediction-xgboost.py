! pip install xgboost
!pip install prophet


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from warnings import filterwarnings
from plotly.subplots import make_subplots
# Ignore warnings
filterwarnings('ignore')


# Load the training and testing datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train


train.info()


train.isnull().sum()


train.describe()


# Handle missing values by filling with the median
train['num_sold'] = train['num_sold'].fillna(train['num_sold'].median())

# Convert the date column to datetime and extract features
train['date'] = pd.to_datetime(train['date'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.dayofweek
train['day_of_year'] = train['date'].dt.dayofyear

test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek
test['day_of_year'] = test['date'].dt.dayofyear

# Add new features such as holidays (customizable)
train['is_holiday'] = train['date'].dt.date.isin(pd.to_datetime(['2025-01-01', '2025-12-25'])).astype(int)
test['is_holiday'] = test['date'].dt.date.isin(pd.to_datetime(['2025-01-01', '2025-12-25'])).astype(int)


sns.set(style="whitegrid")


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], color='blue')
plt.title('Original Time Series of Number Sold')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.show()



rolling_mean = train['num_sold'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_mean, label='Rolling Mean', color='red')
plt.title('Rolling Mean of Number Sold')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


rolling_std = train['num_sold'].rolling(window=30).std()
plt.figure(figsize=(12, 6))
plt.plot(train['date'], rolling_std, color='green')
plt.title('Rolling Standard Deviation of Number Sold')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(train['num_sold'], bins=30, kde=True)
plt.title('Distribution of Number Sold')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x=train['num_sold'])
plt.title('Boxplot of Number Sold')
plt.xlabel('Number Sold')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y='country', data=train, order=train['country'].value_counts().index)
plt.title('Count of Sales per Country')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()



plt.figure(figsize=(12, 6))
sns.countplot(y='store', data=train, order=train['store'].value_counts().index)
plt.title('Count of Sales per Store')
plt.xlabel('Count')
plt.ylabel('Store')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y='product', data=train, order=train['product'].value_counts().index)
plt.title('Count of Sales per Product')
plt.xlabel('Count')
plt.ylabel('Product')
plt.show()


plt.figure(figsize=(12, 6))
for country in train['country'].unique():
    subset = train[train['country'] == country]
    plt.plot(subset['date'], subset['num_sold'], label=country)
plt.title('Time Series of Number Sold by Country')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for store in train['store'].unique():
    subset = train[train['store'] == store]
    plt.plot(subset['date'], subset['num_sold'], label=store)
plt.title('Time Series of Number Sold by Store')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for product in train['product'].unique():
    subset = train[train['product'] == product]
    plt.plot(subset['date'], subset['num_sold'], label=product)
plt.title('Time Series of Number Sold by Product')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plot_acf(train['num_sold'], lags=30)
plt.title('Autocorrelation Function')
plt.show()


plt.figure(figsize=(12, 6))
plot_pacf(train['num_sold'], lags=30)
plt.title('Partial Autocorrelation Function')
plt.show()


decomposition = seasonal_decompose(train['num_sold'], model='additive', period=30)
fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.show()


adf_result = adfuller(train['num_sold'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_mean, label='Rolling Mean', color='red')
plt.title('Combined Time Series and Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_std, label='Rolling Std', color='green')
plt.title('Combined Time Series and Rolling Std')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.hist(train['num_sold'], bins=30, alpha=0.5, label='Number Sold', color='blue')
plt.axvline(rolling_mean.mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.title('Histogram with Mean Line')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='country', y='num_sold', data=train)
plt.title('Boxplot of Number Sold by Country')
plt.ylabel('Number Sold')
plt.xlabel('Country')
plt.show()


# Define features and target variable
X = train.drop(['num_sold', 'date'], axis=1)
y = train['num_sold']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define categorical and numerical columns for preprocessing
categorical_cols = ['country', 'store', 'product']
numerical_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'is_holiday']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
        ('cat', TargetEncoder(), categorical_cols)   # Encode categorical features
    ])

# Apply preprocessing to the training and validation sets
X_train = preprocessor.fit_transform(X_train, y_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(test.drop(['date'], axis=1))

# Build and fit the ARIMA model
model_arima = ARIMA(train['num_sold'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()


# Build and fit the Prophet model
train_prophet = train.rename(columns={'date': 'ds', 'num_sold': 'y'})
model_prophet = Prophet()
model_prophet.fit(train_prophet)

# Evaluate the ARIMA model
y_pred_arima = model_arima_fit.forecast(steps=len(y_val))
print(f'ARIMA MSE: {mean_squared_error(y_val, y_pred_arima)}')

# Define the XGBoost model
xgb_model = XGBRegressor()

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10]
}




# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best parameters and evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_val)

# Calculate MSE for the best model
mse_best = mean_squared_error(y_val, y_pred_best)
print(f'Best XGBoost Model MSE: {mse_best}')


# Make predictions on the test set
test_preds = best_model.predict(X_test)

# Prepare the submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'num_sold': test_preds})
submission.to_csv('submission.csv', index=False)

# Print the best parameters found by Grid Search
print(f"Best parameters from Grid Search: {grid_search.best_params_}")

