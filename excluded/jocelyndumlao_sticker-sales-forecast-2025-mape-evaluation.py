import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer

# Set a visually appealing color palette
sns.set_palette("viridis")
plt.style.use('seaborn-whitegrid')

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col='id')
submission_df =  pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train_df.head().style.background_gradient(cmap='YlOrBr')


test_df.head().style.background_gradient(cmap='YlOrBr')


submission_df.head()


# summary table function
def summary(train_df):
    print(f'data shape: {train_df.shape}')
    summ = pd.DataFrame(train_df.dtypes, columns=['data type'])
    summ['#missing'] = train_df.isnull().sum().values * 100
    summ['%missing'] = train_df.isnull().sum().values / len(train_df)
    summ['#unique'] = train_df.nunique().values
    desc = pd.DataFrame(train_df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = train_df.loc[0].values
    summ['second value'] = train_df.loc[1].values
    summ['third value'] = train_df.loc[2].values

    return summ


summary(train_df).style.background_gradient(cmap='plasma')


train_df.describe().style.background_gradient(cmap='tab20c')


# Time series plot
plt.figure(figsize=(12, 6))
plt.plot(train_df['date'], train_df['num_sold'])
plt.title("Time Series Plot of Number Sold")
plt.xlabel("Date")
plt.ylabel("Number Sold")
plt.xticks(rotation=45)
plt.show()

# Aggregate monthly for seasonality
monthly_sales = train_df.groupby(train_df['date'].dt.to_period('M'))['num_sold'].mean()
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Average Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.show()


# Weekly Seasonality - Boxplots
train_df['dayofweek'] = train_df['date'].dt.dayofweek
plt.figure(figsize=(10, 6))
sns.boxplot(x='dayofweek', y='num_sold', data=train_df)
plt.title("Weekly Sales Distribution")
plt.xlabel("Day of the Week (0=Monday, 6=Sunday)")
plt.ylabel("Number Sold")
plt.show()
print("Explanation: This boxplot shows the distribution of sales across different days of the week.  It helps identify if there is a weekly pattern in sales, such as if sales are higher on weekends. ")

# Rolling Statistics
window_size = 30 # Window size for rolling stats
train_df['rolling_mean'] = train_df['num_sold'].rolling(window=window_size).mean()
train_df['rolling_std'] = train_df['num_sold'].rolling(window=window_size).std()
plt.figure(figsize=(15, 7))
plt.plot(train_df['date'], train_df['num_sold'], label='Original Sales', alpha=0.7)
plt.plot(train_df['date'], train_df['rolling_mean'], label='30-Day Rolling Mean', color='red')
plt.plot(train_df['date'], train_df['rolling_std'], label='30-Day Rolling Std', color='purple')
plt.title("Sales with Rolling Mean and Std")
plt.xlabel("Date")
plt.ylabel("Number Sold")
plt.legend()
plt.xticks(rotation=45)
plt.show()
print("Explanation: Rolling statistics (mean and std) helps smooth out the time series and reveal trends or volatility changes that might not be apparent in the raw sales data. It helps identify periods of significant changes in the sales pattern.")

# Product Sales Comparison
product_sales = train_df.groupby('product')['num_sold'].mean().sort_values()
plt.figure(figsize=(10, 6))
product_sales.plot(kind='bar')
plt.title("Average Sales by Product")
plt.xlabel("Product")
plt.ylabel("Average Number Sold")
plt.xticks(rotation=45)
plt.show()
print("Explanation: Comparing the average sales of different products highlights which products are most popular or have higher sales volume. This can help with product-specific forecasting.")

# Country-Specific Sales
country_sales = train_df.groupby('country')['num_sold'].mean().sort_values()
plt.figure(figsize=(8, 6))
country_sales.plot(kind='bar', color=sns.color_palette("pastel"))
plt.title("Average Sales by Country")
plt.xlabel("Country")
plt.ylabel("Average Number Sold")
plt.xticks(rotation=45)
plt.show()
print("Explanation: Analyzing sales by country provides insights into how sales patterns differ in different geographical regions. This could reveal market-specific trends.")



train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', parse_dates=['date'], index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', parse_dates=['date'], index_col='id')
submission_df =  pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# Check for missing values
print(f"Missing values in train data:\n{train_df.isnull().sum()}")


train_df[['country', 'store', 'product']].drop_duplicates()


test_df[['country', 'store', 'product']].drop_duplicates()


# Feature Engineering
def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df.drop('date', axis=1, inplace=True)
    return df

train_df = create_date_features(train_df.copy())
test_df = create_date_features(test_df.copy())

# Encode Categorical Features
train_df = pd.get_dummies(train_df, columns=['country', 'store', 'product'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'], drop_first=True)

# Align columns (handling train and test might not match all columns)
train_cols = list(train_df.columns)
test_cols = list(test_df.columns)
target_col = 'num_sold'

for col in train_cols:
    if col not in test_cols and col != target_col:
        test_df[col] = 0
for col in test_cols:
    if col not in train_cols:
        train_df[col] = 0

train_cols = list(train_df.columns)
test_cols = list(test_df.columns)

test_df = test_df[train_cols[1:]] # exclude target col



# Fill missing target values with mean
train_df['num_sold'] = train_df['num_sold'].fillna(train_df['num_sold'].mean())


# model training
X_train, X_val, y_train, y_val = train_test_split(train_df.drop('num_sold', axis=1), train_df['num_sold'], test_size=0.2, random_state=42)


# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_val)
lr_mape = mean_absolute_percentage_error(y_val, lr_preds)
print(f"Linear Regression MAPE: {lr_mape:.5f}")


# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_mape = mean_absolute_percentage_error(y_val, rf_preds)
print(f"Random Forest MAPE: {rf_mape:.5f}")


# LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)
lgb_mape = mean_absolute_percentage_error(y_val, lgb_preds)
print(f"LightGBM MAPE: {lgb_mape:.5f}")

# Select the best model
models_mape = {
    'Linear Regression': lr_mape,
    'Random Forest': rf_mape,
    'LightGBM': lgb_mape
}
best_model_name = min(models_mape, key=models_mape.get)
print(f"Best model: {best_model_name} with MAPE: {models_mape[best_model_name]:.5f}")

if best_model_name == 'Linear Regression':
    best_model = lr_model
elif best_model_name == 'Random Forest':
    best_model = rf_model
else:
    best_model = lgb_model



# Prediction 
predictions = best_model.predict(test_df)

# submission
submission = submission_df.copy()
submission['num_sold'] = np.round(predictions).astype(int)  # Ensure integer values if required
submission.to_csv('submission.csv', index=False)  # Use index=False to avoid extra columns

print("\nSubmission file created successfully!")
print("\nSubmission Head:")
submission.head()




