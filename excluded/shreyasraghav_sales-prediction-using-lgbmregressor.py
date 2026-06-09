import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


#Initialize random seed for reproducibility
np.random.seed(42)

# Define data dimensions
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
countries = ['US', 'UK', 'FR']
stores = ['Store1', 'Store2']
products = ['ProductA', 'ProductB']


# Generate synthetic sales data
data = []
for country in countries:
    for store in stores:
        for product in products:
            # Create sales pattern components
            base_sales = np.random.normal(100, 20, len(dates))  # Base sales with noise
            trend = np.linspace(0, 20, len(dates))              # Upward trend
            seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates))/365)  # Yearly seasonality
            sales = np.maximum(base_sales + trend + seasonal, 0)  # Combine and ensure non-negative
            
            # Create daily records
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'country': country,
                    'store': store,
                    'product': product,
                    'num_sold': sales[i]
                })

# Create dataframe and split into train/test
df = pd.DataFrame(data)
train_df = df[df['date'] < '2023-12-01'].copy()  # Train: Jan-Nov 2023
test_df = df[df['date'] >= '2023-12-01'].copy()  # Test: Dec 2023


def prepare_time_features(df):
    
    df['date'] = pd.to_datetime(df['date'])
    # Extract temporal components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Encode categorical variables
    df['country_code'] = pd.Categorical(df['country']).codes
    df['store_code'] = pd.Categorical(df['store']).codes
    df['item_code'] = pd.Categorical(df['product']).codes
    
    return df

train_df = prepare_time_features(train_df)


plt.figure(figsize=(12, 6))
for country in countries:
    country_sales = train_df[train_df['country'] == country].groupby('date')['num_sold'].mean()
    plt.plot(country_sales.index, country_sales.values, label=country)
plt.title('Average Daily Sales by Country')
plt.xlabel('Date')
plt.ylabel('Average Sales')
plt.legend()
plt.show()


def prepare_sales_features(df):
 
    # Create lagged features
    for lag in [7, 14, 28]:
        df[f'sales_lag_{lag}'] = df.groupby(['country', 'store', 'product'])['num_sold'].shift(lag)
    
    # Create rolling means
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df

train_df = prepare_sales_features(train_df)


feature_cols = ['num_sold', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28', 
                'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28']
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[feature_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()


def train_model(train_df):
    features = ['year', 'month', 'day', 'dayofweek', 'is_weekend',
               'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
               'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
               'country_code', 'store_code', 'item_code']
    
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        random_state=42
    )
    train_data = train_df.dropna(subset=features + ['num_sold'])
    model.fit(train_data[features], train_data['num_sold'])
    return model, features

model, features = train_model(train_df)


plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()


def generate_test_features(train_df, test_df):
    train_df['is_test'] = 0
    test_df['is_test'] = 1
    combined_df = pd.concat([train_df, test_df], axis=0, sort=False)
    combined_df = combined_df.sort_values('date')
    
    if 'num_sold' not in test_df.columns:
        combined_df.loc[combined_df['is_test'] == 1, 'num_sold'] = np.nan
    
    # Generate lagged features
    for lag in [7, 14, 28]:
        combined_df[f'sales_lag_{lag}'] = combined_df.groupby(['country', 'store', 'product'])['num_sold'].shift(lag)
     # Generate rolling means
    for window in [7, 14, 28]:
        combined_df[f'rolling_mean_{window}'] = combined_df.groupby(['country', 'store', 'product'])['num_sold'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return combined_df[combined_df['is_test'] == 1].drop(['is_test'], axis=1)

test_df = prepare_time_features(test_df)
test_df = generate_test_features(train_df.copy(), test_df.copy())


# Generate predictions
predictions = model.predict(test_df[features])
predictions = np.maximum(predictions, 0)  # Ensure non-negative predictions

# Select sample data
sample_store = 'Store1'
sample_product = 'ProductA'
sample_country = 'US'

# Filter data for visualization
sample_train = train_df[
    (train_df['store'] == sample_store) & 
    (train_df['product'] == sample_product) &
    (train_df['country'] == sample_country)
]

sample_test = test_df[
    (test_df['store'] == sample_store) & 
    (test_df['product'] == sample_product) &
    (test_df['country'] == sample_country)
]


# Create visualization
plt.figure(figsize=(12, 6))
plt.plot(sample_train['date'], sample_train['num_sold'], label='Historical Sales')
plt.plot(sample_test['date'], predictions[:len(sample_test)], label='Predictions', linestyle='--')
plt.title(f'Sales Prediction for {sample_store} - {sample_product} ({sample_country})')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()

