# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.info()


df_sorted = train.sort_values(by=['country', 'store', 'product', 'date'])

# Group by 'country', 'store', and 'product' to fill missing 'num_sold' values
df_sorted['num_sold'] = df_sorted.groupby(['country', 'store', 'product'])['num_sold'].fillna(method='ffill')

# Then fill remaining missing values using backward fill
df_sorted['num_sold'] = df_sorted.groupby(['country', 'store', 'product'])['num_sold'].fillna(method='bfill')


df_sorted.info()


grouped = df_sorted.groupby(['country', 'store', 'product'])['num_sold'].sum().reset_index()


# 2. Calculate total sales for each country-store combination
total_sales = df_sorted.groupby(['country', 'store'], as_index=False)['num_sold'].sum()

# 3. Merge to get total sales for each country-store combination into the grouped DataFrame
result = pd.merge(grouped, total_sales, on=['country', 'store'], suffixes=('', '_total'))

# 4. Calculate the percentage contribution for each country-store-product combination
result['percentage'] = (result['num_sold'] / result['num_sold_total']) * 100

# Display the result
result


#def plot_total_sales_by_date(country, store, df):
#    # Filter the data for the given country and store
#    filtered_df = df[(df['country'] == country) & (df['store'] == store)]
#    
#    # Group by date and sum the num_sold for all products
#    total_sales_by_date = filtered_df.groupby('date')['num_sold'].sum().reset_index()
#    
#    # Check if data exists for the given combination
#    if total_sales_by_date.empty:
#        print(f"No data available for {country} - {store}.")
#        return
    # Create a line plot for total num_sold by date
##    
#    plt.figure(figsize=(8, 6))
#    sns.lineplot(data=total_sales_by_date, x='date', y='num_sold')

    # Set plot labels and title
#   plt.xlabel('Date')
#   plt.ylabel('Total Number Sold')
#   plt.title(f'Total Sales in {store} ({country}) by Date')

    # Rotate the date labels for better readability
#    plt.xticks(rotation=45)

#    # Show the plot
#    plt.tight_layout()
#    plt.show()


#plot_total_sales_by_date('Finland', 'Discount Stickers', df_sorted)  # You can change this to an
#plot_total_sales_by_date('Norway', 'Discount Stickers', df_sorted)  # You can change this to an
#plot_total_sales_by_date('Singapore', 'Discount Stickers', df_sorted)  # You can change this to an
#plot_total_sales_by_date('Italy', 'Discount Stickers', df_sorted)  # You can change this to an
#plot_total_sales_by_date('Canada', 'Discount Stickers', df_sorted)  # You can change this to an
#plot_total_sales_by_date('Kenya', 'Discount Stickers', df_sorted)  # You can change this to an



def set_sales_for_specific_product_with_ratio(df, target_country, target_store, target_product, source_country, source_store, source_product):
    """
    This method sets the num_sold for a specific product in a source country-store combination to 
    the num_sold values from a target country-store combination, 
    scaled by a ratio calculated as the total num_sold at the target country-store combination 
    divided by the total num_sold at the source country-store combination (for all products).
    
    Parameters:
    - df: The DataFrame containing the sales data.
    - target_country: The country from which to take the target sales values.
    - target_store: The store from which to take the target sales values.
    - target_product: The product whose sales will be used to scale the source product's sales.
    - source_country: The country to which the sales values will be updated.
    - source_store: The store to which the sales values will be updated.
    - source_product: The product whose sales will be updated.
    
    Returns:
    - The updated DataFrame with num_sold for the specific product scaled by the ratio for the source combination.
    """
    
    # Filter the target data for the target country-store combination and the specific product
    target_data = df[(df['country'] == target_country) & 
                     (df['store'] == target_store) & 
                     (df['product'] == target_product)]
    
    # Filter the source data for the source country-store combination and the specific product
    source_data = df[(df['country'] == source_country) & 
                     (df['store'] == source_store) & 
                     (df['product'] == source_product)]
    
    # Calculate total num_sold for all products in the target and source country-store combinations
    total_target_sales = df[(df['country'] == target_country) & 
                            (df['store'] == target_store)]['num_sold'].sum()
    
    total_source_sales = df[(df['country'] == source_country) & 
                            (df['store'] == source_store)]['num_sold'].sum()
    
    # Avoid division by zero
    if total_source_sales == 0:
        raise ValueError(f"Total sales for the source combination ({source_country}-{source_store}) is zero.")
    
    # Calculate the ratio (total num_sold at target / total num_sold at source)
    ratio = total_target_sales / total_source_sales

    print('total_target_sales', total_target_sales)
    print('total_source_sales', total_source_sales)
    print('ratio:', ratio)
   
    df = df.merge(source_data[['date', 'num_sold']], on='date', how='left', suffixes=('', '_source'))

    # Set the num_sold for the source product based on the target num_sold scaled by the ratio
    df.loc[(df['country'] == target_country) & 
           (df['store'] == target_store) & 
           (df['product'] == target_product), 'num_sold'] = df['num_sold_source'] * ratio
    
    # Drop the extra merged column
    df.drop(columns=['num_sold_source'], inplace=True)

    return df



train = set_sales_for_specific_product_with_ratio(df_sorted, 'Canada', 'Discount Stickers', 'Holographic Goose', 'Italy', 'Discount Stickers', 'Holographic Goose')
#train = set_sales_for_specific_product_with_ratio(train, 'Kenya', 'Discount Stickers', 'Holographic Goose', 'Singapore', 'Discount Stickers', 'Holographic Goose')

train[train['country'] == 'Canada'].head(200)


def fill_nulls_for_Kenya(df):
    """
    This method updates the num_sold for each product in the target country-store combination
    by scaling the num_sold of the corresponding product in the source country-store combination.
    The scaling ratio is calculated as the total num_sold at target country-store divided by the 
    total num_sold at source country-store (for all products).

    Parameters:
    - df: The DataFrame containing the sales data.
    - target_country: The country from which to take the target sales values.
    - target_store: The store from which to take the target sales values.
    - source_country: The country from which to take the source sales values.
    - source_store: The store from which to take the source sales values.
    - source_product: The specific product in the source store-country combination to scale.
    
    Returns:
    - The updated DataFrame with num_sold for the target country-store-product combination.
    """

    #total_sales_per_day = df[df['country'] == 'Kenya'].groupby('date')['num_sold'].transform('sum')

    total_sales_per_day = df[(df['country'] == 'Kenya') & (df['store'] == 'Discount Stickers')].groupby('date')['num_sold'].sum().reset_index()
    #total_sales_per_day.rename(columns={'num_sold': 'total_num_sold'}, inplace=True)

    print(total_sales_per_day.head(20))
    
    ratio = 4.75 / 95.25
    print(f"Calculated ratio: {ratio:.2f}")

    # Merge source_data with df to get the corresponding target product for the target country-store combination
    df_merged = df.merge(total_sales_per_day[['date', 'num_sold']], on='date', how='left', suffixes=('', '_source'))
    
    # Now, update num_sold for the target country-store-product combination
    df_merged.loc[(df_merged['country'] == 'Kenya') & 
                  (df_merged['store'] == 'Discount Stickers') & 
                  (df_merged['product'] == 'Holographic Goose'), 'num_sold'] = df_merged['num_sold_source'] * ratio
    
    # Debug: Check if num_sold has been updated
    print("After update:")
    print(df_merged[df_merged['country'] == 'Kenya'][['country', 'store', 'product', 'date', 'num_sold']].head())

    # Drop the extra merged column
    df_merged.drop(columns=['num_sold_source'], inplace=True)
    
    return df_merged


train = fill_nulls_for_Kenya(train)
#train = set_sales_for_specific_product_with_ratio(train, 'Kenya', 'Discount Stickers', 'Holographic Goose', 'Singapore', 'Discount Stickers', 'Holographic Goose')

train[train['country'] == 'Kenya'].head(200)



train.info()


# Feature engineering function
def feature_engineering(df):
    # Generate new columns from date
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['sine_day'] = np.sin(2 * np.pi * df['day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
    df['sine_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sine_year'] = np.sin(2 * np.pi * df['year'])
    df['cos_year'] = np.cos(2 * np.pi * df['year'])

    # Convert categorical columns to category dtype
    for col in ['country', 'store', 'product']:
        df[col] = df[col].astype('category')

    return df

train = feature_engineering(train)


train.info()


# Apply feature engineering and preprocessing
train = train.dropna() #feature_engineering(train)
train.info()
#train = preprocess_data(train)

# Prepare data for training
X = train.drop(columns=['id', 'date', 'num_sold'])
X = pd.get_dummies(X, drop_first=True)
y = np.log1p(train['num_sold'])

# Load test data
test = feature_engineering(test)
X_test = test.drop(columns=['id', 'date'])
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)



# Initialize variables
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_predictions = np.zeros(len(train))
mape_scores = []
test_predictions_list = []

# CatBoost Hyperparameters
params = {
    'iterations': 1000,             # Number of trees to build
    'learning_rate': 0.1,           # Learning rate
    'depth': 6,                     # Depth of the tree
    'loss_function': 'MAPE',        # Loss function for regression
    'cat_features': [],             # List of categorical features (if applicable)
    'random_seed': 42,              # Random seed for reproducibility
    'verbose': 200,                 # Print information every 200 iterations
}


# Train and validate model using 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train CatBoostRegressor
    model = LGBMRegressor()#XGBRegressor()#ExtraTreesRegressor(random_state=42, n_jobs=-1)  # RandomForestRegressor(random_state=42, n_jobs=-1) # CatBoostRegressor(**params) # #
    model.fit(X_train, y_train)

    # Make predictions
    y_val_pred = model.predict(X_val)
    train_predictions[val_idx] = y_val_pred

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    mape_scores.append(mape)
    print(f"Fold {fold}: MAPE = {mape:.4f}")

    # Predict on test data for this fold
    test_pred_fold = model.predict(X_test)
    test_predictions_list.append(test_pred_fold)

# Average test predictions across folds
test_predictions_avg = np.mean(test_predictions_list, axis=0)
test_predictions_avg = np.expm1(test_predictions_avg) 

# Print training MAPE score
print(f"Training MAPE score (5-fold average): {np.mean(mape_scores):.4f}")



# Plot actual vs. predicted values (daily, weekly, monthly)
#def plot_predictions(data, predictions, freq, title):
#    data['predicted'] = np.expm1(predictions)
#    aggregated = data.groupby(pd.Grouper(key='date', freq=freq))[['num_sold', 'predicted']].sum()

#    plt.figure(figsize=(12, 6))
#    plt.plot(aggregated.index, aggregated['num_sold'], label='Actual')
#    plt.plot(aggregated.index, aggregated['predicted'], label='Predicted')
#    plt.title(title)
#    plt.xlabel('Date')
#    plt.ylabel('Num Sold')
#    plt.legend()
#    plt.show()

#plot_predictions(train, train_predictions, 'D', 'Daily Actual vs Predicted')
#plot_predictions(train, train_predictions, 'W', 'Weekly Actual vs Predicted')
#plot_predictions(train, train_predictions, 'M', 'Monthly Actual vs Predicted')


# Print test predictions
test['predicted_num_sold'] = test_predictions_avg
print(test[['id', 'predicted_num_sold']])

# Save predictions
test[['id', 'predicted_num_sold']].to_csv('submission.csv', index=False)


