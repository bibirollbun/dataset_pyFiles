# Import the necessary libraries

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load the datasets

df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


# Print the few rows of the train dataset

df_train.head()


# Print the few rows of the test dataset

df_test.head()


# Print the shape of the data

print('Shape of the data:')
print(df_train.shape)


# print information about the columns in the train dataset

print('Information about the columns:')
print(df_train.info())


# print information about the columns in the test dataset

print('Information about the columns:')
print(df_test.info())


# Print the summary statistics for all variables of the train dataset

print('Summary statistics for all variables:')
df_train.describe()


# Print the summary statistics for all variables of the test dataset

print('Summary statistics for all variables:')
df_test.describe()


# Checking missing vales in train dataset

df_train.isnull().sum()


# Check percentage of missing values

missing_percentage = df_train['num_sold'].isnull().mean() * 100
print(f"Percentage of missing values in 'num_sold': {missing_percentage:.2f}%")


# Impute missing vales with the median(num_sold only have 3.85% data missing)

df_train['num_sold'].fillna(df_train['num_sold'].median(), inplace=True)


df_train.isnull().sum()


# Checking missing vales in test dataset

df_test.isnull().sum()


# Convert 'date' column to datetime

df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])


# Verify the conversion

print(df_train['date'].dtype)
print(df_test['date'].dtype)


# Create new features

for df in [df_train, df_test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


# Display the first few rows with new features

df_train.head()


# Aggregate data by date
sales_trend = df_train.groupby('date')['num_sold'].sum().reset_index()


# Plot sales trend
plt.figure(figsize=(15, 6))
sns.lineplot(data=sales_trend, x='date', y='num_sold')
plt.title('Sales Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines with dashed style
sns.despine()
plt.show()


# Map month numbers to month names using a dictionary
month_mapping = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 
    5: 'May', 6: 'June', 7: 'July', 8: 'August', 
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


# Apply the mapping to create a 'month_name' column
df_train['month_name'] = df_train['month'].map(month_mapping)


# Aggregate sales by month
monthly_sales = df_train.groupby('month_name')['num_sold'].sum().reset_index()


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales['month_name'] = pd.Categorical(monthly_sales['month_name'], categories=month_order, ordered=True)
monthly_sales = monthly_sales.sort_values('month_name')



# Plot monthly sales
plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_sales, x='month_name', y='num_sold', palette='viridis')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
sns.despine()
plt.xticks(rotation=45)
plt.show()


# Aggregate sales by day of the week
day_sales = df_train.groupby('day_of_week')['num_sold'].mean().reset_index()

# Map day numbers to labels
day_sales['day_of_week'] = day_sales['day_of_week'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})


# Plot sales by day of the week
plt.figure(figsize=(10, 5))
sns.barplot(data=day_sales, x='day_of_week', y='num_sold', palette='coolwarm')
plt.title('Average Sales by Day of the Week')
plt.xlabel('Day of the Week', labelpad = 15)
plt.ylabel('Average Sales',labelpad = 15)
sns.despine()
plt.show()


# Aggregate sales by product
product_sales = df_train.groupby('product')['num_sold'].sum().reset_index()


# Plot sales by product
plt.figure(figsize=(10, 5))
sns.barplot(data=product_sales, x='product', y='num_sold', palette='viridis')
plt.title('Total Sales by Product')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
sns.despine()
plt.show()


# Aggregate sales by country
country_sales = df_train.groupby('country')['num_sold'].sum().reset_index()


# Sort the countries by total sales (optional)
country_sales = country_sales.sort_values('num_sold', ascending=False)


# Plot the total sales by country
plt.figure(figsize=(12, 6))
sns.barplot(data=country_sales, x='country', y='num_sold', palette='viridis')
plt.title('Total Sales by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)

# Format y-axis to avoid scientific notation
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

sns.despine()
plt.xticks(rotation=45)  # Rotate country names for better readability
plt.show()


#Plot sales distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_train['num_sold'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sales (num_sold)')
plt.xlabel('Number of Items Sold')
plt.ylabel('Frequency')
sns.despine()
plt.show()


df_train.head()


# Filter for numerical columns only
numerical_cols = df_train.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numerical_cols.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap for Numerical Columns', fontsize=16)
plt.show()


# Aggregate sales by store
store_sales = df_train.groupby('store')['num_sold'].sum().reset_index()


# Plot sales by store
plt.figure(figsize=(8, 5))
sns.barplot(data=store_sales, x='store', y='num_sold', palette='muted')
plt.title('Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
sns.despine()
plt.show()


# Boxplot for sales by product
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train, x='product', y='num_sold', palette='Set2')
plt.title('Boxplot of Sales by Product')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()


def remove_outliers_iqr_per_group(df_train, group_col, target_col):
    """
    Remove outliers from the target column based on IQR within each group.
    """
    def outlier_bounds(x):
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    bounds = df_train.groupby(group_col)[target_col].apply(outlier_bounds)
    lower_bounds = bounds.apply(lambda x: x[0])
    upper_bounds = bounds.apply(lambda x: x[1])

    # Filter rows within bounds
    df_train = df_train.join(lower_bounds.rename('lower_bound'), on=group_col)
    df_train = df_train.join(upper_bounds.rename('upper_bound'), on=group_col)
    return df_train[(df_train[target_col] >= df_train['lower_bound']) & (df_train[target_col] <= df_train['upper_bound'])].drop(['lower_bound', 'upper_bound'], axis=1)

# Remove outliers for 'num_sold' grouped by 'product'
df_train = remove_outliers_iqr_per_group(df_train, group_col='product', target_col='num_sold')



# Before removing outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train, x='product', y='num_sold', palette='viridis')
plt.title('Sales by Product (After Outlier Removal)')
plt.show()


# Aggregate sales by date and country
country_trend = df_train.groupby(['date', 'country'])['num_sold'].sum().reset_index()


# Plot sales trends by country
plt.figure(figsize=(15, 6))
sns.lineplot(data=country_trend, x='date', y='num_sold', hue='country')
plt.title('Sales Trend Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend(title='Country')
sns.despine()
plt.show()


# Aggregate sales by product and country
product_country_sales = df_train.groupby(['product', 'country'])['num_sold'].sum().unstack()


# Plot stacked bar chart
product_country_sales.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Spectral')
plt.title('Sales by Product and Country')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.legend(title='Country')
plt.show()


df_train.drop('month_name', axis= 1, inplace=True)


df_train.head()


# Combine train and test data
df_train['dataset'] = 'train'  # Add a column to differentiate train data
df_test['dataset'] = 'test'    # Add a column to differentiate test data
df_combined = pd.concat([df_train, df_test], axis=0)

# Perform one-hot encoding on combined data
categorical_columns = ['country', 'store', 'product']
df_combined_encoded = pd.get_dummies(df_combined, columns=categorical_columns, drop_first=True)

# Split back into train and test datasets
df_train_encoded = df_combined_encoded[df_combined_encoded['dataset'] == 'train'].drop(columns=['dataset'])
df_test_encoded = df_combined_encoded[df_combined_encoded['dataset'] == 'test'].drop(columns=['dataset', 'num_sold'])

# Check the shapes
print(f"Shape of train data after encoding: {df_train_encoded.shape}")
print(f"Shape of test data after encoding: {df_test_encoded.shape}")


# Define features (X) and target variable (y)
X = df_train_encoded.drop(columns=['num_sold', 'id', 'date'])  # Exclude unnecessary columns
y = df_train_encoded['num_sold']


# Time-based split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train model
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = lgb_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape:.4f}")


# Prepare test features (drop unnecessary columns like 'id' and 'date')
X_test_final = df_test_encoded.drop(columns=['id', 'date'])


# Predict on the test data using the trained LightGBM model
y_pred_test_lgb = lgb_model.predict(X_test_final)


# Round predictions to the nearest integer (since 'num_sold' is likely an integer)
y_pred_test_lgb_rounded = y_pred_test_lgb.round().astype(int)


# Create the submission DataFrame
submission = pd.DataFrame({
    'id': df_test_encoded['id'],  # Use the 'id' column from the test dataset
    'num_sold': y_pred_test_lgb_rounded  # Use the rounded predictions
})


# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)



# Display the first few rows of the submission DataFrame
print(submission.head())

