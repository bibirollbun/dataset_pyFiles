import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")



train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col=0)


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col=0)


print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


train.head()


test.head()


import pandas as pd

def missing_percentage(df):
    missing_data = df.isnull().sum()  # Count missing values per column
    total_data = df.shape[0]  # Total number of rows
    missing_percentage = (missing_data / total_data) * 100  # Calculate percentage of missing values
    return missing_percentage


#Missing values percentage in train
missing_percentage(train)


#Missing values percentage in test
missing_percentage(test)


train = train.dropna().reset_index(drop=True)


missing_percentage(train)


train.describe().T


train.date.dtype


train['date']= pd.to_datetime(train['date'])


train.date.dtype


test['date'] = pd.to_datetime(test['date'])


train.describe().T


print(train['country'].value_counts())


print(train['product'].value_counts())


print(train['store'].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style='whitegrid')

# Group by country and sum the sales
sales_by_country = train.groupby('country')['num_sold'].sum().reset_index()

# Create a figure with a specific size and style
plt.figure(figsize=(14, 8))
plt.rc('axes', titlesize=16, titleweight='bold')  # Customize title size and weight
plt.rc('axes', labelsize=14, labelweight='bold')  # Customize label size and weight
plt.rc('xtick', labelsize=12)  # Customize x-tick label size
plt.rc('ytick', labelsize=12)  # Customize y-tick label size

# Plot sales by country
sns.barplot(data=sales_by_country, x='country', y='num_sold', palette='viridis')
plt.title('Total Sales by Country', fontsize=18, fontweight='bold')
plt.xlabel('Country', fontsize=16, fontweight='bold')
plt.ylabel('Total Sales', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train, x='country', y='num_sold', hue='store')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train, x='store', y='num_sold', hue='country')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train, x='country', y='num_sold', hue='product')


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=train, x='store', y='num_sold', hue='product')


train.head()


df = train.copy()


# Set the date column as the index
df.set_index('date', inplace=True)


df.head()


df.info()


# Set Seaborn style
sns.set(style='whitegrid')

# Group by date and sum sales (total sales over time)
total_sales = df.groupby('date')['num_sold'].sum()

# Create a figure with a specific size and style
plt.figure(figsize=(14, 8))
plt.rc('axes', titlesize=16, titleweight='bold')  # Customize title size and weight
plt.rc('axes', labelsize=14, labelweight='bold')  # Customize label size and weight
plt.rc('xtick', labelsize=12)  # Customize x-tick label size
plt.rc('ytick', labelsize=12)  # Customize y-tick label size

# Plot the total sales over time
plt.plot(total_sales, label='Total Sales', color='royalblue', linewidth=2)
plt.title('Total Sales Over Time', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=16, fontweight='bold')
plt.ylabel('Total Sales', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()



# Set Seaborn style
sns.set(style='whitegrid')

# Calculate the rolling mean and rolling standard deviation
rolling_mean = total_sales.rolling(window=30).mean()  # 30-day window
rolling_std = total_sales.rolling(window=30).std()    # 30-day window

# Create a figure with a specific size and style
plt.figure(figsize=(14, 8))
plt.rc('axes', titlesize=16, titleweight='bold')  # Customize title size and weight
plt.rc('axes', labelsize=14, labelweight='bold')  # Customize label size and weight
plt.rc('xtick', labelsize=12)  # Customize x-tick label size
plt.rc('ytick', labelsize=12)  # Customize y-tick label size

# Plot the total sales, rolling mean, and rolling std
plt.plot(total_sales, label='Total Sales', color='royalblue', alpha=0.6, linewidth=2)
plt.plot(rolling_mean, label='Rolling Mean (30 days)', color='red', linestyle='--', linewidth=2)
plt.plot(rolling_std, label='Rolling Std (30 days)', color='green', linestyle='--', linewidth=2)
plt.title('Total Sales with Rolling Mean and Std', fontsize=18, fontweight='bold', color='darkslategray')
plt.xlabel('Date', fontsize=16, fontweight='bold', color='darkslategray')
plt.ylabel('Total Sales', fontsize=16, fontweight='bold', color='darkslategray')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()



# Extract the year from the index
df['year'] = df.index.year

# Group by year and sum the sales
yearly_sales = df.groupby('year')['num_sold'].sum()

# Plot the yearly sales comparison
plt.figure(figsize=(12, 6))
yearly_sales.plot(kind='bar')
plt.title('Total Sales by Year')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# Set Seaborn style
sns.set(style='whitegrid')

# Create a pivot table with sales by date and product
sales_by_product = df.pivot_table(values='num_sold', index='date', columns='product', aggfunc='sum')

# Create a figure with a specific size and style
plt.figure(figsize=(14, 8))
plt.rc('axes', titlesize=16, titleweight='bold')  # Customize title size and weight
plt.rc('axes', labelsize=14, labelweight='bold')  # Customize label size and weight
plt.rc('xtick', labelsize=12)  # Customize x-tick label size
plt.rc('ytick', labelsize=12)  # Customize y-tick label size

# Plot sales for different products over time
sales_by_product.plot(ax=plt.gca(), colormap='viridis')
plt.title('Sales for Different Products Over Time', fontsize=18, fontweight='bold', color='darkslategray')
plt.xlabel('Date', fontsize=16, fontweight='bold', color='darkslategray')
plt.ylabel('Total Sales', fontsize=16, fontweight='bold', color='darkslategray')
plt.xticks(rotation=45, fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=14, title='Product', title_fontsize='13')
plt.tight_layout()
plt.show()


df.head()


# Extract time-based features
df['year'] = df.index.year
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['week_of_year'] = df.index.isocalendar().week
df['day_of_month'] = df.index.day
df['is_weekend'] = df.index.dayofweek >= 5  # 5 for Saturday, 6 for Sunday


df.head()


# Create Lag Features (e.g., sales from the previous day)
df['lag_1'] = df.groupby(['country', 'store', 'product'])['num_sold'].shift(1)
df['lag_2'] = df.groupby(['country', 'store', 'product'])['num_sold'].shift(2)
df['lag_3'] = df.groupby(['country', 'store', 'product'])['num_sold'].shift(3)


df.head()


# Calculate Daily Sales Difference
df['sales_diff'] = df.groupby(['country', 'store', 'product'])['num_sold'].diff()


df.head()


pip install holidays


import holidays
# Step 5: Create Holiday Feature (Example for Canada)
ca_holidays = holidays.Canada(years=df['year'].unique())
df['is_holiday'] = df.index.isin(ca_holidays)

# Step 6: Create Time Trend Feature (Days since the start of the dataset)
df['time_trend'] = (df.index - df.index.min()).days

# Drop rows with NaN values in lag or sales_diff columns (optional)
df.dropna(subset=['lag_1', 'sales_diff'], inplace=True)

# Final Preprocessed Data
df.head()


missing_percentage(df)


df['lag_1_missing'] = df['lag_1'].isnull().astype(int)
df['lag_2_missing'] = df['lag_2'].isnull().astype(int)
df['lag_3_missing'] = df['lag_3'].isnull().astype(int)
df[['lag_1', 'lag_2', 'lag_3']] = df[['lag_1', 'lag_2', 'lag_3']].fillna(0)  # Replace NaN with 0s after adding missing flags


df.head()


# Reset the index but keep the original index as a column
df = df.reset_index()


import pandas as pd
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Assume 'num_sold' is the target variable
X = df.drop('num_sold', axis=1)  # Features
y = df['num_sold']  # Target


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the splits
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# Convert the 'date' column to datetime format (if it's not already)
X_train['date'] = pd.to_datetime(X_train['date'])

# Extract features like year, month, day, weekday from the 'date' column
X_train['year'] = X_train['date'].dt.year
X_train['month'] = X_train['date'].dt.month
X_train['day'] = X_train['date'].dt.day
X_train['weekday'] = X_train['date'].dt.weekday

# Drop the original 'date' column after extraction
X_train = X_train.drop(columns=['date'])

# Do the same for the test data
X_test['date'] = pd.to_datetime(X_test['date'])
X_test['year'] = X_test['date'].dt.year
X_test['month'] = X_test['date'].dt.month
X_test['day'] = X_test['date'].dt.day
X_test['weekday'] = X_test['date'].dt.weekday
X_test = X_test.drop(columns=['date'])


# Define the encoder (for one-hot encoding categorical columns)
encoder = ColumnTransformer(
    transformers=[
        ('country', OneHotEncoder(), ['country']),  # One-hot encoding for 'country'
        ('store', OneHotEncoder(), ['store']),      # One-hot encoding for 'store'
        ('product', OneHotEncoder(), ['product'])   # One-hot encoding for 'product'
    ],
    remainder='passthrough'  # Keep other columns as they are
)

# Fit and transform on the training data
X_train_encoded = encoder.fit_transform(X_train)

# Now, transform the test data using the same encoder
X_test_encoded = encoder.transform(X_test)


# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_encoded, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_encoded)


# Check the shape of predictions
print(f"Predictions shape: {y_pred.shape}")


# Evaluate the model (optional)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


X_test.head()


# Prepare submission file
submission_df = pd.DataFrame({
    'id': X_test.index,  # 'id' from the test set index
    'num_sold': y_pred  # Predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)


df = pd.read_csv('submission.csv')


df.head()


df.describe()


df.shape

