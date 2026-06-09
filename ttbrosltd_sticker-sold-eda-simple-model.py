# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df.head()


train_df.isnull().sum()


train_df.shape


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(train_df['num_sold'], bins=20, kde=False, color='blue')
plt.title('Distribution of num_sold')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


# Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x= train_df['num_sold'], color='green')
plt.title('Box Plot of num_sold')
plt.xlabel('Number Sold')
plt.show()


print(train_df['num_sold'].describe())


train_df['num_sold'].unique()


train_df['product'].unique()


# Group by product and calculate the sum of num_sold for each product
product_sales = train_df.groupby('product')['num_sold'].sum()

# Print the results
print("Sales for each product:")
for product, sales in product_sales.items():
    print(f"{product} - {sales}")


# Group by product and calculate the mean of num_sold for each product
product_avg_sales = train_df.groupby('product')['num_sold'].mean()

# Print the results
print("Average sales for each product:")
for product, avg_sales in product_avg_sales.items():
    print(f"{product} - {avg_sales:.2f}")


# Replace NaN values in num_sold with the corresponding average sales for each product
def fill_nan_with_avg(row):
    if pd.isna(row['num_sold']):
        return product_avg_sales[row['product']]
    else:
        return row['num_sold']

# Apply the function to the DataFrame
train_df['num_sold'] = train_df.apply(fill_nan_with_avg, axis=1)


train_df['num_sold'].head(10)


train_df.isnull().sum()


train_df.head()


product_to_num = {
    "Holographic Goose": 1,
    "Kaggle": 2,
    "Kaggle Tiers": 3,
    "Kerneler": 4,
    "Kerneler Dark Mode": 5
}

# Apply the mapping to the product column
train_df['product'] = train_df['product'].map(product_to_num)


train_df['product'].head(10)


train_df.head()


# Calculate the correlation matrix between product and num_sold
correlation_matrix = train_df[['product', 'num_sold']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap between Product and Num Sold')
plt.show()


train_df['country'].unique()


pivot_table = train_df.groupby(['country', 'product']).size().unstack(fill_value=0)

# Create a heatmap to visualize the counts
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.5)
plt.title('Product Counts by Country')
plt.xlabel('Product')
plt.ylabel('Country')
plt.show()


# Plotting the counts of each product by country using a bar plot
plt.figure(figsize=(12, 6))
sns.countplot(data= train_df, x='country', hue='product', palette='Set2')
plt.title('Product Counts by Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



country_mapping = {
    'Canada': 1,
    'Finland': 2,
    'Italy': 3,
    'Kenya': 4,
    'Norway': 5,
    'Singapore': 6
}

# Replace country names with numerical values
train_df['country'] = train_df['country'].replace(country_mapping)


train_df['country'].head()


train_df.head()


train_df['date'] = pd.to_datetime(train_df['date'])

# Extract year, month, and day into new columns
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day

# Display the updated dataframe
train_df[['date', 'year', 'month', 'day']].head()


train_df = train_df.drop(columns=['id', 'date'])


train_df.head()


train_df['store'].unique()


# Create a countplot to visualize the frequency of each store type in each country
plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='country', hue='store', palette='Set2')

# Adding titles and labels
plt.title('Frequency of Store Types in Each Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Display the plot
plt.show()


store_mapping = {
    'Discount Stickers': 1,
    'Stickers for Less': 2,
    'Premium Sticker Mart': 3
}

# Replace store names with corresponding numerical values
train_df['store'] = train_df['store'].replace(store_mapping)


train_df.head()


from sklearn.preprocessing import MinMaxScaler

# Normalize the num_sold column
scaler = MinMaxScaler()
train_df['num_sold'] = scaler.fit_transform(train_df[['num_sold']])

train_df.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Assuming your dataset is already loaded as train_df

# Preparing the dataset
X = train_df[['country', 'store', 'product', 'year', 'month', 'day']]  # Features
y = train_df['num_sold']  # Target

# Standardize the features (optional but usually beneficial for linear models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Ridge Regression model
ridge_model = Ridge(alpha=1.0)  

# Create the ensemble model using Voting Regressor
ensemble_model = VotingRegressor(estimators=[('ridge', ridge_model)])

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ensemble_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error of the ensemble model: {mse:.4f}")

# Optionally, you can also print R-squared to check the model's performance
r_squared = ensemble_model.score(X_test, y_test)
print(f"R-squared of the ensemble model: {r_squared:.4f}")

# Calculate the MAPE, avoiding division by zero
non_zero_indices = y_test != 0  # Find indices where y_test is not zero
y_test_non_zero = y_test[non_zero_indices]  # Filter out zero values from y_test
y_pred_non_zero = y_pred[non_zero_indices]  # Filter corresponding predictions

# Now calculate MAPE for non-zero values
mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")





