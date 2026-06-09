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


!unzip '/kaggle/input/sberbank-russian-housing-market/macro.csv.zip'
!unzip '/kaggle/input/sberbank-russian-housing-market/test.csv.zip'
!unzip '/kaggle/input/sberbank-russian-housing-market/train.csv.zip'


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


train.shape


train


df_train = pd.read_csv('train.csv')
df_train


# Display dataset information
print(train.info())
print(train.head())


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Histogram of target variable (price_doc)
sns.histplot(train['price_doc'], bins=30, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Price (RUB)')
plt.show()

# 2. Bar chart for categorical variable (product_type)
sns.countplot(x='product_type', data=train)
plt.title('Distribution of Product Type')
plt.xlabel('Product Type')
plt.ylabel('Count')
plt.show()

# 3. Scatter plot for 'full_sq' vs 'price_doc'
sns.scatterplot(x='full_sq', y='price_doc', data=train)
plt.title('Full Square Area vs Price')
plt.xlabel('Full Square Area')
plt.ylabel('Price (RUB)')
plt.show()

# 4. Scatter plot for 'life_sq' vs 'price_doc'
sns.scatterplot(x='life_sq', y='price_doc', data=train)
plt.title('Living Square Area vs Price')
plt.xlabel('Living Square Area')
plt.ylabel('Price (RUB)')
plt.show()

# 5. Boxplot of 'floor' vs 'price_doc'
sns.boxplot(x='floor', y='price_doc', data=train)
plt.title('Floor vs Price')
plt.xlabel('Floor')
plt.ylabel('Price (RUB)')
plt.xticks(rotation=90)
plt.show()

# 6. Line plot for 'build_year' vs average 'price_doc'
train['build_year'] = pd.to_numeric(train['build_year'], errors='coerce')
build_year_avg_price = train.groupby('build_year')['price_doc'].mean().dropna()
plt.plot(build_year_avg_price, marker='o')
plt.title('Build Year vs Average Price')
plt.xlabel('Build Year')
plt.ylabel('Average Price (RUB)')
plt.xticks(rotation=90)
plt.show()


# Function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Removing outliers for each of the selected attributes
filtered_train = train.copy()  # Make a copy to avoid altering the original dataframe

for column in ['full_sq', 'life_sq', 'floor', 'build_year']:
    filtered_train = remove_outliers(filtered_train, column)

# Display the results
print("Shape before removing outliers:", train.shape)
print("Shape after removing outliers:", filtered_train.shape)


# Import necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to create "before and after" plots for a given attribute
def plot_before_after(attribute, title):
    plt.figure(figsize=(14, 6))

    # Plot histogram before removing outliers
    plt.subplot(1, 2, 1)
    sns.histplot(train[attribute], kde=True, color='blue', bins=50)
    plt.title(f'{title} - Before Removing Outliers')
    plt.xlabel(attribute)
    plt.ylabel('Frequency')

    # Plot histogram after removing outliers
    plt.subplot(1, 2, 2)
    sns.histplot(filtered_train[attribute], kde=True, color='green', bins=50)
    plt.title(f'{title} - After Removing Outliers')
    plt.xlabel(attribute)
    plt.ylabel('Frequency')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Boxplot comparison
    plt.figure(figsize=(14, 6))

    # Boxplot for before filtering outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(x=train[attribute], color='blue')
    plt.title(f'Boxplot of {attribute} - Before Removing Outliers')

    # Boxplot for after filtering outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=filtered_train[attribute], color='green')
    plt.title(f'Boxplot of {attribute} - After Removing Outliers')

    plt.tight_layout()
    plt.show()

# Plot "before and after" for the selected attributes
for attr in ['full_sq', 'life_sq', 'floor', 'build_year']:
    plot_before_after(attr, f'Distribution of {attr}')


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Correlation Heatmap (Filtered Data)
corr_matrix_filtered = filtered_train[['price_doc', 'full_sq', 'life_sq', 'floor', 'build_year']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Filtered Data)')
plt.show()

# 2. Correlation Heatmap (Original Data)
corr_matrix_original = train[['price_doc', 'full_sq', 'life_sq', 'floor', 'build_year']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_original, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Original Data)')
plt.show()

# 3. Trend: Average Price Per Year (Filtered Data)
filtered_train['year'] = pd.to_datetime(filtered_train['timestamp']).dt.year
avg_price_per_year_filtered = filtered_train.groupby('year')['price_doc'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_price_per_year_filtered, marker='o', label='Filtered Data', color='green')
plt.title('Average Price Per Year (Filtered Data)')
plt.xlabel('Year')
plt.ylabel('Average Price (RUB)')
plt.legend()
plt.grid()
plt.show()

# 4. Trend: Average Price Per Year (Original Data)
train['year'] = pd.to_datetime(train['timestamp']).dt.year
avg_price_per_year_original = train.groupby('year')['price_doc'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_price_per_year_original, marker='o', label='Original Data', color='blue')
plt.title('Average Price Per Year (Original Data)')
plt.xlabel('Year')
plt.ylabel('Average Price (RUB)')
plt.legend()
plt.grid()
plt.show()


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Check if 'product_type' exists in the filtered data
if 'product_type' in filtered_train.columns:
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop the first column to avoid the dummy variable trap

    # Perform OneHotEncoding on 'product_type'
    encoded_columns = encoder.fit_transform(filtered_train[['product_type']])

    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['product_type']))

    # Reset the index of both DataFrames to ensure proper concatenation
    filtered_train.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)

    # Concatenate the encoded columns with the original dataset
    filtered_train = pd.concat([filtered_train, encoded_df], axis=1)

    # Drop the original 'product_type' column as it has been encoded
    filtered_train.drop(columns=['product_type'], inplace=True)

    # Display the first few rows of the updated DataFrame
    print(filtered_train.head())
else:
    print("The column 'product_type' does not exist in the filtered dataset.")


from sklearn.preprocessing import OneHotEncoder

# Use OneHotEncoder for 'product_type'
encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid dummy variable trap
encoded_columns = encoder.fit_transform(train[['product_type']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['product_type']))

# Concatenate with original dataset
train = pd.concat([train.reset_index(drop=True), encoded_df], axis=1)
print(train.head())


from sklearn.preprocessing import OneHotEncoder

# Ensure the column names are stripped of any spaces
train.columns = train.columns.str.strip()

# Display the original 'product_type' column before encoding
print("Before Encoding - 'product_type':")
print(train['product_type'].value_counts())
print()

# Use OneHotEncoder for 'product_type'
encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid dummy variable trap
encoded_columns = encoder.fit_transform(train[['product_type']])

# Create a DataFrame for the encoded columns
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['product_type']))

# Concatenate the encoded columns with the original dataset
train_encoded = pd.concat([train.reset_index(drop=True), encoded_df], axis=1)

# Display the transformed dataset after encoding
print("After Encoding - 'product_type':")
print(train_encoded.head())

# Now you can check the columns and confirm encoding has been applied
print("\nColumns after encoding:", train_encoded.columns)


# Mean encode 'product_type'
mean_encoded_product_type = train.groupby('product_type')['price_doc'].mean()

# Map the encoded values back to the original dataset
train['product_type_mean_encoded'] = train['product_type'].map(mean_encoded_product_type)

# Display the first few rows to check the encoding
print(train[['product_type', 'product_type_mean_encoded']].head())


# Checking for Null values in the selected attributes
null_values = train[['full_sq', 'life_sq', 'floor', 'build_year', 'product_type']].isnull().sum()
print("Null values in the selected attributes:")
print(null_values)

# Filling Nulls with default values
# For numeric columns (full_sq, life_sq, floor, build_year), let's fill with 0
train['full_sq'].fillna(0, inplace=True)  # For square footage, assuming no square footage if missing
train['life_sq'].fillna(0, inplace=True)  # Assuming no life square if missing
train['floor'].fillna(-1, inplace=True)   # -1 means unknown floor (not 0, as 0 could mean ground floor)
train['build_year'].fillna(train['build_year'].mode()[0], inplace=True)  # Fill with mode (most frequent year)

# For the categorical column 'product_type', let's fill missing values with the most frequent category
train['product_type'].fillna(train['product_type'].mode()[0], inplace=True)

# Verifying if there are any remaining Null values
print("\nNull values after filling:")
print(train[['full_sq', 'life_sq', 'floor', 'build_year', 'product_type']].isnull().sum())

# Filling Nulls using group-by aggregate (using mean for numeric attributes)
# We will use group-by mean for numeric attributes (e.g., build_year)
train['build_year'] = train.groupby('product_type')['build_year'].transform(lambda x: x.fillna(x.mean()))

# Verifying again if there are any Nulls in 'build_year' after group-by fill
print("\nNull values after group-by aggregate:")
print(train[['build_year']].isnull().sum())


train


# Step 1: Verify if there are any remaining null values in the selected columns after imputation
null_values_after = train[['full_sq', 'life_sq', 'floor', 'build_year', 'product_type']].isnull().sum()
print("Null values after imputation:")
print(null_values_after)

# Step 2: Check a few sample rows from the modified columns to see if the filling worked
print("\nSample data after imputation:")
print(train[['full_sq', 'life_sq', 'floor', 'build_year', 'product_type']].head())

# You can also check the count of unique values in categorical columns after imputation (like product_type)
print("\nUnique values in 'product_type' after filling missing values:")
print(train['product_type'].value_counts())

# If you want to confirm that `build_year` null values were filled with the group mean (check for any null values)
print("\nNull values in 'build_year' after group-by fill:")
print(train['build_year'].isnull().sum())

# For categorical, verify if mode filling worked
print("\n'product_type' missing values after filling with mode:")
print(train['product_type'].isnull().sum())

