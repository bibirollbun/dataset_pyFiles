import pandas as pd
import zipfile

# Path to your zip file
zip_path = '/kaggle/input/sberbank-russian-housing-market/train.csv.zip'

# Extract the train.csv file
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall()  # Extract all files

# Load the dataset
data = pd.read_csv('train.csv')  # Ensure the correct path is provided
print("Data Loaded Successfully")
print(data.head())  # Display the first few rows


# Check column names and datatypes
print("Dataset Information:")
print(data.info())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values[missing_values > 0])


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for continuous variables
data[['price_doc', 'full_sq', 'life_sq']].hist(bins=20, figsize=(12, 6))
plt.suptitle("Histograms for Continuous Attributes")
plt.show()

# Scatter plot for floor vs price_doc
plt.figure(figsize=(8, 6))
sns.scatterplot(x='floor', y='price_doc', data=data)
plt.title('Floor vs Price')
plt.xlabel('Floor')
plt.ylabel('Price')
plt.show()

# Count plot for state
sns.countplot(x='state', data=data)
plt.title('State Distribution')
plt.show()



# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for price_doc and full_sq
data_cleaned = remove_outliers(data, 'price_doc')
data_cleaned = remove_outliers(data_cleaned, 'full_sq')

print(f"Original Dataset Size: {data.shape}")
print(f"Cleaned Dataset Size: {data_cleaned.shape}")



# Correlation matrix
correlation_matrix = data_cleaned[['price_doc', 'full_sq', 'life_sq', 'floor']].corr()

# Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()






# One-hot encoding for 'state'
state_encoded = pd.get_dummies(data_cleaned['state'], prefix='state')
data_cleaned = pd.concat([data_cleaned, state_encoded], axis=1)

# Mean encoding for 'state'
mean_encoded_state = data_cleaned.groupby('state')['price_doc'].mean()
data_cleaned['state_mean_encoded'] = data_cleaned['state'].map(mean_encoded_state)

print(data_cleaned.head())



# Fill missing values with median
data_cleaned['life_sq_median_filled'] = data_cleaned['life_sq'].fillna(data_cleaned['life_sq'].median())

# Fill missing values using group mean
data_cleaned['life_sq_group_mean_filled'] = data_cleaned.groupby('state')['life_sq'].transform(lambda x: x.fillna(x.mean()))

print(data_cleaned[['life_sq', 'life_sq_median_filled', 'life_sq_group_mean_filled']].head())



# Fill missing values in 'life_sq' using group mean based on 'state'
data_cleaned['life_sq_group_mean_filled'] = data_cleaned.groupby('state')['life_sq'].transform(lambda x: x.fillna(x.mean()))

print(data_cleaned[['state', 'life_sq', 'life_sq_group_mean_filled']].head())


