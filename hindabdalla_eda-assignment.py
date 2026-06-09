!unzip -o ../input/sberbank-russian-housing-market/train.csv.zip
!unzip -o ../input/sberbank-russian-housing-market/test.csv.zip 
!unzip -o ../input/sberbank-russian-housing-market/macro.csv.zip
!ls


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('train.csv')
df_train


df = pd.read_csv('train.csv') 
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of house prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price_doc'], kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Plot number of rooms vs. house price
plt.figure(figsize=(10, 6))
sns.boxplot(x='num_room', y='price_doc', data=df)
plt.title('Number of Rooms vs House Price')
plt.xlabel('Number of Rooms')
plt.ylabel('Price')
plt.show()


# Scatter plot to analyze the relationship between living space ('life_sq') and house prices ('price_doc')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='life_sq', y='price_doc', data=df)
plt.title('Living Space (life_sq) vs House Price')
plt.xlabel('Living Space (life_sq)')
plt.ylabel('Price')
plt.show()


# Plot correlation heatmap
corr = df[['price_doc', 'num_room', 'life_sq', 'full_sq']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap')
plt.show()



# Calculate IQR for 'price_doc' to detect outliers
Q1 = df['price_doc'].quantile(0.25)
Q3 = df['price_doc'].quantile(0.75)
IQR = Q3 - Q1

# Define the range for non-outlier values
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print dataset before removing outliers
print("Dataset before removing outliers:")
print(df[['price_doc', 'num_room', 'life_sq', 'full_sq']].describe())

# Filter the data to remove outliers
df_no_outliers = df[(df['price_doc'] >= lower_bound) & (df['price_doc'] <= upper_bound)]

# Print dataset after removing outliers
print("\nDataset after removing outliers:")
print(df_no_outliers[['price_doc', 'num_room', 'life_sq', 'full_sq']].describe())

# You can also visualize the price distribution before and after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(df['price_doc'], kde=True, color='blue', label='Before removing outliers')
sns.histplot(df_no_outliers['price_doc'], kde=True, color='red', label='After removing outliers')
plt.legend()
plt.title('Price Distribution Before and After Removing Outliers')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()



# Explore correlation between numeric features
sns.pairplot(df[['price_doc', 'num_room', 'life_sq', 'full_sq']])
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

# Explore trends: price vs. sub_area (categorical)
plt.figure(figsize=(12, 6))
sns.boxplot(x='sub_area', y='price_doc', data=df)
plt.xticks(rotation=90)
plt.title('Price vs Sub Area')
plt.xlabel('Sub Area')
plt.ylabel('Price')
plt.show()



# One-Hot Encoding for the 'sub_area' categorical feature
df_encoded = pd.get_dummies(df, columns=['sub_area'], drop_first=True)
print(df_encoded.head())



# Mean Encoding for 'sub_area'
mean_encoded = df.groupby('sub_area')['price_doc'].mean().to_dict()
df['sub_area_encoded'] = df['sub_area'].map(mean_encoded)
print(df[['sub_area', 'sub_area_encoded']].head())



# Check for Null values in the dataset
print(df.isnull().sum())



# Fill Null values with the mean of 'life_sq' column
df['life_sq'] = df['life_sq'].fillna(df['life_sq'].mean())

# Check if the Null values have been filled
print(df['life_sq'].isnull().sum())  # This should print 0 if there are no Nulls

# Fill Null values in 'life_sq' using the group mean based on 'sub_area'
df['life_sq'] = df.groupby('sub_area')['life_sq'].transform(lambda x: x.fillna(x.mean()))

# Check if the Null values have been filled after the group-by
print(df['life_sq'].isnull().sum())  # This should print 0 if there are no Nulls



# Display the first few rows of the dataset after filling Null values
print(df[['sub_area', 'life_sq', 'price_doc']].head())


