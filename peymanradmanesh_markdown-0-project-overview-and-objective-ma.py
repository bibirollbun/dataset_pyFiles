# Import core libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# Set visualization aesthetics
plt.style.use('seaborn-darkgrid')
sns.set_palette('Set2')
sns.set_context('talk')

# Paths
INPUT_DIR = '/kaggle/input/playground-series-s5e1/'

# Load data
train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))

# Display first few rows
train_df.head()



# Check dataset shape
print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")

# Show column names
print("\nColumns in dataset:")
print(train_df.columns.tolist())



# Data types and nulls
train_df.info()
print("\nMissing values in train set:")
print(train_df.isnull().sum())

print("\nMissing values in test set:")
print(test_df.isnull().sum())



# Sample rows from datasets
print("Train Sample:")
display(train_df.head())

print("\nTest Sample:")
display(test_df.head())



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(train_df['num_sold'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Sticker Sales (`num_sold`)', fontsize=14)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 2))
sns.boxplot(x=train_df['num_sold'], color='coral')
plt.title('Boxplot of Sticker Sales (`num_sold`)', fontsize=14)
plt.xlabel('Number of Stickers Sold')
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 5))
sns.barplot(data=train_df, x='country', y='num_sold', estimator='mean', palette='Set2')
plt.title('Average Sticker Sales by Country')
plt.ylabel('Mean of num_sold')
plt.xlabel('Country')
plt.grid(True, axis='y')
plt.show()



plt.figure(figsize=(8, 5))
sns.barplot(data=train_df, x='store', y='num_sold', estimator='mean', palette='coolwarm')
plt.title('Average Sticker Sales by Store')
plt.ylabel('Mean of num_sold')
plt.xlabel('Store')
plt.grid(True, axis='y')
plt.show()



print(train_df.columns.tolist())



plt.figure(figsize=(8, 5))
sns.barplot(data=train_df, x='product', y='num_sold', estimator='mean', palette='viridis')
plt.title('Average Sticker Sales by Product')
plt.ylabel('Mean of num_sold')
plt.xlabel('Product')
plt.grid(True, axis='y')
plt.show()



print('train_df exists:', 'train_df' in globals())
print('test_df exists:', 'test_df' in globals())



train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

for df in [train_df, test_df]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday



