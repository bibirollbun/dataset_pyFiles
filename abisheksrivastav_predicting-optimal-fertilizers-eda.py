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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style for better visualization
plt.style.use('seaborn')
sns.set_palette('husl')

# Load the training dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


# 1. Basic Dataset Overview
print("=== Dataset Info ===")
print(train_df.info())
print("\nDataset Shape:", train_df.shape)
print("\nFirst 10 Rows:")
print(train_df.head(10))


# 2. Summary Statistics for Numerical Features
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
print("\n=== Summary Statistics for Numerical Features ===")
print(train_df[numerical_cols].describe())



# 3. Check for Missing Values
print("\n=== Missing Values ===")
print(train_df.isnull().sum())



# 4. Distribution of Numerical Features (Histograms)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


# 5. Distribution of Categorical Features (Count Plots)
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 6. Correlation Matrix for Numerical Features
plt.figure(figsize=(8, 6))
corr_matrix = train_df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# 7. Box Plots: Numerical Features vs. Fertilizer Name
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Fertilizer Name', y=col, data=train_df)
    plt.title(f'{col} vs Fertilizer Name')
    plt.xlabel('Fertilizer Name')
    plt.ylabel(col)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 8. Cross-Tabulation: Categorical Features vs. Fertilizer Name
print("\n=== Soil Type vs Fertilizer Name ===")
soil_fert_crosstab = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'])
print(soil_fert_crosstab)

print("\n=== Crop Type vs Fertilizer Name ===")
crop_fert_crosstab = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
print(crop_fert_crosstab)


# 9. Feature Interactions: Soil Type and Moisture vs. Fertilizer Name
plt.figure(figsize=(12, 6))
sns.boxplot(x='Soil Type', y='Moisture', hue='Fertilizer Name', data=train_df)
plt.title('Moisture vs Soil Type by Fertilizer Name')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 10. Outlier Detection using IQR
print("\n=== Outlier Detection ===")
for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    print(f"{col} - Outliers: {len(outliers)} rows (Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f})")


