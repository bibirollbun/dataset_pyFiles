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


!unzip -o /kaggle/input/sberbank-russian-housing-market/test.csv.zip
!unzip -o /kaggle/input/sberbank-russian-housing-market/train.csv.zip
!unzip -o /kaggle/input/sberbank-russian-housing-market/macro.csv.zip


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset (adjust path as per your setup)
data_path = "/kaggle/working/train.csv"
df = pd.read_csv(data_path)

# Display the first few rows of the dataset
df.head()



df.select_dtypes('object')


# Scatter plot of 'full_sq' vs. 'price_doc'
sns.scatterplot(x=df['full_sq'], y=df['price_doc'])
plt.title("Full Square vs. Price")
plt.xlabel("Full Square (m²)")
plt.ylabel("Price")
plt.show()


!pip install plotly.express -q


import plotly.express as px
fig = px.scatter(df, x='full_sq',y='price_doc',trendline="ols")
fig.show()


df_outliers = df[df['full_sq'] > 5000]
print(df_outliers)



# Highlight values > 5000 in a scatter plot
sns.scatterplot(data=df, x='id', y='full_sq')
plt.axhline(5000, color='red', linestyle='--', label='Threshold: 5000')
plt.legend()
plt.title('Full Square Outliers')
plt.show()


#Plot distribution of 'price_doc'
sns.histplot(df['price_doc'], kde=True, bins=50)
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.show()


#Bar chart of 'material' distribution
sns.countplot(x=df['material'])
plt.title("Material Distribution")
plt.xlabel("Material")
plt.ylabel("Count")
plt.show()



# Remove rows where 'full_sq' > 5000
df_cleaned = df[df['full_sq'] <= 5000]

# Display the shape of the new DataFrame to confirm the outliers are removed
print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")


# Verify no outliers remain
print(df_cleaned[df_cleaned['full_sq'] > 5000])  # Should return an empty DataFrame



# checking outliers in floor
fig = px.scatter(df, x='floor',y='price_doc')
fig.show()



df['floor'].max()


df=df[~(df['floor']==77.0)]


fig = px.scatter(df, x='floor',y='price_doc')
fig.show()


#Plot distribution of 'price_doc'
sns.histplot(df['build_year'], kde=True, bins=50)
plt.title("Distribution of build_year")
plt.xlabel("Build year")
plt.show()


df['build_year'].max()


df[df.build_year > 50000]


df=df[~(df['build_year']==20052009.0)] #outlier removed


df[df.build_year > 50000] #checking


# Correlation matrix
corr_matrix = df[['price_doc', 'full_sq', 'life_sq', 'build_year', 'material']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()




# Trend between 'build_year' and 'price_doc'
sns.lineplot(data=df, x='build_year', y='price_doc', ci=None)
plt.title("Price by Build Year")
plt.xlabel("Build Year")
plt.ylabel("Price")
plt.show()


# Replace missing values in 'material' for clarity
df['material'] = df['material'].fillna('Unknown').astype(str)

# Create a boxplot to show the distribution of house prices for each material type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='material', y='price_doc')
plt.title('House Price Distribution by Material Type', fontsize=16)
plt.xlabel('Material', fontsize=12)
plt.ylabel('Price (RUB)', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# Barplot showing mean house price for each material
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='material', y='price_doc', ci=None, estimator='mean', palette='viridis')
plt.title('Average House Price by Material Type', fontsize=16)
plt.xlabel('Material', fontsize=12)
plt.ylabel('Average Price (RUB)', fontsize=12)
plt.xticks(rotation=45)
plt.show()



from sklearn.preprocessing import LabelEncoder

# Ensure all values in 'material' are strings
df['material'] = df['material'].astype(str)

# Handle missing values by replacing them with 'Unknown'
df['material'] = df['material'].replace('nan', 'Unknown')

# Apply LabelEncoder
le = LabelEncoder()
df['material_encoded'] = le.fit_transform(df['material'])

# Display the first few rows of the updated DataFrame
df[['material', 'material_encoded']].head()



# One-hot encode 'material'
df = pd.get_dummies(df, columns=['material'], prefix='material', drop_first=False)

# Display the first few rows of the DataFrame
df.head()



# Ensure 'build_year' is clean and handle missing values
df['build_year'] = df['build_year'].fillna(0)  # Replace missing years with 0 or any default value

# Perform mean encoding on 'build_year'
mean_encoded = df.groupby('build_year')['price_doc'].mean()
df['build_year_mean_encoded'] = df['build_year'].map(mean_encoded)

# Display the result
df[['build_year', 'build_year_mean_encoded']].head()



# Fill 'life_sq' Nulls with a default value of 0 
#as it is fine to not have living room area size if the overall house size is available.
df['life_sq_filled_default'] = df['life_sq'].fillna(0)

# Check the result
print(df[['life_sq', 'life_sq_filled_default']].head())




df['floor'].isna().sum()


# Fill floor Nulls with group-by median based on build_year
df['floor_filled_group'] = df['floor']
df['floor_filled_group'] = df.groupby('build_year')['floor_filled_group'].transform(
    lambda x: x.fillna(x.median())
)


# Check if there are any remaining Null values
print(df[[ 'floor_filled_group']].isnull().sum())



df

