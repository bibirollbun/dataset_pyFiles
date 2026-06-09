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


# Loading the data set and also viewing it
product_df = pd.read_csv('/kaggle/input/learnplatform-covid19-impact-on-digital-learning/products_info.csv')
district_df = pd.read_csv('/kaggle/input/learnplatform-covid19-impact-on-digital-learning/districts_info.csv')

print("This is the products data sample:- ")
print(product_df.head())

print("This is the districts data sample:- ")
print(district_df.head())


product_df.head()


# Data type checking and handling missing values for the products data set
print("Data Types of the Products Data:- ")
print(product_df.dtypes)

print("\n Missing Values in Product Data:- ")
print(product_df.isnull().sum())
# Data type checking and handling missing values for the district data set
print("\nData Types of the District Data:- ")
print(district_df.dtypes)

print("\n Missing values in the District data:- ")
print(district_df.isnull().sum())





# Let's find out the most common state in the districts data frame so that we can replace the null values in the data frame with the common state
most_common_state = district_df['state'].mode()[0]

# Fill missing values in the 'state' column with the most common state
district_df['state'].fillna(most_common_state, inplace=True)



#Lets check that the districts data set containing the state does or does not containing null values in it 
print("Missing values in districts data after filling the state column")
print(district_df['state'].isnull().sum())
print("Lets check the whole districts dataframe now")
print(district_df.isnull().sum())


# Now lets switch to the product data frame
'''Lets fill up the missing 'Provider/Company Name' values with 'Unknown' 
Then after that fill missing sector and primary essential function values with the most common values
and then check for the missing values in the products dataframe after filling

'''
product_df['Provider/Company Name'].fillna('Unknown', inplace = True)

product_df['Sector(s)'].fillna(product_df['Sector(s)'].mode()[0], inplace=True)
product_df['Primary Essential Function'].fillna(product_df['Primary Essential Function'].mode()[0], inplace=True)
missing_values_product = product_df.isnull().sum()
missing_values_relevant_columns = missing_values_product[["Provider/Company Name", "Sector(s)", "Primary Essential Function"]]

print("Missing Values in Products Data After Handling:")
print(missing_values_relevant_columns)


#Lets perform the Digital Learning analysis
sector_counts = product_df['Sector(s)'].value_counts()
function_counts = product_df['Primary Essential Function'].value_counts()

#Display the counts for sectors and primary essential functions
print("Counts of Product by Sector:")
print(sector_counts)

print("\nCounts of Product by primary essential function: ")
print(function_counts)


import matplotlib.pyplot as plt

# Visualize the distribution of products by sector
plt.figure(figsize=(10, 6))
sector_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Products by Sector')
plt.xlabel('Sector')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualize the distribution of products by primary essential function
plt.figure(figsize=(10, 6))
function_counts.plot(kind='bar', color='lightcoral')
plt.title('Distribution of Products by Primary Essential Function')
plt.xlabel('Primary Essential Function')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Combine the two dataframes into one for analysis
combined_df = pd.concat([sector_counts, function_counts], axis=1)
combined_df.columns = ['Products by Sector', 'Products by Primary Essential Function']

# Display the combined dataframe
print(combined_df)



# Combine the two dataframes into one for analysis
combined_df = pd.concat([sector_counts, function_counts], axis=1)
combined_df.columns = ['Products by Sector', 'Products by Primary Essential Function']

# Group the data by sector and primary essential function and count the occurrences
grouped_df = product_df.groupby(['Sector(s)', 'Primary Essential Function']).size().unstack(fill_value=0)

# Display the breakdown of primary essential functions by sector
print("Breakdown of Primary Essential Functions by Sector:")
print(grouped_df)

























