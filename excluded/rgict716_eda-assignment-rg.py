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


# Unzipping competition files
!unzip -o ../input/sberbank-russian-housing-market/train.csv.zip
!unzip -o ../input/sberbank-russian-housing-market/macro.csv.zip 
!unzip -o ../input/sberbank-russian-housing-market/test.csv.zip
!ls


# Importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Loading and displaying training dataframe
import os
df_train = pd.read_csv('train.csv')
df_train


# Plot 1 - Scatter plot of price_doc vs full_sq
import pandas as pd
df = pd.read_csv('train.csv')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='full_sq', y='price_doc', data=df)
plt.title("Price vs. Full Square")
plt.xlabel("Full Square")
plt.ylabel("Price")
plt.show()


# Plotting a regression line in the graph
plt.figure(figsize=(10, 6))
sns.regplot(x='full_sq', y='price_doc', data=df_train, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Price vs Full Square Footage with Regression Line')
plt.xlabel('Full Square Footage')
plt.ylabel('Price')
plt.show()


# One outlier is the house with full square above 5000 but at a very low price, highlighting it in the full square scatter plot
sns.scatterplot(data=df, x='id', y='full_sq')
plt.axhline(5000, color='red', linestyle='--', label='Threshold: 5000')
plt.legend()
plt.title('Full Square Outliers')
plt.show()


# Removing the outlier value of full square
df_clean = df[df['full_sq'] <= 5000]

# Confirm that the outlier entry was removed
print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_clean.shape}")


# Plot 2 - Histogram of big church count
plt.figure(figsize=(10, 6))
sns.histplot(df['big_church_count_5000'], kde=True, color='pink')
plt.title('Distribution of Big churches')
plt.xlabel('Number of big churches')
plt.ylabel('Frequency')
plt.show()





# Finding outliers
# Calculate IQR for 'big_church_count_5000'
Q1 = df['big_church_count_5000'].quantile(0.25)  # First quartile
Q3 = df['big_church_count_5000'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1

# Determine outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['big_church_count_5000'] < lower_bound) | (df['big_church_count_5000'] > upper_bound)]

# Plot histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['big_church_count_5000'], kde=True, color='pink', label='Data', alpha=0.7)

# Overlay outliers
plt.scatter(
    outliers['big_church_count_5000'], 
    [0] * len(outliers),  # Place points at the bottom of the plot
    color='red', 
    label='Outliers', 
    zorder=10
)

# Add boundaries
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
plt.axvline(upper_bound, color='blue', linestyle='--', label='Upper Bound')

# Add labels and legend
plt.title('Distribution of Big Churches with Highlighted Outliers')
plt.xlabel('Number of Big Churches')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Removal of outliers
df_no_outliers = df[(df['big_church_count_5000'] >= lower_bound) & (df['big_church_count_5000'] <= upper_bound)]

# Plot histogram without outliers
plt.figure(figsize=(10, 6))
sns.histplot(df_no_outliers['big_church_count_5000'], kde=True, color='pink', label='Data Without Outliers', alpha=0.7)
plt.title('Distribution of Big Churches (Outliers Removed)')
plt.xlabel('Number of Big Churches')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Plot 3 - Bar chart of floor number
sns.countplot(x=df['floor'])
plt.title("Floor Numbers")
plt.xlabel("Floor Number")
plt.ylabel("Count")
plt.show()


# Trying to identify outliers in floor frequencies
# Calculate frequency of each floor
floor_counts = df['floor'].value_counts()

# Calculate IQR for floor frequencies
Q1 = floor_counts.quantile(0.25)  # First quartile
Q3 = floor_counts.quantile(0.75)  # Third quartile
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outlier floor numbers
outlier_floors = floor_counts[(floor_counts < lower_bound) | (floor_counts > upper_bound)]

# Print outlier floor numbers
print("Outlier Floors (based on frequency counts):")
print(outlier_floors.index.tolist())


# Plot 4 - Boxplot of number of rooms vs. full square
plt.figure(figsize=(10, 6))
sns.boxplot(x='num_room', y='full_sq', data=df)
plt.title('Number of Rooms vs Full Square')
plt.xlabel('Number of Rooms')
plt.ylabel('Full Square')
plt.show()


outliers = []
# Using interquartile range to define upper and lower bounds
for room, group in df.groupby('num_room'):
    Q1 = group['full_sq'].quantile(0.25)
    Q3 = group['full_sq'].quantile(0.75)
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    room_outliers = group[(group['full_sq'] < lower_bound) | (group['full_sq'] > upper_bound)]
    outliers.append(room_outliers)

# Combine all outliers into a single Dataframe
outliers_df = pd.concat(outliers)

# Print outliers
print("Outliers detected:")
print(outliers_df)

# Visualize boxplot and highlight outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='num_room', y='full_sq', data=df, showfliers=True, color='skyblue')

# Highlight outliers in red
plt.scatter(outliers_df['num_room'], outliers_df['full_sq'], color='red', label='Outliers', alpha=0.8)
plt.title('Number of Rooms vs Full Square (Outliers Highlighted)')
plt.xlabel('Number of Rooms')
plt.ylabel('Full Square')
plt.legend()
plt.show()


# Removal of outliers
df_cleaned = df[~df.index.isin(outliers_df.index)]
print("Data after removing outliers:")
print(df_cleaned.head())


# Plot 5 - Top 10 most frequent sub areas
plt.figure(figsize=(12, 8))
df_train['sub_area'].value_counts().head(10).plot(kind='bar', color='purple')
plt.title('Top 10 Most Frequent Sub Areas')
plt.xlabel('Sub Area')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Correlation matrix between price, full square, room number, maximum floor and material
corr_matrix = df[['price_doc', 'full_sq', 'num_room', 'max_floor', 'material']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Trend between room number and full square
sns.lineplot(data=df, x='num_room', y='full_sq', ci=None)
plt.title("Number of rooms by square footage")
plt.xlabel("Room number")
plt.ylabel("Full square")
plt.show()


# Transforming a categorical variable into a continuous one using one hot encoding
# One-Hot Encoding for 'product_type' column
df_train_encoded = pd.get_dummies(df_train, columns=['product_type'], drop_first=True)

# Show the transformed dataset
print(df_train_encoded.head())


# Transforming a categorical variable into a continuous one by mean encoding - full square
# Ensure the column is clean, handle missing values
df['full_sq'] = df['full_sq'].fillna(0)  # Replace missing values with 0

# Perform mean encoding on 'floor'
mean_encoded = df.groupby('full_sq')['price_doc'].mean()
df['full_sq_mean_encoded'] = df['full_sq'].map(mean_encoded)

# Display the result
df[['full_sq', 'full_sq_mean_encoded']].head()


# Filling nulls of 'kitch_sq' with a default value of 0 because not all houses have a designated separate kitchen
df['kitch_sq_filled_default'] = df['kitch_sq'].fillna(0)

# View new values
print(df[['kitch_sq', 'kitch_sq_filled_default']].head())


# Filling nulls with group-by aggregate - group-by median of room number based on full square
# Since these two seem to have a good correlation, 
# and there are no NA values in full square whereas there are NA values in num_room
# Fill floor Nulls with group-by median based on full_sq
df['num_room_filled_group'] = df['full_sq']
df['num_room_filled_group'] = df.groupby('full_sq')['num_room_filled_group'].transform(
    lambda x: x.fillna(x.median())
)


# Checking if there are any NA values left in 'num_room'
print(df[['num_room_filled_group']].isnull().sum())

