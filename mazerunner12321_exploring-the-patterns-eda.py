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

train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv",index_col='id') 

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# setup consistent style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


print(train_df.shape)
train_df.info()
display(train_df.describe())
train_df.head()


numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns

# Plot histograms and boxplots for each numerical feature
for feature in numerical_features:
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  sns.histplot(train_df[feature], kde=True, bins=30)
  plt.title(f'Histogram of {feature}')
  plt.xlabel(feature)
  plt.ylabel('Frequency')

  plt.subplot(1, 2, 2)
  sns.boxplot(x=train_df[feature])
  plt.title(f'Boxplot of {feature}')
  plt.tight_layout()
  plt.show()

  # statistics
  print(f'Skewness: {train_df[feature].skew():.2f}')
  print(f'Null values: {train_df[feature].isnull().sum()}')



categorical_cols = train_df.select_dtypes(exclude = ['int64','float64']).columns

for col in categorical_cols:
  plt.figure(figsize=(12, 5))
  if col in ['Podcast_Name', 'Episode_Title']:
    top_categories = train_df[col].value_counts().nlargest(10)
    sns.barplot(x = top_categories.index, y=top_categories.values)
  else:
    sns.countplot(x=train_df[col], order=train_df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.show()


  print(f'Unique values: {train_df[col].nunique()}')
  print(f'Null values: {train_df[col].isnull().sum()}')


for feature in numerical_features[:-1]:
  plt.figure(figsize=(8,6))
  sns.scatterplot(
      x = train_df[feature], y=train_df['Listening_Time_minutes'], alpha=0.5
  )
  plt.title(f'Scatter plot of {feature} vs Listening_Time_minutes')
  plt.xlabel(feature)
  plt.ylabel('Listening_Time_minutes')
  plt.show()

corr_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


for col in categorical_cols:
  if col not in ['Podcast_Name', 'Episode_Title']:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=train_df[col], y=train_df['Listening_Time_minutes'])
    plt.title(f'Box plot of {col} vs Listening_Time_minutes')
    plt.xlabel(col)
    plt.ylabel('Listening_Time_minutes')
    plt.xticks(rotation=45)
    plt.show()


print(train_df.isna().sum())
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[numerical_features] = num_imputer.fit_transform(train_df[numerical_features])
train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
print(train_df.isna().sum())




