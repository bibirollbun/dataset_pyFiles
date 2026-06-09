import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  r2_score

import warnings
warnings.filterwarnings('ignore')



df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
display(df_train.head())
print("Shape of Training Data: ", df_train.shape)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
display(df_test.head())
print("Shape of Testing Data:", df_test.shape)


df_train.info()


df_test.info()


df_train.isnull().sum()/len(df_train)*100


df_test.isnull().sum()/len(df_test)*100


df_train['Episode_Length_minutes'].fillna(df_train['Episode_Length_minutes'].mean(),inplace=True)
df_train['Guest_Popularity_percentage'].fillna(df_train['Guest_Popularity_percentage'].mean(),inplace=True)
df_train['Number_of_Ads'].fillna(df_train['Number_of_Ads'].mode()[0],inplace = True)


df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].mean(),inplace=True)
df_test['Guest_Popularity_percentage'].fillna(df_test['Guest_Popularity_percentage'].mean(),inplace=True)


df_train.isnull().sum()


df_test.isnull().sum()


display(df_train['Podcast_Name'].value_counts())

df_train.groupby('Podcast_Name').size().reset_index(name='counts').sort_values(by='counts', ascending=False).head(10)


df_train.groupby(['Genre','Podcast_Name']).size()


df_train.describe()


import matplotlib.pyplot as plt

# Set your numeric features
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

# Create boxplots
for col in numeric_cols:
    plt.figure(figsize=(5, 1.5))
    sns.boxplot(x=df_train[col], color='skyblue')
    plt.title(f'Outlier Check for {col}')
    plt.show()


outlier_cols = ['Episode_Length_minutes','Number_of_Ads']
df_train_clean = df_train.copy()
for col in outlier_cols:
    Q1 = df_train_clean[col].quantile(0.25)
    Q3 = df_train_clean[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_train_clean = df_train_clean[(df_train_clean[col] >= lower) & (df_train_clean[col] <= upper)]




print(df_train.shape)
print(df_train_clean.shape)
df_train_clean.head()


import matplotlib.pyplot as plt

# Set your numeric features
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

# Create boxplots
for col in numeric_cols:
    plt.figure(figsize=(5, 1.5))
    sns.boxplot(x=df_train_clean[col], color='skyblue')
    plt.title(f'Outlier Check for {col}')
    plt.show()


df_test.columns


import matplotlib.pyplot as plt

# Set your numeric features
numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

# Create boxplots
for col in numeric_cols:
    plt.figure(figsize=(5, 1.5))
    sns.boxplot(x=df_test[col], color='skyblue')
    plt.title(f'Outlier Check for {col}')
    plt.show()


df_train_clean.drop(['Episode_Title'], axis = 1, inplace = True)
df_test.drop(['Episode_Title'], axis = 1 , inplace = True)

df_train_clean =df_train_clean.drop(['Podcast_Name'],axis=1)
df_test = df_test.drop(['Podcast_Name'], axis=1)


df_test.columns


df_train_clean.head()


df_train_clean['Genre'].value_counts().unique()


df_train_clean = pd.get_dummies(df_train_clean,columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'])
df_test = pd.get_dummies(df_test, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'])


df_train_clean.head()


df_test.head()


df_train_clean.head()


df_test.head()


df_train_clean.head()


# Pairwise correlations of numeric features
plt.figure(figsize=(8,6))
corr = df_train_clean.select_dtypes(include=['int64','float64']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


df_train.columns


# Scatter + hist matrix for a few numeric features
sns.pairplot(df_train[['Episode_Length_minutes','Genre','Publication_Day','Publication_Time','Number_of_Ads','Episode_Sentiment','Listening_Time_minutes']])
plt.suptitle('Pairplot of Key Numeric Features', y=1.02)
plt.show()







