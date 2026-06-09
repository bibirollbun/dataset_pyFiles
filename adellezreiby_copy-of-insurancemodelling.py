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


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as xp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


train_data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')


train_data


train_data.info()


train_data.isnull().sum()


# missing percentage
missings = train_data.isnull().mean()*100
missings


# handling Age ( mean, mode, median)
sns.boxplot(x=train_data['Age'])
plt.title("BoxPlot of Age")
plt.show()



print(train_data['Age'].mean())
print(train_data['Age'].median())


train_data.describe()


for col in train_data.columns:
    print(train_data[col].value_counts(normalize=True))
    print("-------------")


train_data.info()


del train_data['id']


num = train_data.select_dtypes(include=['int64', 'float64']).columns
cat = train_data.select_dtypes(include=['object', 'category']).columns
print(num.tolist())
print(cat.tolist())



"""
numerical_columns = ['Age' ,'Annual Income' ,'' ]
cat 
train_data.fillna(train_data.mean(), inplace=True)
#gender, Marital Status
"""


for col in num:
    plt.figure(figsize=(15,8))
    sns.histplot(x=train_data[col] , kde=True , bins = 25, data =train_data)
    plt.title(f"Histogram of {col}")
    plt.show()


#test_data.fillna(test_data.mean() , inplace=True )
# missing percentage
for col in num:
    print(f"Column Name : {col}")
    print(train_data[col].isnull().mean()*100)
    
    train_data[col] = train_data[col].fillna(train_data[col].mean())
    print(f"Number of missing after Handling : {train_data[col].isnull().mean()*100}")
    print("________")
    
    


for col in num:
    plt.figure(figsize=(15,8))
    sns.histplot(x=train_data[col] , kde=True , bins = 25 , data= train_data)
    plt.title(f"Histogram of {col}")
    plt.show()


train_data['Marital Status'].mode()[0]


for col in cat:
    print(f"Column Name : {col}")
    print(train_data[col].isnull().mean()*100)
    
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    print(f"Number of missing after Handling : {train_data[col].isnull().mean()*100}")
    print("________")
    
    


# Marital Status - Occupation - Customer Feedback 


train_data['Occupation'] = train_data['Occupation'].fillna('Unknown')
train_data['Customer Feedback'] = train_data['Customer Feedback'].fillna('Unknown')



# 'Number of Dependents' , 'Credit Score'


# Handle missing values for 'Number of Dependents' and 'Credit Score' using median

train_data['Number of Dependents'] = train_data['Number of Dependents'].fillna(train_data['Number of Dependents'].median())
train_data['Credit Score'] = train_data['Credit Score'].fillna(train_data['Credit Score'].median())



print(train_data['Number of Dependents'].isnull().sum())
print(train_data['Credit Score'].isnull().sum())


print("Missing values per column:")
print(train_data.isnull().sum())



#Outlier detection for numerical columns
#Plot boxplots for each numerical column to visually spot outliers.

import matplotlib.pyplot as plt
import seaborn as sns

num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=train_data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()





#handling outliers for each numerical columns

num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Print before clipping info
    print(f"{col}: Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

    train_data[col] = train_data[col].clip(lower=lower_bound, upper=upper_bound)


import matplotlib.pyplot as plt
import seaborn as sns

num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=train_data[col])
    plt.title(f'Boxplot of {col} after outlier handling')
    plt.show()



# ask Ola about it why i did this , is this important to do it??
#i had this idea using the internet because i want it to be sure that i did handle all the outliers.
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before clipping (if you kept a copy of original data)
    # If you didn’t, run this before clipping next time.
    
    # Count outliers after clipping
    outliers_after = train_data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)][col].count()
    
    print(f"{col} outliers after handling: {outliers_after}")





