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


train_data.isna().sum()


train_data.isnull().sum()


# Missing percentage
missings = train_data.isnull().mean()*100
missings


# handling Age ( mean, mode, median)
sns.boxplot(x=train_data['Age'])
plt.title("BoxPlot of Age")
plt.show()


print(train_data['Age'].mean())
print(train_data['Age'].median())


#Handlung
sns.boxplot(x=train_data['Annual Income'])
plt.title("BoxPlot of Annual Income")
plt.show()


print(train_data['Annual Income'].mean())
print(train_data['Annual Income'].median())


sns.boxplot(x=train_data['Health Score'])
plt.title('BoxPlot of Health Score')
plt.show()


print(train_data['Health Score'].mean())
print(train_data['Health Score'].median())


train_data.describe()


train_data.columns


for col in train_data.columns:
    print(train_data[col].value_counts(normalize=True))
    print("----------------------------------------------------------------")


sns.heatmap(train_data.isnull())


del train_data['id']


num = train_data.select_dtypes(include=['int64', 'float64']).columns
cat = train_data.select_dtypes(include=['object', 'category']).columns
print(num.tolist())
print('------------------------------------------------------------------------------')

print(cat.tolist())


for col in num:
    plt.figure(figsize=(15,8))
    sns.histplot(x=train_data[col] , kde=True , bins = 25, data =train_data)
    plt.title(f"Histogram of {col}")
    plt.show()


 mean=train_data['Health Score'].mean()


std=train_data['Health Score'].std()


train_data['Health Score'].isna().sum()


np.random.uniform(mean-std,mean+std, size=train_data['Health Score'].isna().sum())


#test_data.fillna(test_data.mean() , inplace=True )
# missing percentage
for col in num:
    print(f"Column Name : {col}")
    print(train_data[col].isnull().mean()*100)
    m=train_data[col].mean()
    s=train_data[col].std()
    si=train_data[col].isna().sum()
    #pd series
    train_data[col] = train_data[col].fillna(pd.Series(np.random.uniform(m-s, m+s, size=int(si)),index=train_data[train_data[col].isna()].index))
    print(f"Number of missing after Handling : {train_data[col].isnull().mean()*100}")
    print("----------------------------------------------------------------------------")
    


sns.heatmap(train_data.isnull())


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
    


#Handling Outlier 
#Boxplot (Health Score , Age ,Credit Score )


sns.boxplot(x=train_data["Health Score"])
plt.show()


sns.boxplot(x=train_data["Age"])
plt.show()


sns.boxplot(x=train_data["Credit Score"])
plt.show()







