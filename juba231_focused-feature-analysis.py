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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


train.head(10)


print(train.info())


#basic_statistics 
train.describe()


print("ðŸŽ¯ Null values:")
print(train.isnull().sum())


# checking unique values
cat_col = train.select_dtypes(include = object).columns

print("ðŸª¶ Unique values amd there respetive count in Categorical columns:\n")
for col in cat_col:
    print(f"{col}: {train[col].nunique()} unique values ")
    print(train[col].value_counts(), "\n")
print("\n"+ "="*50 + "\n")


num_col = train.select_dtypes(include= 'int64').columns


import matplotlib.pyplot as plt
import seaborn as sns
for col in num_col:
    plt.figure(figsize= (8,6))
    sns.histplot(train[col], kde= True)
    plt.title(f"ðŸ§¿ Distribution of {col}")
    plt.show()




#Neumeric vs Target

for col in ['Temparature', 'Humidity', 'Moisture','Nitrogen','Potassium','Phosphorous']:
    sns.boxplot(x= 'Fertilizer Name',y= col, data= train)
    plt.title(f"{col} vs Fertlizer")
    plt.xticks(rotation= 45)
    plt.show()


train= train.drop(columns=['id'],errors= 'ignore')
sns.heatmap(train.corr(numeric_only= True),annot=True, cmap= 'coolwarm')
plt.title("Correlation Matrix")
plt.show()


#pairwise
#sns.pairplot(train,hue='Fertilizer Name',vars=['Nitrogen','Potassium','Phosphorous'])
#plt.suptitle("N-K-P pairwise interaction by Fertilizer",y=1.02)
#plt.show()

