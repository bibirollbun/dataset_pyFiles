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
import numpy as np
import seaborn as sns                       
import matplotlib.pyplot as plt             
%matplotlib inline
sns.set(color_codes=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


df_train.head()


df_train.describe()


df_train.dtypes


print(df_train.isnull().sum())


for i,col in enumerate(['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Policy Start Date','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']):
    print(col, ':', df_train[col].unique())    


for i,col in enumerate(['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration','Premium Amount']):
    
    start_range = df_train[col].min()
    end_range = df_train[col].max()
    print(f"Column: {col}")
    print(f"  Starting Range: {start_range}")
    print(f"  Ending Range: {end_range}")
    print()



num_buckets = 500
plt.hist(df_train['Premium Amount'], bins=num_buckets, edgecolor='black')

plt.xlabel('Value of Premium Amount')
plt.ylabel('Number of Entries')
plt.title('Histogram of Premium Amount')


plt.show()



columns_to_replace = ['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Policy Start Date','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']
 
# Replace NaN values with "Data Not Available"
df_train[columns_to_replace] = df_train[columns_to_replace].fillna("Data Not Available")



for i,col in enumerate(['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']):
    print(col, ':', df_train[col].unique()) 


cat_cols = ['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']


fig, axes = plt.subplots(5, 2, figsize=(10, 15)) 

for i, col in enumerate(cat_cols):
    row = i // 2  
    col_idx = i % 2  
    
    counts = df_train[col].value_counts()
  
    axes[row, col_idx].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    axes[row, col_idx].set_title(f"Distribution of {col}")

plt.tight_layout()  
plt.show()


num_cols = ['Age','Annual Income','Number of Dependents','Health Score','Previous Claims','Vehicle Age','Credit Score','Insurance Duration']

fig, axes = plt.subplots(4, 2, figsize=(10, 12)) 


colors = ['steelblue', 'forestgreen', 'indianred', 'goldenrod', 'darkorchid', 'teal', 'lightblue', 'olive'] 

for i, col in enumerate(num_cols):
    row = i // 2  
    col_idx = i % 2  
    
    axes[row, col_idx].hist(df_train[col], bins=20, color=colors[i])  
    axes[row, col_idx].set_title(f"Distribution of {col}")

plt.tight_layout()  
plt.show()


le = LabelEncoder()
# Convert categorical columns to numerical using LabelEncoder
for col in cat_cols:
    df_train[col] = le.fit_transform(df_train[col])



for i,col in enumerate(['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']):
    print(col, ':', df_train[col].unique()) 


df_train.dtypes


df_1 = df_train.drop('Policy Start Date', axis=1)
corr_mat = df_1.corr()

plt.figure(figsize=(22,7))
mask = np.zeros_like(corr_mat)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_mat, 
            mask=mask,
            annot=corr_mat.round(2), 
            cmap='coolwarm',  
            vmin=-1, vmax=1,  
            center=0,        
            linewidths=.5)


plt.figure(figsize=(10, 6)) 
plt.scatter(df_train['Credit Score'],df_train['Previous Claims'],  c=df_train['Premium Amount'], cmap='viridis') 
plt.xlabel('Credit Score')
plt.ylabel('Previous Claims')

plt.title('Scatter Plot of Credit Score vs. Previous Claims Colored by Target')
plt.colorbar() 
plt.show()


plt.figure(figsize=(10, 6)) 
plt.scatter(df_train['Annual Income'], df_train['Previous Claims'], c=df_train['Credit Score'], cmap='viridis') 
plt.xlabel('Annual Income')
plt.ylabel('Previous Claims')
plt.title('Scatter Plot of Annual Income vs. Previous Claims Colored by Credit Score')
plt.colorbar() 
plt.show()

