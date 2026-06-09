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
from matplotlib import pyplot as plt
import plotly.express as px

# remove warnings
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv')





df.head()


df.drop(columns=['Unnamed: 0','id'], inplace=True)


df.info()


for col in df.columns:
    print(f"{col} : {df[col].nunique()} {df[col].unique()}\n")


categorical_columns=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for i in categorical_columns:
    df[i]= df[i].astype('category')
    
numerical_columns = [x for x in df.columns.to_list() if x not in categorical_columns]    


df.info()


df.describe()


df.describe(include='category')


df['Customer Type']=df['Customer Type'].apply(lambda x: x.split(" ")[0])


df['Type of Travel']=df['Type of Travel'].apply(lambda x:x.split(" ")[0])


df.isna().sum()


df.dropna(inplace=True)


df.duplicated().any()





sns.set(style="whitegrid")
colors=['#001e45', '#00537a', '#D7C0AE']  

palette={'satisfied': colors[1], 'neutral or dissatisfied': colors[0]}


df.hist(bins=22, figsize=(20, 18), grid = True, color=colors[2])
plt.show()


sns.kdeplot(data=df, x='Age', hue='satisfaction', fill=True, palette={'satisfied': colors[0], 'neutral or dissatisfied': colors[1]})
plt.tight_layout()





len(categorical_columns)


vig_cols = df.drop('satisfaction' , axis = 1).select_dtypes(exclude=[np.number]).columns
fig, ax = plt.subplots(2, 2, figsize = (20, 12))
ax = ax.flatten()
for i in range(len(vig_cols)):
    sns.countplot(x=df[vig_cols[i]], hue=df['satisfaction'], ax=ax[i],  palette=palette)
    ax[i].set_title(vig_cols[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    
for i in range(len(vig_cols), len(ax)):
    fig.delaxes(ax[i])
  
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15, 10))  

sns.heatmap(df.select_dtypes(exclude='category').corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=1, linecolor='black', ax=ax)
plt.tight_layout()

plt.show()


sns.scatterplot(data=df, x='Arrival Delay in Minutes', y='Departure Delay in Minutes', color=colors[2])
plt.tight_layout()
plt.show()


sns.histplot(data=df, x='Flight Distance', kde=True, hue='satisfaction', alpha=0.4,  palette=palette);


len(numerical_columns)


vig_cols = df.drop(columns=['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'] , axis = 1).select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots((len(vig_cols)+2)//4, 4, figsize=(30,20))
ax = ax.flatten()
for i in range(len(vig_cols)):
    sns.countplot(x=df[vig_cols[i]], hue=df['satisfaction'], ax=ax[i],  palette=palette)
    ax[i].set_title(vig_cols[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('satisfaction')
    
for i in range(len(vig_cols), len(ax)):
    fig.delaxes(ax[i])
  
plt.tight_layout()
plt.show()


sns.barplot(data=df, x='Seat comfort', y='Flight Distance', hue='satisfaction', palette=palette);


sns.barplot(data=df, x='Leg room service', y='Flight Distance', hue='satisfaction', palette=palette);


sns.barplot(data=df, x='Class', y='Flight Distance', hue='satisfaction', palette=palette);

