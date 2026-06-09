import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_2=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.head()


train_2.head()


print(train_2.shape)
print(train.shape)


plt.hist(x=train_2['Price'])
plt.title('Distribution of price')


train_2.isnull().sum()


plt.hist(x=train['Price'])
plt.title('Distribution of price')


train.info()


train.isnull().sum()


features_with_na=[feature for feature in train.columns if train[feature].isnull().sum()>1]
features_with_na


rows=int(np.ceil(len(features_with_na)/3))
col=3
fig, axes=plt.subplots(rows,col,figsize=(15, rows*5))
axes=axes.flatten()
for idx, feature in enumerate(features_with_na):
    data=train.copy()
    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['Price'].median().plot.bar(ax=axes[idx],color=['lightblue','orange'])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('') 
#hides any unused subplots
for idx in range(len(features_with_na), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()


numerical_features=[feature for feature in train.columns if train[feature].dtype!='O']
print('Numerical Features of data are :',numerical_features)


sns.boxplot(x='Weight Capacity (kg)', y='Brand',data=train_2)
plt.title('Average weights of bagpack based on brands')


train['Weight Capacity (kg)']=train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())
train_2['Weight Capacity (kg)']=train_2['Weight Capacity (kg)'].fillna(train_2['Weight Capacity (kg)'].median())


categorical_features=[feature for feature in train.columns if train[feature].dtype=='O']
categorical_features


train_len=len(train)
train_merge=pd.concat([train,train_2],axis=0,ignore_index=False)


train_merge.isnull().sum()


rows = int(np.ceil(len(categorical_features) / 3))  # Ensure it aligns with categorical_features
col = 3

# Creating figure and setting size
fig, axes = plt.subplots(rows, col, figsize=(15, rows * 5))
axes = axes.flatten()

# Looping through features and creating subplots
for idx, feature in enumerate(categorical_features):
    data = train_merge.copy()
    
    # Plot on current subplot
    sns.countplot(x=feature, data=data, ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel('')

# Hide any unused subplots
for idx in range(len(categorical_features), len(axes)):  # Aligning with categorical_features
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

