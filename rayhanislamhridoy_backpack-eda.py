import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


Train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
Train.drop(columns="id",inplace=True)
test.drop(columns="id",inplace=True)
train=Train.drop(columns='Price')
target_col=Train["Price"]

cat_cols=train.select_dtypes(include="object").columns
num_cols=train.select_dtypes(include="float").columns
feature_cols=train.columns


train.shape


train.head()


train.describe()


train.select_dtypes(include="object").describe()


train.info()


plt.figure(figsize=(15,18))

plt.subplot(3,2,1)                  
sns.countplot(x='Brand',data=train)

plt.subplot(3,2,2)
sns.countplot(x='Material',data=train)

plt.subplot(3,2,3)
sns.countplot(x='Style',data=train)

plt.subplot(3,2,4)
sns.countplot(x='Color',data=train)


plt.subplot(3,2,5)
sns.countplot(x='Waterproof',data=train)
             
plt.subplot(3,2,6)
sns.countplot(x='Laptop Compartment',data=train)


plt.show()




plt.figure(figsize=(15, 18))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(4, 2, i)
    train[col].value_counts().plot(kind='pie', autopct='%0.2f%%', legend=False)
    plt.title(col)

plt.tight_layout()
plt.show()



plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.histplot(Train["Price"],bins=10,kde=True,color="green")

plt.subplot(1,3,2)
sns.histplot(train["Weight Capacity (kg)"],bins=10,kde=True,color="red")

plt.subplot(1,3,3)
sns.histplot(train["Compartments"],bins=10,kde=True,color="green")


plt.show()


plt.figure(figsize=(15, 18))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=Train[col], y=Train["Price"], palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Price")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()



train.isnull().sum()


def handle_miss(df):
    for col in df.columns:
        if df[col].dtypes=="object":
            df[col].fillna(df[col].mode()[0],inplace=True)
        elif df[col].dtypes=="float64":
            df[col].fillna(df[col].mean(),inplace=True)
    return df



train=handle_miss(train)
train
                 


train.isnull().sum()

