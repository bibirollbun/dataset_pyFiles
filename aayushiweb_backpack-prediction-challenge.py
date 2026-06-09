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


train= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train.head()


extra_train= pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
extra_train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train.head()


train.describe()


train.shape, extra_train.shape


df_train= pd.concat([train,extra_train],axis=0).reset_index(drop=True)
df_train.shape


df_train = df_train[:1994318]


test.head()


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.histplot(df_train["Price"],bins=10,kde=True,color="green")
plt.title("Price Distribution")
plt.xlabel("Price ($)")


plt.subplot(1, 3, 2)
sns.histplot(df_train["Compartments"], bins=10, kde=True, color='yellow')
plt.title("Compartments Distribution")
plt.xlabel("Number of Compartments")

plt.subplot(1, 3, 3)
sns.histplot(df_train["Weight Capacity (kg)"], bins=10, kde=True, color='red')
plt.title("Weight Capacity Distribution")
plt.xlabel("Weight Capacity (kg)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=df_train["Price"], color='blue')
plt.title("Boxplot of Price")

plt.subplot(1, 3, 2)
sns.boxplot(x=df_train["Compartments"], color='green')
plt.title("Boxplot of Compartments")

plt.subplot(1, 3, 3)
sns.boxplot(x=df_train["Weight Capacity (kg)"], color='red')
plt.title("Boxplot of Weight Capacity")

plt.tight_layout()
plt.show()


categorical_features= ["Brand","Material","Size","Laptop Compartment","Waterproof","Style","Color"]
plt.figure(figsize=(15,18))
for i , col in enumerate(categorical_features,1):
    plt.subplot(4,2,i)
    sns.boxplot(x=df_train[col],y=df_train["Price"],palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Price ($)")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()
    


plt.figure(figsize=(12, 6))
sns.countplot(x='Brand', data=df_train, palette='viridis')
plt.title('Brand Distribution')
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(x='Material', data=df_train, palette='Set2')
plt.title('Material Distribution')
plt.xticks(rotation=45)
plt.show()


missing_data = df_train.isnull().sum()
missing_data = missing_data[missing_data > 0] 

if not missing_data.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values)
    plt.title('Missing Value Distribution in df_train')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=90)
    plt.show()
else:
    print("No missing values in the dataset.")





df_train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


test.isnull().sum()


df_train.isnull().sum()


df_train.shape,test.shape


df_train = df_train[:1094318]


def feature_engineering(df):
    size_mapping = {"Small":1, "Medium":2,"Large":3}
    df["Size_Num"] = df["Size"].map(size_mapping)
    df['Compartments_per_Size'] = df['Compartments'] / df['Size_Num']    
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments'] 
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof_Laptop'] = df['Waterproof'] * df['Laptop Compartment']
    df['Is_Durable_Material'] = df['Material'].apply(lambda x: 1 if x in ['Leather', 'Nylon'] else 0)
    df['Is_Lightweight_Material'] = df['Material'].apply(lambda x: 1 if x in ['Canvas', 'Nylon'] else 0)
    df['Luxury_Material'] = df['Material'].apply(lambda x: 1 if x == 'Leather' else 0)
    df['Professional_Style'] = df['Style'].apply(lambda x: 1 if x in ['Messenger', 'Tote'] else 0)
    df['Casual_Style'] = df['Style'].apply(lambda x: 1 if x in ['Backpack', 'Duffle'] else 0)
    df['Is_Premium_Brand'] = df['Brand'].apply(lambda x: 1 if x in ['Nike', 'Under Armour', 'Adidas'] else 0)
    df['Is_Budget_Brand'] = df['Brand'].apply(lambda x: 1 if x == 'Jansport' else 0)
    df['Is_Small'] = df['Size'].apply(lambda x: 1 if x == 'Small' else 0)
    df['Is_Medium'] = df['Size'].apply(lambda x: 1 if x == 'Medium' else 0)
    df['Is_Large'] = df['Size'].apply(lambda x: 1 if x == 'Large' else 0)

    return df

df_train = feature_engineering(df_train)
test = feature_engineering(test)


df_train.dtypes


df_train.columns,test.columns


df_train.isnull().sum()


cat = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color']

df_train[cat] = df_train[cat].fillna('None').astype('string').astype('category')
median_weight = df_train['Weight Capacity (kg)'].median()
df_train['Weight Capacity (kg) categorical'] = df_train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
df_train['Weight Capacity (kg)'] = df_train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')

test[cat] = test[cat].fillna('None').astype('string').astype('category')
test['Weight Capacity (kg) categorical'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight)


df_train.dtypes


y = df_train['Price'] 
df_train = df_train.drop(['Price'],axis=1)
X = df_train
X_test = test


df_train.isnull().sum()


print("Variance:", y.var())
print("Standard Deviation:",y.std())


from scipy.stats import skew
print("Skewness:", skew(y))


scaled_train_data = X
scaled_test_data = X_test


X.columns




