import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
colors = ['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A']
sns.set(style="whitegrid")


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
ss = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


display(train.head())


display(train.info())

print()
print("############################")
print()

display(train.isnull().sum())


missing = train.isnull().sum()
missing = missing[missing > 0] 

if not missing.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values)
    plt.title('Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=90)
    plt.show()


plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
sns.histplot(train["Price"], bins=10, kde=True, color=colors[0])
plt.title("Price Distribution")
plt.xlabel("Price ($)")

plt.subplot(1, 3, 2)
sns.histplot(train["Weight Capacity (kg)"], bins=10, kde=True, color=colors[2])
plt.title("Weight Capacity Distribution")
plt.xlabel("Weight Capacity (kg)")

plt.tight_layout()
plt.show()


cols = ["Price", "Weight Capacity (kg)"]
plt.figure(figsize=(20, 5))
for i , col in enumerate(cols,1):
    plt.subplot(2,2,i)
    sns.boxplot(x=col, y = "Brand", data=train, palette=colors)
    plt.title(col)
    plt.tight_layout()


cols =  ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof", "Style", "Color"]
plt.figure(figsize=(15,8))
for i ,col in enumerate(cols ,1):
    plt.subplot(3,3,i)
    sns.countplot(y=col, data=train, palette=colors)
    plt.title(col)
    
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 15))

for i, col in enumerate(cols, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=train[col], y=train["Price"], palette=colors)
    plt.ylabel("Price")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


def feature_engineering(df):
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Num'] = df['Size'].map(size_mapping)
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

train = feature_engineering(train)
test = feature_engineering(test)


train[cols] = train[cols].fillna('None').astype('string').astype('category')
median_weight = train['Weight Capacity (kg)'].median()
train['Weight Capacity (kg) categorical'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('string')
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_weight).astype('float64')

test[cols] = test[cols].fillna('None').astype('string').astype('category')
test['Weight Capacity (kg) categorical'] = test['Weight Capacity (kg)'].fillna(median_weight).astype('string')
test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(median_weight)


train.info()

