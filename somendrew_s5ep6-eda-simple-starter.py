import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings('ignore')


train =  pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.shape, test.shape


train.head()


train['Fertilizer Name'].value_counts()


train.info()


train.describe()


train.isnull().sum()


# Get categorical columns
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical Columns: " ,cat_cols)
# Get numerical columns
num_cols = train.select_dtypes(include=['number']).columns.tolist()
print("Numerical Columns: ",num_cols)


for col in cat_cols:
    plt.figure(figsize = (12,4))
    sns.countplot(data = train, x = col)
    plt.title(f"Distribution for {col}")


fig, axes  = plt.subplots(len(num_cols),2, figsize=(12, 5 * len(num_cols)))
    
for i, col in enumerate(num_cols):
    #histogram
    sns.histplot(train[col], bins = 30,  kde = True, ax = axes[i,0])
    axes[i, 0].set_title(f'Distribution of {col}')

    #boxplot
    sns.boxplot(x = train[col],ax = axes[i,1])
    axes[i,1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.plot()


train[train.duplicated()]


numeric_df = train.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()


sns.pairplot(train[train.columns], diag_kind='kde')
plt.show()


for cat in cat_cols:
    for num in num_cols:
        plt.figure(figsize=(8, 5))
        sns.stripplot(x=cat, y=num, data=train, jitter=True)
        plt.title(f'Strip Plot of {num} by {cat}')
        plt.xlabel(cat)
        plt.ylabel(num)
        plt.show()




