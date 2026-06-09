import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train= pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test= pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample= pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
train.drop(columns="id",inplace= True)
test.drop(columns="id",inplace= True)



train.iloc[:,8:17].head()


train.head()


train.describe()


train.info()



cat_cols= train.select_dtypes(include='object').columns
num_cols= train.select_dtypes(include=["int64","float64"]).columns



corr= train[num_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot= True, cmap="coolwarm")
plt.show()


train.duplicated().sum()


plt.figure(figsize=(12,10))
for i,col in enumerate(cat_cols, start=1):
    plt.subplot(3, 2, i)
    sns.countplot(x= train[col])
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,8))
for i,col in enumerate(cat_cols, start=1):
    plt.subplot(2,3,i)
    counts= train[col].value_counts()
    plt.pie(counts.values, labels= counts.index ,autopct="%1.1f%%")
    plt.title(col)
plt.show()


 
plt.figure(figsize=(24,16))
for i,col in enumerate(num_cols,start=1):
    plt.subplot(7,3,i)
    sns.distplot(x=train[col], kde= True)
    plt.title(col)
plt.tight_layout()
plt.show()


plt.figure(figsize=(24,16))
for i,col in enumerate(num_cols,start=1):
    plt.subplot(5,4,i)
    sns.boxplot(x=train[col])
    plt.title(col)
    plt.tight_layout()
plt.show()

