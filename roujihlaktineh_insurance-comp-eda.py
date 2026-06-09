import numpy as np
import pandas as pd
import warnings 
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/regression-with-an-insurance-dataset/train.csv")
df


df.isnull().sum()


df.info()


count_age_na = df ["Age"].isnull().sum()
count_age_na


(count_age_na/1200000 )*100


df.shape[0]


for i in df.columns :
     col_nun = df[i].isnull().sum()  
     na_per = ( col_nun / df.shape[0] ) * 100
    
     print(f" missing in {i} is :{na_per.round(2)} %")
    


df.duplicated().sum()


df['Policy Start Date'] = df['Policy Start Date'].astype("datetime64[ns]")


cat = df.select_dtypes("object").columns
cat



for i in cat : 
    df[i] = df[i].astype("category")

df.info()


df = df .drop("id"   , axis=1)
df


num = df.select_dtypes(["float64"  , "int64"]).columns
num


for col in cat :
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)


for col in num:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(),inplace=True)


df.isnull().sum().sum()


df[num].hist(bins=30,  figsize=(15 ,10), layout=(3, 3))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Premium Amount'],kde=True, bins=50)
plt.title('Distribution of Premium Amount')
plt.show()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Premium Amount'])
plt.title("Box Plot of Premium Amount")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="Smoking Status", y="Premium Amount" , data=df)
plt.title("Premium Amount by Smoking")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x="Policy Type", y="Premium Amount" , data=df)
plt.title("Premium Amount by Policy ")
plt.show()


corr = df[num].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr , annot=True, cmap="coolwarm" , fmt="2f")
plt.title("Correlation Matrix")
plt.show()


df["Year"] = df["Policy Start Date"].dt.year
df["Month"] =  df["Policy Start Date"].dt.month
df["Day"] = df["Policy Start Date"].dt.day



fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()
for i, col in enumerate(cat):
    sns.countplot(y=col, data=df, ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2 , 2, figsize=(14, 10))
sns.countplot(x="Year", data=df, ax=axes[0, 0])
sns.countplot(x="Month", data=df, ax=axes[0, 1])
sns.countplot(x="Day", data=df, ax=axes[1, 0])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3,  3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(num[:9]):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.show


for col in num :
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col]< lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

