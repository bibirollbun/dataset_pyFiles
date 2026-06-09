import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


train.head()


train.info()


test.info()


train.describe()


train.drop(columns=["id"], inplace=True)
# test.drop(columns=["id"], inplace=True)


train["diagnosed_diabetes"] = train["diagnosed_diabetes"].astype(bool)


train["diagnosed_diabetes"].value_counts()


train.dtypes


cat_cols = [
    "gender", "ethnicity", "education_level",
    "income_level", "smoking_status", "employment_status"
]


for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


train.duplicated().sum()


plt.figure(figsize=(5,4))
sns.countplot(data=train, x="diagnosed_diabetes")
plt.title("Target Distribution")
plt.xlabel("Diabetes (False/True)")
plt.ylabel("Count")
plt.show()


train["diagnosed_diabetes"].value_counts(normalize=True)


numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns


numeric_cols


train[numeric_cols].hist(figsize=(18,18), bins=30)
plt.suptitle("Numeric Feature Distributions", y=1)
plt.show()


plt.figure(figsize=(18,18))
corr = train[numeric_cols.tolist() + ["diagnosed_diabetes"]].corr()
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()


for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.kdeplot(data=train, x=col, hue="diagnosed_diabetes", common_norm=False)
    plt.title(f"Distribution of {col} by Diabetes Status")
    plt.show()



cat_cols = train.select_dtypes(include="category").columns
cat_cols


for col in cat_cols:
    plt.figure(figsize=(7,4))
    sns.countplot(data=train, x=col)
    plt.title(f"Category Distribution: {col}")
    plt.xticks(rotation=45)
    plt.show()



for col in cat_cols:
    plt.figure(figsize=(7,4))
    sns.barplot(data=train, x=col, y="diagnosed_diabetes")
    plt.title(f"Diabetes Rate by {col}")
    plt.xticks(rotation=45)
    plt.ylabel("Mean Diabetes Rate")
    plt.show()



for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(data=train, x="diagnosed_diabetes", y=col)
    plt.title(f"{col} vs Diabetes (Boxplot)")
    plt.show()





