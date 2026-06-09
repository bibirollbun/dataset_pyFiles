%pip install -qq scikit-learn lightgbm xgboost --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.linear_model import Lasso, Ridge

from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor

import optuna

import seaborn as sns


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


df_train = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/train.csv",
    usecols=[
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "Calories",
    ],
)
df_test = pd.read_csv(
    "/kaggle/input/playground-series-s5e5/test.csv",
    usecols=[
        "id",
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
    ],
)
df_origin = pd.read_csv(
    "/kaggle/input/calories-burnt-prediction/calories.csv",
    usecols=[
        "Age",
        "Gender",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "Calories",
    ],
)


df_train.drop_duplicates(keep="first", inplace=True)
df_origin.drop_duplicates(keep="first", inplace=True)


df_train


df_train.describe()



df_test.drop("id", axis=1).describe()



df_origin.describe()



df_train_copy = df_train.copy()

df_train_copy["Sex"] = np.where((df_train_copy["Sex"] == "male"), 1, 0).astype("int8")

corr = df_train_copy.corr()

corr["Calories"].sort_values()


plt.figure(figsize=(16, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

del df_train_copy


df_test_copy = df_test.copy()

df_test_copy["Sex"] = np.where((df_test_copy["Sex"] == "male"), 1, 0).astype("int8")

corr = df_test_copy.corr()

plt.figure(figsize=(16, 12))

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)

del df_test_copy


df_train.plot.kde(subplots=True, figsize=(16, 20), layout=(4, 2))


df_origin.plot.kde(subplots=True, figsize=(16, 20), layout=(4, 2))


df_test.drop("id", axis=1).plot.kde(subplots=True, figsize=(16, 20), layout=(3, 2))


df_train["Sex"].value_counts().plot.hist()



df_origin["Gender"].value_counts().plot.hist()



df_test["Sex"].value_counts().plot.hist()



plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Age"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Age"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Height"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Height"].sort_values().unique()



df_train = df_train[(df_train["Height"] < 217) & (df_train["Height"] > 129)]

plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Height"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Height"].sort_values().unique()



df_origin = df_origin[(df_origin["Height"] < 217) & (df_origin["Height"] > 132)]
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Weight"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Weight"].sort_values().unique()



df_train = df_train[df_train["Weight"] < 124]
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Weight"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Weight"].sort_values().unique()



df_origin = df_origin[df_origin["Weight"] < 124]
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Duration"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Duration"])
plt.title("Outlier Detection via Boxplot")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Heart_Rate"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Heart_Rate"].sort_values().unique()



df_train = df_train[df_train["Heart_Rate"] < 126]
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Heart_Rate"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Heart_Rate"].sort_values().unique()



df_origin = df_origin[df_origin["Heart_Rate"] < 128]



plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Body_Temp"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Body_Temp"].sort_values().unique()



df_train = df_train[df_train["Body_Temp"] > 37.9]



plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Body_Temp"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_origin["Body_Temp"].sort_values().unique()



df_origin = df_origin[df_origin["Body_Temp"] > 38.0]



plt.figure(figsize=(10, 6))
sns.boxplot(x=df_train["Calories"])
plt.title("Outlier Detection via Boxplot")
plt.show()


df_train["Calories"].sort_values().unique()



df_train = df_train[df_train["Calories"] < 289]



plt.figure(figsize=(10, 6))
sns.boxplot(x=df_origin["Calories"])
plt.title("Outlier Detection via Boxplot")
plt.show()

