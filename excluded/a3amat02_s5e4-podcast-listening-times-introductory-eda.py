import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_squared_error

from IPython.core.display import display, HTML

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, rotation=rotation, label_type=location)
    axes.set_ylabel("")
    axes.set_xlabel("")
    axes.set_yticklabels(())

rfr = RandomForestRegressor()
abr = AdaBoostRegressor()
gbr = GradientBoostingRegressor()
etr = ExtraTreesRegressor()
svr = SVR()
lnr = LinearRegression()
xgbr = XGBRegressor()
lgbr = LGBMRegressor(verbose=-100)

models_r = [abr, gbr, etr,
         lnr, xgbr, lgbr]

names_r = ["Ada Boost", "Gradient Boosting", "Extra Trees",
        "Linear Regression", "XGBoost", "LightGBM"]

def training_regression():
    r2s, mses = [], []

    for i in models_r:
        i.fit(x_train, y_train)
        pred = i.predict(x_test)
        r2s += [r2_score(pred, y_test)*100]
        mses += [mean_squared_error(pred, y_test, squared=False)]

    dd = pd.DataFrame({"r2": r2s, "rmse": mses}, index=names_r)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    index = 0
    dd = dd.sort_values("r2", ascending=False)
    dd["r2"] = round(dd["r2"], 2)
    dd["r2"].plot(kind="bar", ax=axes[index])
    bar_labels(axes[index])
    axes[index].set_title("r2 scores")

    index += 1

    dd = dd.sort_values("rmse", ascending=True)
    dd["rmse"] = round(dd["rmse"], 4)
    dd["rmse"].plot(kind="bar", ax=axes[index])
    bar_labels(axes[index])
    axes[index].set_title("rmse scores")

    plt.tight_layout()
    plt.show()

df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

df.drop("id", axis=1, inplace=True)

df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].median(), inplace=True)
df["Number_of_Ads"].fillna(0, inplace=True)
df["Guest_Popularity_percentage"].fillna(0.0, inplace=True)
df["Listening_Time_minutes"].fillna(df["Listening_Time_minutes"].median(), inplace=True)
df["Host_Popularity_percentage"] = df["Host_Popularity_percentage"].apply(lambda x: min(x, 100.0))

df["Episode_Title"] = df["Episode_Title"].apply(lambda x: x.split()[1])
df["Episode_Title"] = df["Episode_Title"].astype(int)

df["Number_of_Ads"] = df["Number_of_Ads"].apply(lambda x: min(x, 3))
df["Number_of_Ads"] = df["Number_of_Ads"].astype(int)

cats = [i for i in df.columns if df[i].nunique() <= 48]
nums = [i for i in df.columns if i not in cats]

index = 0

for _ in range(2):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    for i in range(3):
        df[cats[index]].value_counts()[:10].plot(kind="bar", ax=axes[i])
        bar_labels(axes[i])
        axes[i].set_title(cats[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(ncols=5, figsize=(15, 6))

for i, j in enumerate(nums):
    sns.histplot(df, x=j, kde=True, ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


index = 0

display(HTML("<h2>Average listening times</h2>"))

for _ in range(2):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    for i in range(3):
        grouped = df.groupby(cats[index])
        grouped[nums[-1]].mean().sort_values(ascending=False)[:10].plot(kind="bar", ax=axes[i])
        bar_labels(axes[i], 90, "center")
        axes[i].set_title(cats[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()


fig, axes = plt.subplots(ncols=4, figsize=(15, 6))

display(HTML("<h2>Listening times distribution</h2>"))

for i, j in enumerate(nums[:-1]):
    sns.scatterplot(df, x=j, y=nums[-1], ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


for i in cats:
    if i != "Number of Ads":
        df[i] = LabelEncoder().fit_transform(df[i].values)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

training_regression()

