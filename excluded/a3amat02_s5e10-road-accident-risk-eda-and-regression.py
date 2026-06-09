import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

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
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score, mean_squared_error

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, rotation=rotation, label_type=location)
    axes.set_ylabel("")
    axes.set_xlabel("")
    axes.set_yticklabels(())

xgbr = XGBRegressor()
lgbr = LGBMRegressor(verbose=-100)
cbr = CatBoostRegressor(verbose=False)

models_r = [xgbr, lgbr, cbr]

names_r = ["XGBoost", "LightGBM", "CatBoost"]

def training_regression():
    r2s, mses = [], []

    for i in models_r:
        i.fit(x_train, y_train)
        pred = i.predict(x_test)
        r2s += [r2_score(pred, y_test)*100]
        mses += [mean_squared_error(pred, y_test)]

    dd = pd.DataFrame({"r2": r2s, "mse": mses}, index=names_r)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    index = 0
    dd = dd.sort_values("r2", ascending=False)
    dd["r2"] = round(dd["r2"], 2)
    dd["r2"].plot(kind="bar", ax=axes[index])
    bar_labels(axes[index])
    axes[index].set_title("r2 scores")

    index += 1

    dd = dd.sort_values("mse", ascending=True)
    dd["mse"] = round(dd["mse"], 4)
    dd["mse"].plot(kind="bar", ax=axes[index])
    bar_labels(axes[index])
    axes[index].set_title("mse scores")

    plt.tight_layout()
    plt.show()

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")

df.drop("id", axis=1, inplace=True)

cats = [i for i in df.columns if df[i].dtype == 'O']
nums = [i for i in df.columns if i not in cats]

index = 0

for j in [4, 4, 5]:
    fig, axes = plt.subplots(ncols=j, figsize=(15, 6))
    for i in range(j):
        if df.columns[index] in cats:
            df[df.columns[index]].value_counts().plot(kind="bar", ax=axes[i])
            bar_labels(axes[i])
        else:
            sns.histplot(df, x=df.columns[index], kde=True, ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
        
        axes[i].set_title(df.columns[index].replace('_', ' '))

        index += 1

    plt.tight_layout()
    plt.show()


index = 0

for _ in range(3):
    fig, axes = plt.subplots(ncols=4, figsize=(15, 6))
    for i in range(4):
        if df.columns[index] in cats:
            df.groupby(df.columns[index])[df.columns[-1]].mean().plot(kind="bar", ax=axes[i])
            bar_labels(axes[i])
            axes[i].set_title("Average road accident risk for {}".format(df.columns[index].replace('_', ' ')))
        else:
            sns.scatterplot(df, x=df.columns[index], y=df.columns[-1], ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].set_title("{} vs Road accident risk".format(df.columns[index].replace('_', ' ')))

        index += 1

    plt.tight_layout()
    plt.show()


encoders = dict()

for i in cats:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)
    encoders[i] = le

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

training_regression()

