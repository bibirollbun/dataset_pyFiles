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

from sklearn.metrics import r2_score, mean_squared_log_error

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

models_r = [rfr, abr, gbr, etr,
         lnr, svr, xgbr, lgbr]

names_r = ["Random Forest", "Ada Boost", "Gradient Boosting", "Extra Trees",
        "Linear Regression", "Support Vector Machine", "XGBoost", "LightGBM"]

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

df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

df.drop("id", inplace=True, axis=1)

fig, axes = plt.subplots(figsize=(15, 6))

df["Sex"].value_counts().plot(kind="bar", ax=axes)
bar_labels(axes)
axes.set_title("Gender")

plt.show()

fig, axes = plt.subplots(ncols=7, figsize=(15, 6))

for i, j in enumerate(df.columns[1:]):
    sns.histplot(df, x=j, kde=True, ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=7, figsize=(15, 6))

for i, j in enumerate(df.columns[1:]):
    sns.kdeplot(df, x=j, hue=df.columns[0], ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


grouped = df.groupby(df.columns[0])

fig, axes = plt.subplots(figsize=(15, 6))

grouped[df.columns[-1]].mean().plot(kind="bar", ax=axes)
bar_labels(axes)
axes.set_title("Calories per gender")
plt.show()


fig, axes = plt.subplots(ncols=6, figsize=(15, 6))

for i, j in enumerate(df.columns[1:-1]):
    sns.scatterplot(df, x=j, y=df.columns[-1], ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"].values)


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'max_depth': 10,
        'learning_rate': 0.00075,
        'reg_alpha': 2,
        'reg_lambda': 1,
        'max_delta_step': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.55,
        'enable_categorical': True,
        'device': "cuda",
        'min_child_weight': 10,                 # Minimum sum of instance weight needed in a child
        'gamma': 0.1,                           # Minimum loss reduction required to make a further partition
        'tree_method': 'gpu_hist',              # Efficient histogram-based tree growth on GPU
        'grow_policy': 'lossguide',             # Use 'depthwise' or 'lossguide' (good for large datasets)
        'sampling_method': 'uniform',           # Or 'gradient_based'
        'max_bin': 512,    
    }

model = XGBRegressor(**params)

model.fit(x_train, y_train)

pred = model.predict(x_test)

msle = mean_squared_log_error(y_test, pred, squared=False)

print("XGBoost RMSLE score: ", msle)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

test_set["Sex"] = le.transform(test_set["Sex"].values)

test_x = test_set.iloc[:, 1:].values

res = model.predict(test_x)

test_set["Calories"] = np.expm1(res)

sub = test_set[["id", "Calories"]]


sub.to_csv("submission.csv", index=False)

