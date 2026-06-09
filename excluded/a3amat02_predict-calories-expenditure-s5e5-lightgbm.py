import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_squared_log_error

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, rotation=rotation, label_type=location)
    axes.set_ylabel("")
    axes.set_xlabel("")
    axes.set_yticklabels(())

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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = LGBMRegressor(verbose=-100, metric="rmse")

model.fit(x_train, y_train)

pred = model.predict(x_test)

msle = mean_squared_log_error(y_test, pred, squared=False)

print("LightGBM RMSLE score: ", msle)


test_set = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

test_set["Sex"] = le.transform(test_set["Sex"].values)

test_x = test_set.iloc[:, 1:].values

res = model.predict(test_x)

test_set["Calories"] = [max(i, 0) for i in res]

sub = test_set[["id", "Calories"]]


sub.to_csv("submission.csv", index=False)

