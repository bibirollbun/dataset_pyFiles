!pip install -U imbalanced-learn


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, label_type=location, rotation=rotation)
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_yticklabels(())
  

df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

df.drop("id", axis=1, inplace=True)

cats = ["job", "marital", "education", "default",
       "housing", "loan", "contact", "month",
       "poutcome", "y"]

nums = [i for i in df.columns if i not in cats]

index = 0

for j in [5, 4, 4, 4]:
    fig, axes = plt.subplots(ncols=j, figsize=(15, 6))
    for i in range(j):
        if df.columns[index] in cats:
            df[df.columns[index]].value_counts()[:12].plot(kind="bar", ax=axes[i])
            bar_labels(axes[i])
            axes[i].set_title(df.columns[index].replace('_', ' '))
        else:
            sns.histplot(df, x=df.columns[index], kde=True, ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
            axes[i].set_title(df.columns[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()


grouped = df.groupby(cats[-1])

index = 0

for _ in range(4):
    fig, axes = plt.subplots(ncols=4, figsize=(15, 6))
    for i in range(4):
        if df.columns[index] in cats:
            grouped[df.columns[index]].value_counts().unstack().plot(kind="bar", stacked=True, ax=axes[i])
            bar_labels(axes[i], 0, "center")
        else:
            sns.kdeplot(df, x=df.columns[index], hue=cats[-1], ax=axes[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")
        axes[i].set_title(df.columns[index].replace('_', ' '))
        index += 1
    plt.tight_layout()
    plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

df = pd.concat([df, test_df.drop("id", axis=1)])

for i in cats[:-1]:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)


test_df.iloc[:, 1:] = df[df["y"].isna()].iloc[:, :-1]

x = df[df["y"].notna()].iloc[:, :-1].values
y = df[df["y"].notna()].iloc[:, -1].values

smote = SMOTE()

x, y = smote.fit_resample(x, y)


model = LGBMClassifier()

model.fit(x, y)


test_x = test_df.iloc[:, 1:].values

pred = model.predict_proba(test_x)[:, 1]

test_df["y"] = pred


test_df = test_df[["id", "y"]]

test_df.to_csv("submission.csv", index=False)

