import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, label_type=location, rotation=rotation)
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_yticklabels(())

df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")

df.drop("id", axis=1, inplace=True)

cats = ["Soil Type", "Crop Type", "Fertilizer Name"]
nums = [i for i in df.columns if i not in cats]

fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

for i, j in enumerate(cats):
    df[j].value_counts().plot(kind="bar", ax=axes[i])
    bar_labels(axes[i])
    axes[i].set_title(j)
plt.tight_layout()
plt.show()

index = 0

for _ in range(2):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    for i in range(3):
        sns.histplot(df, x=nums[index], kde=True, ax=axes[i])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(nums[index])
        index += 1
    plt.tight_layout()
    plt.show()


grouped = df.groupby(cats[-1])

fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

for i, j in enumerate(cats[:-1]):
    grouped[j].value_counts().unstack().plot(kind="bar", stacked=True, ax=axes[i])
    bar_labels(axes[i], 0, "center")
    axes[i].set_title(j)
plt.tight_layout()
plt.show()

index = 0

for _ in range(2):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    for i in range(3):
        sns.kdeplot(df, x=nums[index], hue=cats[-1], ax=axes[i])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(nums[index])
        index += 1
    plt.tight_layout()
    plt.show()


encoders = []
for i in cats:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)
    encoders += [le]

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

model = XGBClassifier()

model.fit(x, y)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

index = 0

for i in cats[:-1]:
    test_df[i] = encoders[index].transform(test_df[i].values)
    index += 1

test_x = test_df.iloc[:, 1:].values

predictions = model.predict_proba(test_x)


answers = []

for i in range(len(predictions)):
    top = predictions[i]
    indices = [[j, top[j]] for j in range(7)]
    indices = sorted(indices, key=lambda x: x[1], reverse=True)
    labels = []
    for j in range(3):
        if indices[j][1] >= 0.14:
            labels += [indices[j][0]]
    
    if len(labels) < 1:
        labels = [x[0] for x in indices[:3]]

    answers += [' '.join(encoders[-1].inverse_transform(labels))]


test_df["Fertilizer Name"] = answers

sub = test_df[["id", "Fertilizer Name"]]

sub.to_csv("submission.csv", index=False)

