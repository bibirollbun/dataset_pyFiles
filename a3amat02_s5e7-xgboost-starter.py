import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def bar_labels(axes, rotation=0, location="edge"):
    for container in axes.containers:
        axes.bar_label(container, rotation=rotation, label_type=location)
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_yticklabels(())

def handling_missing(df):
    df["Stage_fear"] = df["Stage_fear"].fillna("Unknown")
    df["Drained_after_socializing"] = df["Drained_after_socializing"].fillna("Unknown")

    classes = ["Yes", "No", "Unknown"]

    for i in classes:
        for j in classes:
            temp_df = df[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j)]
            time_spent_alone = temp_df[temp_df["Time_spent_Alone"].notna()]["Time_spent_Alone"].median()
            Social_event_attendance = temp_df[temp_df["Social_event_attendance"].notna()]["Social_event_attendance"].median()
            Going_outside = temp_df[temp_df["Going_outside"].notna()]["Going_outside"].median()
            Friends_circle_size = temp_df[temp_df["Friends_circle_size"].notna()]["Friends_circle_size"].median()
            Post_frequency = temp_df[temp_df["Post_frequency"].notna()]["Post_frequency"].median()
            df.loc[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j) & (df["Time_spent_Alone"].isna()), "Time_spent_Alone"] = time_spent_alone
            df.loc[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j) & (df["Social_event_attendance"].isna()), "Social_event_attendance"] = Social_event_attendance
            df.loc[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j) & (df["Going_outside"].isna()), "Going_outside"] = Going_outside
            df.loc[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j) & (df["Friends_circle_size"].isna()), "Friends_circle_size"] = Friends_circle_size
            df.loc[(df["Stage_fear"] == i) & (df["Drained_after_socializing"] == j) & (df["Post_frequency"].isna()), "Post_frequency"] = Post_frequency

    return df

labels_index = {"Extrovert": 0, "Introvert": 1}
index_labels = {1: "Introvert", 0: "Extrovert"}

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

df = handling_missing(df)

df.drop("id", axis=1, inplace=True)

cats = ["Stage_fear", "Drained_after_socializing", "Personality"]
nums = [i for i in df.columns if i not in cats]

fig, axes = plt.subplots(ncols=3, figsize=(15, 6))

for i, j in enumerate(cats):
    df[j].value_counts().plot(kind="bar", ax=axes[i])
    bar_labels(axes[i])
    axes[i].set_title(j.replace('_', ' '))
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


grouped = df.groupby(cats[-1])

fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

for i, j in enumerate(cats[:-1]):
    grouped[j].value_counts().unstack().plot(kind="bar", stacked=True, ax=axes[i])
    bar_labels(axes[i], 0, "center")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols=5, figsize=(15, 6))

for i, j in enumerate(nums):
    sns.kdeplot(df, x=j, hue=cats[-1], ax=axes[i])
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].set_title(j.replace('_', ' '))
plt.tight_layout()
plt.show()


df["Personality"] = df["Personality"].map(labels_index)

encoders = []

for i in cats[:-1]:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].values)
    encoders += [le]

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = XGBClassifier(scale_pos_weight=1,
               learning_rate=0.1,
                     depth=10)

model.fit(x_train, y_train)

pred = model.predict(x_test)

score = accuracy_score(pred, y_test)
report = classification_report(pred, y_test)
cm = confusion_matrix(pred, y_test)

sns.heatmap(cm, annot=True)
plt.title("XGBoost: {}%".format(round(score*100, 2)))
plt.show()

print(report)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

test_df = handling_missing(test_df)

index = 0

for i in cats[:-1]:
    test_df[i] = encoders[index].transform(test_df[i].values)

test_x = test_df.iloc[:, 1:].values
pred = model.predict(test_x)

test_df["Personality"] = pred
test_df["Personality"] = test_df["Personality"].map(index_labels)
test_df = test_df[["id", "Personality"]]
test_df.to_csv("submission.csv", index=False)

