import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


TRAIN_PATH = "/kaggle/input/playground-series-s5e3/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e3/test.csv"
SUBMISSION_PATH = "/kaggle/input/playground-series-s5e3/sample_submission.csv"


train_set = pd.read_csv(TRAIN_PATH)
test_set = pd.read_csv(TEST_PATH)
submission_set = pd.read_csv(SUBMISSION_PATH)

train_set.head()


train_set.drop(columns=["id", "day"], inplace=True)
test_set.drop(columns=["id", "day"], inplace=True)
test_set.fillna(0, inplace=True)

train_set.head(1)


train_set.info()


train_set.describe()


train_set.isna().sum()


train_set.duplicated().sum()


fig = plt.figure(figsize=(12, 6))
sns.heatmap(train_set.corr(), annot=True,cmap="Blues")
plt.xticks(rotation=45, ha='right')
plt.show()


sns.pairplot(train_set, hue="rainfall")
plt.show()


fig = plt.figure(figsize=(12, 6))

NFEATURES = len(train_set.columns[:-1])
NCOLS = 4
NROWS = int(NFEATURES / NCOLS) + (NFEATURES % NCOLS > 0)

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 8))

axes = axes.flatten()
for idx, col in enumerate(train_set.columns[:-1]):
    sns.violinplot(x=col, y="rainfall", data=train_set, ax=axes[idx])

fig.tight_layout()

plt.show()


fig = plt.figure(figsize=(12, 6))

NFEATURES = len(train_set.columns[:-1])
NCOLS = 4
NROWS = int(NFEATURES / NCOLS) + (NFEATURES % NCOLS > 0)

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 8))

axes = axes.flatten()
for idx, col in enumerate(train_set.columns[:-1]):
    sns.boxplot(x=col, y="rainfall", data=train_set, ax=axes[idx])

fig.tight_layout()

plt.show()


X_train = train_set.iloc[:, :-1]
y_train = train_set['rainfall'].values.reshape(-1, 1)


lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred_lr = lr.predict(test_set)
submission_file['rainfall'] = y_pred_lr


submission_file.to_csv("submission_file_lr.csv", index=False)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)


y_pred_rf = rf.predict(test_set)
submission_file['rainfall'] = y_pred_rf


submission_file.to_csv("submission_file_rf.csv", index=False)




