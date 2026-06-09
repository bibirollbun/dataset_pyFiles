import numpy as np
import pandas as pd

train_data = pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/train.csv")

train_data.head()


# Column names
print(", \n".join(train_data.columns))


# Total count of null values per column
train_data.isnull().sum()


# Total count of zeros per column
(train_data == 0).sum()


train_data[["HOMELESS_RATE", "RACE_BLACK_NH_PCT", "RACE_NATIVE_NH_PCT", "RACE_PACIFIC_NH_PCT", "MULTI_PERSON_NONFAMILY_HH_PCT"]].head()


import matplotlib.pyplot as plt
import math

graph_count = len(train_data.columns)
col_count = 4
row_count = math.ceil(graph_count / col_count)

fig, axes = plt.subplots(row_count, col_count)
fig.set_size_inches(col_count*3, row_count*3)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.25, hspace=0.4)
axes = axes.ravel()
for i in range(graph_count):
    axes[i].hist(train_data[train_data.columns[i]], bins=15)
    axes[i].set_title(train_data.columns[i])


correlations = train_data.iloc[:, 1:].corr()
correlations["HOMELESS_RATE"].sort_values()


from sklearn.ensemble import RandomForestRegressor

X = train_data.iloc[:, 2:]
y = train_data.iloc[:, 1]

model = RandomForestRegressor()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

