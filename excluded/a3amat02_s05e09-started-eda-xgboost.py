import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")

df.drop("id", axis=1, inplace=True)

index = 0

for _ in range(2):
    fig, axes = plt.subplots(ncols=5, figsize=(15, 6))
    for i in range(5):
        sns.histplot(df, x=df.columns[index], ax=axes[i])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(df.columns[index].replace("_", " "))
        index += 1
    plt.tight_layout()
    plt.show()


index = 0

for _ in range(3):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
    for i in range(3):
        sns.scatterplot(df, x=df.columns[index], y=df.columns[-1], ax=axes[i])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_title(df.columns[index].replace("_", " "))
        index += 1
    plt.tight_layout()
    plt.show()


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = XGBRegressor()

model.fit(x_train, y_train)

pred = model.predict(x_test)

r2 = r2_score(pred, y_test)
mse = mean_squared_error(pred, y_test)

print("XGBoost\nR2 score: {}\nMSE score: {}".format(r2, mse))

test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

test_x = test.iloc[:, 1:].values

pred = model.predict(test_x)

test["BeatsPerMinute"] = pred

test = test[["id", "BeatsPerMinute"]]

test.to_csv("submission.csv", index=False)

