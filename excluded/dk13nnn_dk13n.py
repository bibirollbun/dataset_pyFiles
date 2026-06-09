# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install uv
!uv pip install xgboost scikit-learn pandas


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


train_df = pd.read_csv(r"/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv")
test_df  = pd.read_csv(r"/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv")


train_df.head()


test_df.head()


print("DepTime:", train_df["DepTime"].min(), "->", train_df["DepTime"].max())
print("Distance:", test_df["Distance"].min(), "->", test_df["Distance"].max())


print(train_df["Distance"].describe(percentiles=[0.25,0.5,0.75]))


# Hoặc chia theo ngưỡng cố định (ví dụ <500, 500-1000, >1000)
bins_dist = [0, 300,500, 1500, 5000]
labels_dist = ["Short", "Medium", "Long","VeryLong"]

train_df["Distance_bin"] = pd.cut(train_df["Distance"], bins=bins_dist, labels=labels_dist, right=False)
test_df["Distance_bin"] = pd.cut(test_df["Distance"], bins=bins_dist, labels=labels_dist, right=False)

train_df.head()


train_df["DepHour"] = train_df["DepTime"] // 100 - 24*((train_df["DepTime"]//100) > 24)
test_df["DepHour"] = test_df["DepTime"] // 100 - 24*((test_df["DepTime"]//100) > 24)
# Chia khung giờ: 0-6 (đêm), 6-12 (sáng), 12-18 (chiều), 18-24 (tối)
bins_time = [0, 6, 12, 18, 24]
labels_time = ["Night", "Morning", "Afternoon", "Evening"]

train_df["DepTime_bin"] = pd.cut(train_df["DepHour"], bins=bins_time, labels=labels_time, right=False)
test_df["DepTime_bin"] = pd.cut(test_df["DepHour"], bins=bins_time, labels=labels_time, right=False)


train_df = train_df.drop(columns=["DepTime","Distance","DepHour"])
test_df = test_df.drop(columns=["DepTime","Distance","DepHour"])



OHCol = ["DepTime_bin","Distance_bin","Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
Train = pd.get_dummies(train_df,columns=OHCol)
Test = pd.get_dummies(test_df,columns=OHCol)

Train_Label = Train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
Train = Train.drop(columns=["dep_delayed_15min"])

Test = Test.reindex(columns=Train.columns, fill_value=0)
Train.head()


Test.head()


X_train,X_valid,Y_train,Y_valid = train_test_split(
    Train, Train_Label, test_size=0.3, random_state=17, stratify=Train_Label
)

bst = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    colsample_bytree = 0.6,
    random_state=17,
    n_jobs=-1,
    booster = "dart",
    objective="binary:logistic",
    gamma=0.6,
    alpha=3,
    subsample = 0.7,
)

THRESHOLD = 0.5


bst.fit(
    X_train,Y_train,
    eval_set = [(X_valid,Y_valid)],
    early_stopping_rounds=30,      
)

y_pred = bst.predict(X_valid, iteration_range=(0,bst.best_iteration+1))

print("Best iteration:", bst.best_iteration)
print("Accuracy:", accuracy_score(Y_valid, y_pred))
print("ROC-AUC:", roc_auc_score(Y_valid, y_pred))



probs = bst.predict_proba(Test, iteration_range=(0, bst.best_iteration + 1))[:,1]

print(probs)
pd.Series(probs,
          name='dep_delayed_15min').to_csv('xgb_2feat.csv',
                                           index_label='id', header=True)

