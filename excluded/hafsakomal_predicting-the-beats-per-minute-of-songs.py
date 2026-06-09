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


import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)




X = train.drop(columns=["BeatsPerMinute", "id"])   # drop target + id
y = train["BeatsPerMinute"]

X_test = test.drop(columns=["id"])                 # drop only id


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# 6. Train model with callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[early_stopping(100), log_evaluation(100)]
)


valid_preds = model.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_preds, squared=False)
print("Validation RMSE:", rmse)


test_preds = model.predict(X_test)


submission = sample_submission.copy()
submission["BeatsPerMinute"] = test_preds
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created:", submission.shape)
submission.head()




