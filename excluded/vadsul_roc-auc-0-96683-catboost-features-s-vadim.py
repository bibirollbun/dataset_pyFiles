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



# загружаем данные
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col="id")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv", index_col="id")

# проверим размеры
print(train.shape)  # (кол-во строк, кол-во колонок)
print(test.shape)
print(sample_submission.shape)


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col="id")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv", index_col="id")

def add_features(df):
    df["duration_sq"] = df["duration"]**2
    df["balance_log"] = np.log1p(df["balance"].clip(lower=0))
    df["day_month"] = df["day"]  # можно объединить с month, если хочешь
    return df

train = add_features(train)
test = add_features(test)

y = train["y"]
X = train.drop(columns=["y"])
X_test = test.copy()

cat_features = X.select_dtypes(include=["object"]).columns.tolist()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    eval_metric="AUC",
    cat_features=cat_features,
    verbose=200,
    random_state=42
)

model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)

val_pred = model.predict_proba(X_val)[:, 1]
print("Validation ROC AUC:", roc_auc_score(y_val, val_pred))

y_test_pred = model.predict_proba(X_test)[:, 1]
sample_submission["y"] = y_test_pred
sample_submission.to_csv("submission_catboost.csv", index=False)



print(X_test.head())

