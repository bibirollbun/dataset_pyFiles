import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder


import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.shape


test.shape


train.info()


test.info()


X = train.drop(columns = ['diagnosed_diabetes'])

y = train['diagnosed_diabetes']


y


y.isna().sum()


y.fillna(0, inplace=True)


X.head()


cat_cols = X.select_dtypes(include="object").columns


for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42
)


model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(100)
    ]
)


val_preds = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, val_preds)

print(f"Validation ROC-AUC: {auc:.4f}")


model.fit(X, y)

test_preds = model.predict_proba(test)[:, 1]


submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()




