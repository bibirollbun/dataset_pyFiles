import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head(3)


X = train.drop(columns=["y", "id"])
y = train["y"]
X_test = test.drop(columns=["id"]).copy()

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.038993224953339976,
    "num_leaves": 112,
    "max_depth": 0,
    "n_estimators": 916,
    "feature_fraction": 0.7264763348129452,
    "bagging_fraction": 0.8937118335575807,
    "bagging_freq": 5,
    "seed": 42,
    "verbosity": -1
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
print("ROC AUC:", roc_auc_score(y_valid, y_valid_pred))

y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

submission = pd.DataFrame({
    'id': test['id'],
    'y': y_test_pred 
})

submission.to_csv('submission.csv', index=False)







