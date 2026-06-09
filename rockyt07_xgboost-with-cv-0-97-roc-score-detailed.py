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
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score





train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
train_df.head()








def preprocess(train, test, target="y"):
    train = train.copy()
    test  = test.copy()
    
    test_ids = test["id"]
    
    train = train.drop(columns=["id"])
    test  = test.drop(columns=["id"])
    
    categorical_cols = ["job", "marital", "education", "default",
                        "housing", "loan", "contact", "month", "poutcome"]
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
    
    return train, test, test_ids

train_processed, test_processed, test_ids = preprocess(train_df, test_df, target="y")
X = train_processed.drop(columns=["y"])
y = train_processed["y"]






xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "lambda": 1,
    "alpha": 0,
    "tree_method": "hist",   # change to "gpu_hist" if GPU available
    "seed": 42
}





dtrain = xgb.DMatrix(X, label=y)

cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=1000,
    nfold=3,
    stratified=True,
    early_stopping_rounds=20,
    metrics="auc",
    seed=42,
    verbose_eval=100
)

best_round = len(cv_results)
best_auc = cv_results["test-auc-mean"].max()

print(f"âœ… Best round from CV: {best_round}")
print(f"âœ… Best CV ROC-AUC: {best_auc:.6f}")





final_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=best_round
)





dtest = xgb.DMatrix(test_processed)
test_preds = final_model.predict(dtest)





submission = pd.DataFrame({"id": test_ids, "y": test_preds})
submission.to_csv("/kaggle/working/submission.csv", index=False, float_format="%.6f")

print("âœ… Submission saved to /kaggle/working/submission.csv")
print(submission.head(10).to_string(index=False, formatters={"y": "{:.6f}".format}))




