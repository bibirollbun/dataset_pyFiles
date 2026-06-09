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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train_df["age_comorbidity"] = train_df["age_at_hct"] * train_df["comorbidity_score"]
test_df["age_comorbidity"] = test_df["age_at_hct"] * test_df["comorbidity_score"]
columns_to_drop = ["efs", "efs_time", "ID", "cyto_score_detail", "gvhd_proph"]


X_train = train_df.drop(columns=columns_to_drop, axis=1)
y_train = train_df["efs_time"].values


num_features = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()


preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(drop="first", min_frequency=0.05, sparse=False))
    ]), cat_features)
])


X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)


X_tr, X_val, y_tr, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)


params = {
    "objective": "survival:cox",    # Cox proportional hazards model
    "eval_metric": "cox-nloglik",   # Negative log partial likelihood
    "max_depth": 4,
    "subsample": 0.7,
    "tree_method": "hist",
    "verbosity": 0,
}


evals = [(dtrain, "train"), (dval, "eval")]
bst = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)


X_test = test_df.drop(columns=["ID", "cyto_score_detail", "gvhd_proph"], axis=1)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)


dtest = xgb.DMatrix(X_test_processed)
risk_scores = bst.predict(dtest)


submission = pd.DataFrame({
    "ID": test_df["ID"],
    "prediction": risk_scores
})


submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

