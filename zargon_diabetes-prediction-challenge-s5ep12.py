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


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test.head()


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


target = "diagnosed_diabetes"

X = train_df.drop(columns=[target])
y = train_df[target]

test_ids = test_df["id"]
X_test = test_df.copy()


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

if "id" in numeric_cols:
    numeric_cols.remove("id")


numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


pipeline.fit(X_train, y_train)


val_preds = pipeline.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {auc:.4f}")


# pipeline.fit(X, y)


test_preds = pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission1.csv", index=False)




