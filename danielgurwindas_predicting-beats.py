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


import re
import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


df_train.info()


df_train.isna().sum()


df_test.isna().sum()


df_train.head()


# ---------- Utility functions ----------

def infer_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    if "target" in train.columns and "target" not in test.columns:
        return "target"
    train_only = [c for c in train.columns if c not in test.columns]
    if len(train_only) == 1:
        return train_only[0]
    return train.columns[-1]   # fallback: last column


def infer_id_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if re.search(r"id$", c, flags=re.I)]
    if candidates:
        return candidates[0]
    return df.columns[0]


def detect_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique()
        if nunique <= 2:
            return "binary"
        if pd.api.types.is_integer_dtype(y) and nunique <= 20:
            return "multiclass"
        return "regression"
    else:
        return "multiclass"


def build_pipeline(X: pd.DataFrame, task: str) -> Pipeline:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if task in ("binary", "multiclass"):
        model = HistGradientBoostingClassifier(random_state=42)
    else:
        model = HistGradientBoostingRegressor(random_state=42)

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def cross_validate(pipe, X, y, task):
    if task == "binary":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        return "ROC AUC", np.mean(scores)
    elif task == "multiclass":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        return "Accuracy", np.mean(scores)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scorer = make_scorer(lambda yt, yp: math.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=rmse_scorer)
        return "RMSE", -np.mean(scores)


# ---------- Main pipeline ----------

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

target = infer_target(train, test)
id_col = infer_id_column(test)

X = train.drop(columns=[target])
y = train[target]
task = detect_task(y)

pipe = build_pipeline(X, task)

metric_name, cv_score = cross_validate(pipe, X, y, task)
print(f"{metric_name} (CV): {cv_score:.4f}")

# Fit on full training data
pipe.fit(X, y)

# Predict on test
if task == "binary":
    preds = pipe.predict_proba(test)[:, 1]
elif task == "multiclass":
    preds = pipe.predict(test)
else:
    preds = pipe.predict(test)


# Save submission
submission = pd.DataFrame({
    id_col: test[id_col].values,
    target: preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

