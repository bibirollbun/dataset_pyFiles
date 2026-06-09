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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
#train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

#train_full = pd.concat([train, train_extra], ignore_index=True)
train_full = train

y_train = train_full["Price"]
X_train = train_full.drop(columns=["id", "Price"])
X_test = test.drop(columns=["id"])

X_train["Material_Size"] = X_train["Material"] + "_" + X_train["Size"].astype(str)
X_test["Material_Size"] = X_test["Material"] + "_" + X_test["Size"].astype(str)

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

rf = RandomForestRegressor(random_state=42)

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=2, #can be changed to 5 or 10 for better performance
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", search)
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "price": predictions
})
submission.to_csv("submission.csv", index=False)
print("Done!")


pd.read_csv("submission.csv").head()



