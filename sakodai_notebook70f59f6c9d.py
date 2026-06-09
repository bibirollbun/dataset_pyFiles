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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train_data = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2021/train.csv")
test_data = pd.read_csv("/kaggle/input/tabular-playground-series-apr-2021/test.csv")


for df in [train_data, test_data]:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)


X = train_data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y = train_data["Survived"]
test_X = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)


categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
categorical_cols += ["Pclass", "Sex", "Embarked", "Title"]
categorical_cols = list(set(categorical_cols))
numerical_cols = [c for c in X.columns if X[c].dtype in ["int64", "float64"] and c not in ["Survived"]]


numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0),
    "LightGBM": LGBMClassifier(random_state=0),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=0)
}


best_model = None
best_score = 0

print("モデル比較中\n")
for name, model in models.items():
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    avg_score = scores.mean()
    print(f"{name}: Average CV accuracy = {avg_score:.4f}")
    if avg_score > best_score:
        best_score = avg_score
        best_model = model


print(f"\n選んだモデル: {best_model.__class__.__name__}（精度: {best_score:.4f}）")


final_clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
final_clf.fit(X, y)


preds = final_clf.predict(test_X)


output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": preds
})
output.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv を出力")

