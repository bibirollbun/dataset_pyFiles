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


train_data = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")


test_data = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")


train_data.info()


train_data = train_data.dropna(subset=["is_cheating"])


train_data


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X_train = train_data.drop(columns=["user_hash", "is_cheating","high_conf_clean"])
y_train = train_data["is_cheating"]


X_train


pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])


pipeline.fit(X_train, y_train)


test_df = test_data.drop(columns=["user_hash"])


X_test = test_df


test_preds = pipeline.predict(X_test)


print(test_preds[:20])


proba = pipeline.predict_proba(X_test)


print(proba[:5])


from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, cv=5,scoring="accuracy")

print("CV Accuracy:", scores.mean())



from sklearn.linear_model import LogisticRegression

logreg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


logreg_pipeline.fit(X_train, y_train)


test_preds = pipeline.predict(X_test)


logreg_acc = cross_val_score(
    logreg_pipeline, X_train, y_train,
    cv=5,
    scoring="accuracy"
)

print("Logistic Regression Accuracy (CV):", logreg_acc.mean())


from sklearn.tree import DecisionTreeClassifier

tree_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        random_state=42
    ))
])



tree_pipeline.fit(X_train,y_train)


test_preds = pipeline.predict(X_test)


tree_acc = cross_val_score(
    tree_pipeline, X_train, y_train,
    cv=5,
    scoring="accuracy"
)

print("Decision tree Regression Accuracy (CV):", logreg_acc.mean())


submission = pd.read_csv("/kaggle/input/mercor-cheating-detection/sample_submission.csv")
submission = submission.drop(columns=["prediction"])

submission["is_cheating"] = test_preds

submission.head()







