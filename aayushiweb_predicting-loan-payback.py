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


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


test.head()


target = "loan_paid_back"

num_cols = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate"
]

cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]

cols = num_cols + cat_cols


train.head()


train.info


train.isna().sum()


train.isnull().sum()


sns.histplot(train[target], kde=True)
plt.title("Distribution of Target - loan_paid_back")
plt.show()


sns.countplot(x=train[target])
plt.title("Counts of Target Classes")
plt.show()


train[num_cols + [target]].corr()[target].sort_values(ascending=False)


for col in num_cols:
    sns.boxplot(x=train[target], y=train[col])
    plt.title(f"{col} vs {target}")
    plt.show()


for col in cat_cols:
    print(f"\n--- {col} ---")
    print(train.groupby(col)[target].mean().sort_values(ascending=False))


for col in cat_cols:
    sns.barplot(data=train, x=col, y=target)
    plt.xticks(rotation=45)
    plt.title(f"{col} Mean {target}")
    plt.show()


train["subgrade"] = train["grade_subgrade"].str[1:].astype(int)
train["grade"]    = train["grade_subgrade"].str[0]
test["subgrade"] = test["grade_subgrade"].str[1:].astype(int)
test["grade"]    = test["grade_subgrade"].str[0]
num_cols = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate", "subgrade"
]

cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade"
]

cols = num_cols + cat_cols


train.drop(columns=["grade_subgrade"], inplace=True)
test.drop(columns=["grade_subgrade"], inplace=True)


train[["grade", "subgrade"]].head()


grade_mean = train.groupby("grade")[target].mean()
train["grade_te"] = train["grade"].map(grade_mean)


subgrade_mean = train.groupby("subgrade")[target].mean()
train["subgrade_te"] = train["subgrade"].map(subgrade_mean)


train.drop(columns=["grade", "subgrade"], inplace=True)


num_cols = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate", "subgrade_te"
]

cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose"
]

cols = num_cols + cat_cols


train[["grade_te", "subgrade_te"]].head()


test["grade_te"] = test["grade"].map(grade_mean)
test["subgrade_te"] = test["subgrade"].map(subgrade_mean)


X = train[cols]
y = (train[target] >= 0.6).astype(int) 


from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")


model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.0015,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

clf = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))


lgb_model = clf.named_steps["model"]
importance = lgb_model.feature_importances_

plt.figure(figsize=(12,5))
plt.bar(range(len(importance)), importance)
plt.title("Feature Importance")
plt.show()


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "model__n_estimators": [400,600,800,1000],
    "model__learning_rate": [0.05, 0.03, 0.01],
    "model__max_depth": [-1, 5, 7, 9],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    clf,
    param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_clf = search.best_estimator_
print(search.best_params_)


# Use best model from search
best_clf = search.best_estimator_

# Predict
pred = best_clf.predict_proba(test)[:, 1]

# Create submission
sub = pd.DataFrame({
    "id": test["id"],
    target: pred
})

# Save
sub.to_csv("submission2.csv", index=False)
print("✅ submission2.csv created!")


sub.head()


sub.mean()


sub.min()







