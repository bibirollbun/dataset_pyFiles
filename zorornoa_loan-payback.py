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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


TARGET = "loan_paid_back"


print("\nMissing values:")
print(train.isnull().sum())


# Basic visualization
plt.figure(figsize=(8,5))
sns.countplot(x=TARGET, data=train)
plt.title("Loan Payback Distribution")
plt.show()


# Numeric correlations
num_cols = train.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(10,8))
sns.heatmap(train[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numeric Feature Correlation")
plt.show()


def feature_engineer(df):
    df = df.copy()
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["income_per_interest"] = df["annual_income"] / (df["interest_rate"] + 1)
    df["credit_to_debt_ratio"] = df["credit_score"] / (df["debt_to_income_ratio"] + 0.01)
    df["is_high_earner"] = (df["annual_income"] > df["annual_income"].median()).astype(int)
    return df

train = feature_engineer(train)
test = feature_engineer(test)


y = train[TARGET]
X = train.drop(columns=[TARGET, "id"])
X_test = test.drop(columns=["id"])

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder="drop"
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(**params))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best parameters:", study.best_params)


best_params = study.best_params
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(**best_params))
])
final_model.fit(X, y)


xgb_model = final_model.named_steps["classifier"]
importances = xgb_model.feature_importances_
# Extract feature names after one-hot
encoded_features = final_model.named_steps["preprocessor"].transformers_[0][2] + \
    list(final_model.named_steps["preprocessor"].transformers_[1][1]
         .named_steps["encoder"].get_feature_names_out(cat_features))
feat_imp = pd.Series(importances, index=encoded_features).sort_values(ascending=False)[:20]

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 20 Feature Importances")
plt.show()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
final_model.fit(X_train, y_train)
preds = final_model.predict_proba(X_valid)[:,1]
roc_auc = roc_auc_score(y_valid, preds)
print(f"Validation ROC AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_valid, preds)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


submission = sample.copy()
submission["loan_paid_back"] = final_model.predict_proba(X_test)[:,1]
submission.to_csv("submission.csv", index=False)
print(" Submission saved as submission.csv")




