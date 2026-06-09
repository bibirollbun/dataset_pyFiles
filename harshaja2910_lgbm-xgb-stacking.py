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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42




train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
print(train_df.shape, test_df.shape)
print(train_df.info())
print(train_df.head())

print("\n\n-------------Test-------------------")
print(test_df.info())
print(test_df.head())


# Uniqueness of all object columns
#Job
print("Job: ", train_df["job"].unique())

# Martial
print("Martial: ", train_df["marital"].unique())   

# Education
print("Education: ", train_df["education"].unique())   

# Default
print("Default: ", train_df["default"].unique())   

# Housing
print("Housing: ", train_df["housing"].unique()) 

# Loan
print("Loan: ", train_df["loan"].unique()) 

# contact
print("Contact: ", train_df["contact"].unique()) 

# poutcome
print("poutcome: ", train_df["poutcome"].unique()) 


# y
print("y: ", train_df["y"].unique()) 

print(train_df['y'].value_counts())


sns.countplot(x="y", data=train_df)
plt.title("Target Distribution (y)")
plt.show() # conclude that class is imbalance

print(train_df['y'].value_counts(normalize=True))


cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

for col in cat_cols:
    plt.figure(figsize=(10,5))
    prop = train_df.groupby(col)["y"].mean().sort_values(ascending=False)  # subscription rate
    sns.barplot(x=prop.index, y=prop.values, palette="viridis", errorbar="sd")
    plt.title(f"Subscription Rate by {col}")
    plt.ylabel("Proportion of y=1")
    plt.xticks(rotation=45)
    plt.show()


train = train_df.copy()
test = test_df.copy()

X = train.drop(["id", "y"], axis=1)
y = train["y"]
X_test = test.drop("id", axis=1)

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ================================================================
# Train/Validation split (before preprocessing)
# ================================================================

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)



# ================================================================
# Preprocessing
# ================================================================
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

# Fit only on training
preprocess.fit(X_tr)

X_tr_proc  = preprocess.transform(X_tr)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)


param_grids = {
    "dt": {
        "max_depth": [3, 5, 7, 9, None],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf": [1, 2, 5, 10, 20]
    },
    "xgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },
    "lgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [-1, 7, 10],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 50, 70]
    }
}

base_models = {
    "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "xgb": xgb.XGBClassifier(
        eval_metric="auc", use_label_encoder=False, tree_method="hist", random_state=RANDOM_STATE, early_stopping_rounds=50
    ),
    "lgb": lgb.LGBMClassifier(
        objective="binary", random_state=RANDOM_STATE, force_col_wise=True
    )
}



best_models = {}

for name, model in base_models.items():
    print(f"ğŸ”� Tuning {name}...")

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[name],
        n_iter=10,
        scoring="roc_auc",
        n_jobs=1,   # âš  safer than -1 when using fit_params with xgb/lgb
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1
    )

    # Prepare conditional fit_params safely
    if name == "xgb":
        fit_params = {
            "eval_set": [(X_val_proc, y_val)],
            "verbose": False
        }
    elif name == "lgb":
       fit_params = {
            "eval_set": [(X_val_proc, y_val)],
            "callbacks": [lgb.early_stopping(50, verbose=False)]
       }
    else:
        fit_params = {}

    search.fit(X_tr_proc, y_tr, **fit_params)

    best_models[name] = search.best_estimator_
    print(f"âœ… {name} Best AUC: {roc_auc_score(y_val, search.predict_proba(X_val_proc)[:, 1]):.5f}")


val_preds = []
weights = []

for name, model in best_models.items():
    preds = model.predict_proba(X_val_proc)[:,1]
    auc = roc_auc_score(y_val, preds)
    val_preds.append(preds)
    weights.append(max(auc, 0.0001))
    print(f"{name} AUC on val: {auc:.5f}")

# Soft Voting (weighted)
soft_vote_val = np.average(val_preds, axis=0, weights=weights)
soft_auc = roc_auc_score(y_val, soft_vote_val)
print(f"\nSoft Voting AUC: {soft_auc:.5f}")

# Stacking

stack_base_models = {
    "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "xgb": xgb.XGBClassifier(
        eval_metric="auc", use_label_encoder=False, tree_method="hist",
        random_state=RANDOM_STATE, n_estimators=best_models["xgb"].n_estimators,
        learning_rate=best_models["xgb"].learning_rate,
        max_depth=best_models["xgb"].max_depth
    ),
    "lgb": lgb.LGBMClassifier(
        objective="binary", random_state=RANDOM_STATE, force_col_wise=True,
        n_estimators=best_models["lgb"].n_estimators,
        learning_rate=best_models["lgb"].learning_rate,
        max_depth=best_models["lgb"].max_depth,
        num_leaves=best_models["lgb"].num_leaves
    )
}

stack = StackingClassifier(
    estimators=[(n, m) for n, m in stack_base_models.items()],
    final_estimator=LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
    cv=3, n_jobs=-1
)

stack.fit(X_tr_proc, y_tr)
stack_auc = roc_auc_score(y_val, stack.predict_proba(X_val_proc)[:,1])
print(f"Stacking AUC: {stack_auc:.5f}")


# ================================================================
# Retrain on full data & Predict Test
# ================================================================
X_proc = preprocess.transform(X)
if stack_auc > soft_auc:
    print("âœ… Using STACKING for final submission")
    stack.fit(X_proc, y)
    test_preds = stack.predict_proba(X_test_proc)[:,1]
else:
    print("âœ… Using SOFT VOTING for final submission")
    for name, model in best_models.items():
        model.fit(X_proc, y)
    test_preds = np.average(
        [m.predict_proba(X_test_proc)[:,1] for m in best_models.values()],
        axis=0, weights=weights
    )





# ================================================================
# Save Submission
# ================================================================

sub = pd.DataFrame({
    "id": test.index,   # or whatever the ID column is called in test.csv
    "y": test_preds
})
sub.to_csv("submission.csv", index=False)
print("ğŸ�‰ submission.csv saved!")

