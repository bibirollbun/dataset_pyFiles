import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train_df.columns


train_df.describe()


train_df.head()


train_df['efs'].value_counts(normalize=True)


train_df.isnull().mean().sort_values(ascending=False).head(15)


sns.histplot(data=train_df, x="age_at_hct", hue="efs", bins=30)
plt.show()


sns.histplot(data=train_df, x="donor_age", hue="efs", bins=30)
plt.show()


sns.histplot(data=train_df, x="race_group", hue="efs", bins=30)
plt.show()


X = train_df.drop(columns="efs")
y = train_df["efs"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=13
)


cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()


plt.figure(figsize=(20,12))
sns.heatmap(train_df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")


numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])



models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42))
    ]),
    "Gradient Boosting (sklearn)": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42))
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42
        ))
    ]),
    "LightGBM": Pipeline([
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    results.append({
        "Model": name,
        "ROC-AUC Mean": scores.mean(),
        "ROC-AUC Std": scores.std()
    })
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC Mean", ascending=False)
results_df


X_train_cb = X_train.copy()
X_val_cb = X_val.copy()
X_cb = X.copy()
test_cb = test_df.copy()

for df in [X_train_cb, X_val_cb, X_cb, test_cb]:
    for col in cat_cols:
        df[col] = df[col].fillna("missing").astype(str)


cat_features = [X_train_cb.columns.get_loc(col) for col in cat_cols]


cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    random_seed=42
)

cat_scores = []
for train_idx, val_idx in cv.split(X_train_cb, y_train):
    X_tr, X_va = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    cat_model.fit(X_tr, y_tr, cat_features=cat_features)
    
    preds = cat_model.predict_proba(X_va)[:, 1]
    cat_scores.append(roc_auc_score(y_va, preds))

print(f"CatBoost CV ROC-AUC: {np.mean(cat_scores):.4f} ± {np.std(cat_scores):.4f}")


best_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100,
    random_seed=42
)

best_model.fit(X_train_cb, y_train, cat_features=cat_features)

val_preds = best_model.predict_proba(X_val_cb)[:, 1]
val_score = roc_auc_score(y_val, val_preds)
print("Holdout ROC-AUC:", val_score)


test_ids = test_cb["ID"].copy()

for col in X_cb.columns:
    if col not in test_cb.columns:
        test_cb[col] = np.nan

test_cb = test_cb[X_cb.columns]

for col in cat_cols:
    if col in test_cb.columns:
        test_cb[col] = test_cb[col].fillna("missing").astype(str)

test_preds = best_model.predict_proba(test_cb)[:, 1]

submission = pd.DataFrame({
    "ID": test_ids,
    "prediction": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

