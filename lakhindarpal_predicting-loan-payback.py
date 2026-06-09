import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import joblib

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
print("Train Shape", train.shape)
train.sample(5)


test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
print("Test Shape", test.shape)
test.sample(5)


print("Missing values in train:", train.isna().sum().sum())
print("Missing values in test:", test.isna().sum().sum())


train.info()


target = "loan_paid_back"
cat_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]
num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
]


train.drop(columns=[target, "id"]).describe()


for col in cat_cols:
    print(train[col].value_counts(), "\n")


print(train[target].value_counts() * 100 / train.shape[0])
train[target].value_counts()


sns.countplot(x="loan_paid_back", data=train)
plt.title("Target Variable Distribution")
plt.show()


# Categorical Features vs Target
n_cols = 2
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

for i, col in enumerate(cat_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.countplot(x=col, hue=target, data=train, ax=ax)
    ax.set_title(f"{col} vs {target}")
    ax.tick_params(axis="x", rotation=30)

# Hide empty subplots
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j // n_cols, j % n_cols])

plt.tight_layout()
plt.show()


# Numeric Feature Distributions
n_cols = 2
n_rows = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

for i, col in enumerate(num_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.histplot(data=train, x=col, hue=target, bins=30, kde=True, ax=ax)
    ax.set_title(f"{col} Distribution by {target}")

for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j // n_cols, j % n_cols])

plt.tight_layout()
plt.show()


# Boxplots (Numeric vs Target)
n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

for i, col in enumerate(num_cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.boxplot(x=target, y=col, data=train, ax=ax)
    ax.set_title(f"{col} vs {target}")

for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j // n_cols, j % n_cols])

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Numeric Features")
plt.show()


sns.barplot(x="education_level", y="annual_income", hue="loan_paid_back", data=train)
plt.title("Average Income by Education Level and Loan Status")
plt.xticks(rotation=30)
plt.show()


X = train.drop(columns=[target, "id"])
y = train[target]
X_test = test.drop(columns=["id"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]


cat_model = CatBoostClassifier(
    task_type="GPU",
    devices="0",
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    cat_features=cat_cols,
    eval_metric="AUC",
    scale_pos_weight=pos_weight,
    early_stopping_rounds=50,
    verbose=100,
)

cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

cat_valid_pred = cat_model.predict_proba(X_valid)[:, 1]

print("CatBoost AUC:", roc_auc_score(y_valid, cat_valid_pred))


cat_test_pred = cat_model.predict_proba(X_test)[:, 1]

pd.DataFrame({"id": test["id"], "loan_paid_back": cat_test_pred}).to_csv(
    "submission_catboost.csv", index=False
)


for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_valid[c] = X_valid[c].astype("category")
    X_test[c] = X_test[c].astype("category")


lgb_model = LGBMClassifier(
    device="gpu",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    objective="binary",
    metric="auc",
    categorical_features=cat_cols,
    scale_pos_weight=pos_weight,
    early_stopping_rounds=50,
    verbose=100,
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
)

lgb_valid_pred = lgb_model.predict_proba(X_valid)[:, 1]

print("LightGBM AUC:", roc_auc_score(y_valid, lgb_valid_pred))


lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]

pd.DataFrame({"id": test["id"], "loan_paid_back": lgb_test_pred}).to_csv(
    "submission_lightgbm.csv", index=False
)


xgb_model = XGBClassifier(
    device="cuda",
    n_estimators=1000,
    learning_rate=0.05,
    tree_method="hist",
    max_depth=8,
    objective="binary:logistic",
    enable_categorical=True,
    eval_metric="auc",
    scale_pos_weight=pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)

xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]

print("XGBoost AUC:", roc_auc_score(y_valid, xgb_valid_pred))


xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

pd.DataFrame({"id": test["id"], "loan_paid_back": xgb_test_pred}).to_csv(
    "submission_xgboost.csv", index=False
)


# Stack validation and test predictions
meta_train = np.vstack([cat_valid_pred, lgb_valid_pred, xgb_valid_pred]).T
meta_test = np.vstack([cat_test_pred, lgb_test_pred, xgb_test_pred]).T

# Train meta model on stacked validation predictions
meta_model = LogisticRegression(max_iter=500)
meta_model.fit(meta_train, y_valid)

# Predict on the same validation folds
meta_valid_pred = meta_model.predict_proba(meta_train)[:, 1]

print("MetaModel AUC:", roc_auc_score(y_valid, meta_valid_pred))


meta_pred = meta_model.predict_proba(meta_test)[:, 1]
pd.DataFrame({"id": test["id"], "loan_paid_back": meta_pred}).to_csv(
    "submission.csv", index=False
)


# Save all the models
joblib.dump(cat_model, "catboost_model.pkl")
joblib.dump(lgb_model, "lightgbm_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(meta_model, "meta_model.pkl")


models = {
    "CatBoost": cat_valid_pred,
    "LightGBM": lgb_valid_pred,
    "XGBoost": xgb_valid_pred,
    "Meta Model": meta_valid_pred
}

plt.figure(figsize=(7, 7))
for name, preds in models.items():
    fpr, tpr, _ = roc_curve(y_valid, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Base Models vs Meta Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




