import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import joblib

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
print("Train Shape", train.shape)
train.sample(5)


og = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")
print("og Shape", og.shape)
og.sample(5)


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print("Test Shape", test.shape)
test.sample(5)


print("Missing values in train:", train.isna().sum().sum())
print("Missing values in test:", test.isna().sum().sum())
print("Missing values in og:", og.isna().sum().sum())


# merge train with og
train = pd.concat([train.drop(columns=["id"]), og], join="inner", ignore_index=True)
train = train.drop_duplicates()
print("New Train Shape", train.shape)


train.info()


target = "diagnosed_diabetes"
cat_cols = train.select_dtypes(include="object").columns.tolist()
num_cols = (
    train.select_dtypes(include="number").drop(columns=[target]).columns.tolist()
)


train.drop(columns=[target]).describe()


train[num_cols].skew().sort_values(ascending=False)


for col in cat_cols:
    print(train[col].value_counts(), "\n")


print(train[target].value_counts() * 100 / train.shape[0])
train[target].value_counts()


sns.countplot(x=target, data=train)
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


plt.figure(figsize=(20, 16))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Numeric Features")
plt.show()


# Load saved stacking features

model_prefixes = ["lgbm", "xgb", "cat"]
model_names = ["lightgbm", "xgboost", "catboost"]

y = train[target].values

def normalize_rank(arr):
    return rankdata(arr) / len(arr)

stack_train = np.column_stack([
    normalize_rank(np.load(f"/kaggle/input/ps-s5e12-stratified-k-fold-{name}/{prefix}_oof.npy"))
    for prefix, name in zip(model_prefixes, model_names)
])

stack_test = np.column_stack([
    normalize_rank(np.load(f"/kaggle/input/ps-s5e12-stratified-k-fold-{name}/{prefix}_pred.npy"))
    for prefix, name in zip(model_prefixes, model_names)
])

print("Stack train shape:", stack_train.shape)
print("Stack test shape:", stack_test.shape)


# Train meta model
Cs = np.logspace(-2, 1, 15)

meta_model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegressionCV(
        Cs=Cs,
        cv=5,              # internal CV to find best C
        penalty="l2",
        scoring="roc_auc",
        max_iter=5000,
        n_jobs=-1,
        refit=True
    ))
])

meta_model.fit(stack_train, y)

oof_meta = meta_model.predict_proba(stack_train)[:, 1]
meta_auc = roc_auc_score(y, oof_meta)
print("Meta-model OOF AUC:", meta_auc)


final_preds = meta_model.predict_proba(stack_test)[:, 1]
# light clipping for safety
final_preds = np.clip(final_preds, 0.001, 0.999)

submission = pd.DataFrame({
    "id": test["id"],
    target: final_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


np.save("meta_oof.npy", oof_meta)
np.save("meta_preds.npy", final_preds)
joblib.dump(meta_model, "meta_model.pkl")

print("Done")


pd.read_csv("/kaggle/working/submission.csv")




