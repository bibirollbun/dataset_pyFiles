import os, sys, warnings, json
warnings.filterwarnings("ignore")
RND = 42

# presentation header
title = "Machine Learning Pipeline for Multi-Class Prediction"
subtitle = "Developing a Reliable Multi-Class Classifier with Modern ML Techniques"
print("="*len(title))
print(title)
print(subtitle)
print("="*len(title), "\n")

# imports
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.base import clone

# Try to use LightGBM if already present in the environment for a stronger baseline
use_lgb = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    use_lgb = True
except Exception:
    use_lgb = False

print("Using LightGBM:", use_lgb)
print()

n_train = 5000
n_test = 2000
n_features = 20
n_classes = 3

X, y = make_classification(
    n_samples=n_train + n_test,
    n_features=n_features,
    n_informative=8,
    n_redundant=4,
    n_classes=n_classes,
    flip_y=0.03,
    class_sep=1.0,
    random_state=RND
)

X_train = X[:n_train].copy(); y_train = y[:n_train].copy()
X_test  = X[n_train:].copy()

train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(n_features)])
train["target"] = y_train
train["id"] = np.arange(1, len(train) + 1)

test = pd.DataFrame(X_test, columns=[f"f_{i}" for i in range(n_features)])
test["id"] = np.arange(len(train) + 1, len(train) + 1 + len(test))

# Add two categorical features by binning two numeric columns
for col in ["f_0", "f_1"]:
    train[f"{col}_cat"] = pd.qcut(train[col], q=4, labels=[f"{col}_A", f"{col}_B", f"{col}_C", f"{col}_D"])
    test[f"{col}_cat"]  = pd.qcut(test[col],  q=4, labels=[f"{col}_A", f"{col}_B", f"{col}_C", f"{col}_D"])

train = train.sample(frac=1, random_state=RND).reset_index(drop=True)
test  = test.sample(frac=1, random_state=RND+1).reset_index(drop=True)

out_dir = "./data"
os.makedirs(out_dir, exist_ok=True)
train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

# create a sample_submission matching expected format: id + class_0..class_{n-1}
sample_submission = pd.DataFrame({"id": test["id"].values})
for c in range(n_classes):
    sample_submission[f"class_{c}"] = 1.0 / n_classes
sample_submission.to_csv(os.path.join(out_dir, "sample_submission.csv"), index=False)

print("Saved synthetic data to ./data (train.csv, test.csv, sample_submission.csv)")
print("Train shape:", train.shape, "Test shape:", test.shape)
print()

ID_COL = "id"; TARGET = "target"
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
num_cols = train[features].select_dtypes(include=["int64","float64","int","float"]).columns.tolist()
cat_cols = train[features].select_dtypes(include=["object","category"]).columns.tolist()

print("Numeric features:", num_cols)
print("Categorical features:", cat_cols)
print()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
], remainder="drop")

if use_lgb:
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=RND, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=300, random_state=RND, n_jobs=-1)

pipe = Pipeline([("preproc", preprocessor), ("clf", model)])

X = train.drop([ID_COL, TARGET], axis=1)
y = train[TARGET].values
X_test = test.drop([ID_COL], axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
oof_proba = np.zeros((len(X), n_classes))
test_proba = np.zeros((len(X_test), n_classes))
fold_stats = []
print("Starting 5-fold Stratified CV...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    clf = clone(pipe)
    clf.fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)
    oof_proba[val_idx] = p_val
    test_proba += clf.predict_proba(X_test) / skf.n_splits
    loss = log_loss(y_val, p_val)
    acc = accuracy_score(y_val, p_val.argmax(axis=1))
    fold_stats.append({"fold": fold, "log_loss": float(loss), "accuracy": float(acc)})
    print(f"Fold {fold} - log_loss: {loss:.5f} - accuracy: {acc:.4f}")

mean_loss = np.mean([f["log_loss"] for f in fold_stats])
mean_acc  = np.mean([f["accuracy"] for f in fold_stats])
print(f"\nMean CV log_loss: {mean_loss:.5f} - Mean CV accuracy: {mean_acc:.4f}\n")


final_clf = clone(pipe)
final_clf.fit(X, y)
final_test_proba = final_clf.predict_proba(X_test)
blend_test_proba = 0.5 * test_proba + 0.5 * final_test_proba

# prepare submission matching sample_submission
prob_cols = [c for c in sample_submission.columns if c != "id"]
if len(prob_cols) != n_classes:
    prob_cols = [f"class_{i}" for i in range(n_classes)]
submission = pd.DataFrame({"id": test["id"].values})
for i, col in enumerate(prob_cols):
    submission[col] = blend_test_proba[:, i]
submission = submission[["id"] + prob_cols]
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)

# save oof proba for local validation inspection
pd.DataFrame(oof_proba, columns=prob_cols).to_csv(os.path.join(out_dir, "oof_proba.csv"), index=False)

print("Wrote submission.csv to:", submission_path)
print("\nSubmission preview (first 6 rows):")
display_df = submission.head(6).copy()
try:
    # ace_tools helper (available in python_user_visible environment) to show DataFrame nicely
    import ace_tools as tools
    tools.display_dataframe_to_user("submission_preview", display_df)
except Exception:
    print(display_df.to_string(index=False))

# print OOF metrics summary
from math import isfinite
oof_loss = log_loss(y, oof_proba)
oof_acc  = accuracy_score(y, oof_proba.argmax(axis=1))
print(f"\nOOF log_loss: {oof_loss:.5f}  OOF accuracy: {oof_acc:.4f}")

# list key files created
print("\nFiles created in working directory (top-level):")
for f in sorted(os.listdir(".")):
    if f in {"data", "submission.csv"} or f.endswith(".csv"):
        print(" -", f)
print("\nInside ./data/:", sorted(os.listdir(out_dir))[:50])

# final JSON summary for easy copy-paste into a Kaggle write-up section
summary = {
    "title": title,
    "subtitle": subtitle,
    "n_train": n_train,
    "n_test": n_test,
    "n_features": n_features,
    "n_classes": n_classes,
    "cv_folds": 5,
    "mean_cv_log_loss": round(float(mean_loss), 6),
    "mean_cv_accuracy": round(float(mean_acc), 6),
    "oof_log_loss": round(float(oof_loss), 6),
    "oof_accuracy": round(float(oof_acc), 6),
    "submission_path": submission_path,
    "data_dir": out_dir
}
print("\nSummary:")
print(json.dumps(summary, indent=2))


