# ðŸ”¥ Predicting Loan Payback 


# IMPORT LIB.


import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from lightgbm import LGBMClassifier

SEED = 42
FOLDS = 5
np.random.seed(SEED)

DATA_DIR = "/kaggle/input/playground-series-s5e11"


# LOAD DATA


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET_COL = "loan_paid_back"
ID_COL = "id"

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("\nTrain columns:", train.columns.tolist())

print("\nTarget description:")
print(train[TARGET_COL].describe())
print("\nTarget value counts (normalized):")
print(train[TARGET_COL].value_counts(normalize=True))

plt.figure(figsize=(4,3))
sns.countplot(x=TARGET_COL, data=train)
plt.title("Target Distribution: loan_paid_back")
plt.show()


# FEATURE ENGINEERING (RATIOS)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Adapt to column names present in this competition
    # Common columns in S5E11: loan_amount, annual_income, interest_rate,
    # debt_to_income_ratio, credit_score, term, employment_length, etc.

    if {"loan_amount", "annual_income"}.issubset(df.columns):
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1e-3)

    if {"credit_score", "annual_income"}.issubset(df.columns):
        df["credit_to_income_ratio"] = df["credit_score"] / (df["annual_income"] + 1e-3)

    if {"interest_rate", "debt_to_income_ratio"}.issubset(df.columns):
        df["interest_debt_product"] = df["interest_rate"] * df["debt_to_income_ratio"]

    if {"loan_amount", "term"}.issubset(df.columns):
        df["monthly_loan_burden"] = df["loan_amount"] / (df["term"] + 1e-3)

    return df

train_fe = add_ratio_features(train)
test_fe  = add_ratio_features(test)

ratio_cols = [
    c for c in ["loan_to_income_ratio",
                "credit_to_income_ratio",
                "interest_debt_product",
                "monthly_loan_burden"]
    if c in train_fe.columns
]

print("\nRatio features added:", ratio_cols)

# Simple ratio vs target plots
for col in ratio_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=TARGET_COL, y=col, data=train_fe, showfliers=False)
    plt.title(f"{col} vs {TARGET_COL}")
    plt.tight_layout()
    plt.show()


# PREPROCESSING & ENCODING


y = train_fe[TARGET_COL].copy()
train_ids = train_fe[ID_COL].copy()
test_ids  = test_fe[ID_COL].copy()

# Drop ID and target from feature frames
X = train_fe.drop(columns=[ID_COL, TARGET_COL])
X_test = test_fe.drop(columns=[ID_COL])

# Combine for consistent encoding
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

num_cols = combined.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# Numeric: median imputation
for col in num_cols:
    med = combined[col].median()
    combined[col] = combined[col].fillna(med)

# Categorical: fill + Ordinal encode
if cat_cols:
    combined[cat_cols] = combined[cat_cols].fillna("Unknown")
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined[cat_cols] = ord_enc.fit_transform(combined[cat_cols])

# Split back to train & test
X_proc = combined.iloc[:len(X), :].reset_index(drop=True)
X_test_proc = combined.iloc[len(X):, :].reset_index(drop=True)

print("\nProcessed X shape     :", X_proc.shape)
print("Processed X_test shape:", X_test_proc.shape)


# BASELINE LIGHTGBM + FEATURE IMPORTANCE


baseline_params = {
    "n_estimators": 1500,
    "learning_rate": 0.02,
    "max_depth": -1,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "binary",
    "random_state": SEED
}

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_base = np.zeros(len(X_proc))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_proc, y), 1):
    print(f"\nBaseline Fold {fold}/{FOLDS}")

    X_tr, X_va = X_proc.iloc[tr_idx], X_proc.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    base_model = LGBMClassifier(**baseline_params)
    base_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc"
        # no verbose / verbose_eval to avoid version issues
    )

    oof_base[va_idx] = base_model.predict_proba(X_va)[:, 1]

baseline_auc = roc_auc_score(y, oof_base)
print(f"\nBaseline OOF ROC-AUC: {baseline_auc:.6f}")

# Fit on full data for feature importance
full_base_model = LGBMClassifier(**baseline_params)
full_base_model.fit(X_proc, y)

importances = pd.Series(full_base_model.feature_importances_, index=X_proc.columns)
imp_df = importances.sort_values(ascending=False).reset_index()
imp_df.columns = ["feature", "importance"]

plt.figure(figsize=(8,6))
imp_df.head(25).set_index("feature")["importance"][::-1].plot(kind="barh")
plt.title("Baseline Feature Importance (Top 25)")
plt.tight_layout()
plt.show()


# FEATURE PRUNING + SAFE HYPERPARAMETER TUNING


n_features = X_proc.shape[1]
keep_fraction = 0.7  # keep top 70% features
min_keep = 25        # always keep at least 25 features

k_keep = max(min_keep, int(n_features * keep_fraction))
top_features = imp_df.head(k_keep)["feature"].tolist()

print(f"Total features: {n_features}")
print(f"Keeping top {k_keep} features based on importance.")

X_pruned = X_proc[top_features].copy()
X_test_pruned = X_test_proc[top_features].copy()

print("Pruned X shape     :", X_pruned.shape)
print("Pruned X_test shape:", X_test_pruned.shape)

print("\n===== SAFE HYPERPARAMETER TUNING (RandomizedSearchCV) =====")

tune_param_grid = {
    "num_leaves": [31, 63, 95],
    "max_depth": [5, 7, 9, -1],
    "learning_rate": [0.01, 0.02],
    "min_child_samples": [50, 100, 150],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0.0, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 3.0],
}

base_for_search = LGBMClassifier(
    n_estimators=2000,
    objective="binary",
    random_state=SEED
)

search = RandomizedSearchCV(
    estimator=base_for_search,
    param_distributions=tune_param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    random_state=SEED,
    verbose=1
)

search.fit(X_pruned, y)

print("\nBest params from tuning:")
print(search.best_params_)
print(f"CV AUC from tuning: {search.best_score_:.6f}")

best_params = search.best_params_


# FINAL CV TRAINING WITH TUNED MODEL


oof_final = np.zeros(len(X_pruned))
test_final = np.zeros(len(X_test_pruned))

skf_final = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf_final.split(X_pruned, y), 1):
    print(f"\nTuned Fold {fold}/{FOLDS}")

    X_tr, X_va = X_pruned.iloc[tr_idx], X_pruned.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    tuned_model = LGBMClassifier(
        **best_params,
        n_estimators=2000,
        objective="binary",
        random_state=SEED
    )

    tuned_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc"
    )

    oof_final[va_idx] = tuned_model.predict_proba(X_va)[:, 1]
    test_final += tuned_model.predict_proba(X_test_pruned)[:, 1] / FOLDS

final_oof_auc = roc_auc_score(y, oof_final)
print(f"\nFinal Tuned OOF ROC-AUC: {final_oof_auc:.6f}")


# Feature importance from tuned model on full data
final_model_full = LGBMClassifier(
    **best_params,
    n_estimators=2000,
    objective="binary",
    random_state=SEED
)
final_model_full.fit(X_pruned, y)

final_importances = pd.Series(final_model_full.feature_importances_, index=X_pruned.columns)
final_imp_df = final_importances.sort_values(ascending=False).reset_index()
final_imp_df.columns = ["feature", "importance"]

plt.figure(figsize=(8,6))
final_imp_df.head(25).set_index("feature")["importance"][::-1].plot(kind="barh")
plt.title("Tuned Model Feature Importance (Top 25)")
plt.tight_layout()
plt.show()

# OOF prediction distribution
plt.figure(figsize=(6,4))
sns.histplot(oof_final, kde=True, bins=50)
plt.title("OOF Predicted Probabilities (Tuned Model)")
plt.xlabel("Predicted loan_paid_back probability")
plt.tight_layout()
plt.show()


# SUBMISSION


test_pred_clipped = np.clip(test_final.astype(float), 0.0, 1.0)

submission = pd.DataFrame({
    "id": test_ids.astype(int),
    "loan_paid_back": test_pred_clipped
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
print("\nSubmission preview:")
display(submission.head())




