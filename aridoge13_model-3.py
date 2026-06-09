import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")



# Load Data 
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")



# Define Categoricals
cat_cols = ["gender",
    "marital_status",
    "education_level",
    "loan_purpose",
    "grade_subgrade"]


# Handle Missing Values
skip_cols = {"id", "loan_paid_back"}
for col in train.columns:
    if col in skip_cols:
        continue
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Missing")
        if col in test.columns:
            test[col] = test[col].fillna("Missing")
    else:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

        

# Mapping Employment Status
def normalize_cat_series(s):
    return s.astype(str).str.lower().str.strip().str.replace("-", " ").str.replace("_", " ")

emp_map = {
    "unemployed": 0,
    "retired": 1,
    "student": 2,
    "self-employed": 3,  
    "employed": 4
}

# normalize original columns (do not overwrite yet)
train["employment_status_norm"] = normalize_cat_series(train["employment_status"])
test["employment_status_norm"]  = normalize_cat_series(test["employment_status"])

# map to rank, fill missing/unseen with -1
train["employment_status_rank"] = train["employment_status_norm"].map(emp_map).fillna(-1).astype(float)
test["employment_status_rank"]  = test["employment_status_norm"].map(emp_map).fillna(-1).astype(float)

        


# Encode Categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    # transform test with unseen -> "Unknown"
    test[col] = test[col].astype(str).map(lambda s: s if s in le.classes_ else "Unknown")
    if "Unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "Unknown")
    test[col] = le.transform(test[col])
    encoders[col] = le



# Feature Engineering
print("Features engineered:")
def add_features(df):
    df["monthly_income"] = df["annual_income"] / 12.0
    # protect against zero debt_to_income_ratio
    df["income_to_loan_ratio"] = 1.0 / (df["debt_to_income_ratio"].replace(0, np.nan))
    df["income_to_loan_ratio"] = df["income_to_loan_ratio"].fillna(0.0)
    df["credit_to_dti"] = df["credit_score"] * np.log1p(df["debt_to_income_ratio"])
    df["monthly_loan_payment"] = (df["loan_amount"] / 12.0) + ((df["loan_amount"] * df["interest_rate"]) / 12.0)
    df["credit_income_power"] = df["credit_score"] * np.log1p(df["annual_income"])
    df["log_income"] = np.log1p(df["annual_income"])
    df["log_loan"] = np.log1p(df["loan_amount"])
    df["debt_interest_pressure"] = df["debt_to_income_ratio"] * df["interest_rate"]
    # avoid divide by zero
    df["income_leverage"] = np.log1p(df["annual_income"]) / (np.log1p(df["loan_amount"]) + 1e-9)
    df["interest_income_ratio"] = df["interest_rate"] / (df["annual_income"] + 1e-6)
    df["loan_to_credit_ratio"] = df["loan_amount"] / (df["credit_score"] + 1e-6)
    df["stress_factor"] = df["debt_to_income_ratio"] * df["loan_to_credit_ratio"]
    # employment interactions (ensure employment_status_rank exists)
    df["emp_credit_ratio"] = df["employment_status_rank"] / (df["credit_score"] + 1e-6)
    df["emp_income_power"] = df["employment_status_rank"] * np.log1p(df["annual_income"])
    df["emp_debt_pressure"] = df["employment_status_rank"] * df["debt_to_income_ratio"]
    return df

train = add_features(train)
test  = add_features(test)


# Feature and Target Separation
X = train.drop(columns=["id", "loan_paid_back", "employment_status", "employment_status_norm"])
y = train["loan_paid_back"].astype(int)
X_test = test.drop(columns=["id", "employment_status", "employment_status_norm"], errors="ignore")


# Ensure same column order
X_test = X_test[X.columns]




skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

feature_names = X.columns.tolist()
print("Training features:", len(feature_names))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    model = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1
    )

    model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(200), log_evaluation(0)]
)

    
    # per-fold metrics:
    fold_oof = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, fold_oof)
    print(f" Fold {fold} AUC: {fold_auc:.4f}")

    oof_preds[valid_idx] = fold_oof
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# -------------------------
# CV summary
# -------------------------
cv_auc = roc_auc_score(y, oof_preds)
cv_pr  = average_precision_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")
print(f"Overall CV PR-AUC:  {cv_pr:.4f}")




# Feature Importance
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=feature_importance.head(30), y="feature", x="importance", orient="h")
plt.title("LightGBM Feature Importance (last fold)")
plt.tight_layout()
plt.show()



#SHAP Analysis
print("Running SHAP analysis...")
sample = X.sample(n=min(1000, len(X)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample)



# Save Model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/lgbm_last_fold.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
joblib.dump(emp_map, "artifacts/emp_map.pkl")
print("Saved model + encoders to artifacts/")


# Create Submission
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

