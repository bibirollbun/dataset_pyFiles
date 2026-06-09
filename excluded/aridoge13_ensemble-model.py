import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import os
import joblib
import xgboost as xgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# Load Data 
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Define Categoricals
cat_cols = ["gender", "marital_status", "education_level", "loan_purpose", "grade_subgrade"]

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
    "self employed": 3,
    "employed": 4
}

train["employment_status_norm"] = normalize_cat_series(train["employment_status"])
test["employment_status_norm"]  = normalize_cat_series(test["employment_status"])

train["employment_status_rank"] = train["employment_status_norm"].map(emp_map).fillna(-1).astype(float)
test["employment_status_rank"]  = test["employment_status_norm"].map(emp_map).fillna(-1).astype(float)

# Encode Categoricals
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = test[col].astype(str).map(lambda s: s if s in le.classes_ else "Unknown")
    if "Unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "Unknown")
    test[col] = le.transform(test[col])
    encoders[col] = le

# Feature Engineering
print("Features engineered:")
def add_features(df):
    df["monthly_income"] = df["annual_income"] / 12.0
    df["income_to_loan_ratio"] = 1.0 / (df["debt_to_income_ratio"].replace(0, np.nan))
    df["income_to_loan_ratio"] = df["income_to_loan_ratio"].fillna(0.0)
    df["credit_to_dti"] = df["credit_score"] * np.log1p(df["debt_to_income_ratio"])
    df["monthly_loan_payment"] = (df["loan_amount"] / 12.0) + ((df["loan_amount"] * df["interest_rate"]) / 12.0)
    df["credit_income_power"] = df["credit_score"] * np.log1p(df["annual_income"])
    df["log_income"] = np.log1p(df["annual_income"])
    df["log_loan"] = np.log1p(df["loan_amount"])
    df["debt_interest_pressure"] = df["debt_to_income_ratio"] * df["interest_rate"]
    df["income_leverage"] = np.log1p(df["annual_income"]) / (np.log1p(df["loan_amount"]) + 1e-9)
    df["interest_income_ratio"] = df["interest_rate"] / (df["annual_income"] + 1e-6)
    df["loan_to_credit_ratio"] = df["loan_amount"] / (df["credit_score"] + 1e-6)
    df["stress_factor"] = df["debt_to_income_ratio"] * df["loan_to_credit_ratio"]
    df["emp_credit_ratio"] = df["employment_status_rank"] / (df["credit_score"] + 1e-6)
    df["emp_income_power"] = df["employment_status_rank"] * np.log1p(df["annual_income"])
    df["emp_debt_pressure"] = df["employment_status_rank"] * df["debt_to_income_ratio"]
    return df

train = add_features(train)
test = add_features(test)


# ============================
# ADVANCED FEATURE ENGINEERING
# ============================

# ===== Target Encoding (Safe, KFold, No Leakage) =====
def target_encode(train, test, col, target="loan_paid_back", n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train[f"{col}_te"] = 0
    test_out = []

    for tr_idx, val_idx in kf.split(train):
        tr, val = train.iloc[tr_idx], train.iloc[val_idx]
        means = tr.groupby(col)[target].mean()
        train.loc[train.index[val_idx], f"{col}_te"] = val[col].map(means)

    full_means = train.groupby(col)[target].mean()
    test[f"{col}_te"] = test[col].map(full_means)

    global_mean = train[target].mean()
    train[f"{col}_te"].fillna(global_mean, inplace=True)
    test[f"{col}_te"].fillna(global_mean, inplace=True)

    return train, test

# apply target encoding
for col in ["grade_subgrade", "loan_purpose"]:
    train, test = target_encode(train, test, col)


# ===== Group Statistics Based on Employment Status =====
emp_grp = train.groupby("employment_status_norm")

credit_mean = emp_grp["credit_score"].mean()
credit_std  = emp_grp["credit_score"].std()
dti_mean    = emp_grp["debt_to_income_ratio"].mean()
inc_mean    = emp_grp["annual_income"].mean()

def emp_stat(row, dic):
    return dic.get(row["employment_status_norm"], np.nan)

train["credit_minus_emp_mean"] = train.apply(lambda r: r["credit_score"] - emp_stat(r, credit_mean), axis=1)
test["credit_minus_emp_mean"]  = test.apply(lambda r: r["credit_score"] - emp_stat(r, credit_mean), axis=1)

train["credit_emp_z"] = train.apply(lambda r: (r["credit_score"] - emp_stat(r, credit_mean)) / 
                                                 (credit_std.get(r["employment_status_norm"], 1)), axis=1)
test["credit_emp_z"]  = test.apply(lambda r: (r["credit_score"] - emp_stat(r, credit_mean)) /
                                                 (credit_std.get(r["employment_status_norm"], 1)), axis=1)

train["dti_over_emp_mean"] = train["debt_to_income_ratio"] / train["employment_status_norm"].map(dti_mean)
test["dti_over_emp_mean"]  = test["debt_to_income_ratio"] / test["employment_status_norm"].map(dti_mean)

train["income_over_emp_mean"] = train["annual_income"] / train["employment_status_norm"].map(inc_mean)
test["income_over_emp_mean"]  = test["annual_income"] / test["employment_status_norm"].map(inc_mean)


# ===== Interaction Features =====
train["emp_x_credit"] = train["employment_status_rank"] * train["credit_score"]
test["emp_x_credit"]  = test["employment_status_rank"] * test["credit_score"]

train["emp_x_dti_inv"] = train["employment_status_rank"] * (1 / (train["debt_to_income_ratio"] + 1e-6))
test["emp_x_dti_inv"]  = test["employment_status_rank"] * (1 / (test["debt_to_income_ratio"] + 1e-6))

train["credit_x_dti"] = train["credit_score"] * train["debt_to_income_ratio"]
test["credit_x_dti"]  = test["credit_score"] * test["debt_to_income_ratio"]

train["interest_x_dti"] = train["interest_rate"] * train["debt_to_income_ratio"]
test["interest_x_dti"]  = test["interest_rate"] * test["debt_to_income_ratio"]

train["loan_x_income"] = train["loan_amount"] / (train["annual_income"] + 1e-6)
test["loan_x_income"]  = test["loan_amount"] / (test["annual_income"] + 1e-6)


# ===== Ratio Features =====
train["fico_to_income"] = train["credit_score"] / (train["annual_income"] + 1e-6)
test["fico_to_income"]  = test["credit_score"] / (test["annual_income"] + 1e-6)

train["loan_to_fico"] = train["loan_amount"] / (train["credit_score"] + 1e-6)
test["loan_to_fico"]  = test["loan_amount"] / (test["credit_score"] + 1e-6)

train["rate_to_fico"] = train["interest_rate"] / (train["credit_score"] + 1e-6)
test["rate_to_fico"]  = test["interest_rate"] / (test["credit_score"] + 1e-6)

train["rate_to_income"] = train["interest_rate"] / (np.log1p(train["annual_income"]) + 1e-6)
test["rate_to_income"]  = test["interest_rate"] / (np.log1p(test["annual_income"]) + 1e-6)


# ===== Outlier Flags =====
train["low_fico_flag"] = (train["credit_score"] < 580).astype(int)
test["low_fico_flag"]  = (test["credit_score"] < 580).astype(int)

train["high_dti_flag"] = (train["debt_to_income_ratio"] > 0.6).astype(int)
test["high_dti_flag"]  = (test["debt_to_income_ratio"] > 0.6).astype(int)

train["high_interest_flag"] = (train["interest_rate"] > train["interest_rate"].median()).astype(int)
test["high_interest_flag"]  = (test["interest_rate"] > test["interest_rate"].median()).astype(int)


# ===== KMeans Cluster Feature =====
cluster_features = ["credit_score", "debt_to_income_ratio", "annual_income"]
kmeans = KMeans(n_clusters=6, random_state=42)

train["cluster6"] = kmeans.fit_predict(train[cluster_features])
test["cluster6"]  = kmeans.predict(test[cluster_features])

# ============================
# END ADVANCED FEATURE ENGINEERING
# ============================



# Feature and Target Separation
X = train.drop(columns=["id", "loan_paid_back", "employment_status", "employment_status_norm"])
y = train["loan_paid_back"].astype(int)
X_test = test.drop(columns=["id", "employment_status", "employment_status_norm"], errors="ignore")
X_test = X_test[X.columns]

# Model weights
rf_weight = 0.2
xgb_weight = 0.4
lgb_weight = 0.4

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_oof = rf.predict_proba(X_val)[:, 1]
    rf_test = rf.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=4,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        use_label_encoder=False
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=200
    )
    xgb_oof = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]

    # LightGBM
    lgb_model = LGBMClassifier(
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
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(200), log_evaluation(0)]
    )
    lgb_oof = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test = lgb_model.predict_proba(X_test)[:, 1]

    # Blend predictions
    blended_oof = (
        rf_weight * rf_oof +
        xgb_weight * xgb_oof +
        lgb_weight * lgb_oof
    )

    blended_test = (
        rf_weight * rf_test +
        xgb_weight * xgb_test +
        lgb_weight * lgb_test
    )

    fold_auc = roc_auc_score(y_val, blended_oof)
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

    oof_preds[valid_idx] = blended_oof
    test_preds += blended_test / skf.n_splits

# CV summary
cv_auc = roc_auc_score(y, oof_preds)
cv_pr = average_precision_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")
print(f"Overall CV PR-AUC:  {cv_pr:.4f}")

# ===== RETRAIN FINAL ENSEMBLE ON FULL DATASET =====
print("\n===== Retraining final ensemble on full dataset =====")

rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X, y)

xgb_final = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=4,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    tree_method="hist",
    use_label_encoder=False
)
xgb_final.fit(X, y)

lgb_final = LGBMClassifier(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
lgb_final.fit(X, y)

# Create Ensemble Wrapper Class
class EnsembleClassifier:
    def __init__(self, rf, xgb, lgb, weights=(0.2, 0.4, 0.4)):
        self.rf = rf
        self.xgb = xgb
        self.lgb = lgb
        self.rf_weight, self.xgb_weight, self.lgb_weight = weights

    def predict_proba(self, X):
        rf_pred = self.rf.predict_proba(X)[:, 1]
        xgb_pred = self.xgb.predict_proba(X)[:, 1]
        lgb_pred = self.lgb.predict_proba(X)[:, 1]
        
        ensemble_pred = (
            self.rf_weight * rf_pred +
            self.xgb_weight * xgb_pred +
            self.lgb_weight * lgb_pred
        )
        return ensemble_pred

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

# Create and save ensemble
ensemble_model = EnsembleClassifier(rf_final, xgb_final, lgb_final)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(ensemble_model, "artifacts/ensemble_model.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
joblib.dump(emp_map, "artifacts/emp_map.pkl")
print("Saved ensemble model to artifacts/ensemble_model.pkl")

# Feature Importance (from LightGBM)
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_final.feature_importances_
}).sort_values("importance", ascending=False)

sns.barplot(data=feature_importance.head(30), y="feature", x="importance", orient="h")
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png")
plt.show()

# SHAP Analysis
print("Running SHAP analysis...")
sample = X.sample(n=min(1000, len(X)), random_state=42)
explainer = shap.TreeExplainer(lgb_final)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample, plot_type = "bar")
shap.summary_plot(shap_values, sample)
plt.savefig("artifacts/shap_summary.png")
plt.show()

# Create Submission
final_preds = ensemble_model.predict_proba(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": final_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

