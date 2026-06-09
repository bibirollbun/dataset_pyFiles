import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import shap 
from scipy.spatial.distance import jensenshannon
import os
import warnings 


# Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
independent = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Independent Shape:', independent.shape)


train.info()
independent.info()


# Map Smoking Status
def normalize_cat(s):
    return (
        s.astype(str)
         .str.lower().str.strip()
         .str.replace("-", " ", regex=False)
         .str.replace("_", " ", regex=False)
    )

smk_map = {"never":0, "former":1, "current":2}

for df in (train, test, independent):
    if "smoking_status" in df.columns:
        df["smk_norm"] = normalize_cat(df["smoking_status"])
        df["smk_rank"] = df["smk_norm"].map(smk_map).fillna(-1).astype(int)
        # Drop smk_norm as it's string/object type that causes issues
        df.drop(columns=["smk_norm"], inplace=True, errors='ignore')
    else:
        df["smk_rank"] = -1





# Define Prediction Target
predict = "diagnosed_diabetes"

# categorical columns must be object dtype in train
cat_cols = train.select_dtypes(include="object").columns.tolist()

# numeric columns must be numeric IN ALL datasets
num_cols = [
    col for col in train.columns
    if col not in [predict, "id"]
    and np.issubdtype(train[col].dtype, np.number)
    and np.issubdtype(test[col].dtype, np.number)
    and np.issubdtype(independent[col].dtype, np.number)
]

# base = cat + num (minus id/target)
base = cat_cols + num_cols


# Handle Missing Values in ALL datasets
datasets = [train, test, independent]

for df in datasets:
    for col in df.columns:
        if col in ["id", "diagnosed_diabetes"]:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(df[col].median())


# Encode Categoricals
encoders = {}

for col in cat_cols:

    # Fit on union of categories from all datasets
    all_values = (
        train[col].astype(str).tolist() +
        test[col].astype(str).tolist() +
        independent[col].astype(str).tolist()
    )

    le = LabelEncoder()
    le.fit(all_values)

    # Transform all datasets
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    independent[col] = le.transform(independent[col].astype(str))

    # Ensure XGBoost-compatible dtype
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)
    independent[col] = independent[col].astype(int)

    encoders[col] = le

print("Categorical encoding completed for train/test/independent.")



# Data Visualisation 
# Creating summary counts for training data
train_summary = train["diagnosed_diabetes"].value_counts().reset_index()
train_summary.columns = ["diagnosed_diabetes", "count"]
train_summary["source"] = "train"

# Creating summary counts for the independent diabetes data
ind_summary = independent["diagnosed_diabetes"].value_counts().reset_index()
ind_summary.columns = ["diagnosed_diabetes", "count"]
ind_summary["source"] = "independent"



# Mapping Diagnosis of diabetes for both datasets
# Data Visualisation 
# Creating summary counts for training data
train_summary = train["diagnosed_diabetes"].value_counts().reset_index()
train_summary.columns = ["diagnosed_diabetes", "count"]
train_summary["source"] = "train"

# Creating summary counts for the independent diabetes data
ind_summary = independent["diagnosed_diabetes"].value_counts().reset_index()
ind_summary.columns = ["diagnosed_diabetes", "count"]
ind_summary["source"] = "independent"



# Mapping Diagnosis of diabetes for both datasets
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.barplot(
    data=train_summary,
    x="source",
    y="count",
    hue="diagnosed_diabetes",
    palette="rainbow",
    ax=axs[0]
)
axs[0].set_title("Train Dataset")
axs[0].set_xlabel("Dataset")
axs[0].set_ylabel("Count")

sns.barplot(
    data=ind_summary,
    x="source",
    y="count",
    hue="diagnosed_diabetes",
    palette="rainbow",
    ax=axs[1]
)
axs[1].set_title("Independent Dataset")
axs[1].set_xlabel("Dataset")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


#Visualisation of all the other features of the training, test and independent dataset as an overlapping gridplot
def jsd(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return jensenshannon(p, q)

results = []

for col in num_cols:
    t = train[col].values
    e = test[col].values
    o = independent[col].values
    
    # hist bins must match
    bins = 40
    hist_train, _ = np.histogram(t, bins=bins, density=True)
    hist_test, _  = np.histogram(e, bins=bins, density=True)
    hist_ind, _   = np.histogram(o, bins=bins, density=True)

    results.append({
        "Feature": col,
        "Train vs Test": jsd(hist_train, hist_test),
        "Train vs Independent": jsd(hist_train, hist_ind),
        "Test vs Independent": jsd(hist_test, hist_ind)
    })

df_jsd = pd.DataFrame(results).set_index("Feature")

plt.figure(figsize=(10, len(num_cols)*0.35))
sns.heatmap(df_jsd, annot=True, cmap="coolwarm_r")
plt.title("Distribution Divergence Heatmap (Jensen–Shannon Divergence)")
plt.show()



# Only from TRAIN 
cat_cols = train.select_dtypes(include="object").columns.tolist()

# Removing smoking_status because we are using smk_rank instead
if "smoking_status" in cat_cols:
    cat_cols.remove("smoking_status")


# Raw Feature Matrix for Baseline SHAP
predict = "diagnosed_diabetes"
y = train[predict].astype(int)
X = train.drop(columns=["id", predict, "smoking_status"], errors="ignore")
X_test = test.reindex(columns=X.columns, fill_value=0)
X_ind = independent.reindex(columns=X.columns, fill_value=0)

# Baseline XGBoost Model
baseline_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

baseline_model.fit(X, y)
print("Baseline model trained for SHAP.")


# Sample for SHAP
train_sample = X.sample(5000, random_state=42)
test_sample  = X_test.sample(5000, random_state=42)
ind_sample   = X_ind.sample(5000, random_state=42)


# SHAP Analysis
print("Independent dtypes before SHAP:")
print(independent.dtypes)

explainer = shap.TreeExplainer(baseline_model)

shap_train = explainer(train_sample)
shap_test  = explainer(test_sample)
shap_ind   = explainer(ind_sample)

print("Generating SHAP summary for TRAIN vs INDEPENDENT...")

shap.summary_plot(shap_train, train_sample, plot_type="dot")
shap.summary_plot(shap_ind,  ind_sample,  plot_type="dot")



def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.divide(a, b)
        return np.where(np.isfinite(r), r, 0.0)

def add_features(df):

    df = df.copy()

    # ======================================================
    # 0. Ensure numeric stability (convert relevant cols)
    # ======================================================
    numeric_cols = [
        "age", "physical_activity_minutes_per_week", "diet_score",
        "sleep_hours_per_day", "screen_time_hours_per_day", "bmi",
        "waist_to_hip_ratio", "systolic_bp", "diastolic_bp",
        "heart_rate", "cholesterol_total", "hdl_cholesterol",
        "ldl_cholesterol", "triglycerides", "family_history_diabetes",
        "hypertension_history", "cardiovascular_history", "smk_rank"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ======================================================
    # 1. **Stable SHAP features** (top-scoring & low drift)
    # ======================================================

    # Physical Activity — top SHAP feature
    pa = df["physical_activity_minutes_per_week"]
    df["pa_log"] = np.log1p(pa.clip(lower=0))
    df["pa_bin"] = pd.cut(pa, bins=[-1, 60, 150, 300, 2000], labels=[0,1,2,3]).astype(int)
    df["pa_inverse"] = 1 / (pa + 1)

    # Family history interactions (extremely stable)
    df["fam_bmi"] = df["family_history_diabetes"] * df["bmi"]
    df["fam_tg"]  = df["family_history_diabetes"] * df["triglycerides"]

    # Diet + Sleep (both low drift)
    df["diet_sleep"] = df["diet_score"] * df["sleep_hours_per_day"]
    df["screen_sleep_ratio"] = safe_div(df["screen_time_hours_per_day"],
                                        df["sleep_hours_per_day"] + 1e-6)

    # ======================================================
    # 2. Blood pressure features (moderate drift → transformations)
    # ======================================================
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map_bp"]         = df["diastolic_bp"] + df["pulse_pressure"] / 3
    df["bp_ratio"]       = safe_div(df["systolic_bp"], df["diastolic_bp"] + 1e-6)
    df["bp_product"]     = df["systolic_bp"] * df["diastolic_bp"]

    # ======================================================
    # 3. Lipid ratios (need transformations due to drift)
    # ======================================================
    df["chol_hdl"] = safe_div(df["cholesterol_total"], df["hdl_cholesterol"] + 1e-6)
    df["ldl_hdl"]  = safe_div(df["ldl_cholesterol"], df["hdl_cholesterol"] + 1e-6)
    df["tg_hdl"]   = safe_div(df["triglycerides"], df["hdl_cholesterol"] + 1e-6)
    df["non_hdl"]  = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["atherogenic_index"] = np.log1p(df["tg_hdl"])

    # ======================================================
    # 4. Anthropometrics (stable / mildly drifting)
    # ======================================================
    df["bmi2"]     = df["bmi"] ** 2
    df["bmi_whr"]  = df["bmi"] * df["waist_to_hip_ratio"]
    df["bmi_age"]  = df["bmi"] * df["age"]
    df["whr_age"]  = df["waist_to_hip_ratio"] * df["age"]

    # ======================================================
    # 5. Physical activity × Health interactions (safe)
    # ======================================================
    df["pa_bmi"]   = pa * df["bmi"]
    df["pa_age"]   = pa * df["age"]
    df["pa_tg"]    = safe_div(pa, df["triglycerides"] + 1e-6)
    df["screen_pa_ratio"] = safe_div(df["screen_time_hours_per_day"], pa + 1e-6)

    # ======================================================
    # 6. Smoking features **only rank interactions**, no original signal
    # (Because smk_rank has catastrophic drift.)
    # ======================================================
    df["smk_bmi"] = df["smk_rank"] * df["bmi"]
    df["smk_age"] = df["smk_rank"] * df["age"]
    df["smk_tg"]  = df["smk_rank"] * df["triglycerides"]

    # ======================================================
    # 7. Medical history × clinical interactions (stable)
    # ======================================================
    df["htn_sysbp"]  = df["hypertension_history"] * df["systolic_bp"]
    df["cardio_tg"]  = df["cardiovascular_history"] * df["triglycerides"]

    # ======================================================
    # 8. Age bins (SHAP confirmed age is strong + stable)
    # ======================================================
    df["age_group"] = pd.cut(df["age"],
                             bins=[-1, 30, 45, 60, 200],
                             labels=[0,1,2,3]).astype(int)

    # ======================================================
    # 9. Clip heavy-tailed features (essential for leaderboard)
    # ======================================================
    clip_ranges = {
        "bmi": (10, 60),
        "waist_to_hip_ratio": (0.5, 1.5),
        "triglycerides": (10, 2000),
        "hdl_cholesterol": (10, 200),
        "ldl_cholesterol": (10, 400),
        "systolic_bp": (80, 220),
        "diastolic_bp": (40, 140),
        "physical_activity_minutes_per_week": (0, 2000)
    }

    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # ======================================================
    # 10. Final cleanup
    # ======================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df



# Feature and Target Separation

# Columns to drop from training
drop_cols = ["id", "diagnosed_diabetes", "smoking_status"]

# X and y
X = train.drop(columns=drop_cols, errors="ignore")
y = train["diagnosed_diabetes"].astype(int)

# Test features
X_test = test.drop(columns=["id", "smoking_status"], errors="ignore")

# Ensure same columns & correct order
X_test = X_test.reindex(columns=X.columns, fill_value=0)



# Stratified K-Folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

feature_names = X.columns.tolist()
print("Training features:", len(feature_names))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n===== Fold {fold} =====")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    xgbm = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.015,
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
        random_state=42 + fold,
        verbosity=0
    )

    # Train
    xgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=False
    )

    # OOF predictions
    oof_preds[valid_idx] = xgbm.predict_proba(X_val)[:, 1]

    # Test predictions
    test_preds += xgbm.predict_proba(X_test)[:, 1] / skf.n_splits



# Evaluate cross-validated performance
cv_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")

# Precision-Recall AUC
cv_pr = average_precision_score(y, oof_preds)
print(f"Overall CV PR-AUC: {cv_pr:.4f}")


# FEATURE IMPORTANCE (last fold model)
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'feature': X.columns,                     
    'importance': xgbm.feature_importances_   
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance, y='feature', x='importance', orient='h')
plt.title('XGBoost Feature Importance (Last Fold)')
plt.tight_layout()
plt.show()


# SHAP ANALYSIS
print("Running SHAP analysis...")

# sample from X_val of LAST fold
sample = X_val.sample(n=min(1000, len(X_val)), random_state=42)

explainer = shap.TreeExplainer(xgbm) 
shap_values = explainer(sample)

shap.summary_plot(shap_values, sample, plot_type="bar")
shap.summary_plot(shap_values, sample)



# Save final fold model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(xgbm, "artifacts/xgb_model.pkl")
joblib.dump(encoders, "artifacts/label_encoders.pkl")
print("XGBoost model and encoders saved to artifacts/")


# Use CV-averaged predictions already computed in test_preds
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds 
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")


