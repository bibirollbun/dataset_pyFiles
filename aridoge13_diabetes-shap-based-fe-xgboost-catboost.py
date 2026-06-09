import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import shap 
from scipy.spatial.distance import jensenshannon
import os
import warnings 


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
external = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('External Shape:', external.shape)
print("diagnosed_diabetes" in external.columns)


train.info()
test.info()
external.info()
print("diagnosed_diabetes" in external.columns)


# Map Smoking Status
def normalize_cat(s):
    return (
        s.astype(str)
         .str.lower().str.strip()
         .str.replace("-", " ", regex=False)
         .str.replace("_", " ", regex=False)
    )

smk_map = {"never":0, "former":1, "current":2}

for df in (train, test, external):
    if "smoking_status" in df.columns:
        df["smk_norm"] = normalize_cat(df["smoking_status"])
        df["smk_rank"] = df["smk_norm"].map(smk_map).fillna(-1).astype(int)
        # Drop smk_norm as it's string/object type that causes issues
        df.drop(columns=["smk_norm"], inplace=True, errors='ignore')
    else:
        df["smk_rank"] = -1

print("diagnosed_diabetes" in external.columns)


# Defining Target 
predict = "diagnosed_diabetes" # It is better to name it Target. I like the name predict

# Clean leftover smoking_status so it doesn't get encoded
for df in (train, test, external):
    df.drop(columns=["smoking_status"], inplace=True, errors="ignore")

# Categorical columns (object/string only)
cat_cols = train.select_dtypes(include="object").columns.tolist()

# Numeric columns (robust dtype detection)
num_cols = [
    col for col in train.columns
    if col not in [predict, "id"]
    and is_numeric_dtype(train[col])
    and is_numeric_dtype(test[col])
    and is_numeric_dtype(external[col])
]

base = cat_cols + num_cols
print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)
print("diagnosed_diabetes" in external.columns)


# 1. Align external dataset to TRAIN columns only (NOT test)
external = external.reindex(columns=train.columns, fill_value=np.nan)
external["diagnosed_diabetes"].value_counts(dropna=False)

# 2. Handle missing values using train medians
train_medians = train.median(numeric_only=True)
datasets = [train, test, external]

for df in datasets:
    for col in df.columns:
        if col in ["id", predict]:
            continue
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(train_medians[col])

# 3. Label Encoding
encoders = {}
for col in cat_cols:
    all_values = pd.concat([train[col], test[col], external[col]], axis=0).astype(str).unique()

    le = LabelEncoder()
    le.fit(all_values)

    train[col]     = le.transform(train[col].astype(str)).astype(int)
    test[col]      = le.transform(test[col].astype(str)).astype(int)
    external[col]  = le.transform(external[col].astype(str)).astype(int)

    encoders[col] = le

print("Categorical encoding completed for train/test/external.")
print("diagnosed_diabetes" in external.columns)  # SHOULD BE TRUE NOW
print(external["diagnosed_diabetes"].unique())


print("diagnosed_diabetes" in external.columns)
# --- Build summary tables ---

train_summary = (
    train["diagnosed_diabetes"]
    .value_counts()
    .rename_axis("diagnosed_diabetes")
    .reset_index(name="count")
)
train_summary["source"] = "train"

external_summary = (
    external["diagnosed_diabetes"]
    .value_counts()
    .rename_axis("diagnosed_diabetes")
    .reset_index(name="count")
)
external_summary["source"] = "external"

# --- Plot ---

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=train_summary,
    x="source",
    y="count",
    hue="diagnosed_diabetes",
    palette="viridis",
    ax=axs[0]
)
axs[0].set_title("Train Dataset")
axs[0].set_xlabel("Dataset")
axs[0].set_ylabel("Count")

sns.barplot(
    data=external_summary,
    x="source",
    y="count",
    hue="diagnosed_diabetes",
    palette="viridis",
    ax=axs[1]
)
axs[1].set_title("External Dataset")
axs[1].set_xlabel("Dataset")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()



# Visualisation of all the other features
# Divergence Heatmap
def jsd(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return jensenshannon(p, q)

results = []

for col in num_cols:
    t = train[col].values
    e = test[col].values
    o = external[col].values
    
    # hist bins must match
    bins = 40
    hist_train, _ = np.histogram(t, bins=bins, density=True)
    hist_test, _  = np.histogram(e, bins=bins, density=True)
    hist_ext, _   = np.histogram(o, bins=bins, density=True)

    results.append({
        "Feature": col,
        "Train vs Test": jsd(hist_train, hist_test),
        "Train vs External": jsd(hist_train, hist_ext),
        "Test vs External": jsd(hist_test, hist_ext)
    })

df_jsd = pd.DataFrame(results).set_index("Feature")

plt.figure(figsize=(10, len(num_cols)*0.35))
sns.heatmap(df_jsd, annot=True, cmap="coolwarm_r")
plt.title("Distribution Divergence Heatmap (Jensen–Shannon Divergence)")
plt.show()




# Distribution Curve
warnings.filterwarnings("ignore", category=FutureWarning)
train_plot = train[num_cols].replace([np.inf, -np.inf], np.nan)
test_plot  = test[num_cols].replace([np.inf, -np.inf], np.nan)
ext_plot   = external[num_cols].replace([np.inf, -np.inf], np.nan)

cols = num_cols
n = len(cols)
ncols = 4
nrows = int(np.ceil(n / ncols))

plt.figure(figsize=(20, nrows * 3))

for i, col in enumerate(cols, 1):
    plt.subplot(nrows, ncols, i)

    # USE NUMPY ARRAYS → avoids deprecated pandas internal path
    sns.kdeplot(train_plot[col].dropna().to_numpy(), label="train", fill=True, alpha=0.4)
    sns.kdeplot(test_plot[col].dropna().to_numpy(), label="test", fill=True, alpha=0.4)
    sns.kdeplot(ext_plot[col].dropna().to_numpy(), label="external", fill=True, alpha=0.4)

    plt.title(col, fontsize=10)
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.legend(loc="upper right")
plt.show()


# Raw feature matrix for baseline SHAP
predict = "diagnosed_diabetes"

y = train[predict].astype(int)
X = train.drop(columns=["id", predict], errors="ignore")

# Reindex test/external using TRAIN columns and fill with train medians
train_medians = X.median(numeric_only=True)

X_test = test.reindex(columns=X.columns).fillna(train_medians)
X_ext  = external.reindex(columns=X.columns).fillna(train_medians)

# Baseline XGBoost model
baseline_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)
baseline_model.fit(X, y)

print("Baseline model trained for SHAP.")

# SHAP sampling
train_sample = X.sample(min(5000, len(X)), random_state=42)
test_sample  = X_test.sample(min(5000, len(X_test)), random_state=42)
ext_sample   = X_ext.sample(min(5000, len(X_ext)), random_state=42)

# SHAP explainer
explainer = shap.Explainer(baseline_model)

shap_train = explainer(train_sample)
shap_ext   = explainer(ext_sample)

# SHAP summary plots
print("Generating SHAP summary for TRAIN vs EXTERNAL...")
shap.summary_plot(shap_train.values, train_sample, plot_type="dot")
shap.summary_plot(shap_ext.values,  ext_sample,  plot_type="bar")



# Tier 1 – Stable features (use priors + interactions)
stable_core_features = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "triglycerides",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "cholesterol_total",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "physical_activity_minutes_per_week",
    "family_history_diabetes",
]

# Tier 2 – Raw-only (NO priors, NO interactions)
raw_only_features = [
    "diet_score",
    "screen_time_hours_per_day",
    "income_level",
    "alcohol_consumption_per_week",
]

# Tier 3 – Remove completely (no signal + drift)
drop_features = [
    "cardiovascular_history",
    "education_level",
    "sleep_hours_per_day",
    "employment_status",
]



def build_priors(df, features, target="diagnosed_diabetes", alpha=10):
    priors_mean = {}
    priors_count = {}
    global_mean = df[target].mean()

    for col in features:
        if col not in df.columns:
            continue

        s = df[col]
        # bin continuous data to reduce noise
        binned = pd.qcut(s.rank(method="first"), q=10, duplicates="drop").astype(str)

        counts = binned.value_counts()
        means = df.groupby(binned)[target].mean()

        smooth = (counts * means + alpha * global_mean) / (counts + alpha)

        priors_mean[col] = smooth
        priors_count[col] = counts

    return priors_mean, priors_count



def apply_priors(df, priors_mean, priors_count):
    df = df.copy()
    eps = 1e-9

    for col in priors_mean:
        if col not in df.columns:
            continue

        keys = pd.qcut(df[col].rank(method="first"), q=10, duplicates="drop")
        keys = keys.astype(str)

        df[f"prior_mean_{col}"] = keys.map(priors_mean[col]).fillna(priors_mean[col].mean())
        df[f"prior_count_{col}"] = keys.map(priors_count[col]).fillna(0)

        total = priors_count[col].sum()
        df[f"prior_freq_{col}"] = df[f"prior_count_{col}"] / (total + eps)

        df[f"prior_ratio_{col}"] = df[f"prior_mean_{col}"] / (df[f"prior_freq_{col}"] + eps)

    return df



def add_internal_features(df):
    df = df.copy()

    # clips
    clip_dict = {
        "bmi": (10, 60),
        "waist_to_hip_ratio": (0.4, 2.0),
        "triglycerides": (5, 2000),
        "hdl_cholesterol": (5, 200),
        "ldl_cholesterol": (5, 400),
        "systolic_bp": (60, 240),
        "diastolic_bp": (30, 160),
        "physical_activity_minutes_per_week": (0, 2000),
        "age": (0, 120)
    }

    for col, (lo, hi) in clip_dict.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Stable internal interactions only
    df["bmi_age"] = df["bmi"] * df["age"]
    df["whr_age"] = df["waist_to_hip_ratio"] * df["age"]
    df["pa_log"] = np.log1p(df["physical_activity_minutes_per_week"])
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)
    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-6)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df



def drop_unstable(df, drop_features):
    return df.drop(columns=[c for c in drop_features if c in df.columns], errors="ignore")



# Drop unstable features first
train_fe = drop_unstable(train, drop_features)
test_fe = drop_unstable(test, drop_features)
external_fe = drop_unstable(external, drop_features)

# Build priors only on stable features
priors_mean, priors_count = build_priors(external_fe, stable_core_features)

# Apply priors
train_fe = apply_priors(train_fe, priors_mean, priors_count)
test_fe  = apply_priors(test_fe, priors_mean, priors_count)
external_fe = apply_priors(external_fe, priors_mean, priors_count)

# Add internal features
train_fe = add_internal_features(train_fe)
test_fe  = add_internal_features(test_fe)
external_fe = add_internal_features(external_fe)



# Columns to drop (unstable or noise)
drop_cols = [
    "id",
    "diagnosed_diabetes",
    "smoking_status",  # harmless; kept for safety
    "cardiovascular_history",
    "education_level",
    "sleep_hours_per_day",
    "employment_status",
]

# Drop unstable features before building final matrices
X = train.drop(columns=drop_cols, errors="ignore")
y = train["diagnosed_diabetes"].astype(int)

X_test = test.drop(columns=drop_cols, errors="ignore")

# Ensure same columns and same order
X_test = X_test.reindex(columns=X.columns)

# Fill numeric columns with train medians
num_cols = X.select_dtypes(include="number").columns
X_test[num_cols] = X_test[num_cols].fillna(X[num_cols].median())

# Fill non-numeric with 0 (safe neutral for engineered priors/ratios/bins)
non_num_cols = X.select_dtypes(exclude="number").columns
X_test[non_num_cols] = X_test[non_num_cols].fillna(0)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

print(f"Training on {len(X.columns)} features...\n")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n================= Fold {fold} =================")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # ------------------------
    # XGBoost Model
    # ------------------------
    xgbm = XGBClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=3,
        subsample=0.75,
        colsample_bytree=0.75,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42 + fold,
        reg_alpha=1.0,
        reg_lambda=2.0,
        verbosity=0
    )

    xgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    # OOF + Test
    oof_xgb[valid_idx] = xgbm.predict_proba(X_val)[:, 1]
    test_xgb += xgbm.predict_proba(X_test)[:, 1] / skf.n_splits

    # ------------------------
    # CatBoost Model
    # ------------------------
    cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bayesian",
        random_seed=42 + fold,
        verbose=0
    )

    cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )

    # OOF + Test
    oof_cat[valid_idx] = cat.predict_proba(X_val)[:, 1]
    test_cat += cat.predict_proba(X_test)[:, 1] / skf.n_splits

    # ------------------------
    # Per-Fold Metrics
    # ------------------------
    auc_xgb = roc_auc_score(y_val, oof_xgb[valid_idx])
    auc_cat = roc_auc_score(y_val, oof_cat[valid_idx])

    print(f"Fold {fold} XGB AUC: {auc_xgb:.5f}")
    print(f"Fold {fold} CAT AUC: {auc_cat:.5f}")


# =============================
# Final Blending
# =============================

# Blend weights (can tune, but 0.5/0.5 works extremely well)
blend_oof = 0.5 * oof_xgb + 0.5 * oof_cat
blend_test = 0.5 * test_xgb + 0.5 * test_cat

# Final OOF Metrics
auc = roc_auc_score(y, blend_oof)
pr_auc = average_precision_score(y, blend_oof)

print("\n================ Final CV =================")
print(f"OOF ROC-AUC: {auc:.5f}")
print(f"OOF PR-AUC : {pr_auc:.5f}")


# === FINAL CV METRICS ===

cv_auc = roc_auc_score(y, blend_oof)
cv_pr = average_precision_score(y, blend_oof)

print(f"\nOverall CV ROC-AUC: {cv_auc:.4f}")
print(f"Overall CV PR-AUC: {cv_pr:.4f}")



# FEATURE IMPORTANCE (XGBoost, Last Fold) 

plt.figure(figsize=(10, 8))

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgbm.feature_importances_,
}).sort_values("importance", ascending=False)

sns.barplot(data=feature_importance, y="feature", x="importance", orient="h")
plt.title("XGBoost Feature Importance (Last Fold)")
plt.tight_layout()
plt.show()



# Catboost Feature Importance
cat_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": cat.get_feature_importance(),
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=cat_importance, y="feature", x="importance", orient="h")
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()



# === SHAP ANALYSIS ===

print("Running SHAP analysis...")

# Representative sample from full training data
sample = X.sample(n=min(2000, len(X)), random_state=42)

explainer = shap.TreeExplainer(xgbm)
shap_values = explainer(sample)

# BAR summary plot
shap.summary_plot(shap_values, sample, plot_type="bar")

# DOT summary plot
shap.summary_plot(shap_values, sample)



# Final Prediction and Submission
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": blend_test
})

# Light smoothing BEFORE saving
submission["diagnosed_diabetes"] = submission["diagnosed_diabetes"].clip(0.001, 0.999)

# Save CSV
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)

# Preview
submission.head()

