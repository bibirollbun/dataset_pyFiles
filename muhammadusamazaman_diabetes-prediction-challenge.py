# ============================================================
# EDA Notebook Template (Playground Series S5E12)
# Target: diagnosed_diabetes (predict probability)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: seaborn for nicer plots (usually available on Kaggle)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# ----------------------------
# 1) Load Data
# ----------------------------
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test  shape:", test.shape)
print("\nTrain columns:\n", train.columns.tolist())
print("\nTest columns:\n", test.columns.tolist())

# ----------------------------
# 2) Basic info
# ----------------------------
print("\n--- Train info ---")
print(train.info())

print("\n--- Test info ---")
print(test.info())

# ----------------------------
# 3) Identify Target and ID
# ----------------------------
TARGET = "diagnosed_diabetes"
ID_COL = "id" if "id" in train.columns else None

assert TARGET in train.columns, f"Target column '{TARGET}' not found in train."

feature_cols = [c for c in train.columns if c != TARGET]
if ID_COL is not None:
    print("\nDetected ID column:", ID_COL)

X_train = train[feature_cols].copy()
y_train = train[TARGET].copy()
X_test  = test.copy()

# Ensure train/test have same feature columns
missing_in_test = set(X_train.columns) - set(X_test.columns)
missing_in_train = set(X_test.columns) - set(X_train.columns)
print("\nColumns in train but not in test:", missing_in_test)
print("Columns in test but not in train:", missing_in_train)

# ----------------------------
# 4) Missing values
# ----------------------------
def missing_report(df, name="df"):
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        print(f"\n{name}: No missing values ✅")
    else:
        print(f"\n{name}: Missing values found ❗")
        display(miss)

missing_report(train, "train")
missing_report(test, "test")

# ----------------------------
# 5) Duplicate rows check (optional)
# ----------------------------
dup_train = train.duplicated().sum()
dup_test = test.duplicated().sum()
print("\nDuplicate rows:")
print("Train duplicates:", dup_train)
print("Test duplicates :", dup_test)

# ----------------------------
# 6) Separate numeric and categorical
# ----------------------------
num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Some columns may be encoded as int but are categorical (we detect common ones)
# Based on your columns list, these are categorical-like:
likely_cat = ["gender", "ethnicity", "education_level", "income_level", "smoking_status", "employment_status"]
for c in likely_cat:
    if c in X_train.columns and c not in cat_cols:
        cat_cols.append(c)
        if c in num_cols:
            num_cols.remove(c)

# Binary flags (kept separate for quick summaries)
binary_flags = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]
bin_cols = [c for c in binary_flags if c in X_train.columns]

print("\nNumeric columns:", len(num_cols))
print(num_cols)

print("\nCategorical columns:", len(cat_cols))
print(cat_cols)

print("\nBinary flag columns:", len(bin_cols))
print(bin_cols)

# ----------------------------
# 7) Target distribution
# ----------------------------
print("\n--- Target distribution ---")
print(y_train.value_counts())
print("\n--- Target distribution (normalized) ---")
print(y_train.value_counts(normalize=True))

plt.figure()
y_train.value_counts().plot(kind="bar")
plt.title("Target Distribution: diagnosed_diabetes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 8) Quick statistical summary
# ----------------------------
print("\n--- Numeric summary (train) ---")
display(train[num_cols].describe().T)

print("\n--- Numeric summary (test) ---")
display(test[num_cols].describe().T)

# ----------------------------
# 9) Categorical levels + counts
# ----------------------------
print("\n--- Categorical cardinality ---")
for c in cat_cols:
    if c in X_train.columns:
        print(f"{c}: nunique={X_train[c].nunique()}")

# Top categories
for c in cat_cols:
    if c in X_train.columns:
        print(f"\nTop values for {c}:")
        display(X_train[c].value_counts().head(10))

# ----------------------------
# 10) Binary flag rates by target (useful signal check)
# ----------------------------
for c in bin_cols:
    if c in train.columns:
        tab = pd.crosstab(train[c], train[TARGET], normalize="columns")
        print(f"\nBinary flag: {c} (normalized by target class)")
        display(tab)

# ----------------------------
# 11) Correlation with target (numeric only)
# ----------------------------
corr_df = train[num_cols + [TARGET]].corr(numeric_only=True)[TARGET].sort_values(ascending=False)
print("\n--- Correlation with target (numeric) ---")
display(corr_df)

plt.figure(figsize=(6, 8))
corr_df.drop(index=TARGET, errors="ignore").head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Numeric Correlations with Target")
plt.xlabel("Correlation")
plt.show()

# ----------------------------
# 12) Skewness (numeric)
# ----------------------------
skew = train[num_cols].skew(numeric_only=True).sort_values(ascending=False)
print("\n--- Skewness (train numeric) ---")
display(skew.head(20))

# ----------------------------
# 13) Train vs Test distribution drift (quick check)
#     We'll compare a small set of important numeric features.
# ----------------------------
important_num = [
    "age", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day"
]
important_num = [c for c in important_num if c in num_cols]

def plot_train_test_dist(col):
    tr = train[col].dropna()
    te = test[col].dropna()

    plt.figure()
    if _HAS_SNS:
        sns.kdeplot(tr, label="train", common_norm=False)
        sns.kdeplot(te, label="test", common_norm=False)
    else:
        # fallback histogram density
        plt.hist(tr, bins=50, density=True, alpha=0.5, label="train")
        plt.hist(te, bins=50, density=True, alpha=0.5, label="test")

    plt.title(f"Train vs Test Distribution: {col}")
    plt.legend()
    plt.show()

print("\n--- Plotting train vs test distributions for key numeric features ---")
for col in important_num[:10]:
    plot_train_test_dist(col)

# ----------------------------
# 14) Boxplots for outliers (optional small subset)
# ----------------------------
subset_for_box = ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol_total", "triglycerides"]
subset_for_box = [c for c in subset_for_box if c in num_cols]

for col in subset_for_box:
    plt.figure()
    if _HAS_SNS:
        sns.boxplot(x=train[col])
    else:
        plt.boxplot(train[col].dropna(), vert=False)
    plt.title(f"Boxplot (train): {col}")
    plt.show()

# ----------------------------
# 15) Grouped target rate by categories (very useful)
# ----------------------------
for c in cat_cols:
    if c in train.columns:
        grp = train.groupby(c)[TARGET].mean().sort_values(ascending=False)
        print(f"\n--- Target rate by {c} (mean diagnosed_diabetes) ---")
        display(grp.head(20))

        plt.figure(figsize=(8, 4))
        grp.head(15).plot(kind="bar")
        plt.title(f"Top 15 categories by target rate: {c}")
        plt.ylabel("Mean diagnosed_diabetes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# ----------------------------
# 16) Pairwise relationships (optional, limited to a few)
# ----------------------------
# WARNING: Full pairplots are heavy with 700k rows. We'll sample.
SAMPLE_N = 5000
sample_df = train.sample(SAMPLE_N, random_state=42)

pair_cols = ["age", "bmi", "waist_to_hip_ratio", "systolic_bp", "cholesterol_total", TARGET]
pair_cols = [c for c in pair_cols if c in sample_df.columns]

if _HAS_SNS and len(pair_cols) >= 3:
    sns.pairplot(sample_df[pair_cols], hue=TARGET, corner=True, plot_kws={"alpha": 0.3, "s": 10})
    plt.show()
else:
    print("\nSkipping pairplot (either seaborn unavailable or insufficient columns).")

# ----------------------------
# 17) Quick notes for modeling readiness
# ----------------------------
print("\n--- Modeling Readiness Checklist ---")
print("1) Drop ID for training (keep for submission).")
print("2) Encode categoricals carefully (label encoding or CatBoost).")
print("3) Use Stratified CV due to possible class imbalance.")
print("4) Focus on probabilistic metrics (logloss) and calibration later.")



# def add_features(df):
#     df = df.copy()

#     df["bmi_age"] = df["bmi"] * df["age"]
#     df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
#     df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)

#     df["lifestyle_risk"] = (
#         (df["physical_activity_minutes_per_week"] < 150).astype(int) +
#         (df["sleep_hours_per_day"] < 6).astype(int)
#     )

#     return df



# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import log_loss

# from catboost import CatBoostClassifier


# train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
# test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# TARGET = "diagnosed_diabetes"
# ID_COL = "id"



# def add_features_v2(df):
#     df = df.copy()

#     # Existing
#     df["bmi_age"] = df["bmi"] * df["age"]
#     df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
#     df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)

#     # NEW (important)
#     df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
#     df["total_lipid_load"] = df["cholesterol_total"] + df["triglycerides"]

#     df["activity_deficit"] = (df["physical_activity_minutes_per_week"] < 150).astype(int)
#     df["sleep_deficit"] = (df["sleep_hours_per_day"] < 6).astype(int)

#     df["lifestyle_risk"] = (
#         df["activity_deficit"] +
#         df["sleep_deficit"] +
#         df["smoking_status"].isin(["Current", "Former"]).astype(int)
#     )

#     df["metabolic_risk"] = (
#         (df["bmi"] > 30).astype(int) +
#         (df["waist_to_hip_ratio"] > 0.9).astype(int) +
#         (df["triglycerides"] > 150).astype(int)
#     )

#     return df



# train = add_features(train)
# test  = add_features(test)



# X = train.drop(columns=[TARGET])
# y = train[TARGET]

# X_test = test.copy()

# # Drop ID (VERY IMPORTANT)
# X = X.drop(columns=[ID_COL])
# X_test = X_test.drop(columns=[ID_COL])



# cat_cols = [
#     "gender",
#     "ethnicity",
#     "education_level",
#     "income_level",
#     "smoking_status",
#     "employment_status"
# ]

# cat_features = [X.columns.get_loc(col) for col in cat_cols]



# N_SPLITS = 7
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# oof_preds = np.zeros(len(X))
# test_preds = np.zeros(len(X_test))

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"Fold {fold+1}")

#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

#     model = CatBoostClassifier(
#         iterations=800,
#         learning_rate=0.04,
#         depth=9,
#         l2_leaf_reg=6,
#         loss_function="Logloss",
#         eval_metric="Logloss",
#         random_seed=42,
#         verbose=200,
#         cat_features=cat_features,
#         early_stopping_rounds=100
#     )

#     model.fit(
#         X_tr, y_tr,
#         eval_set=(X_val, y_val),
#         use_best_model=True
#     )

#     oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
#     test_preds += model.predict_proba(X_test)[:, 1] / 7


# from sklearn.metrics import log_loss
# print("CV LogLoss:", log_loss(y, oof_preds))



# submission = pd.DataFrame({
#     "id": test["id"],
#     "diagnosed_diabetes": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# submission.head()



# ============================================================
# LIGHTGBM BASELINE (FAST & STRONG)
# Playground Series S5E12
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# ----------------------------
# 1. Load Data
# ----------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"
ID_COL = "id"

# ----------------------------
# 2. Feature Engineering (v2)
# ----------------------------
def add_features_v2(df):
    df = df.copy()

    df["bmi_age"] = df["bmi"] * df["age"]
    df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1)
    df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["total_lipid_load"] = df["cholesterol_total"] + df["triglycerides"]

    df["activity_deficit"] = (df["physical_activity_minutes_per_week"] < 150).astype(int)
    df["sleep_deficit"] = (df["sleep_hours_per_day"] < 6).astype(int)

    df["lifestyle_risk"] = (
        df["activity_deficit"] +
        df["sleep_deficit"] +
        df["smoking_status"].isin(["Current", "Former"]).astype(int)
    )

    df["metabolic_risk"] = (
        (df["bmi"] > 30).astype(int) +
        (df["waist_to_hip_ratio"] > 0.9).astype(int) +
        (df["triglycerides"] > 150).astype(int)
    )

    return df

train = add_features_v2(train)
test  = add_features_v2(test)

# ----------------------------
# 3. Split X / y
# ----------------------------
y = train[TARGET]
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# ----------------------------
# 4. Encode Categorical Features
# ----------------------------
cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# ----------------------------
# 5. LightGBM Dataset
# ----------------------------
train_data = lgb.Dataset(X, label=y)

# ----------------------------
# 6. Model Parameters (Stable & Fast)
# ----------------------------
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,
    "verbosity": -1,
    "seed": 42
}

# ----------------------------
# 7. Train Model (NO CV – FAST)
# ----------------------------
model = lgb.train(
    params,
    train_data,
    num_boost_round=800
)

# ----------------------------
# 8. Predict on Test
# ----------------------------
test_preds = model.predict(X_test)

print("Prediction range:", test_preds.min(), test_preds.max())




# ----------------------------
# 9. Create Submission
# ----------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


