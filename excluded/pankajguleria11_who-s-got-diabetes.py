# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

TRAIN_PATH = '/kaggle/input/playground-series-s5e12/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e12/test.csv'
SAMPLE_SUB_PATH = r'/kaggle/input/playground-series-s5e12/sample_submission.csv'

TARGET = "diagnosed_diabetes"


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert TARGET in train_df.columns, f"Target '{TARGET}' not found in train file."

X_raw = train_df.drop(columns=[TARGET])
y = train_df[TARGET].astype(int)
X_test_raw = test_df.copy()


print("==== BASIC INFO ====")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nColumns:", list(train_df.columns))
print("\nTarget distribution:\n", y.value_counts())
print("Target rate:", y.mean())

print("\n==== DTYPES (Train) ====")
print(X_raw.dtypes.value_counts())


# Checking Missingness
missing_train = X_raw.isna().mean().sort_values(ascending=False)
missing_test = X_test_raw.isna().mean().sort_values(ascending=False)

print("\n==== TOP MISSING (Train) ====")
print((missing_train.head(15) * 100).round(2))

print("\n==== TOP MISSING (Test) ====")
print((missing_test.head(15) * 100).round(2))


# Identifying the base column types
base_num_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
base_cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric cols:", len(base_num_cols))
print("Categorical cols:", len(base_cat_cols))


# Plot 1: Target distribution
plt.figure()
y.value_counts().sort_index().plot(kind="bar")
plt.title("Target Distribution (diagnosed_diabetes)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Plot 2: Missingness (top 20)
plt.figure(figsize=(10, 4))
(missing_train.head(20)[::-1] * 100).plot(kind="barh")
plt.title("Top 20 Columns by Missing % (Train)")
plt.xlabel("Missing %")
plt.tight_layout()
plt.show()


# Plot 3: Numeric histograms (up to 6 highest-variance numeric features)
if len(base_num_cols) > 0:
    variances = X_raw[base_num_cols].var(numeric_only=True).sort_values(ascending=False)
    top_hist = variances.head(6).index.tolist()
    for c in top_hist:
        plt.figure()
        X_raw[c].hist(bins=30)
        plt.title(f"Distribution: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# Plot 4: Correlation heatmap (numeric only; if enough numeric cols)
if len(base_num_cols) >= 3:
    corr = train_df[base_num_cols + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(9, 6))
    plt.imshow(corr, aspect="auto")
    plt.title("Correlation Heatmap (Numeric + Target)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# Plot 5: Boxplots of a few key numeric features vs target (top correlated)
if len(base_num_cols) >= 3:
    corr_target = train_df[base_num_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).abs()
    top_box = corr_target.sort_values(ascending=False).head(3).index.tolist()
    for c in top_box:
        plt.figure()
        data0 = train_df.loc[train_df[TARGET] == 0, c].dropna()
        data1 = train_df.loc[train_df[TARGET] == 1, c].dropna()
        plt.boxplot([data0, data1], labels=["No Diabetes (0)", "Diabetes (1)"])
        plt.title(f"{c}: Distribution by Target")
        plt.ylabel(c)
        plt.tight_layout()
        plt.show()


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic feature engineering that is robust to unknown schemas.
    Works for tabular medical-style datasets.
    """
    out = df.copy()

    # Row-level missing information
    out["fe_missing_count"] = out.isna().sum(axis=1)
    out["fe_missing_frac"] = out["fe_missing_count"] / max(out.shape[1], 1)

    # Numeric aggregations
    num_cols = out.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(num_cols) > 0:
        out["fe_num_sum"] = out[num_cols].sum(axis=1, skipna=True)
        out["fe_num_mean"] = out[num_cols].mean(axis=1, skipna=True)
        out["fe_num_std"] = out[num_cols].std(axis=1, skipna=True)
        out["fe_num_min"] = out[num_cols].min(axis=1, skipna=True)
        out["fe_num_max"] = out[num_cols].max(axis=1, skipna=True)

        # Add log1p features for the most skewed numeric columns (up to 5)
        skew_abs = out[num_cols].skew(numeric_only=True).abs().sort_values(ascending=False)
        top_skew = skew_abs.head(5).index.tolist()
        for c in top_skew:
            col = out[c]
            if col.min(skipna=True) < 0:
                shift = -col.min(skipna=True) + 1.0
                out[f"fe_log1p_shift_{c}"] = np.log1p(col + shift)
            else:
                out[f"fe_log1p_{c}"] = np.log1p(col)

    # Simple ratio features if common columns exist
    # (safe divisions)
    def safe_div(a, b):
        return a / np.where(b == 0, np.nan, b)

    cols = set(out.columns.str.lower())
    # BMI already might exist; still add classic ratios if present
    if "systolic_bp" in out.columns and "diastolic_bp" in out.columns:
        out["fe_pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
        out["fe_bp_ratio"] = safe_div(out["systolic_bp"], out["diastolic_bp"])

    if "waist" in cols and "hip" in cols:
        # Only if exact columns exist; otherwise dataset may already contain waist_to_hip_ratio
        pass

    # Categorical frequency encoding (for up to 20 categorical columns)
    cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols[:20]:
        freq = out[c].value_counts(dropna=False)
        out[f"fe_freq_{c}"] = out[c].map(freq).astype("float64")

    return out

X_fe = add_feature_engineering(X_raw)
X_test_fe = add_feature_engineering(X_test_raw)

# Align train/test columns after feature engineering
X_fe, X_test_fe = X_fe.align(X_test_fe, join="left", axis=1)

# Recompute types after FE
num_cols = X_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_fe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\n==== AFTER FEATURE ENGINEERING ====")
print("Train FE shape:", X_fe.shape)
print("Test  FE shape:", X_test_fe.shape)
print("Numeric cols:", len(num_cols))
print("Categorical cols:", len(cat_cols))


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

if HAS_XGB:
    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
else:
    # Solid baseline if xgboost is unavailable
    model = LogisticRegression(max_iter=3000, n_jobs=-1)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


X_train, X_val, y_train, y_val = train_test_split(
    X_fe, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)
val_pred = pipeline.predict_proba(X_val)[:, 1]
holdout_auc = roc_auc_score(y_val, val_pred)
print(f"\nHoldout ROC-AUC: {holdout_auc:.5f}")

# CV AUC (5-fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_fe, y), start=1):
    X_tr, X_va = X_fe.iloc[tr_idx], X_fe.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    pipeline.fit(X_tr, y_tr)
    p = pipeline.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    cv_scores.append(auc)
    print(f"Fold {fold} ROC-AUC: {auc:.5f}")

print(f"CV ROC-AUC mean: {np.mean(cv_scores):.5f} | std: {np.std(cv_scores):.5f}")


if HAS_XGB:
    # Extract feature names after preprocessing
    # Numeric names are kept; categorical become one-hot names
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols) if len(cat_cols) > 0 else np.array([])
    feature_names = np.concatenate([np.array(num_cols), cat_names])

    importances = pipeline.named_steps["model"].feature_importances_
    imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp = imp.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 5))
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


pipeline.fit(X_fe, y)
test_pred = pipeline.predict_proba(X_test_fe)[:, 1]

# Ensure submission has target column
if TARGET not in sample_sub.columns:
    sample_sub[TARGET] = 0.0

sample_sub[TARGET] = test_pred

sample_sub.head()


OUT_PATH = '/kaggle/working/final_submission.csv'
sample_sub.to_csv(OUT_PATH, index=False)

print("\nSaved submission to:", OUT_PATH)
print(sample_sub.head())




