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


# ============================================================
# Bike Sharing Demand — Kaggle (with Graphs)
# Robust loader (.csv or .csv.zip) + EDA plots + CV model pick
# RMSLE objective; writes submission.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, zipfile, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional seaborn for nicer plots
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------------
# Helpers
# ----------------------------
def info(msg=""):
    print(f"[INFO] {msg}")

def read_csv_or_zip(csv_path, zip_path):
    """Read CSV if present; otherwise read from .zip containing the CSV."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_members:
                raise FileNotFoundError(f"No .csv found inside {zip_path}")
            with zf.open(csv_members[0]) as f:
                return pd.read_csv(f)
    else:
        raise FileNotFoundError(f"Neither {csv_path} nor {zip_path} found.")

def find_bike_files(base="/kaggle/input"):
    """
    Find train.csv(.zip), test.csv(.zip), sampleSubmission.csv anywhere under /kaggle/input.
    Returns (train_path, test_path, sample_path)
    """
    train_path = test_path = sample_path = None
    for root, _, files in os.walk(base):
        fl = {f.lower(): f for f in files}
        # train
        if "train.csv" in fl:
            train_path = os.path.join(root, fl["train.csv"])
        elif "train.csv.zip" in fl and train_path is None:
            train_path = os.path.join(root, fl["train.csv.zip"])
        # test
        if "test.csv" in fl:
            test_path = os.path.join(root, fl["test.csv"])
        elif "test.csv.zip" in fl and test_path is None:
            test_path = os.path.join(root, fl["test.csv.zip"])
        # sample
        if "samplesubmission.csv" in fl:
            sample_path = os.path.join(root, fl["samplesubmission.csv"])
        if train_path and test_path and sample_path:
            break

    if not (train_path and test_path and sample_path):
        info("Could not auto-find all files. Listing /kaggle/input for debugging:")
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if os.path.isdir(p):
                    print(" -", p, "->", os.listdir(p)[:10])
        except:
            pass
        raise FileNotFoundError(
            "Attach the **Bike Sharing Demand** dataset via 'Add data'. "
            "Expected train.csv(.zip), test.csv(.zip), sampleSubmission.csv"
        )

    return train_path, test_path, sample_path

# ----------------------------
# 1) Load data (robust)
# ----------------------------
TRAIN_PATH, TEST_PATH, SAMPLE_PATH = find_bike_files("/kaggle/input")
info(f"Found:\n  TRAIN:  {TRAIN_PATH}\n  TEST:   {TEST_PATH}\n  SAMPLE: {SAMPLE_PATH}")

train = read_csv_or_zip(TRAIN_PATH[:-4], TRAIN_PATH) if TRAIN_PATH.lower().endswith(".zip") else pd.read_csv(TRAIN_PATH)
test  = read_csv_or_zip(TEST_PATH[:-4],  TEST_PATH)  if TEST_PATH.lower().endswith(".zip")  else pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH)

info(f"Shapes -> train: {train.shape}, test: {test.shape}")
assert "datetime" in train.columns and "count" in train.columns, "Train must have 'datetime' and 'count'"

# ----------------------------
# 2) Utility: RMSLE scorer
# ----------------------------
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0)))**2))
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# ----------------------------
# 3) Feature engineering
# ----------------------------
def add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_interactions(df):
    df = df.copy()
    df["temp_atemp_diff"] = df["temp"] - df["atemp"]
    df["temp_humidity"] = df["temp"] * df["humidity"]
    return df

train_fe = add_interactions(add_time_features(train))
test_fe  = add_interactions(add_time_features(test))

# Remove leakage columns (casual + registered make up count)
for c in ["casual", "registered"]:
    if c in train_fe.columns:
        train_fe = train_fe.drop(columns=c)

# ----------------------------
# 4) EDA GRAPHS
# ----------------------------
# A) Target distribution
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.histplot(train_fe["count"], bins=50, kde=False)
else:
    plt.hist(train_fe["count"], bins=50)
plt.title("Target Distribution: count")
plt.xlabel("count"); plt.ylabel("frequency")
plt.tight_layout(); plt.show()

# B) Daily trend (mean count per day)
daily = train_fe.copy()
daily["date"] = pd.to_datetime(daily["datetime"]).dt.date
daily_mean = daily.groupby("date")["count"].mean()
plt.figure(figsize=(12,4))
plt.plot(daily_mean.index, daily_mean.values)
plt.title("Daily Mean Count Over Time"); plt.xlabel("date"); plt.ylabel("mean count")
plt.tight_layout(); plt.show()

# C) Hourly pattern
hourly = train_fe.groupby("hour")["count"].mean()
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.barplot(x=hourly.index, y=hourly.values)
else:
    plt.bar(hourly.index, hourly.values)
plt.title("Average Count by Hour"); plt.xlabel("hour"); plt.ylabel("avg count")
plt.tight_layout(); plt.show()

# D) By month and weekday
month_avg = train_fe.groupby("month")["count"].mean()
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.barplot(x=month_avg.index, y=month_avg.values)
else:
    plt.bar(month_avg.index, month_avg.values)
plt.title("Average Count by Month"); plt.xlabel("month"); plt.ylabel("avg count")
plt.tight_layout(); plt.show()

dow_avg = train_fe.groupby("dayofweek")["count"].mean()
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.barplot(x=dow_avg.index, y=dow_avg.values)
else:
    plt.bar(dow_avg.index, dow_avg.values)
plt.title("Average Count by Day of Week (0=Mon)"); plt.xlabel("dayofweek"); plt.ylabel("avg count")
plt.tight_layout(); plt.show()

# E) Numeric feature histograms
num_cols_quick = ["temp","atemp","humidity","windspeed"]
num_cols_quick = [c for c in num_cols_quick if c in train_fe.columns]
if num_cols_quick:
    train_fe[num_cols_quick].hist(figsize=(10,6), bins=30)
    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout(); plt.show()

# F) Correlation heatmap (numeric only)
num_only = train_fe.select_dtypes(include=[np.number])
if num_only.shape[1] >= 2:
    corr = num_only.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plt.figure(figsize=(10,8))
    if HAS_SNS:
        sns.heatmap(corr, cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap (numeric)")
    plt.tight_layout(); plt.show()

# ----------------------------
# 5) Prepare ML matrices
# ----------------------------
target = "count"
y = train_fe[target].values
X = train_fe.drop(columns=[target])

# Categorical-like columns
cat_cols = [c for c in ["season","holiday","workingday","weather","year","month","hour","dayofweek","is_weekend"] if c in X.columns]
# Numerical columns
num_cols = [c for c in X.columns if c not in cat_cols and c != "datetime"]

# Drop raw datetime (we already extracted parts)
if "datetime" in X.columns: X = X.drop(columns=["datetime"])
if "datetime" in test_fe.columns: test_fe = test_fe.drop(columns=["datetime"])

numeric_pipe = SimpleImputer(strategy="median")
categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

# ----------------------------
# 6) Models (log-target wrapper)
# ----------------------------
class LogTargetWrapper:
    """Train on log1p(y); predict in original space via expm1; clip >= 0."""
    def __init__(self, base_model):
        self.base = base_model
        self._fitted = False
    def fit(self, X, y):
        y_log = np.log1p(np.maximum(y, 0))
        self.base.fit(X, y_log)
        self._fitted = True
        return self
    def predict(self, X):
        y_log_pred = self.base.predict(X)
        y_pred = np.expm1(y_log_pred)
        return np.clip(y_pred, 0, None)
    def get_params(self, deep=True): return {"base_model": self.base}
    def set_params(self, **params):
        if "base_model" in params:
            self.base = params.pop("base_model")
        for k, v in params.items():
            setattr(self.base, k, v)
        return self

ridge = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogTargetWrapper(Ridge(alpha=2.0, random_state=42)))
])

rf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogTargetWrapper(RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=42, n_jobs=-1,
        min_samples_split=4, min_samples_leaf=1
    )))
])

models = {"Ridge": ridge, "RandomForest": rf}

if HAS_XGB:
    xgb = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogTargetWrapper(XGBRegressor(
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )))
    ])
    models["XGB"] = xgb

# ----------------------------
# 7) Cross-validation (RMSLE) + CV bar chart
# ----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_means = {}
cv_stds = {}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X, y, cv=kf, scoring=rmsle_scorer, n_jobs=-1)
    # scores are negative (because greater_is_better=False); invert for readability
    means = -scores.mean(); stds = scores.std()
    cv_means[name], cv_stds[name] = means, stds
    print(f"{name} RMSLE (5-fold): mean={means:.5f} ± {stds:.5f}")

# CV score bar plot (lower is better)
plt.figure(figsize=(6,4))
plt.bar(cv_means.keys(), cv_means.values())
plt.title("CV RMSLE by Model (lower is better)")
plt.ylabel("RMSLE")
plt.tight_layout(); plt.show()

best_name = min(cv_means, key=cv_means.get)
print(f"\nBest by CV RMSLE: {best_name} (mean={cv_means[best_name]:.5f})")
best_model = models[best_name]

# ----------------------------
# 8) Holdout split for permutation importance
# ----------------------------
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_tr, y_tr)
# Permutation importance on validation set (works for any model)
try:
    result = permutation_importance(best_model, X_va, y_va, n_repeats=10, random_state=42, n_jobs=-1, scoring=None)
    importances = result.importances_mean
    # Get feature names after preprocessor
    prep = best_model.named_steps["prep"]
    num_names = list(num_cols)
    cat_names = []
    if "cat" in prep.named_transformers_:
        ohe = prep.named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = num_names + cat_names
    # Align lengths
    n = min(len(importances), len(feature_names))
    fi = pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
    fi = fi.sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(8,6))
    if HAS_SNS:
        sns.barplot(data=fi, x="importance", y="feature")
    else:
        plt.barh(fi["feature"][::-1], fi["importance"][::-1]); plt.gca().invert_yaxis()
    plt.title(f"Permutation Importance (Top 20) — {best_name}")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Permutation importance skipped:", e)

# ----------------------------
# 9) Fit best on FULL training, predict test, save submission
# ----------------------------
best_model.fit(X, y)
test_X = test_fe.copy()
# The pipeline handles selecting/encoding columns
preds = best_model.predict(test_X)
preds = np.round(np.clip(preds, 0, None)).astype(int)

submission = sample_sub.copy()  # expected columns: datetime,count
cols_lower = [c.lower() for c in submission.columns]
if not ("datetime" in cols_lower and "count" in cols_lower):
    raise ValueError(f"sampleSubmission must have columns ['datetime','count'], got: {list(submission.columns)}")
dt_col = submission.columns[cols_lower.index("datetime")]
cnt_col = submission.columns[cols_lower.index("count")]
submission[cnt_col] = preds
submission.to_csv("submission.csv", index=False)
info("Saved: submission.csv")
print(submission.head())


