
# ==== Setup & Imports ====
import os, gc, math, random, sys, datetime, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

# Try LightGBM/XGBoost; keep graceful fallbacks if missing
LGBM_AVAILABLE = True
XGB_AVAILABLE = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, early_stopping
except Exception as e:
    LGBM_AVAILABLE = False
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGB_AVAILABLE = False

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)

# Paths for Kaggle vs local
KAGGLE_INPUT = Path("/kaggle/input/agriyield-2025")
LOCAL_INPUT = Path("./")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./")

def data_path(filename: str):
    # Prefer Kaggle input path if exists
    kaggle_candidate = KAGGLE_INPUT / filename
    if kaggle_candidate.exists():
        return kaggle_candidate
    # Else default to local path
    return LOCAL_INPUT / filename

print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)
print("LightGBM available:", LGBM_AVAILABLE)
print("XGBoost available:", XGB_AVAILABLE)





# ==== Load Data ====
train_path = data_path("train.csv")
test_path = data_path("test.csv")
sub_path = data_path("sample_submission.csv")

assert train_path.exists(), f"Missing {train_path}. Place train.csv in working dir or Kaggle input."
assert test_path.exists(), f"Missing {test_path}. Place test.csv in working dir or Kaggle input."
assert sub_path.exists(), f"Missing {sub_path}. Place sample_submission.csv accordingly."

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sub_path)

print("Train shape:", train.shape)
print("Test  shape:", test.shape)
display(train.head())
display(test.head())
display(sample_submission.head())

# Try to infer target & id columns
candidate_targets = ["yield","Yield","target","Target","y"]
target_col = None
for c in candidate_targets:
    if c in train.columns:
        target_col = c
        break
assert target_col is not None, f"Could not infer target column from {candidate_targets}. Please set manually."

# Infer ID column from sample_submission (first column usually id, second is target name)
id_col = sample_submission.columns[0]
sub_target_name = sample_submission.columns[1]

print("Inferred target:", target_col)
print("Inferred id:", id_col, "| submission target col:", sub_target_name)




# ==== Quick EDA (lightweight) ====
numeric_cols = [c for c in train.columns if c not in [target_col] and pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in train.columns if c not in [target_col] and not pd.api.types.is_numeric_dtype(train[c])]

print(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")

# Missing stats
miss_train = train.isna().mean().sort_values(ascending=False).head(20)
miss_test = test.isna().mean().sort_values(ascending=False).head(20)

print("Top 20 missing ratios (train):")
display(miss_train.to_frame("missing_ratio").T)
print("Top 20 missing ratios (test):")
display(miss_test.to_frame("missing_ratio").T)

# Target distribution summary
print("Target summary:")
display(train[target_col].describe())

# Simple histogram (matplotlib, single plot, no custom colors)
plt.figure()
train[target_col].hist(bins=50)
plt.title("Target Distribution")
plt.xlabel(target_col)
plt.ylabel("Count")
plt.show()




# ==== Preprocessing ====
# Numeric: impute missing with median
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical: impute then OneHot
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# ==== Base Model Selection ====
def make_lgbm():
    return LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )

def make_xgb():
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist"
    )

def make_rf():
    return RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        random_state=SEED,
        n_jobs=-1
    )

# Try LightGBM -> XGBoost -> RandomForest
BASE_MODEL_NAME = "LightGBM" if LGBM_AVAILABLE else ("XGBoost" if XGB_AVAILABLE else "RandomForest")
if BASE_MODEL_NAME == "LightGBM":
    base_model = make_lgbm()
elif BASE_MODEL_NAME == "XGBoost":
    base_model = make_xgb()
else:
    base_model = make_rf()

print("Using base model:", BASE_MODEL_NAME)

# ==== Build pipeline ====
from sklearn.base import clone
model = Pipeline(steps=[("preprocess", preprocessor),
                       ("regressor", base_model)])





# ==== 5-Fold CV ====
X = train.drop(columns=[target_col])
y = train[target_col].astype(float)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pipe = clone(model)
    if BASE_MODEL_NAME in ["LightGBM", "XGBoost"]:
        # Enable early stopping via fit params accessed from final step
        if BASE_MODEL_NAME == "LightGBM":
            pipe.fit(X_tr, y_tr,
                     regressor__eval_set=[(X_va, y_va)],
                     regressor__eval_metric="rmse",
                     regressor__callbacks=[],
                     regressor__verbose=False)
        else:
            pipe.fit(X_tr, y_tr,
                     regressor__eval_set=[(X_va, y_va)],
                     regressor__eval_metric="rmse",
                     regressor__verbose=False,
                     regressor__early_stopping_rounds=100)
    else:
        pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_va)
    rmse = math.sqrt(mean_squared_error(y_va, pred))
    fold_scores.append(rmse)
    oof[va_idx] = pred

    print(f"Fold {fold}: RMSE = {rmse:.5f}")
    gc.collect()

cv_rmse = math.sqrt(mean_squared_error(y, oof))
print(f"\nOOF RMSE: {cv_rmse:.5f}")
print("Fold RMSEs:", fold_scores)



# ==== Train on Full Data & Predict Test ====
full_model = clone(model)

if BASE_MODEL_NAME in ["LightGBM", "XGBoost"]:
    # Use a validation split for monitoring even in full-fit (optional)
    # Here we just fit directly to all data.
    full_model.fit(X, y)
else:
    full_model.fit(X, y)

test_pred = full_model.predict(test)

# Create submission
submission = sample_submission.copy()
submission[submission.columns[1]] = test_pred
sub_name = f"submission_{BASE_MODEL_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
submission.to_csv(sub_name, index=False)
print("Saved:", sub_name)
display(submission.head())




# ==== Feature Importances (if model supports) ====
def plot_importance(fitted_pipeline, top_n=30):
    # Retrieve names after preprocessing
    pre = fitted_pipeline.named_steps["preprocess"]
    reg = fitted_pipeline.named_steps["regressor"]

    # Get processed feature names
    feat_names = []
    # Numeric
    feat_names.extend(pre.transformers_[0][2])
    # Categorical
    if len(pre.transformers_) > 1 and hasattr(pre.transformers_[1][1].named_steps["onehot"], "get_feature_names_out"):
        ohe = pre.transformers_[1][1].named_steps["onehot"]
        cat_base = pre.transformers_[1][2]
        cat_names = ohe.get_feature_names_out(cat_base).tolist()
        feat_names.extend(cat_names)

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        top_feats = [feat_names[i] for i in idx]
        top_vals = importances[idx]

        plt.figure()
        plt.barh(range(len(top_feats))[::-1], top_vals[::-1])
        plt.yticks(range(len(top_feats))[::-1], top_feats[::-1])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("This regressor does not expose feature_importances_.")

plot_importance(full_model, top_n=30)




# ==== Save Model ====
import joblib

model_path = f"agriyield_{BASE_MODEL_NAME.lower()}_model.pkl"
joblib.dump(full_model, model_path)
print("Saved model to:", model_path)


