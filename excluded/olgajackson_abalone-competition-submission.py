# Load Python packages

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


# =========================
# Kaggle: Abalone Regression (Playground S4E4)
# Two models + diagnostics + 3 submission files
# =========================
import os, sys, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Try XGBoost if present in Kaggle image (it usually is)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False



# -------------------------
# Add data from the competion then print to verify the correct files are loaded
# -------------------------
DATA_DIR = "/kaggle/input/playground-series-s4e4"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("\nColumns:", list(train.columns))
print(train.head())

# -------------------------
# Quick sanity checks
# -------------------------
print("\nMissing values in train:")
print(train.isna().sum())


# Separate Target from features and drop the row identifier
TARGET = "Rings"
ID_COL = "id"
y = train[TARGET].copy()
X = train.drop([TARGET, ID_COL], axis=1)
X_test = test.drop([ID_COL], axis=1)

# Identify col types for pre-processing, detect which ones need one-hot encoding and which ones are already numeric.
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)


# -------------------------
# Define Metric for Cross-Valudation (CV): RMSLE scorer per competition instructions
# -------------------------
def rmsle(y_true, y_pred):
    # ensure no negatives; clip tiny values to 0
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)#lower is better


# -------------------------
# Preprocessing
# - OneHotEncode "Sex" as a feature to be encoded
# - Scale numeric (for linear model stability)
# -------------------------
preprocess_for_linear = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ],
    remainder="drop"
)

preprocess_for_tree = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ],
    remainder="drop"
)


# -------------------------
# Model A: Ridge (linear) with log1p target
# -------------------------
# Handle multicollinearity
ridge = Ridge(alpha=3.0, random_state=42)
# Ensures the same preprocessing is applied in cross-validationand the test set
ridge_pipe = Pipeline(steps=[
    ("prep", preprocess_for_linear),
    ("reg", ridge)
])
#regularized linear model train on log1p(Rings), invert prediction with expm1
ridge_ttr = TransformedTargetRegressor(
    regressor=ridge_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

# Cross-Validation with 5 folds (no negative predictions - can't have negative number of rings)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
ridge_cv = cross_val_score(ridge_ttr, X, y, cv=cv, scoring=rmsle_scorer, n_jobs=-1)
print("\nModel A (Ridge) 5-fold CV RMSLE (mean ± std):",
      f"{-ridge_cv.mean():.5f} ± {ridge_cv.std():.5f}")

# Fit on full training
ridge_ttr.fit(X, y)
ridge_pred_test = ridge_ttr.predict(X_test)
ridge_pred_test = np.maximum(0, ridge_pred_test)


# -------------------------
# Model B: XGBoost (or GradientBoosting fallback)
# -------------------------
# Non-linear tree-based model to a model perfomance on non-linear relationships

if HAS_XGB:
    tree_model = XGBRegressor(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist"
    )
else:
    tree_model = GradientBoostingRegressor(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

tree_pipe = Pipeline(steps=[
    ("prep", preprocess_for_tree),
    ("reg", tree_model)
])

tree_cv = cross_val_score(tree_pipe, X, y, cv=cv, scoring=rmsle_scorer, n_jobs=-1)
print("Model B (Tree)  5-fold CV RMSLE (mean ± std):",
      f"{-tree_cv.mean():.5f} ± {tree_cv.std():.5f}")

tree_pipe.fit(X, y)
tree_pred_test = tree_pipe.predict(X_test)
tree_pred_test = np.maximum(0, tree_pred_test)


# -------------------------
# Ensemble (avg of both models)
# -------------------------
ensemble_pred = (ridge_pred_test + tree_pred_test) / 2.0
ensemble_pred = np.maximum(0, ensemble_pred)
print(ensemble_pred)


# -------------------------
# Model interpretation
# -------------------------
# 1) Extract Standardized Ridge coefficients 
ridge_pipe_fitted = ridge_ttr.regressor_  # Pipeline inside TTR
ohe = ridge_pipe_fitted.named_steps["prep"].named_transformers_["cat"]#scaling
scaler = ridge_pipe_fitted.named_steps["prep"].named_transformers_["num"]
feature_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols

coef = ridge_pipe_fitted.named_steps["reg"].coef_.ravel()
coef_df = pd.DataFrame({"feature": feature_names, "ridge_coef": coef})
coef_df = coef_df.sort_values("ridge_coef", ascending=False)
print("\nTop + coefficients (Ridge):")
print(coef_df.head(10).to_string(index=False))
print("\nTop − coefficients (Ridge):")
print(coef_df.tail(10).to_string(index=False))


# 2) Check for feature importance for tree model (if available)
try:
    importances = tree_pipe.named_steps["reg"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    print("\nTop features (Tree importance):")
    print(imp_df.sort_values("importance", ascending=False).head(12).to_string(index=False))
except Exception:
    pass


# -------------------------
# Assumption checks for linear model (on log scale)
# -------------------------
# Inspect residuals directly on log-scale
X_tr, X_va, y_tr, y_va = train_test_split(X, np.log1p(y), test_size=0.2, random_state=42)
lin_log_pipe = Pipeline([("prep", preprocess_for_linear), ("reg", Ridge(alpha=3.0, random_state=42))])
lin_log_pipe.fit(X_tr, y_tr)
y_va_pred_log = lin_log_pipe.predict(X_va)
resid = y_va - y_va_pred_log

# Residual diagnostics 
print("\nResidual diagnostics (validation, log-scale target):")
print(f"Mean resid: {resid.mean():.6f}")
print(f"Std  resid: {resid.std():.6f}")

# Normality (skew/kurtosis)
from scipy.stats import skew, kurtosis
print(f"Skew(resid): {skew(resid):.4f}")
print(f"Kurt(resid): {kurtosis(resid):.4f}")

# Homoscedasticity: Breusch–Pagan test
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
Xv = lin_log_pipe.named_steps["prep"].transform(X_va)
Xv_sm = sm.add_constant(pd.DataFrame(Xv))
bp_stat, bp_p, _, _ = het_breuschpagan(resid, Xv_sm)
print(f"Breusch–Pagan p-value: {bp_p:.6f}  (p<0.05 suggests heteroscedasticity)")

# Multicollinearity (VIF) on numeric-only features (for simplicity)
from statsmodels.stats.outliers_influence import variance_inflation_factor
num_mat = StandardScaler().fit_transform(train[num_cols])
vif = [variance_inflation_factor(num_mat, i) for i in range(num_mat.shape[1])]
vif_df = pd.DataFrame({"feature": num_cols, "VIF": vif}).sort_values("VIF", ascending=False)
print("\nVIF (numeric features):")
print(vif_df.to_string(index=False))


# -------------------------
# Save submissions
# -------------------------
def make_submit(preds, filename):
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: preds})
    sub.to_csv(filename, index=False)
    print("Wrote:", filename)

os.makedirs("/kaggle/working", exist_ok=True)
make_submit(ridge_pred_test,    "/kaggle/working/submission_ridge.csv")
make_submit(tree_pred_test,     "/kaggle/working/submission_tree.csv")
make_submit(ensemble_pred,      "/kaggle/working/submission_ensemble.csv")


print("\nAll done. Use 'Save Version' → 'Save & Submit' to send one of the CSVs above.")

