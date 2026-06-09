# ================================================================
# Diabetes Prediction Challenge - Full Ensemble
# Models: Logistic + RandomForest + LightGBM + XGBoost (+ CatBoost)
# ================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Try to import CatBoost (optional)
try:
    from catboost import CatBoostClassifier
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False
    print("CatBoost not installed, skipping CatBoost model.")

# --------------------------
# 1️⃣ Load data
# --------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

TARGET = "diagnosed_diabetes"
ID_COL = "id"

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]

print("Train shape:", X.shape, " Test shape:", X_test.shape)

# --------------------------
# 2️⃣ Split numeric / categorical
# --------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# --------------------------
# 3️⃣ Preprocessor for models that need numeric input
# --------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Fit preprocessor on full training data
X_enc = preprocessor.fit_transform(X)
X_test_enc = preprocessor.transform(X_test)

# --------------------------
# 4️⃣ Define models
# --------------------------

# 4.1 Logistic Regression
log_clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    solver="lbfgs"
)

# 4.2 Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# 4.3 XGBoost
xgb_clf = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

# 4.4 LightGBM
lgb_clf = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1
)

# 4.5 CatBoost (optional, works directly on raw X with categorical features)
if USE_CATBOOST:
    cat_features_idx = [X.columns.get_loc(col) for col in categorical_features]
    cb_clf = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=0
    )

# --------------------------
# 5️⃣ Train models
# --------------------------
print("Training Logistic Regression...")
log_clf.fit(X_enc, y)

print("Training RandomForest...")
rf_clf.fit(X_enc, y)

print("Training XGBoost...")
xgb_clf.fit(X_enc, y)

print("Training LightGBM...")
lgb_clf.fit(X_enc, y)

if USE_CATBOOST:
    print("Training CatBoost...")
    cb_clf.fit(X, y, cat_features=cat_features_idx)

# --------------------------
# 6️⃣ Predict probabilities on test set
# --------------------------
pred_log = log_clf.predict_proba(X_test_enc)[:, 1]
pred_rf  = rf_clf.predict_proba(X_test_enc)[:, 1]
pred_xgb = xgb_clf.predict_proba(X_test_enc)[:, 1]
pred_lgb = lgb_clf.predict_proba(X_test_enc)[:, 1]

if USE_CATBOOST:
    pred_cb  = cb_clf.predict_proba(X_test)[:, 1]

# --------------------------
# 7️⃣ Blend / Ensemble
# --------------------------
if USE_CATBOOST:
    # Weighted average (tweak weights if you like)
    preds_ensemble = (
        0.15 * pred_log +
        0.20 * pred_rf  +
        0.30 * pred_lgb +
        0.35 * pred_xgb
        # You can add CatBoost too, for example:
        # 0.25 * pred_cb and adjust others to sum to 1
    )
else:
    preds_ensemble = (
        0.20 * pred_log +
        0.25 * pred_rf  +
        0.30 * pred_lgb +
        0.25 * pred_xgb
    )

# Clip just in case
preds_ensemble = np.clip(preds_ensemble, 0.000001, 0.999999)

# --------------------------
# 8️⃣ Build submission
# --------------------------
submission = sample.copy()                     # keeps IDs & column names correct
submission[TARGET] = preds_ensemble

print(submission.head())
print("Submission shape:", submission.shape)

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


