# ===============================
# Maternal Risk Prediction - XGBoost
# Kaggle Notebook (final fixed)
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ===============================
# Load Data
# ===============================
train = pd.read_csv("/kaggle/input/mlolympiadbd2025/train.csv")
test  = pd.read_csv("/kaggle/input/mlolympiadbd2025/test.csv")

# Separate target
y = train["RiskLevel"]

# Drop target + Id from training features
X = train.drop(columns=["RiskLevel", "Id"])

# Drop only Id from test features
X_test = test.drop(columns=["Id"])

# Convert object columns → numeric codes
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category").cat.codes
for col in X_test.columns:
    if X_test[col].dtype == "object":
        X_test[col] = X_test[col].astype("category").cat.codes

# ===============================
# Model Training with CV
# ===============================
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit final model
model.fit(X, y)

# ===============================
# Predict on Test Set
# ===============================
test_preds = model.predict(X_test)

# ===============================
# Save Submission
# ===============================
submission = pd.DataFrame({
    "Id": test["Id"],          # keep original Id
    "RiskLevel": test_preds
})
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv saved!")


