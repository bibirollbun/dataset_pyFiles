import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# ==========================
# Data Preprocessing
# ==========================

# Encode categorical column
le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

# Features and target
X = train.drop(columns=["id", "Calories"])
y = train["Calories"]
X_test = test.drop(columns=["id"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train/Validation split for model evaluation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================
# Model Definition
# ==========================

# Base models
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=5, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.07, max_depth=5, random_state=42)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Final estimator
final_estimator = Ridge(alpha=1.0)

# Stacking Regressor
stack = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

# ==========================
# Model Training
# ==========================

stack.fit(X_train, y_train)

# Validation
y_pred = stack.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# ==========================
# Prediction & Submission
# ==========================

final_preds = stack.predict(X_test_scaled)
sample_submission["Calories"] = final_preds
sample_submission.to_csv("submission.csv", index=False)
print("Saved submission to final_submission.csv")


