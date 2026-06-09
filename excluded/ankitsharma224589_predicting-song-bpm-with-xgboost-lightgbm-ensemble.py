import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV




# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Load saved base models
rf_model = joblib.load("/kaggle/input/saved-model/RandomForest.pkl")
xgb_model = joblib.load("/kaggle/input/saved-model/XGBoost.pkl")
lgb_model = joblib.load("/kaggle/input/saved-model/LightGBM.pkl")


sns.heatmap(train.corr(), annot=False, cmap="coolwarm")
plt.show()



X = train.drop(["id", "BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_val)

print("RMSE:", mean_squared_error(y_val, preds, squared=False))



# Predict on validation set
pred_rf = rf_model.predict(X_val)
pred_xgb = xgb_model.predict(X_val)
pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)

# Compute RMSE
rmse_rf = mean_squared_error(y_val, pred_rf, squared=False)
rmse_xgb = mean_squared_error(y_val, pred_xgb, squared=False)
rmse_lgb = mean_squared_error(y_val, pred_lgb, squared=False)

print(f"RandomForest RMSE: {rmse_rf:.4f}")
print(f"XGBoost RMSE: {rmse_xgb:.4f}")
print(f"LightGBM RMSE: {rmse_lgb:.4f}")



# Stack predictions from base models (now including LightGBM)
stacked_preds = np.column_stack((pred_rf, pred_xgb, pred_lgb))

# Ridge regression with scaling
blend_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0], store_cv_values=True))
])

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(stacked_preds):
    X_tr, X_val_blend = stacked_preds[train_idx], stacked_preds[val_idx]
    y_tr, y_val_blend = y_val.iloc[train_idx], y_val.iloc[val_idx]
    
    blend_model.fit(X_tr, y_tr)
    val_preds = blend_model.predict(X_val_blend)
    rmse = mean_squared_error(y_val_blend, val_preds, squared=False)
    cv_scores.append(rmse)

# Train on full validation set for final blend model
blend_model.fit(stacked_preds, y_val)
final_preds = blend_model.predict(stacked_preds)

print("Stacked Blended RMSE on full validation:", mean_squared_error(y_val, final_preds, squared=False))
print("Blending CV RMSEs:", cv_scores)
print("Average CV RMSE:", np.mean(cv_scores))
print("Selected Ridge alpha:", blend_model.named_steps['ridge'].alpha_)


# Prepare test data
test_X = test.drop(["id"], axis=1)

# Get test predictions from each base model
test_pred_rf = rf_model.predict(test_X)
test_pred_xgb = xgb_model.predict(test_X)
test_pred_lgb = lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration_)

# Stack test predictions
stacked_test_preds = np.column_stack((test_pred_rf, test_pred_xgb, test_pred_lgb))

# Use the learned blending model for final predictions
final_test_preds = blend_model.predict(stacked_test_preds)

# Optional: Clip predictions to realistic BPM range
final_test_preds = np.clip(final_test_preds, 40, 220)

# Save submission
submission["BeatsPerMinute"] = final_test_preds
submission.to_csv("submission.csv", index=False)

print("âœ… Stacked submission with RF + XGB + LGB saved as submission.csv")
print("Prediction range:", final_test_preds.min(), "to", final_test_preds.max())
print("Submission shape:", submission.shape)

# Print learned blending weights
if hasattr(blend_model, "coef_"):
    print("ðŸ“Š Blending Weights:", blend_model.coef_)
elif hasattr(blend_model[-1], "coef_"):
    print("ðŸ“Š Blending Weights:", blend_model[-1].coef_)


