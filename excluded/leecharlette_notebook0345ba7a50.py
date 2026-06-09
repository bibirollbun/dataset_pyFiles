import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Data Type Optimization ===
def optimize_dataframe(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

# === Load datasets ===
train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")

train.drop(columns=["timestamp"], inplace=True, errors="ignore")
test.drop(columns=["timestamp"], inplace=True, errors="ignore")

# === Load top features only
top_features = pd.read_csv("/kaggle/input/shapfeature/shap_selected_features.csv")["feature"].tolist()

# Subset
X = train[top_features]
y = train["label"]
test = test[top_features]

# Optimize types
X = optimize_dataframe(X)
test = optimize_dataframe(test)

# === Train-val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === XGBoost training with L1 and L2 regularization
model = xgb.XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=True)

# === Evaluate
y_pred_val = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred_val, squared=False)
r2 = r2_score(y_val, y_pred_val)
print(f"\nðŸ“Š Final Top-Only Model Evaluation")
print(f"âœ… Validation RMSE: {rmse:.5f}")
print(f"âœ… Validation RÂ²:   {r2:.5f}")

# === Predict and Save submission
y_test_pred = model.predict(test)
submission["prediction"] = y_test_pred
submission.to_csv("submission.csv", index=False)
print("ðŸ“¦ Saved: submission_top_only_l1l2.csv")


