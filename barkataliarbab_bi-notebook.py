import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# --- Load train data ---
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

# --- Encode categorical columns ---
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- Features / target ---
X = df.drop(["y","id"], axis=1)
y = df["y"]

# --- Handle class imbalance ---
scale_pos_weight = (len(y) - y.sum()) / y.sum()
print("scale_pos_weight =", scale_pos_weight)

# --- XGBoost model ---
base_model = XGBClassifier(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    eval_metric="logloss",
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
    random_state=42
)

# --- Stratified K-Fold CV ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train base model
    base_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
    
    # Calibrate probabilities
    cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    cal_model.fit(X_val, y_val)
    
    # Store calibrated predictions
    oof_preds[val_idx] = cal_model.predict_proba(X_val)[:,1]

# --- Load test data ---
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# --- Encode categorical columns in test ---
for col in cat_cols:
    if col in test.columns:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].astype(str))

# --- Final predictions on test set ---
final_preds = cal_model.predict_proba(test.drop("id", axis=1))[:,1]

# --- Save submission ---
submission = pd.DataFrame({"id": test["id"], "y": final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")


