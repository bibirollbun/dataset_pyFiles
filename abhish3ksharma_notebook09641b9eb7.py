# =================== FULL PIPELINE (Single Kaggle Cell) ===================

# Cell 0 - Setup
# !pip install lightgbm scikit-learn --quiet

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# LightGBM
try:
    import lightgbm as lgb
except Exception as e:
    print("LightGBM not installed:", e)
    lgb = None

print("Libraries loaded. LightGBM available:", bool(lgb))

# =================== Cell 1 - Load Data ===================
TRAIN_PATH = "/kaggle/input/carnival-risk-analytics-challenge/train.csv"
TEST_PATH = "/kaggle/input/carnival-risk-analytics-challenge/test.csv"
SUB_PATH = "/kaggle/input/carnival-risk-analytics-challenge/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SUB_PATH)

print("Train shape:", train.shape, "| Test shape:", test.shape)

# =================== Cell 2 - Detect TARGET & ID ===================
# Force column names to match submission format
TARGET = "Premium Amount"
ID_COL = "id"

print("Using TARGET:", TARGET)
print("Using ID_COL:", ID_COL)

# =================== Cell 3 - Preprocessor ===================
def build_preprocessor(X, cat_threshold=50):
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if TARGET in num_cols:
        num_cols.remove(TARGET)

    low_card_cols = [c for c in cat_cols if X[c].nunique() <= cat_threshold]
    high_card_cols = [c for c in cat_cols if X[c].nunique() > cat_threshold]

    print(f"Numerical: {len(num_cols)}, Low-card cat: {len(low_card_cols)}, High-card cat: {len(high_card_cols)}")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_low_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    cat_high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if low_card_cols: transformers.append(("cat_low", cat_low_pipe, low_card_cols))
    if high_card_cols: transformers.append(("cat_high", cat_high_pipe, high_card_cols))

    return ColumnTransformer(transformers, remainder="drop")

# =================== Cell 4 - Target Transform + Split ===================
y_original = train[TARGET].values
y_log = np.log1p(y_original)  # log(1 + y)

X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
preprocessor = build_preprocessor(X)

X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# =================== Cell 5 - LightGBM Model ===================
if lgb:
    model = Pipeline([
        ("preproc", preprocessor),
        ("model", lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    print("\nTraining LightGBM...")
    model.fit(X_train, y_train)

    # Validation
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    y_val_original = np.expm1(y_val)

    rmse_original = mean_squared_error(y_val_original, val_preds, squared=False)
    rmse_log = mean_squared_error(y_val, val_preds_log, squared=False)

    print(f"\nLightGBM RMSE (Original Scale): {rmse_original:.5f}")
    print(f"LightGBM RMSE (Log Scale): {rmse_log:.5f}")

else:
    raise ImportError("LightGBM not found.")

# =================== Cell 6 - Train on Full Data & Predict Test ===================
print("\nRetraining on full dataset for final predictions...")
full_X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
full_y = np.log1p(train[TARGET].values)
model.fit(full_X, full_y)

# Predict test set
test_X = test.drop(columns=[ID_COL], errors="ignore")
test_preds_log = model.predict(test_X)
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0  # Clamp negatives

# =================== Cell 7 - Save Submission ===================
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_preds
})

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)

print("\n✅ Submission file saved to:", output_path)
print(submission.head())
print(f"\nPrediction stats → Min: {submission[TARGET].min():.4f}, Max: {submission[TARGET].max():.4f}, Mean: {submission[TARGET].mean():.4f}")


