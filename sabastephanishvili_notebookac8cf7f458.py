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


# ======================================================
# 0. Imports & utilities
# ======================================================
print("STEP 0: Imports")
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# ensure plots show in notebooks
plt.ioff()

# ======================================================
# 1. Load data
# ======================================================
print("\nSTEP 1: Load data")
train = pd.read_csv("/kaggle/input/playground-series-s3e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s3e1/test.csv")
print("Train rows:", train.shape[0], "Test rows:", test.shape[0])

# ======================================================
# 2. Train / validation split (80/20)
# ======================================================
print("\nSTEP 2: Train/Validation split (80/20)")
trainset = train.sample(frac=0.8, random_state=42)
valset = train.drop(trainset.index)
print("Trainset:", trainset.shape, "Valset:", valset.shape)

# ======================================================
# 3. Prepare data function (cleaning, FE, 80% row drop on TRAIN only)
# ======================================================
print("\nSTEP 3: Define prepare_data()")
def prepare_data(data, status="Train", info_dct=None):
    """
    Cleans and feature-engineers dataframe.
    - status="Train": learns medians, drops id, drops high-constant rows (>=80% identical)
      and returns info_dct to reuse on Test/Val.
    - status="Test": applies stored info_dct (no row dropping).
    """
    df = data.copy()
    categorical_cols = []  # none in this dataset; kept for structure

    if status == "Train":
        info_dct = {}

        # Drop id column (identifier)
        unnecessary_columns = ["id"]
        df = df.drop(columns=unnecessary_columns, errors="ignore")
        info_dct["unnecessary_columns"] = unnecessary_columns

        # Fill numeric missings with medians
        missing_medians = {}
        num_cols_local = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        for col in num_cols_local:
            median = df[col].median()
            missing_medians[col] = median
            df[col] = df[col].fillna(median)
        info_dct["missing_medians"] = missing_medians

        # Feature engineering (ratios) if columns exist
        fe_cols = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
        if all(col in df.columns for col in fe_cols):
            # guard division by zero by adding tiny epsilon
            eps = 1e-9
            df["rooms_per_household"] = df["AveRooms"] / (df["AveOccup"] + eps)
            df["bedrooms_per_room"] = df["AveBedrms"] / (df["AveRooms"] + eps)
            df["population_per_household"] = df["Population"] / (df["AveOccup"] + eps)

        # Placeholder for dummy columns tracking (if any categorical)
        info_dct["dummy_columns"] = {}

        # 80% mostly-constant row drop (TRAIN only)
        threshold = 0.8
        row_constant_ratio = df.apply(lambda row: row.value_counts().max() / len(row), axis=1)
        rows_dropped = row_constant_ratio >= threshold
        info_dct["constant_row_threshold"] = threshold
        info_dct["num_constant_rows_dropped"] = int(rows_dropped.sum())
        df = df[~rows_dropped].reset_index(drop=True)

    else:  # status == "Test" or "Val"
        # Apply same column drops
        df = df.drop(columns=info_dct["unnecessary_columns"], errors="ignore")
        # Fill missings with stored medians
        for col, median in info_dct["missing_medians"].items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        # Same feature engineering
        fe_cols = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
        if all(col in df.columns for col in fe_cols):
            eps = 1e-9
            df["rooms_per_household"] = df["AveRooms"] / (df["AveOccup"] + eps)
            df["bedrooms_per_room"] = df["AveBedrms"] / (df["AveRooms"] + eps)
            df["population_per_household"] = df["Population"] / (df["AveOccup"] + eps)

    return df, info_dct

# ======================================================
# 4. Separate X/y and apply prepare_data() to train & val
# ======================================================
print("\nSTEP 4: Separate X/y and clean data")
target_col = "MedHouseVal"

X_train_raw = trainset.drop(columns=[target_col])
y_train = trainset[target_col].reset_index(drop=True)

X_val_raw = valset.drop(columns=[target_col])
y_val = valset[target_col].reset_index(drop=True)

X_train, info_dct = prepare_data(X_train_raw, status="Train")
X_val, _ = prepare_data(X_val_raw, status="Test", info_dct=info_dct)

print("After cleaning - X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
print("Rows dropped from training by 80% rule:", info_dct.get("num_constant_rows_dropped", 0))

# ======================================================
# 5. One-hot encoding (get_dummies) and align columns
# ======================================================
print("\nSTEP 5: One-hot encode and align columns")
X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)
X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
print("After dummies - X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

# ======================================================
# 6. Scaling numeric columns
# ======================================================
print("\nSTEP 6: Scale numeric columns")
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
print("Scaled numeric columns:", num_cols)

# ======================================================
# 7. Define evaluation helper
# ======================================================
print("\nSTEP 7: Define evaluation helper")
def evaluate_model(model, X_val_local, y_val_local, model_name="Model"):
    preds = model.predict(X_val_local)
    rmse = mean_squared_error(y_val_local, preds) ** 0.5
    mae = mean_absolute_error(y_val_local, preds)
    r2 = r2_score(y_val_local, preds)
    print(f"\n{model_name} performance:")
    print("  RMSE:", rmse)
    print("  MAE :", mae)
    print("  R²  :", r2)
    return rmse, mae, r2

# ======================================================
# 8. Train baseline & tuned models (Linear, DT tuned, RF tuned)
# ======================================================
print("\nSTEP 8: Train models (Linear, DecisionTree (Grid), RandomForest (Grid))")

# 8a. Linear Regression (baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_val, y_val, "Linear Regression")

# 8b. Decision Tree tuning (medium grid)
dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    "max_depth": [None, 10, 20],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 5, 10],
    "max_features": [None, "sqrt"],
    "ccp_alpha": [0.001, 0.01]
}
grid_dt = GridSearchCV(dt, param_grid_dt, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
print("\nBest Decision Tree params:", grid_dt.best_params_)
evaluate_model(best_dt, X_val, y_val, "Decision Tree (tuned)")

# 8c. Random Forest tuning (medium grid)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
    "bootstrap": [True]
}
grid_rf = GridSearchCV(rf, param_grid_rf, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print("\nBest Random Forest params:", grid_rf.best_params_)
evaluate_model(best_rf, X_val, y_val, "Random Forest (tuned)")

# ======================================================
# 9. Compare models in a table
# ======================================================
print("\nSTEP 9: Comparison table")
lr_rmse, lr_mae, lr_r2 = evaluate_model(lr_model, X_val, y_val, "Linear Regression (again)")
dt_rmse, dt_mae, dt_r2 = evaluate_model(best_dt, X_val, y_val, "Decision Tree (tuned) (again)")
rf_rmse, rf_mae, rf_r2 = evaluate_model(best_rf, X_val, y_val, "Random Forest (tuned) (again)")

results = pd.DataFrame({
    "RMSE": [lr_rmse, dt_rmse, rf_rmse],
    "MAE":  [lr_mae, dt_mae, rf_mae],
    "R²":   [lr_r2, dt_r2, rf_r2]
}, index=["Linear Regression", "Decision Tree", "Random Forest"])

print("\nValidation results:\n", results)

# ======================================================
# 10. Diagnostics: residual plot and feature importance for RF
# ======================================================
print("\nSTEP 10: Diagnostics")
final_model_for_diag = best_rf  # pick tuned RF for diagnostics
residuals = y_val - final_model_for_diag.predict(X_val)
# Residual scatter (not shown inline if running as script, but created)
plt.figure(figsize=(6,4))
plt.scatter(final_model_for_diag.predict(X_val), residuals, alpha=0.3)
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions")

# Feature importances
if hasattr(final_model_for_diag, "feature_importances_"):
    importances = final_model_for_diag.feature_importances_
    fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 feature importances (Random Forest):\n", fi.head(10))

# ======================================================
# 11. Retrain best model on full training data (train+val) and prepare test
# ======================================================
print("\nSTEP 11: Retrain best model on full data and prepare test set for submission")
# Build full training data (use cleaned & processed X_train and X_val)
X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train.reset_index(drop=True), y_val.reset_index(drop=True)], axis=0)

# Refit scaler on X_full numeric columns to be safe (fit on X_train only is also acceptable)
# Here we reuse existing scaler but it's okay—scaler was fit on X_train earlier.
final_model = best_rf  # choose best model (you can switch to best_dt or lr_model)
final_model.fit(X_full, y_full)

# Prepare test set (Kaggle-safe: DO NOT drop or rearrange test rows)
print("Preparing test set — keeping all rows and original id order")
X_test_raw = test.copy()  # keep the original test DataFrame intact
test_ids = X_test_raw["id"].copy()

# Apply same prepare_data cleaning (status="Test")
X_test, _ = prepare_data(X_test_raw, status="Test", info_dct=info_dct)

# One-hot encode and align to X_full's columns (use X_train columns as reference)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale numeric cols
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ======================================================
# 12. Predict on test and save submission (Kaggle-safe)
# ======================================================
print("\nSTEP 12: Predict on test and save submission.csv")
test_preds = final_model.predict(X_test)
test_preds = np.nan_to_num(test_preds)

submission = pd.DataFrame({
    "id": test_ids,
    "target": test_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file created at /kaggle/working/submission.csv — rows:", submission.shape[0])


