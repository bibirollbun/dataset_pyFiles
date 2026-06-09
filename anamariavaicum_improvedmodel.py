import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")



# Fill missing weight capacity
weight_cap = train["Weight Capacity (kg)"].median()
train["Weight Capacity (kg)"].fillna(weight_cap, inplace=True)
test["Weight Capacity (kg)"].fillna(weight_cap, inplace=True)

# Extract numeric size values
train["Size_Num"] = pd.to_numeric(train["Size"].str.extract(r"(\d+)")[0], errors="coerce")
test["Size_Num"] = pd.to_numeric(test["Size"].str.extract(r"(\d+)")[0], errors="coerce")
size_median = train["Size_Num"].median()
train["Size_Num"].fillna(size_median, inplace=True)
test["Size_Num"].fillna(size_median, inplace=True)

# Brand-based aggregated features
train["Brand_MeanPrice"] = train["Brand"].map(train.groupby("Brand")["Price"].mean())
train["Brand_Count"] = train["Brand"].map(train["Brand"].value_counts())
test["Brand_MeanPrice"] = test["Brand"].map(train.groupby("Brand")["Price"].mean()).fillna(train["Price"].mean())
test["Brand_Count"] = test["Brand"].map(train["Brand"].value_counts()).fillna(1)

# Log transform of weight
train["Weight_Log"] = np.log1p(train["Weight Capacity (kg)"].clip(lower=0))
test["Weight_Log"] = np.log1p(test["Weight Capacity (kg)"].clip(lower=0))

# Encode boolean-like columns
for col in ["Laptop Compartment", "Waterproof"]:
    train[col] = train[col].map({"Yes": 1, "No": 0})
    test[col] = test[col].map({"Yes": 1, "No": 0})



num_cols = ["Weight Capacity (kg)", "Weight_Log", "Size_Num", "Brand_MeanPrice", 
            "Brand_Count", "Laptop Compartment", "Waterproof", "Compartments"]

cat_cols = ["Brand", "Material", "Size", "Style", "Color"]

# Remove any constant/empty columns if they exist
num_cols = [col for col in num_cols if not train[col].isna().all()]
cat_cols = [col for col in cat_cols if not train[col].isna().all()]

# Pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])



X = train.drop(columns=["id", "Price"])
y = np.log1p(train["Price"])  # Log transform the target
X_test = test.drop(columns=["id"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)



lgb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", lgbm.LGBMRegressor(
        learning_rate=0.04,
        n_estimators=250,
        num_leaves=32,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=101
    ))
])

print("Training LightGBM...")
lgb_pipeline.fit(X_train, y_train)
lgb_val_pred = np.expm1(lgb_pipeline.predict(X_val))
lgb_rmse = mean_squared_error(np.expm1(y_val), lgb_val_pred, squared=False)
print(f"LightGBM RMSE: {lgb_rmse:.4f}")



# Manual numerical imputation for Random Forest
from sklearn.impute import SimpleImputer

X_train_num = X_train[num_cols].copy()
X_val_num = X_val[num_cols].copy()
X_test_num = X_test[num_cols].copy()

imputer = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(imputer.fit_transform(X_train_num), columns=num_cols)
X_val_num = pd.DataFrame(imputer.transform(X_val_num), columns=num_cols)
X_test_num = pd.DataFrame(imputer.transform(X_test_num), columns=num_cols)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=101)
rf_model.fit(X_train_num, y_train)
rf_val_pred = np.expm1(rf_model.predict(X_val_num))
rf_rmse = mean_squared_error(np.expm1(y_val), rf_val_pred, squared=False)
print(f"Random Forest RMSE: {rf_rmse:.4f}")



lgb_preds = np.expm1(lgb_pipeline.predict(X_test))
rf_preds = np.expm1(rf_model.predict(X_test_num))

# Dynamic ensemble weighting
if lgb_rmse < rf_rmse:
    final_preds = 0.75 * lgb_preds + 0.25 * rf_preds
    print("Weighted more on LightGBM")
else:
    final_preds = 0.5 * lgb_preds + 0.5 * rf_preds
    print("Equal weighting")

submission["Price"] = final_preds
submission.to_csv("improved_model_v2_submission.csv", index=False)
print("Submission saved: improved_model_v2_submission.csv")


