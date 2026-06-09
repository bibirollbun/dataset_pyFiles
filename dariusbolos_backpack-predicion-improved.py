import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

# Backup of target
y = train["Price"]

# Fill missing numerical
train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())

# Fill and encode categorical features
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", 
                    "Waterproof", "Style", "Color"]

label_encoders = {}
for col in categorical_cols:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")
    
    le = LabelEncoder()
    le.fit(list(train[col]) + list(test[col]))  # fit on both train+test to avoid mismatches
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Features
X = train.drop(columns=["id", "Price"])
X_test = test.drop(columns=["id"])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=False)

# Validation
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)

# Test prediction
test_preds = model.predict(X_test)
submission["Price"] = test_preds
submission.to_csv("improved_submission.csv", index=False)


