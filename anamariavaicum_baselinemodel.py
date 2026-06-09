import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


# Impute missing numeric values with median
weight_imputer = SimpleImputer(strategy='median')
train_df["Weight Capacity (kg)"] = weight_imputer.fit_transform(train_df[["Weight Capacity (kg)"]])
test_df["Weight Capacity (kg)"] = weight_imputer.transform(test_df[["Weight Capacity (kg)"]])

# Fill missing categorical values with a placeholder
cat_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
train_df[cat_features] = train_df[cat_features].fillna("None")
test_df[cat_features] = test_df[cat_features].fillna("None")


encoders = {}
for col in cat_features:
    encoder = LabelEncoder()
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])
    encoders[col] = encoder


X_train_all = train_df.drop(columns=["id", "Price"])
y_train_all = train_df["Price"]
X_test_final = test_df.drop(columns=["id"])

# Create a validation split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=1)


rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=None,
    random_state=1
)
rf.fit(X_train, y_train)


y_valid_preds = rf.predict(X_valid)
val_rmse = mean_squared_error(y_valid, y_valid_preds, squared=False)
print(f"Validation RMSE: {val_rmse:.3f}")


test_preds = rf.predict(X_test_final)
submission_df["Price"] = test_preds
submission_df.to_csv("rf_baseline_alternative.csv", index=False)
print("Submission file saved: rf_baseline_alternative.csv")

