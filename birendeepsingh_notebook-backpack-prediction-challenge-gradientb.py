# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra_train = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Combine original and extra training data
train_combined = pd.concat([train, extra_train], ignore_index=True)

# Separate target and features
X = train_combined.drop(columns=["Price"])
y = train_combined["Price"]

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=["float64", "int64"]).drop(columns=["id"]).columns.tolist()

# Preprocessing for numerical data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

# Preprocessing for categorical data
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Model pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model_pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")

# Predict on test set
test_preds = model_pipeline.predict(test)

# Generate submission
submission = pd.DataFrame({
    "id": test["id"],
    "Price": test_preds
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")


