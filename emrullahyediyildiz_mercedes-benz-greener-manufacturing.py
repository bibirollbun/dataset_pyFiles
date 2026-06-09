import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor




train = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip")
test  = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip")
sample = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip")

print("âœ… Train shape:", train.shape)
print("âœ… Test shape :", test.shape)


# Show first rows
train.head()


# ======================
# Quick EDA
# ======================

print("Dataset shape:", train.shape)
print("\nInfo:")
print(train.info())
print("\nStatistical summary:")
print(train.describe())

# Missing values check
missing = train.isnull().sum()
print("\nMissing values:", missing[missing > 0])

# Find categorical columns
cat_cols = [col for col in train.columns if train[col].dtype == "object"]
print("\nCategorical columns:", cat_cols)

# Print unique values for each categorical column
for col in cat_cols:
    print(f"\n{col} categories: {train[col].unique()}")



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(train["y"], kde=True, bins=50, color="steelblue")
plt.title("Target Variable Distribution (y)", fontsize=14)
plt.xlabel("y", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()



# Example: First categorical column (X0)
cat_col = "X0"

plt.figure(figsize=(10,5))
sns.countplot(data=train, x=cat_col, order=train[cat_col].value_counts().index, palette="viridis")
plt.title(f"Category Distribution: {cat_col}", fontsize=14)
plt.xlabel(cat_col, fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()



# ID droppen
train = train.drop('ID', axis=1)

# One-hot encoding
X = pd.get_dummies(train.drop('y', axis=1))
y = train['y']

# train test 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Model
gbr = GradientBoostingRegressor(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9
)
gbr.fit(X_train, y_train)

# Validation
y_pred = gbr.predict(X_val)
r2  = r2_score(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)

print(f"R2:   {r2:.4f}")
print(f"RMSE: {rmse:.4f}")



joblib.dump(gbr, 'gb_model.pkl')
joblib.dump(X.columns, 'columns.pkl')
print('âœ… gb_model.pkl ve columns.pkl has been saved.')


# Rebuild X and X_test (one-hot + align) to guarantee X_test exists
X = pd.get_dummies(train.drop(columns=["y"]))
y = train["y"]

X_test = pd.get_dummies(test.drop(columns=["ID"]))
X_test, _ = X_test.align(X, join="right", axis=1, fill_value=0)

# (optional) if you need a split again:
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



gbr.fit(X, y)                   
test_preds = gbr.predict(X_test)  

submission = pd.DataFrame({"ID": test["ID"], "y": test_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv created:", submission.shape)


sub = pd.read_csv("submission.csv")
print(sub.shape)
print(sub.head())



# Ensure you already have: y_val (Series) and y_pred (np.array) from your model

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1) True vs. Predicted ---
plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred, alpha=0.4)
mn, mx = np.min([y_val.min(), y_pred.min()]), np.max([y_val.max(), y_pred.max()])
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)  # ideal line
plt.xlabel("True values (y)")
plt.ylabel("Predicted values")
plt.title("True vs. Predicted")
plt.tight_layout()
plt.show()

# --- 2) Residual Plot ---
residuals = y_val - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Predicted values")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# --- Column types ---
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# --- Preprocessing ---
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# --- Model ---
gbr = GradientBoostingRegressor(random_state=42)

# --- Pipeline ---
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", gbr)
])

# --- Fit ---
pipe.fit(X_train, y_train)

# --- Save full pipeline ---
joblib.dump(pipe, "gb_pipeline.pkl")
print("âœ… Saved pipeline as gb_pipeline.pkl")


