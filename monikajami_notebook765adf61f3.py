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


# -------------------------
# Kaggle: Playground Series S4E12 - Insurance Premium Prediction (with Visualization)
# -------------------------

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ---------- 1) Load datasets ----------
train_df = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e12/sample_submission.csv")

TARGET = 'Premium Amount'

print("Train shape:", train_df.shape)
print("Test shape: ", test_df.shape)

# ---------- 2) Prepare X and y ----------
X = train_df.drop(columns=[TARGET])
y_orig = train_df[TARGET]

# Log-transform target for skewed distribution
y = np.log1p(y_orig)

# Optional: create a small validation split for visualization
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ---------- 3) Preprocessing ----------
numeric_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
                    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation',
                        'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status',
                        'Exercise Frequency', 'Property Type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ---------- 4) Build pipeline with HistGradientBoosting ----------
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    ))
])

# ---------- 5) Train the model ----------
model.fit(X_train, y_train)
print("Model training completed!")

# ---------- 6) Save the model ----------
joblib.dump(model, 'histgb_insurance_model.joblib')
print("Model saved as histgb_insurance_model.joblib")

# ---------- 7) Predict on validation set ----------
y_val_pred = model.predict(X_val)
y_val_pred_orig = np.expm1(y_val_pred)
y_val_orig = np.expm1(y_val)

# Calculate RMSE
rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig))
print(f"Validation RMSE: {rmse_val:.2f}")

# ---------- 8) Visualizations ----------
plt.figure(figsize=(10,5))
plt.scatter(y_val_orig, y_val_pred_orig, alpha=0.3)
plt.plot([0, max(y_val_orig)], [0, max(y_val_orig)], 'r--', lw=2)  # perfect prediction line
plt.xlabel("Actual Premium Amount")
plt.ylabel("Predicted Premium Amount")
plt.title("Actual vs Predicted Premium Amount (Validation Set)")
plt.show()

plt.figure(figsize=(10,5))
plt.hist(y_val_orig, bins=50, alpha=0.5, label='Actual')
plt.hist(y_val_pred_orig, bins=50, alpha=0.5, label='Predicted')
plt.xlabel("Premium Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Actual vs Predicted Premiums")
plt.legend()
plt.show()

# ---------- 9) Predict on test data ----------
y_test_pred = model.predict(test_df)
y_test_pred_orig = np.expm1(y_test_pred)  # convert back from log scale

# ---------- 10) Create submission ----------
submission = sample_submission.copy()
submission['Premium Amount'] = y_test_pred_orig
submission.to_csv("submission.csv", index=False)
print("submission.csv created! Ready to submit.")


