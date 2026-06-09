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
# Improved Version: Insurance Premium Prediction (S4E12)
# -------------------------

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
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

# Instead of plain log1p, use Quantile transformation for better scaling of skewed target
qt = QuantileTransformer(output_distribution='normal', random_state=42)
y = qt.fit_transform(y_orig.values.reshape(-1, 1)).ravel()

# ---------- 3) Validation split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ---------- 4) Preprocessing ----------
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

# ---------- 5) Improved model ----------
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=8,
        learning_rate=0.03,
        l2_regularization=0.1,
        min_samples_leaf=20,
        random_state=42
    ))
])

# ---------- 6) Train ----------
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ---------- 7) Save ----------
joblib.dump((model, qt), 'histgb_insurance_model_improved.joblib')
print("Model saved as histgb_insurance_model_improved.joblib")

# ---------- 8) Validation predictions ----------
y_val_pred = model.predict(X_val)

# Inverse transform to original scale
y_val_pred_orig = qt.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
y_val_orig = qt.inverse_transform(y_val.reshape(-1, 1)).ravel()

# RMSE
rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig))
print(f"Validation RMSE: {rmse_val:.2f}")

# ---------- 9) Visualizations ----------
plt.figure(figsize=(10,5))
plt.scatter(y_val_orig, y_val_pred_orig, alpha=0.3)
plt.plot([0, max(y_val_orig)], [0, max(y_val_orig)], 'r--', lw=2)
plt.xlabel("Actual Premium Amount")
plt.ylabel("Predicted Premium Amount")
plt.title("Actual vs Predicted Premium Amount (Improved Model)")
plt.show()

plt.figure(figsize=(10,5))
plt.hist(y_val_orig, bins=50, alpha=0.5, label='Actual')
plt.hist(y_val_pred_orig, bins=50, alpha=0.5, label='Predicted')
plt.xlabel("Premium Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Actual vs Predicted Premiums (Improved Model)")
plt.legend()
plt.show()

# ---------- 10) Test predictions ----------
y_test_pred = model.predict(test_df)
y_test_pred_orig = qt.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()

# ---------- 11) Submission ----------
submission = sample_submission.copy()
submission['Premium Amount'] = y_test_pred_orig
submission.to_csv("submission_improved.csv", index=False)
print("✅ submission_improved.csv created! Ready to submit.")

