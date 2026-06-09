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


!pip install pygam


# ============================================================
# Abalone Competition: Spline Regression + GAM Submission
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s, f
from functools import reduce

# -----------------------------
# 2. LOAD DATA
# -----------------------------
# Adjust paths if running locally or in Kaggle
train_path = "/kaggle/input/playground-series-s4e4/train.csv"
test_path = "/kaggle/input/playground-series-s4e4/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

y = train["Rings"]
X = train.drop(columns=["Rings"])
X_test = test.copy()

# -----------------------------
# 3. PREPROCESSING
# -----------------------------
# Identify numeric and categorical columns
numeric_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

# Standardize numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numeric_cols])
X_test_num = scaler.transform(X_test[numeric_cols])

# One-hot encode categorical features
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
if len(cat_cols) > 0:
    X_cat = encoder.fit_transform(X[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
else:
    X_cat = np.empty((X.shape[0], 0))
    X_test_cat = np.empty((X_test.shape[0], 0))

# Combine numeric + categorical
X_processed = np.hstack([X_num, X_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# -----------------------------
# 4. MODEL A: SPLINE REGRESSION
# -----------------------------
# Transform numeric features with splines
spline = SplineTransformer(degree=3, n_knots=5, include_bias=False)
X_spline = spline.fit_transform(X_num)
X_test_spline = spline.transform(X_test_num)

# Combine spline numeric + categorical features
X_spline_full = np.hstack([X_spline, X_cat])
X_test_spline_full = np.hstack([X_test_spline, X_test_cat])

# Fit linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_spline_full, y)

# Predict on test set
y_pred_spline = lin_reg.predict(X_test_spline_full)
y_pred_spline = np.round(np.clip(y_pred_spline, a_min=1, a_max=None)).astype(int)

# -----------------------------
# 5. MODEL B: GAM
# -----------------------------
# Build GAM terms: splines for numeric, linear for categorical
gam_terms = []
for i in range(X_processed.shape[1]):
    if i < len(numeric_cols):
        gam_terms.append(s(i))
    else:
        gam_terms.append(f(i))

# Fit GAM with gridsearch - this takes time to run
gam = LinearGAM(reduce(lambda a, b: a + b, gam_terms)).gridsearch(X_processed, y, progress=False)

# Predict on test set and clip the negative number for Kaggle Submission
y_pred_gam = gam.predict(X_test_processed)
y_pred_gam = np.round(np.clip(y_pred_gam, a_min=1, a_max=None)).astype(int)

# -----------------------------
# 6. CREATE SUBMISSIONS
# -----------------------------
# Try loading sample_submission.csv, to fix a previous error "sample_submission - file is not found"
try:
    sample_sub = pd.read_csv("/kaggle/input/playground-series-s4e4/sample_submission.csv")
except FileNotFoundError:
    # Create minimal submission using test IDs
    if "ID" in X_test.columns:
        sample_sub = pd.DataFrame({"ID": X_test["ID"]})
    else:
        # If no ID column, create row numbers
        sample_sub = pd.DataFrame({"ID": np.arange(len(X_test))})

# Spline Regression submission
submission_spline = sample_sub.copy()
submission_spline["Rings"] = y_pred_spline
submission_spline.to_csv("submission_spline.csv", index=False)
print("Spline regression submission saved as 'submission_spline.csv'")

# GAM submission
submission_gam = sample_sub.copy()
submission_gam["Rings"] = y_pred_gam
submission_gam.to_csv("submission_gam.csv", index=False)
print("GAM submission saved as 'submission_gam.csv'")

# -----------------------------
# 7. Print
# -----------------------------
print("All submissions created successfully.")



# -----------------------------
# 8. Model Comparison
# -----------------------------
from sklearn.metrics import mean_squared_error, r2_score

# Spline Regression
y_pred_train_spline = lin_reg.predict(X_spline_full)
rmse_spline = np.sqrt(mean_squared_error(y, y_pred_train_spline))
r2_spline = r2_score(y, y_pred_train_spline)
print(f"Spline Regression RMSE: {rmse_spline:.3f}, R²: {r2_spline:.3f}")

# GAM
y_pred_train_gam = gam.predict(X_processed)
rmse_gam = np.sqrt(mean_squared_error(y, y_pred_train_gam))
r2_gam = r2_score(y, y_pred_train_gam)
print(f"GAM RMSE: {rmse_gam:.3f}, R²: {r2_gam:.3f}")


