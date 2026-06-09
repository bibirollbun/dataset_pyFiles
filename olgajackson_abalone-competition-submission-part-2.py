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


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Load competition data from Kaggle directory (S4E4 Abalone dataset)
train_path = "/kaggle/input/playground-series-s4e4/train.csv"
test_path = "/kaggle/input/playground-series-s4e4/test.csv"
submission_path = "/kaggle/input/playground-series-s4e4/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)


# Features & target for training dataset
X = train.drop(columns=["id", "Rings"])  # Drop ID + target from the features
y = train["Rings"] #target

# Features for test dataset
X_test = test.drop(columns=["id"])  # Test features only

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns between train and test
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# Model 1: Regularization (Ridge & Lasso)
# =========================================
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_valid)
ridge_rmse = mean_squared_error(y_valid, ridge_preds, squared=False)

lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_valid)
lasso_rmse = mean_squared_error(y_valid, lasso_preds, squared=False)

print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Lasso RMSE: {lasso_rmse:.4f}")


# ================================================
# Model 2: Principal Components Regression (PCR)
# ================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Keep enough components to explain 95% variance
pca = PCA(0.95, random_state=42)
X_train_pca = pca.fit_transform(X_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

linreg = LinearRegression()
linreg.fit(X_train_pca, y_train)
pcr_preds = linreg.predict(X_valid_pca)
pcr_rmse = mean_squared_error(y_valid, pcr_preds, squared=False)

print(f"PCR RMSE: {pcr_rmse:.4f}")


# ======================================
# Choose best model & predict test set
# ======================================

final_model = ridge
final_model.fit(X, y)  # Retrain on all data

final_preds = final_model.predict(X_test)

# Clip to avoid negative predictions
final_preds = np.clip(final_preds, a_min=0, a_max=None)

# Create submission to KAggle Competition
submission["Rings"] = final_preds
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")



import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# =====================================================
# Assumption Checks for Ridge as the final model
# =====================================================

# Fit model on full training data
ridge_final = Ridge(alpha=1.0, random_state=42)
ridge_final.fit(X, y)

y_pred = ridge_final.predict(X)
residuals = y - y_pred

# 1. Linearity: observed vs predicted
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=y, alpha=0.4)
plt.xlabel("Predicted Rings")
plt.ylabel("Actual Rings")
plt.title("Linearity Check: Predicted vs Actual")
plt.show()




# 2. Normality of residuals
import warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")


plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals Distribution")
plt.show()

sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()

shapiro_test = stats.shapiro(residuals[:5000])  # sample if large dataset
print("Shapiro-Wilk Test for Normality:", shapiro_test)


# 3. Homoscedasticity: residuals vs fitted
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Homoscedasticity Check")
plt.show()


# 4. Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Numeric data is used for VIF
X_vif = X.select_dtypes(include=[np.number]).copy()

# Add constant for intercept
X_vif = sm.add_constant(X_vif)

# Compute VIFs
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                   for i in range(X_vif.shape[1])]

# Drop the constant row for clarity
vif_data = vif_data[vif_data["Feature"] != "const"]

print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values(by="VIF", ascending=False).head(10))


