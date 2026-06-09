# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load data
sample_submission = pd.read_csv("/kaggle/input/playground-series-s4e4/sample_submission.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s4e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e4/test.csv")


# Data preprocessing
def preprocess_data(data):
    # Encode categorical variable 'Sex' (M=1, F=2, I=0)
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 2, 'I': 0})
    # Fill missing values if any (none in this dataset)
    data.fillna(0, inplace=True)
    return data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)


# Separate features and target variable for training
X = train_data.drop(columns=["id", "Rings"])
y = train_data["Rings"]


# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the features for regularization and PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print(X_train_scaled)


# Model 1: Regularization with Lasso
lasso_model = Lasso(alpha=0.1) # Alpha is the regularization parameter
lasso_model.fit(X_train_scaled, y_train)


# Predictions and evaluation for Lasso
y_pred_lasso = lasso_model.predict(X_val_scaled)
lasso_mse = mean_squared_error(y_val, y_pred_lasso)
lasso_r2 = r2_score(y_val, y_pred_lasso)
print("Lasso Regression Results:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"R^2 Score: {lasso_r2}")


# Feature importance from Lasso
lasso_coefficients = pd.Series(lasso_model.coef_, index=X.columns).sort_values()
print("\nLasso Feature Coefficients:")
print(lasso_coefficients)


# Model 2: Principal Components Regression (PCR)
# Apply PCA to reduce dimensionality
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)


# Select the number of principal components explaining 95% variance
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance_ratio >= 0.95) + 1
print(f"\nNumber of Principal Components explaining 95% variance: {n_components}")


# Train linear regression on selected principal components
pcr_model = LinearRegression()
pcr_model.fit(X_train_pca[:, :n_components], y_train)


# Predictions and evaluation for PCR
y_pred_pcr = pcr_model.predict(X_val_pca[:, :n_components])
pcr_mse = mean_squared_error(y_val, y_pred_pcr)
pcr_r2 = r2_score(y_val, y_pred_pcr)
print("\nPrincipal Components Regression Results:")
print(f"Mean Squared Error: {pcr_mse}")
print(f"R^2 Score: {pcr_r2}")


# Assumptions Investigation
# Residual analysis for Lasso model
residuals_lasso = y_val - y_pred_lasso
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_lasso, residuals_lasso)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Lasso Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")


# Residual analysis for PCR model
residuals_pcr = y_val - y_pred_pcr
plt.subplot(1, 2, 2)
plt.scatter(y_pred_pcr, residuals_pcr)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("PCR Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


# Submission
test_data_scaled = scaler.transform(test_data.drop(columns=["id"]))
test_predictions_lasso = lasso_model.predict(test_data_scaled)
submission_df = pd.DataFrame({
"id": test_data["id"],
"Rings": test_predictions_lasso
})
submission_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")




