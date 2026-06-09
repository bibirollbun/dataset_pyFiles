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


# Step 1: Load and Inspect Data

# Import required library
import pandas as pd

# Define file paths (already provided earlier)
train_path = "/kaggle/input/playground-series-s5e9/train.csv"
test_path = "/kaggle/input/playground-series-s5e9/test.csv"
# submission_path = "/mnt/data/sample_submission.csv"

# Load the datasets
train_df = pd.read_csv(train_path)   # Training dataset with target BeatsPerMinute
test_df = pd.read_csv(test_path)     # Test dataset without target
# submission_df = pd.read_csv(submission_path)  # Sample submission format

# Inspect shapes of datasets
print("Train shape:", train_df.shape)        # Should include target
print("Test shape:", test_df.shape)          # Should exclude target
# print("Submission shape:", submission_df.shape)  # Should have 2 columns: ID, BeatsPerMinute

# Preview first few rows
print("\nTrain preview:")
print(train_df.head())

print("\nTest preview:")
print(test_df.head())

# print("\nSubmission preview:")
# print(submission_df.head())

# Check for missing values in train and test
print("\nMissing values in Train:")
print(train_df.isnull().sum())

print("\nMissing values in Test:")
print(test_df.isnull().sum())



# Step 2: Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution of the target variable
plt.figure(figsize=(8, 5))
sns.histplot(train_df['BeatsPerMinute'], bins=50, kde=True)
plt.title("Distribution of Target: BeatsPerMinute")
plt.xlabel("BeatsPerMinute")
plt.ylabel("Frequency")
plt.show()

# 2. Summary statistics of features
print("\nSummary statistics of training data:")
print(train_df.describe().T)

# 3. Correlation heatmap (features + target)
plt.figure(figsize=(10, 8))
corr_matrix = train_df.drop(columns=['id']).corr()   # drop 'id' since it's not a feature
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Features and Target")
plt.show()

# 4. Correlation of each feature with target
target_corr = corr_matrix['BeatsPerMinute'].sort_values(ascending=False)
print("\nCorrelation of Features with BeatsPerMinute:")
print(target_corr)

# 5. Pairplot for a few important features (based on correlation)
top_features = target_corr.drop('BeatsPerMinute').abs().sort_values(ascending=False).head(4).index.tolist()
sns.pairplot(train_df[top_features + ['BeatsPerMinute']], diag_kind="kde")
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.show()



from sklearn.preprocessing import StandardScaler


# Step 1: Separate features and target from train_df

X_train = train_df.drop(columns=["BeatsPerMinute"])   # Features
y_train = train_df["BeatsPerMinute"]                  # Target

# Test set (no labels provided)
X_test = test_df.copy()

print("Shapes before scaling:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)


# Step 2: Preprocessing (Scaling)

def preprocess_data(X_train, X_test):
    """
    Applies StandardScaler to training and test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Run preprocessing
X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

print("\nAfter scaling:")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)



# Step 4: Baseline Model Training with Cross-Validation (Linear Regression Only)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Define Linear Regression model
lin_reg = LinearRegression()

# Evaluate model with cross-validation
scores = cross_val_score(
    lin_reg, X_train_scaled, y_train,
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1
)
rmse_scores = np.sqrt(-scores)  # Convert negative MSE to RMSE

print("\nLinear Regression Results (CPU):")
print(f"Mean RMSE: {rmse_scores.mean():.4f}")
print(f"Std RMSE: {rmse_scores.std():.4f}")




# Step 6: Validation for Best Model (Linear Regression)
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import numpy as np

# Use the Linear Regression model defined in Step 4
best_model = lin_reg  

# Perform 10-Fold Cross Validation with different metrics
cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_rmse = np.sqrt(-cross_val_score(best_model, X_train_scaled, y_train, 
                                   scoring="neg_mean_squared_error", cv=cv))
cv_mae = -cross_val_score(best_model, X_train_scaled, y_train, 
                          scoring="neg_mean_absolute_error", cv=cv)
cv_r2 = cross_val_score(best_model, X_train_scaled, y_train, 
                        scoring="r2", cv=cv)

print("Validation Results for Linear Regression:")
print(f"RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
print(f"R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")


# Step 7: Residual Analysis

# Fit model on full training data
best_model.fit(X_train_scaled, y_train)

# Predict on training set
y_pred = best_model.predict(X_train_scaled)

# Calculate residuals
residuals = y_train - y_pred

# Residual plot
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.2)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted BPM")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot for Linear Regression")
plt.show()

# Error distribution
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Distribution of Residuals")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()




# Final Submission using Linear Regression

import pandas as pd

# Train best model on full training data
best_model.fit(X_train_scaled, y_train)

# Predict on test set
y_test_pred = best_model.predict(X_test_scaled)

# Example: IDs should start from 524164
start_id = 524164  

submission = pd.DataFrame({
    "id": range(start_id, start_id + len(y_test_pred)),
    "BeatsPerMinute": y_test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission.head())




