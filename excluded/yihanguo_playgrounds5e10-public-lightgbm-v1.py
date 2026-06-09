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


import pandas as pd

# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

# Print dataset shapes
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)

# Preview the first 5 rows of training data
train.head()


# Check data types and basic info
train.info()
train.describe()


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.hist(train['accident_risk'], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Accident Risk")
plt.xlabel("Accident Risk")
plt.ylabel("Frequency")
plt.show()


import seaborn as sns

# Prepare data for correlation: convert booleans to int for correlation computation
corr_df = train.copy()
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
corr_df[bool_cols] = corr_df[bool_cols].astype(int)
# Compute correlations between numeric features and target
corr_matrix = corr_df[['num_lanes','curvature','speed_limit','num_reported_accidents',
                       'road_signs_present','public_road','holiday','school_season',
                       'accident_risk']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Features and Target")
plt.show()


# Identify feature columns (exclude id and target)
features = [col for col in train.columns if col not in ["id", "accident_risk"]]

# Separate into X (features) and y (target)
X = train[features].copy()
y = train["accident_risk"].copy()

# Convert categorical columns to 'category' dtype for LightGBM
categorical_cols = ["road_type", "lighting", "weather", "time_of_day"]
for col in categorical_cols:
    X[col] = X[col].astype("category")

# Convert boolean columns to integers (0/1)
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
for col in bool_cols:
    X[col] = X[col].astype(int)

# Do the same preprocessing for the test features
X_test = test[features].copy()
for col in categorical_cols:
    X_test[col] = X_test[col].astype("category")
for col in bool_cols:
    X_test[col] = X_test[col].astype(int)

print("Preprocessed training features shape:", X.shape)
print("Preprocessed test features shape:", X_test.shape)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
rmse_scores = []

for train_idx, valid_idx in kf.split(X):
    # Split data into training and validation sets for this fold
    X_train_cv = X.iloc[train_idx]
    X_valid_cv = X.iloc[valid_idx]
    y_train_cv = y.iloc[train_idx]
    y_valid_cv = y.iloc[valid_idx]
    
    # Initialize LightGBM regressor (baseline parameters)
    model_cv = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    # Train the model on the training fold
    model_cv.fit(X_train_cv, y_train_cv, categorical_feature=categorical_cols)
    
    # Predict on the validation fold
    y_pred_cv = model_cv.predict(X_valid_cv)
    # Calculate RMSE for this fold
    rmse = mean_squared_error(y_valid_cv, y_pred_cv, squared=False)
    print(f"Fold {fold} RMSE: {rmse:.5f}")
    rmse_scores.append(rmse)
    fold += 1

# Compute average RMSE across folds
avg_rmse = np.mean(rmse_scores)
print(f"Average RMSE across 5 folds: {avg_rmse:.5f}")


# Train final model on all training data
final_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y, categorical_feature=categorical_cols)


# Generate predictions for the test set
test_predictions = final_model.predict(X_test)
print("Test predictions sample:", test_predictions[:5])


# Create submission dataframe
submission = sample_sub.copy()
submission['accident_risk'] = test_predictions

# Save to CSV (this will create a file for submission)
submission.to_csv("submission.csv", index=False)

# Preview the first few lines of the submission file
submission.head()




