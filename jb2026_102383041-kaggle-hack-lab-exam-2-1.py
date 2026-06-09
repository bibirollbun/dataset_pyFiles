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


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')

# Verify the first few rows of the data
print(train_data.head())
print(test_data.head())


# Dynamically determine feature columns
if 'id' in train_data.columns:
    X = train_data.drop(columns=["id", "target"])
else:
    X = train_data.drop(columns=["target"])

# Ensure the target column is correct
if 'target' in train_data.columns:
    y = train_data["target"]
else:
    raise KeyError("Target column not found in training data.")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)





# Model Training-Step 1: Train a Random Forest to calculate feature importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Model Evaluation-Step 2: Extract feature importances
feature_importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Data Preprocessing-Step 3: Select the top N important features (e.g., top 10)
top_features = feature_importance_df['Feature'].head(10).values  # Adjust N as needed
print(f"Top Features: {top_features}")

# Use only the top N features for training and validation
X_train_top = X_train_scaled[:, :10]  # Select top 10 features from scaled data
X_val_top = X_val_scaled[:, :10]      # Ensure the same selection for validation

# Model Training-Step 4: Train a new Random Forest model on the selected features
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_top, y_train)

# Model Evaluation-Step 5: Evaluate the model
y_val_pred = model.predict(X_val_top)
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R² with Selected Features: {r2:.4f}")



# Prepare the test data (only top N features)
X_test = test_data.drop(columns=["id"]) if 'id' in test_data.columns else test_data
X_test_scaled = scaler.transform(X_test)
X_test_top = X_test_scaled[:, :10]  # Use only the top N features

# Make predictions on the test data
test_predictions = model.predict(X_test_top)

# Save predictions to a CSV file
submission = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Predictions saved to submission.csv")

