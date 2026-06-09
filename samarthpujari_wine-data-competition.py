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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("/kaggle/input/mastering-ordinal-regression-with-wine-data/train.csv")

# Features and target variable
X = data.drop(columns=["quality", "id"])  # Drop 'quality' and 'id' columns
y = data["quality"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the KFold cross-validator
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Placeholder for out-of-fold predictions
oof_preds = np.zeros(len(y))

# Track QWK scores for each fold
qwk_scores = []

# Cross-validation
for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"Fold {fold + 1}")
    
    # Split the data into training and validation sets
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    val_preds = model.predict(X_valid)
    val_preds_rounded = np.round(val_preds).astype(int)  # Round predictions to integers
    val_preds_rounded = np.clip(val_preds_rounded, 3, 8)  # Clip predictions to valid range (3 to 8)
    
    # Calculate QWK for the current fold
    qwk = cohen_kappa_score(y_valid, val_preds_rounded, weights="quadratic")
    qwk_scores.append(qwk)
    print(f"Fold {fold + 1} QWK: {qwk:.4f}")
    
    # Store out-of-fold predictions
    oof_preds[valid_idx] = val_preds_rounded

# Calculate the mean QWK score across all folds
mean_qwk = np.mean(qwk_scores)
print(f"Mean QWK across all folds: {mean_qwk:.4f}")

# Predict on the test set (assuming a test dataset exists)
test_data = pd.read_csv("/kaggle/input/mastering-ordinal-regression-with-wine-data/test.csv")
test_ids = test_data["id"]
X_test = test_data.drop(columns=["id"])
X_test_scaled = scaler.transform(X_test)

# Generate predictions on the test set
test_preds = model.predict(X_test_scaled)
test_preds_rounded = np.round(test_preds).astype(int)  # Round predictions to integers
test_preds_rounded = np.clip(test_preds_rounded, 3, 8)  # Clip predictions to valid range

# Prepare the submission file
submission = pd.DataFrame({
    "id": test_ids,
    "quality": test_preds_rounded
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")





