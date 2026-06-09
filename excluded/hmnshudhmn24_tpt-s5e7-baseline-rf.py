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


# =======================================================
# Playground Series S5E7: Introvert vs Extrovert
# Full Kaggle-ready notebook
# Handles:
# 1. Categorical features via one-hot encoding
# 2. Missing values via filling
# 3. Baseline RandomForest
# =======================================================

# --------------------------
# Imports
# --------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# --------------------------
# Load Data
# --------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# --------------------------
# Separate Features and Target
# --------------------------
X_train = train.drop(columns=["id", "Personality"])
y_train = train["Personality"]

X_test = test.drop(columns=["id"])

# --------------------------
# Identify categorical columns
# --------------------------
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_cols)

# --------------------------
# Combine train and test for consistent encoding
# --------------------------
all_data = pd.concat([X_train, X_test], axis=0)

# One-hot encode categorical columns
all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols)

# --------------------------
# Fill missing values
# --------------------------
# Fill numeric NaNs with median
all_data_encoded = all_data_encoded.fillna(all_data_encoded.median())

# Split back into train and test
X_train_encoded = all_data_encoded.iloc[:len(X_train), :]
X_test_encoded  = all_data_encoded.iloc[len(X_train):, :]

print("Encoded train shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)

# --------------------------
# Train RandomForest Baseline
# --------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_encoded, y_train)

# --------------------------
# Predict on Test Set
# --------------------------
y_pred = clf.predict(X_test_encoded)

# --------------------------
# Create Submission
# --------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": y_pred
})

# Save to Kaggle working directory
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved submission.csv ✅")

# Verify submission file
print("Files in /kaggle/working:", os.listdir("/kaggle/working"))
submission.head()


