# 1. Important Packages

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# 2. Define Data Paths

base_path = '/kaggle/input/intelligent-systems-ecu-csg-2341-sem-2-2024-02'
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path, 'test.csv')


# 3. Load the Data

print("Loading training data...")
train_df = pd.read_csv(train_path)
print("Training data shape:", train_df.shape)

print("Loading test data...")
test_df = pd.read_csv(test_path)
print("Test data shape:", test_df.shape)

# Inspect the first few rows to verify data structure
print("Training data preview:")
print(train_df.head())


# 4. Prepare the Data

# If "Label" is missing, raise an error
if 'Label' not in train_df.columns:
    raise ValueError("Expected 'Label' column not found in training data.")

# Optional: drop 'index' column from both train and test if it exists 
# so that it does not become a modeling feature
if 'index' in train_df.columns:
    train_df.drop('index', axis=1, inplace=True)
if 'index' in test_df.columns:
    # But keep a copy of test index for final submission
    test_index = test_df['index'].values  
    test_df.drop('index', axis=1, inplace=True)
else:
    # If the test set really has no 'index' column, 
    # just create a range-based index
    test_index = np.arange(len(test_df))

# Separate features (X) from target (y)
X = train_df.drop('Label', axis=1)
y = train_df['Label']

# Convert string labels to 0/1 numeric:
# "normal" -> 0, "attack" -> 1
label_map = {'normal': 0, 'attack': 1}
y = y.map(label_map)

# Check for missing values
if X.isnull().values.any():
    print("Missing values detected in training data. Filling with median.")
    X = X.fillna(X.median())
if test_df.isnull().values.any():
    print("Missing values detected in test data. Filling with median.")
    test_df = test_df.fillna(test_df.median())


# 5. Split Data for Local Evaluation

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Train a Machine Learning Model

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Training the model on the training split...")
clf.fit(X_train, y_train)


# 7. Evaluate the Model Locally

y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print("Validation ROC AUC: {:.4f}".format(roc_auc))


# 8. Retrain on the Full Training Data

print("Retraining the model on the full training data...")
clf.fit(X, y)


# 9. Generate Predictions for the Test Data and Save the Submission File.

print("Generating predictions on the test data...")
numeric_preds = clf.predict(test_df)  # This will be 0 or 1
submission = pd.DataFrame({
    'index': test_index,       # Use the original index from test data
    'Label': numeric_preds     # Numeric labels (0=normal, 1=attack)
})

submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print("Submission file created and saved as:", submission_path)


