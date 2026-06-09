# Rainfall Prediction using Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Load the data from Kaggle dataset
train_path = '/kaggle/input/playground-series-s5e3/train.csv'
test_path = '/kaggle/input/playground-series-s5e3/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Separate features and target
X = train_data.drop(columns=['rainfall', 'id'])
y = train_data['rainfall']

# Handle missing values using median
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_data_imputed = pd.DataFrame(imputer.transform(test_data.drop(columns=['id'])), columns=X.columns)

# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on validation data
val_preds = model.predict_proba(X_valid)[:, 1]
roc_auc = roc_auc_score(y_valid, val_preds)
print(f"Validation ROC AUC: {roc_auc}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_valid, val_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# Predict on test data
test_preds = model.predict_proba(test_data_imputed)[:, 1]

# Prepare submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds
})

# Save submission to Kaggle working directory
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created: /kaggle/working/submission.csv")

# Note: Ensure the dataset name is correct in the path before running the code.


