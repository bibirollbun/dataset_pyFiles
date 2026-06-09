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

# Load the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Display the first few rows of the training dataset
print("First few rows of the training dataset:")
display(train_df.head())

# Display basic information about the dataset
print("\nBasic information of the training dataset:")
train_df.info()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the FutureWarning from Seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Load the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Replace infinite values with NaN
train_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

# Drop rows where wind direction is missing
train_df = train_df.dropna(subset=['winddirection'])

# Verify that missing values are removed
print("Missing values after dropping rows:")
print(train_df.isnull().sum())

# Summary statistics of numerical features
print("\nSummary statistics of numerical features in training dataset:")
display(train_df.describe())

# Distribution of the target variable 'rainfall'
plt.figure(figsize=(6, 4))
sns.histplot(train_df['rainfall'], bins=20, kde=True)
plt.title("Distribution of Target Variable: Rainfall")
plt.xlabel("Rainfall (0: No Rain, 1: Rain)")
plt.ylabel("Frequency")
plt.show()

# Visualizing feature distributions
train_df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import pandas as pd

# Select features and target variable
X = train_df.drop(columns=['id', 'rainfall'])  # Drop 'id' and target 'rainfall'
y = train_df['rainfall']

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_val_scaled)
y_pred_log_reg_prob = log_reg.predict_proba(X_val_scaled)[:, 1]

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_val_scaled)
y_pred_rf_prob = rf.predict_proba(X_val_scaled)[:, 1]

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_val_scaled)
y_pred_xgb_prob = xgb_model.predict_proba(X_val_scaled)[:, 1]

# Evaluate models based on accuracy and ROC AUC score
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
predictions = [y_pred_log_reg_prob, y_pred_rf_prob, y_pred_xgb_prob]

for model, pred in zip(models, predictions):
    accuracy = accuracy_score(y_val, (pred > 0.5).astype(int))
    roc_auc = roc_auc_score(y_val, pred)
    print(f"{model} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

# Select the best model (based on ROC AUC for example)
best_model = xgb_model  # Replace with the best model after evaluating

# Prepare the test dataset for predictions
X_test = test_df.drop(columns=['id'])  # Drop 'id' column for prediction
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set using the best model
y_pred_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Create submission file
submission = pd.DataFrame({'id': test_df['id'], 'rainfall': y_pred_test_prob})
submission.to_csv('submission.csv', index=False)

print("\nSubmission file created: 'submission.csv'")


