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

# Define dataset paths
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)

# Display basic information
print("Train Dataset Info:")
train_df.info()
print("\nTrain Dataset Sample:")
display(train_df.head())

print("\nTest Dataset Info:")
test_df.info()
print("\nTest Dataset Sample:")
display(test_df.head())

print("\nSample Submission:")
display(submission_df.head())

# Check for missing values
print("\nMissing Values in Train Dataset:")
print(train_df.isnull().sum())

print("\nMissing Values in Test Dataset:")
print(test_df.isnull().sum())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check for duplicate rows
print(f"Duplicate rows in train dataset: {train_df.duplicated().sum()}")

# Display basic statistics
print("\nBasic Statistics:")
display(train_df.describe())

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.drop(columns=["id"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



from sklearn.model_selection import train_test_split

# Drop 'id' and redundant highly correlated features
#train_df = train_df.drop(columns=["id", "temparature", "dewpoint", "mintemp"])

# Define features and target
X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]

# Split into train and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)



!pip install tabpfn --quiet


# Load test dataset
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
test_df = pd.read_csv(test_path)


test_df


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

# Train a single TabPFN model
clf = TabPFNClassifier(device="cpu")  # Use "cuda" if available
clf.fit(X_train, y_train)

# Predict probabilities on validation set
y_proba = clf.predict_proba(X_val)[:, 1]  # Get probability of rainfall

# Try different probability thresholds
thresholds = np.linspace(0.1, 0.9, 50)  # Test 50 threshold values between 0.1 and 0.9
best_auc = 0
best_threshold = 0

for t in thresholds:
    y_pred_adjusted = (y_proba >= t).astype(int)  # Convert probabilities to 0/1 labels
    auc = roc_auc_score(y_val, y_pred_adjusted)
    
    if auc > best_auc:
        best_auc = auc
        best_threshold = t

print(f"Best Threshold: {best_threshold:.3f} | Best AUC: {best_auc:.5f}")



# Store 'id' column separately
test_ids = test_df["id"]


# Predict probabilities on test set
test_proba = clf.predict_proba(test_df)[:, 1]  # Get probability of rainfall

# Apply best threshold
test_predictions = (test_proba >= best_threshold).astype(int)

# Create submission file
submission_df = pd.DataFrame({"id": test_ids, "rainfall": test_predictions})
submission_df.to_csv("submission.csv", index=False)

print(f"Submission file 'submission.csv' created with threshold {best_threshold:.3f}")


