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


# List files in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


test = pd.read_csv('/kaggle/input//playground-series-s5e3/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


train


test


## Exploratory Data Analysis (EDA)
# Check for missing values
print("Missing values in train dataset:")
print(train.isnull().sum())
print("\nMissing values in test dataset:")
print(test.isnull().sum())


# Visualize feature distributions
import matplotlib.pyplot as plt
train.hist(figsize=(15, 10))
plt.suptitle('Feature Distributions in Train Dataset', y=1.01)
plt.show()


# Compare train/test distributions
import seaborn as sns

# Visualize feature distributions - Train vs Test (excluding 'rainfall')
for col in train.columns[2:-1]:  # Exclude 'id' and 'rainfall'
    if col in test.columns:  # Only plot columns that exist in both datasets
        plt.figure(figsize=(8, 4))
        sns.kdeplot(train[col], label='Train', shade=True)
        sns.kdeplot(test[col], label='Test', shade=True)
        plt.title(f'{col} Distribution - Train vs Test')
        plt.legend()
        plt.show()



# Analyze correlations
correlation_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Prepare features and target
target = 'rainfall'
features = [col for col in train.columns if col not in ['id', target]]

X = train[features]
y = train[target]
X_test = test[features]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train XGBoost
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {roc_auc:.4f}")

# Make predictions on the test set
test_preds = model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


