# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import resample

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# File paths
train_path = '/kaggle/input/GiveMeSomeCredit/cs-training.csv'
test_path = '/kaggle/input/GiveMeSomeCredit/cs-test.csv'

# Load CSVs
train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

# Quick look
print(train_df.head())
print(train_df.info())


# Fill missing values using direct assignment
train_df['MonthlyIncome'] = train_df['MonthlyIncome'].fillna(train_df['MonthlyIncome'].median())
train_df['NumberOfDependents'] = train_df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median())

test_df['MonthlyIncome'] = test_df['MonthlyIncome'].fillna(test_df['MonthlyIncome'].median())
test_df['NumberOfDependents'] = test_df['NumberOfDependents'].fillna(test_df['NumberOfDependents'].median())


# Separate majority and minority classes
train_majority = train_df[train_df.SeriousDlqin2yrs==0]
train_minority = train_df[train_df.SeriousDlqin2yrs==1]

# Upsample minority class
train_minority_upsampled = resample(train_minority,
                                    replace=True,
                                    n_samples=len(train_majority),
                                    random_state=42)

# Combine back to balanced DataFrame
train_balanced = pd.concat([train_majority, train_minority_upsampled])

# Features and target
x_balanced = train_balanced.drop('SeriousDlqin2yrs', axis=1)
y_balanced = train_balanced['SeriousDlqin2yrs']


rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(x_balanced, y_balanced)


# Predictions
y_pred = rf.predict(x_balanced)
y_pred_proba = rf.predict_proba(x_balanced)[:,1]

# ROC-AUC Score
print("ROC AUC Score:", roc_auc_score(y_balanced, y_pred_proba))

# Classification Report
print(classification_report(y_balanced, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_balanced, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


feature_importances = pd.DataFrame({
    'feature': x_balanced.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importances)

# Plot top 5 features
plt.figure(figsize=(6,4))
sns.barplot(x='importance', y='feature', data=feature_importances.head(5))
plt.title('Top 5 Feature Importances')
plt.show()


# Copy test set
X_test = test_df.copy()

# Drop target if exists
if 'SeriousDlqin2yrs' in X_test.columns:
    X_test = X_test.drop('SeriousDlqin2yrs', axis=1)

# Keep only features used in training
X_test = X_test[x_balanced.columns]

# Predict
test_pred = rf.predict(X_test)
test_df['SeriousDlqin2yrs_pred'] = test_pred

# Save predictions
test_df[['SeriousDlqin2yrs_pred']].to_csv('/kaggle/working/rf_predictions.csv', index=False)
print("Predictions saved to /kaggle/working/rf_predictions.csv")


pd.read_csv("/kaggle/working/rf_predictions.csv")

