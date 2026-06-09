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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


print(train_df.head())
print(train_df.info())


# Check target distribution
sns.countplot(x=train_df['y'])
plt.title("Target Class Distribution")
plt.show()


# Features and target
X = train_df.drop(columns=['id', 'y'])
y = train_df['y']


# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
test_encoded = pd.get_dummies(test_df.drop(columns=['id']), drop_first=True)


test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_encoded)


# Train model
svm = LinearSVC(max_iter=10000, dual=False)
svm.fit(X_train_scaled, y_train)


# Validation performance
y_pred = svm.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# Train on full data and predict test set
svm.fit(scaler.fit_transform(X), y)
test_preds = svm.predict(test_scaled)


# Save submission
submission = sample_submission.copy()
submission['y'] = test_preds
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")


sub = pd.read_csv("submission.csv")
sub

