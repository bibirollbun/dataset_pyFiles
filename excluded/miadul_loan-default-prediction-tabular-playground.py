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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



train.info()
train.describe()
train.isnull().sum()



# ===============================
# ğŸ”� STEP 4: Target Variable Distribution
# ===============================
sns.countplot(x='loan_paid_back', data=train, palette='coolwarm')
plt.title("Target Variable Distribution (Loan Paid Back)")
plt.show()



# ===============================
# ğŸ“Š STEP 5: Numeric Feature Analysis
# ===============================
numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

train[numeric_features].hist(figsize=(10, 8), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Numeric Feature Distributions")
plt.show()



# ===============================
# ğŸ§  STEP 6: Encode Categorical Columns
# ===============================
categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))

train.head()



# ===============================
# âš™ï¸� STEP 7: Feature & Target Split
# ===============================
X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



# ===============================
# ğŸ”¢ STEP 8: Scale Numeric Columns
# ===============================
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_valid[numeric_features] = scaler.transform(X_valid[numeric_features])
test[numeric_features] = scaler.transform(test[numeric_features])



# ===============================
# ğŸ¤– STEP 9: Model Training (Random Forest)
# ===============================
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

print("Validation Accuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))



# ===============================
# ğŸ“‰ STEP 10: Confusion Matrix
# ===============================
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# ===============================
# ğŸš€ STEP 11: Test Prediction & Submission
# ===============================
test_pred = model.predict(test.drop('id', axis=1))
submission = sample.copy()
submission['loan_paid_back'] = test_pred
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as submission.csv")


