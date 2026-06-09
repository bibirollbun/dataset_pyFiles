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


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset
heart_train = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
heart_test = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv")
sample_submission = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/sample_submission.csv")


heart_train.head()


heart_test.head()


sample_submission.head()


heart_train.isnull().sum()


heart_train.info()


heart_test.isnull().sum()


sample_submission.isnull().sum()


heart_train.describe()


heart_train['HeartDisease'].value_counts()


X = heart_train.drop(columns='HeartDisease',axis=1)
y = heart_train['HeartDisease']


print(X)


print(y)


# Step 2: One-hot encode categorical features
X_encoded = pd.get_dummies(X)


X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,test_size=0.2,stratify=y,random_state=42)


print(X.shape, X_train.shape, X_test.shape)


log_model = LogisticRegression(max_iter=1000)


log_model.fit(X_train,y_train)


# Step 6: Evaluate
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")


# Save original (for display later)
heart_test_display = heart_test.copy()

# One-hot encode categorical columns
heart_test_encoded = pd.get_dummies(heart_test)

# Align with training features
heart_test_encoded = heart_test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict
predictions = log_model.predict(heart_test_encoded)
probabilities = log_model.predict_proba(heart_test_encoded)[:, 1]  # probability of class 1

# Add predictions to original data
heart_test_display['Predicted_HeartDisease'] = predictions
heart_test_display['Probability_HeartDisease'] = probabilities

# View results
print(heart_test_display.head())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {accuracy_rf:.2f}")


# Predict classes for X_test
rf_class_preds = rf_model.predict(X_test)

# View first 10 predictions
print("Predicted Classes:", rf_class_preds[:10])


# Predict probabilities for X_test
rf_prob_preds = rf_model.predict_proba(X_test)

# rf_prob_preds[:, 1] gives the probability of class 1 (Heart Disease = Yes)
print("Predicted Probabilities:", rf_prob_preds[:10, 1])


# Read and preprocess the test data
heart_test = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')

# Drop label if present
if 'HeartDisease' in heart_test.columns:
    heart_test = heart_test.drop('HeartDisease', axis=1)

# Encode & align columns
heart_test_encoded = pd.get_dummies(heart_test)
heart_test_encoded = heart_test_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predict
test_preds = rf_model.predict(heart_test_encoded)

# Add prediction to the original DataFrame
heart_test['Predicted_HeartDisease'] = test_preds

# View results
print(heart_test.head())


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=heart_test, x='Predicted_HeartDisease')
plt.title("Prediction Summary: Heart Disease")
plt.xticks([0, 1], ['No Disease', 'Heart Disease'])
plt.ylabel('Count')
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# For Logistic Regression
y_pred_log = log_model.predict(X_test)
y_proba_log = log_model.predict_proba(X_test)[:, 1]

# For Random Forest
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("ğŸ”� Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_proba_log))

print("\nğŸŒ² Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))


heart_test['id'] = heart_test.index  # or use 'Id' column if available
heart_test.rename(columns={'Predicted_HeartDisease': 'target'}, inplace=True)

# Save the submission file
heart_test[['id', 'target']].to_csv("submission.csv", index=False)

