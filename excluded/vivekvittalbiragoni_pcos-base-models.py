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



train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


train.head()


train.info()


train.describe(include='all').T


train.isnull().sum()




# Drop the 'ID' column
train = train.drop(columns=['ID'], axis=1)
test = test.drop(columns=['ID'], axis=1)

# Convert categorical columns to category type (if applicable)
for col in train.select_dtypes(include=['object']).columns:
    train[col] = train[col].astype("category")

# Create count plots with 4 rows and 4 columns
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 20))  # Adjusted to fit 13 features

ax = ax.flatten()  # Flatten for easier iteration

for index, col in enumerate(train.columns):
    sns.countplot(x=col, data=train, ax=ax[index])
    ax[index].set_title(col, fontsize=14)  # Make feature names clearer
    ax[index].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for clarity

# Hide any unused subplots (since we have 13 features in a 4x4 grid)
for i in range(index + 1, len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout(pad=1, w_pad=1, h_pad=3)
plt.show()




cat_col = train.select_dtypes(include =['object','category']).columns
cat_col


num_col = train.select_dtypes(exclude =['object','category']).columns
num_col


cat_col =['Age', 'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism','Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency','Exercise_Type', 'Exercise_Duration', 'Sleep_Hours','Exercise_Benefit']

for col in cat_col:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


for col in num_col:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(test[col].mean())


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder

encoders = {}  # Dictionary to store encoders for each categorical column

for col in cat_col:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])  # Fit & transform on train
    
    # Store encoder for future use
    encoders[col] = le

    # Transform test set safely (handle unseen labels)
    test[col] = test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)



le = LabelEncoder()
train['PCOS'] = le.fit_transform(train['PCOS'])


corr = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')



X = train.drop(columns=['PCOS'])
y = train['PCOS']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify =y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probability scores for class 1 (positive class)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # ✅ Use probabilities

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_score = roc_auc_score(y_test, y_pred_prob)

# Print ROC AUC Score
print(f"ROC AUC Score: {roc_score:.4f}")

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()




from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve


model = XGBClassifier(n_jobs=-1)
model.fit(X_train, y_train)

# Get probability scores for class 1 (positive class)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # ✅ Use probabilities

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_score = roc_auc_score(y_test, y_pred_prob)

# Print ROC AUC Score
print(f"ROC AUC Score: {roc_score:.4f}")

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"XGBClassifier (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve


model = LGBMClassifier()
model.fit(X_train, y_train)

# Get probability scores for class 1 (positive class)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # ✅ Use probabilities

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_score = roc_auc_score(y_test, y_pred_prob)

# Print ROC AUC Score
print(f"ROC AUC Score: {roc_score:.4f}")

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"LGBMClassifier (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve


model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Get probability scores for class 1 (positive class)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # ✅ Use probabilities

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_score = roc_auc_score(y_test, y_pred_prob)

# Print ROC AUC Score
print(f"ROC AUC Score: {roc_score:.4f}")

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"CatBoostClassifier (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


model = CatBoostClassifier(verbose=0)
model.fit(X, y)

# Get probability scores for class 1 (positive class)
y_pred_prob = model.predict_proba(test)[:, 1]  # ✅ Use probabilities



y_pred_prob


submission = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/sample_submission.csv')
submission['PCOS'] = y_pred_prob


submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")




