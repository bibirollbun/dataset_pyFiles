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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head()


train.info()


train.describe()


train.duplicated().sum()


unknown_counts = (train == 'unknown').sum()
print(unknown_counts)

total_unknown = (train == 'unknown').sum().sum()
print("Total unknown values:", total_unknown)


train = train.drop(columns=['id'])


train.isnull().sum()


train = train.replace("unknown", np.nan)


train.isnull().sum()


train['job'] = train['job'].fillna(train['job'].mode()[0])
train['education'] = train['education'].fillna(train['education'].mode()[0])

train['contact'] = train['contact'].fillna("unknown")
train['poutcome'] = train['poutcome'].fillna("unknown")


train.isnull().sum()


train.head()


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='y', data=train)
plt.title("Distribution of Target Variable (y)")
plt.show()

print(train['y'].value_counts(normalize=True))


plt.figure(figsize=(8,5))
sns.histplot(train['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10,5))
sns.countplot(y='job', data=train, order=train['job'].value_counts().index)
plt.title("Job Distribution")
plt.show()


sns.countplot(x='marital', data=train)
plt.title("Marital Status Distribution")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(y='education', data=train, order=train['education'].value_counts().index)
plt.title("Education Distribution")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x='y', y='balance', data=train)
plt.title("Balance vs Subscription (y)")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='y', y='duration', data=train)
plt.title("Call Duration vs Subscription (y)")
plt.show()


numerical_features = train.select_dtypes(include=['float64', 'int64'])

corr = numerical_features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=train[col])
    plt.title(col)

plt.tight_layout()
plt.show()


import numpy as np

balance_cap = train['balance'].quantile(0.99)
train['balance'] = np.where(train['balance'] > balance_cap, balance_cap, train['balance'])

duration_cap = train['duration'].quantile(0.99)
train['duration'] = np.where(train['duration'] > duration_cap, duration_cap, train['duration'])

campaign_cap = train['campaign'].quantile(0.99)
train['campaign'] = np.where(train['campaign'] > campaign_cap, campaign_cap, train['campaign'])

previous_cap = train['previous'].quantile(0.99)
train['previous'] = np.where(train['previous'] > previous_cap, previous_cap, train['previous'])

train['pdays_binary'] = np.where(train['pdays'] == -1, 0, 1)
test['pdays_binary'] = np.where(test['pdays'] == -1,0,1)

# train = train.drop(columns=['pdays'])


print(train[['balance','duration','campaign','previous','pdays','pdays_binary']].describe(percentiles=[0.5,0.9,0.95,0.99]))

import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['age','balance','duration','campaign','pdays','previous']
plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols,1):
    plt.subplot(2,3,i)
    sns.boxplot(x=train[col])
    plt.title(col)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

binary_cols = ['default', 'housing', 'loan', 'y']
test_binary_cols = ['default', 'housing', 'loan']
le = LabelEncoder()

for col in binary_cols:
    train[col] = le.fit_transform(train[col])
    
for col in test_binary_cols:
    test[col] = le.fit_transform(test[col])

categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)


train.head()


print("Shape after encoding:", train.shape)


from sklearn.preprocessing import StandardScaler

num_cols = ['age','balance','duration','campaign','pdays','previous','day']

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.fit_transform(test[num_cols])

train.head()


from sklearn.model_selection import train_test_split

X = train.drop(columns=['y'])
y = train['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logreg_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg_balanced.fit(X_train, y_train)

y_pred_balanced = logreg_balanced.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))
print("\nClassification Report:\n", classification_report(y_test, y_pred_balanced))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# --------------------------
# Get predicted probabilities for both models
# --------------------------
y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]    # Probability of class 1 for Random Forest
y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]  # Probability of class 1 for XGBoost

# --------------------------
# ROC Curve
# --------------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)     # False Positive Rate & True Positive Rate for RF
roc_auc_rf = auc(fpr_rf, tpr_rf)                          # Area Under Curve for RF

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb) # FPR & TPR for XGB
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)                      # AUC for XGB

# Plot ROC Curves
plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.2f})")
plt.plot([0,1],[0,1],'k--')  # Diagonal line (baseline)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------
# Precision-Recall Curve
# --------------------------
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_pred_prob_rf)  # Precision & Recall for RF
avg_prec_rf = average_precision_score(y_test, y_pred_prob_rf)        # Average Precision for RF

prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_pred_prob_xgb) # Precision & Recall for XGB
avg_prec_xgb = average_precision_score(y_test, y_pred_prob_xgb)        # Average Precision for XGB

# Plot Precision-Recall Curves
plt.figure(figsize=(8,6))
plt.plot(rec_rf, prec_rf, label=f"Random Forest (AP = {avg_prec_rf:.2f})")
plt.plot(rec_xgb, prec_xgb, label=f"XGBoost (AP = {avg_prec_xgb:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample Submission shape:", sample_sub.shape)


test_data = test.copy()
test_ids = test_data["id"]
test = test.drop(['id',"job_unknown" , "education_unknown"] , axis=1)


y_pred = rf.predict(test)

submission = pd.DataFrame({
    "id": test_ids,
    "prediction": y_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")

