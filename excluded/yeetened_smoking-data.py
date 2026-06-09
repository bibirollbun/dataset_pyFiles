import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


train_data = pd.read_csv('/kaggle/input/smoking-binary-prediction-using-bio-signals/train.csv')
test_data = pd.read_csv('/kaggle/input/smoking-binary-prediction-using-bio-signals/test.csv')


print("First 5 rows of the train dataset:")
print(train_data.head())

print("\nGet train dataset info:")
print(train_data.info())

print("\nSummarize statistics:")
print(train_data.describe())

print("\nCheck for missing values:")
print(train_data.isnull().sum())


X = train_data.drop(columns=['id', 'smoking'])
y = train_data['smoking']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)


# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Support Vector Machine
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, knn_model.predict_proba(X_test)[:, 1])



# Print results for Logistic Regression
print("Logistic Regression")
print(f"Accuracy: {accuracy_log_reg}")
print(f"ROC AUC Score: {roc_auc_log_reg}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Print results for Random Forest
print("Random Forest")
print(f"Accuracy: {accuracy_rf}")
print(f"ROC AUC Score: {roc_auc_rf}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Print results for SVM
print("Support Vector Machine")
print(f"Accuracy: {accuracy_svm}")
print(f"ROC AUC Score: {roc_auc_svm}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Print results for KNN
print("K-Nearest Neighbors")
print(f"Accuracy: {accuracy_knn}")
print(f"ROC AUC Score: {roc_auc_knn}")
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))



# ROC Curve for all models
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1])

plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (area = {roc_auc_log_reg:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (area = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (area = {roc_auc_svm:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (area = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='best')
plt.show()

