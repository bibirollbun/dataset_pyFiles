# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import warnings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')

print("Train shape:", train.shape)
print("\nUstunlar:\n", train.columns)

sns.countplot(data=train, x='Exited')
plt.title('Exited (Target) Taqsimoti')
plt.show()

train.info()


train.head()


print(train['Geography'].value_counts())



print("Yo'qolgan qiymatlar:\n", train.isnull().sum())

train.describe()



warnings.filterwarnings('ignore')
plt.figure(figsize=(8, 4))
sns.histplot(data=train, x='Age', hue='Exited', kde=True, bins=30)
plt.title('Yosh va Churn o‘rtasidagi bog‘liqlik')
plt.show()



plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Geography', hue='Exited')
plt.title('Geography va Churn')
plt.show()



plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Gender', hue='Exited')
plt.title('Gender va Churn')
plt.show()



numeric_cols = train.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Korrelatsion matritsa')
plt.show()



corr_matrix = numeric_cols.corr()
exited_corr = corr_matrix['Exited'].sort_values(ascending=False)

print("Exited bilan eng bog‘liq ustunlar:\n")
print(exited_corr)



X = train.drop(['Exited', 'CustomerId', 'Surname'], axis=1)  
y = train['Exited'] 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])

X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  


X = train.drop(['Exited', 'CustomerId', 'Surname'], axis=1)  
y = train['Exited'] 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
X_train['Gender'] = label_encoder.fit_transform(X_train['Gender'])
X_valid['Gender'] = label_encoder.transform(X_valid['Gender'])  # Valid ma'lumotlari uchun ham kodlash

X_train = pd.get_dummies(X_train, columns=['Geography'], drop_first=True)
X_valid = pd.get_dummies(X_valid, columns=['Geography'], drop_first=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)  # Valid ma'lumotlarini faqat transform qilamiz



rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train_scaled, y_train)

y_valid_pred_rf = rf_model.predict(X_valid_scaled)
test_accuracy_rf = accuracy_score(y_valid, y_valid_pred_rf)
print(f"Random Forest Test Accuracy: {test_accuracy_rf:.4f}")

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_valid, y_valid_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_valid, y_valid_pred_rf))

y_prob_rf = rf_model.predict_proba(X_valid_scaled)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_valid, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Random Forest)')
plt.legend(loc="lower right")
plt.show()



svc_model = SVC(probability=True, random_state=42)

svc_model.fit(X_train_scaled, y_train)

y_valid_pred_svc = svc_model.predict(X_valid_scaled)
test_accuracy_svc = accuracy_score(y_valid, y_valid_pred_svc)
print(f"SVC Test Accuracy: {test_accuracy_svc:.4f}")

print("\nConfusion Matrix (SVC):")
print(confusion_matrix(y_valid, y_valid_pred_svc))

print("\nClassification Report (SVC):")
print(classification_report(y_valid, y_valid_pred_svc))

y_prob_svc = svc_model.predict_proba(X_valid_scaled)[:, 1]
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_valid, y_prob_svc)
roc_auc_svc = auc(fpr_svc, tpr_svc)

plt.figure()
plt.plot(fpr_svc, tpr_svc, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (SVC)')
plt.legend(loc="lower right")
plt.show()



logreg_model = LogisticRegression(random_state=42)

logreg_model.fit(X_train_scaled, y_train)

y_valid_pred_logreg = logreg_model.predict(X_valid_scaled)
test_accuracy_logreg = accuracy_score(y_valid, y_valid_pred_logreg)
print(f"Logistic Regression Test Accuracy: {test_accuracy_logreg:.4f}")

print("\nConfusion Matrix (Logistic Regression):")
print(confusion_matrix(y_valid, y_valid_pred_logreg))

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_valid, y_valid_pred_logreg))

y_prob_logreg = logreg_model.predict_proba(X_valid_scaled)[:, 1]
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_valid, y_prob_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

plt.figure()
plt.plot(fpr_logreg, tpr_logreg, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logreg)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Logistic Regression)')
plt.legend(loc="lower right")
plt.show()



gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

y_valid_pred_gb = gb_model.predict(X_valid_scaled)
test_accuracy_gb = accuracy_score(y_valid, y_valid_pred_gb)
print(f"Gradient Boosting Test Accuracy: {test_accuracy_gb:.4f}")

print("\nConfusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_valid, y_valid_pred_gb))

print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_valid, y_valid_pred_gb))

y_prob_gb = gb_model.predict_proba(X_valid_scaled)[:, 1]
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_valid, y_prob_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.figure()
plt.plot(fpr_gb, tpr_gb, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_gb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Gradient Boosting)')
plt.legend(loc="lower right")
plt.show()



knn_model = KNeighborsClassifier()

knn_model.fit(X_train_scaled, y_train)

y_valid_pred_knn = knn_model.predict(X_valid_scaled)
test_accuracy_knn = accuracy_score(y_valid, y_valid_pred_knn)
print(f"KNN Test Accuracy: {test_accuracy_knn:.4f}")

print("\nConfusion Matrix (KNN):")
print(confusion_matrix(y_valid, y_valid_pred_knn))

print("\nClassification Report (KNN):")
print(classification_report(y_valid, y_valid_pred_knn))

y_prob_knn = knn_model.predict_proba(X_valid_scaled)[:, 1]
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_valid, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure()
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (KNN)')
plt.legend(loc="lower right")
plt.show()



test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')

X_test = test.drop(['CustomerId', 'Surname'], axis=1)


X_test['Gender'] = label_encoder.transform(X_test['Gender'])


X_test = pd.get_dummies(X_test, columns=['Geography'], drop_first=True)

X_test_scaled = scaler.transform(X_test)

y_prob_test = gb_model.predict_proba(X_test_scaled)[:, 1]



output = pd.DataFrame({
    'id': test['CustomerId'],
    'PredictedExitedProb': y_prob_test,  
     
})


output = output.drop_duplicates(subset='id', keep='first')

output.to_csv('predictions.csv', index=False)



