





import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


url="/kaggle/input/binaryclassificationwithabankchurndataset/train.csv"
df = pd.read_csv(url)
df.sample(10)


print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Data Types and Missing Values:")
df.info()

print("\nStatistical Summary of Numerical Columns:")
print(df.describe())


print(df.isnull().sum())


sns.countplot(x='Exited', data=df)
plt.title('Distribution of Exited (Target Variable)')
plt.show()


df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})



df = pd.get_dummies(df, columns=['Geography'], drop_first=True)



def age_category(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle_Aged'
    else:
        return 'Senior'

df['AgeCategory'] = df['Age'].apply(age_category)

df = pd.get_dummies(df, columns=['AgeCategory'], drop_first=False)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

df[num_cols] = scaler.fit_transform(df[num_cols])



df.drop(['Surname', 'CustomerId', 'id'], axis=1, inplace=True)



y = df['Exited']

X = df.drop('Exited', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))



from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))



import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))



from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(LogisticRegression(random_state=42), param_grid=params, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

best_lr_model = grid.best_estimator_
y_pred_best_lr = best_lr_model.predict(X_test)

print("Tuned Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_best_lr))



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))
print("Tuned Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf_tuned))



import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1]
}

grid_xgb = GridSearchCV(xgb_clf, param_grid=param_grid_xgb, cv=5, n_jobs=-1)
grid_xgb.fit(X_train, y_train)

print("Best parameters for XGBoost:", grid_xgb.best_params_)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test)

print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_tuned))
print("Tuned XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb_tuned))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:,1] 

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-score
f1 = f1_score(y_test, y_pred)

# AUC-ROC
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")



fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0,1], [0,1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# XGBoost modeli uchun bashorat
y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]  # Class 1 ehtimoli

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-score
f1 = f1_score(y_test, y_pred)

# AUC-ROC
auc = roc_auc_score(y_test, y_proba)

# Natijalarni chop etish
print(f"XGBoost Accuracy: {accuracy:.4f}")
print(f"XGBoost Precision: {precision:.4f}")
print(f"XGBoost Recall: {recall:.4f}")
print(f"XGBoost F1-score: {f1:.4f}")
print(f"XGBoost AUC-ROC: {auc:.4f}")



fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.grid(True)
plt.show()



import pandas as pd
from sklearn.preprocessing import StandardScaler

# Test faylni yuklash
test_url = '/kaggle/input/binaryclassificationwithabankchurndataset/test.csv'
test_df = pd.read_csv(test_url)

# Asl mijoz ID'larini saqlab qoâ€˜yamiz
customer_ids = test_df['CustomerId'].copy()

# Gender ustunini raqamlarga aylantirish
test_df['Gender'] = test_df['Gender'].map({'Male': 0, 'Female': 1})

# Geography ustunini one-hot encoding qilish
test_df = pd.get_dummies(test_df, columns=['Geography'], drop_first=True)

# Yosh kategoriyasini yaratish
def age_category(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle_Aged'
    else:
        return 'Senior'

test_df['AgeCategory'] = test_df['Age'].apply(age_category)
test_df = pd.get_dummies(test_df, columns=['AgeCategory'], drop_first=False)

# Scaling qilish (oldingi scaler bilan)
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Ortiqcha ustunlarni olib tashlash
test_df.drop(['Surname', 'CustomerId'], axis=1, inplace=True)

# Yetishmayotgan ustunlar uchun 0 qoâ€˜shish
missing_cols = set(X_train.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0

# Modelga mos ustun tartibi
test_df = test_df[X_train.columns]

# âœ… XGBoost modelidan ehtimollar bilan bashorat
test_predictions = best_xgb.predict_proba(test_df)[:, 1]

# âœ… Toâ€˜gâ€˜ri submission fayl yaratish (id = CustomerId)
submission = pd.DataFrame({
    'id': customer_ids,
    'Exited': test_predictions
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission fayl yaratildi: submission.csv (XGBoost bilan, xatosiz)")



submission.head(20)




