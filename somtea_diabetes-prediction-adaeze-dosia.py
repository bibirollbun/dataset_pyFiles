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


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head(10)


df.info()


df.isna().sum()


df.drop('id', axis=1, inplace=True)


df.info()


df.corr(numeric_only=True)['diagnosed_diabetes'].sort_values()


df.diagnosed_diabetes.value_counts()


df.dtypes


categorical = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
for cols in categorical:
    print(df[cols].value_counts(), "\n")
    print(df.groupby(cols)['diagnosed_diabetes'].mean(), "\n")


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('diagnosed_diabetes')
df[numeric_cols].describe()


import seaborn as sns
import matplotlib.pyplot as plt


df[numeric_cols].hist(figsize=(15,10), bins=30)
plt.tight_layout()
plt.show()



numeric_col = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(15,10))
sns.heatmap(numeric_col.corr(), annot=True,fmt=".3f", cmap='Blues')
plt.show()


df.alcohol_consumption_per_week.value_counts()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


hot_encoder = OneHotEncoder()
gender_values = hot_encoder.fit_transform(df.gender.values.reshape(-1, 1)).toarray()

gender_cats = hot_encoder.categories_[0]

df[gender_cats] = gender_values



df.drop(columns='gender', inplace=True)


oneHot_cols = ['ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

df = pd.get_dummies(df, columns=oneHot_cols, drop_first=True)


df.head(5)


df.columns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


numeric_cols = [
    'age', 'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week', 'diet_score',
    'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides'
]



scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])


log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("logistic regression Accuracy:", accuracy_score(y_test, log_preds))
print("logistic regression Precision:", precision_score(y_test, log_preds))
print("logistic regression Recall:", recall_score(y_test, log_preds))
print("logistic regression F1:", f1_score(y_test, log_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_preds))

print("\nClassification Report:")
print(classification_report(y_test, log_preds))


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)


print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(" Random Forest Precision:", precision_score(y_test, rf_preds))
print(" Random Forest Recall:", recall_score(y_test, rf_preds))
print(" Random Forest F1:", f1_score(y_test, rf_preds))


from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1','l2'],
    'solver': ['liblinear']
}

grid_log = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=5, n_jobs=-1)
grid_log.fit(X_train_scaled, y_train)

best_log = grid_log.best_estimator_
log_preds_tuned = best_log.predict(X_test_scaled)


param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
rf_preds_tuned = best_rf.predict(X_test)



print("===== Logistic Regression (Tuned) =====")
print("Accuracy:", accuracy_score(y_test, log_preds_tuned))
print("Precision:", precision_score(y_test, log_preds_tuned))
print("Recall:", recall_score(y_test, log_preds_tuned))
print("F1-Score:", f1_score(y_test, log_preds_tuned))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_preds_tuned))

print("\nClassification Report:")
print(classification_report(y_test, log_preds_tuned))


print("===== Random Forest (Tuned) =====")
print("Accuracy:", accuracy_score(y_test, rf_preds_tuned))
print("Precision:", precision_score(y_test, rf_preds_tuned))
print("Recall:", recall_score(y_test, rf_preds_tuned))
print("F1-Score:", f1_score(y_test, rf_preds_tuned))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds_tuned))

print("\nClassification Report:")
print(classification_report(y_test, rf_preds_tuned))

