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


# ============================================
# A — Acquire Data
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()



# ============================================
# B — Basic Understanding
# ============================================

print("Shape:", df.shape)
df.info()
df.describe()
df.columns



# ============================================
# C — Clean Data
# ============================================

# Missing values
df.isnull().sum()

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.head()



# ============================================
# D — Data Exploration (EDA)
# ============================================



df.hist(figsize=(12,8))
plt.show()

sns.pairplot(df.sample(300), diag_kind='kde')



# ============================================
# E — Encode Features
# ============================================

label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le



X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# I — Initialize Models
# ============================================

log_model = LogisticRegression(max_iter=2000)
rf_model = RandomForestClassifier()



# ============================================
# K — Key Model Training
# ============================================

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)



# ============================================
# L — Log Evaluation Metrics
# ============================================

def evaluate(model):
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

print("Logistic Regression:")
evaluate(log_model)

print("Random Forest:")
evaluate(rf_model)



# ============================================
# M — Model Tuning
# ============================================

params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
}

grid = GridSearchCV(rf_model, params, cv=3)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
best_rf



# ============================================
# N — Normalize / Scale Data
# ============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# ============================================
# O — Optimize Features
# ============================================

importances = rf_model.feature_importances_
fi = pd.DataFrame({"Feature": X.columns, "Importance": importances})
fi.sort_values("Importance", ascending=False).head(10)



# ============================================
# P — Plot Important Graphs
# ============================================

# ROC Curve
plt.figure(figsize=(6,4))
proba = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()



# Trying tuned model
evaluate(best_rf)


# ============================================
# S — Save Model
# ============================================

import joblib
joblib.dump(best_rf, "model.pkl")



# ============================================
# T — Test with New Inputs
# ============================================

sample = X_test.iloc[0:1]
best_rf.predict(sample)



# ============================================
# U — Use Error Analysis
# ============================================

preds = best_rf.predict(X_test)
errors = X_test[preds != y_test]
errors.head()



# ============================================
# X — eXport Results
# ============================================

submission = pd.DataFrame({
    "ID": df.index,
    "Prediction": best_rf.predict(X)
})
submission.to_csv("submission.csv", index=False)


