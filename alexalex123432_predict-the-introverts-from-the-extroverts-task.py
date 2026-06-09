import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna(train[col].mode()[0])
    else:
        train[col] = train[col].fillna(train[col].median())

le = LabelEncoder()
cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = le.fit_transform(train[col])

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_val, log_model.predict(X_val))

best_dt_acc = 0
best_depth = 0
for depth in [3, 5, 7, 10, 15]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_val, dt.predict(X_val))
    if acc > best_dt_acc:
        best_dt_acc = acc
        best_depth = depth

print(f"Logistic Regression Accuracy: {log_acc:.4f}")
print(f"Best Decision Tree Accuracy: {best_dt_acc:.4f} (Depth: {best_depth})")

if log_acc > best_dt_acc:
    best_model = log_model
    print("Selected Model: Logistic Regression")
else:
    best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_model.fit(X_train, y_train)
    print("Selected Model: Decision Tree")

print("\nBest Model Parameters:")
print(best_model.get_params())

