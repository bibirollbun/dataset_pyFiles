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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")



print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"sample_submission: (sample_submission.shape)")


display(train.head())


import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Distribution of Target Variable
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x=train["Personality"], palette="coolwarm")
plt.title("Distribution of Personality Types")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.show()

# ==============================
# 2. Correlation Heatmap (Numeric Features)
# ==============================
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# ==============================
# 3. Boxplot: Time spent alone vs Personality
# ==============================
plt.figure(figsize=(8,5))
sns.boxplot(x="Personality", y="Time_spent_Alone", data=train, palette="Set2")
plt.title("Time Spent Alone by Personality")
plt.xlabel("Personality")
plt.ylabel("Time Spent Alone")
plt.show()

# ==============================
# 4. Countplot: Stage Fear vs Personality
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x="Stage_fear", hue="Personality", data=train, palette="viridis")
plt.title("Stage Fear vs Personality")
plt.xlabel("Stage Fear (Yes=1, No=0)")
plt.ylabel("Count")
plt.show()



# ==============================
# Preprocessing
# ==============================
binary_cols = ['Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Stage_fear']
for col in binary_cols:
    if col in train.columns:
        train[col] = train[col].map({'Yes': 1, 'No': 0})
    if col in test.columns:
        test[col] = test[col].map({'Yes': 1, 'No': 0})

# Encode target variable
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])  # Extrovert=1, Introvert=0

# Drop non-feature columns
drop_cols = [col for col in ['id', 'Personality'] if col in train.columns]
X = train.drop(columns=drop_cols)
y = train['Personality']

# Ensure numeric features
X = X.apply(pd.to_numeric, errors='coerce')
test_features = test.drop(columns=['id']) if 'id' in test.columns else test
test_features = test_features.apply(pd.to_numeric, errors='coerce')

# ==============================
# Handle missing values with SimpleImputer
# ==============================
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
test_features = imputer.transform(test_features)



# ==============================
# Train-test split (validation)
# ==============================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Train Logistic Regression
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# Validation performance
# ==============================
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ==============================
# Retrain on full data
# ==============================
model.fit(X, y)


# ==============================
# Train-test split (validation)
# ==============================
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Feature Scaling (SVM এর জন্য খুব গুরুত্বপূর্ণ)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==============================
# Train Support Vector Machine
# ==============================
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# ==============================
# Validation performance
# ==============================
y_pred = svm_model.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ==============================
# Retrain on full data
# ==============================
X_scaled = scaler.fit_transform(X)
svm_model.fit(X_scaled, y)


# ==============================
# Train-test split (validation)
# ==============================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Decision Tree
# ==============================
dt_model = DecisionTreeClassifier(
    criterion='gini',      # বা 'entropy'
    max_depth=None,        # চাইলে overfitting কমাতে depth limit দিতে পারো
    random_state=42
)

dt_model.fit(X_train, y_train)

# ==============================
# Validation performance
# ==============================
y_pred = dt_model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ==============================
# Retrain on full data
# ==============================
dt_model.fit(X, y)


dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


# ==============================
# Train-test split (validation)
# ==============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Random Forest
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=None,        # চাইলে depth limit দিতে পারো
    random_state=42,
    n_jobs=-1              # faster training
)

rf_model.fit(X_train, y_train)

# ==============================
# Validation performance
# ==============================
y_pred = rf_model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ==============================
# Retrain on full data
# ==============================
rf_model.fit(X, y)


# ==============================
# Import Libraries
# ==============================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==============================
# Train-test split
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 1️⃣ Logistic Regression
# ==============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_val)
log_acc = accuracy_score(y_val, y_pred_log)

# ==============================
# 2️⃣ SVM (with scaling)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_val_scaled)
svm_acc = accuracy_score(y_val, y_pred_svm)

# ==============================
# 3️⃣ Decision Tree
# ==============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_val)
dt_acc = accuracy_score(y_val, y_pred_dt)

# ==============================
# 4️⃣ Random Forest
# ==============================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, y_pred_rf)

# ==============================
# Plot Accuracy Comparison
# ==============================
models = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"]
accuracies = [log_acc, svm_acc, dt_acc, rf_acc]

plt.figure()
plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.ylim(0.9, 1.0)

plt.show()

# Print accuracies
print("Logistic Regression:", log_acc)
print("SVM:", svm_acc)
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)




# ==============================
# Predict on test set
# ==============================
test_pred = model.predict(test_features)
test_pred_labels = le.inverse_transform(test_pred)

# ==============================
# Create submission file
# ==============================
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": test_pred_labels
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv file created successfully!")


