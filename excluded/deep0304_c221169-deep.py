import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

# Label encoding
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Define features and labels
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Tune Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Load and process test data
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/test_dataset_exam.csv")
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])

X_test = test_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Predict on test set
test_data['satisfaction'] = best_model.predict(X_test)
test_data['satisfaction'] = label_encoders['satisfaction'].inverse_transform(test_data['satisfaction'])

# Save submission
test_data.rename(columns={'id': 'ID'}, inplace=True)
test_data[['ID', 'satisfaction']].to_csv("submission.csv", index=False)

# Feature importance plot
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importances")
plt.show()


