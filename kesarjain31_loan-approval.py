# -----------------------------
# Loan Approval Prediction - Classification Mode (Fixed)
# -----------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

# 2. Handle missing values
for col in train.select_dtypes(include='number').columns:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())

for col in train.select_dtypes(include='object').columns:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])

# 3. Encode categorical columns
le = LabelEncoder()
for col in train.select_dtypes(include='object').columns:
    train[col] = le.fit_transform(train[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

# 4. Data Exploration Graphs
sns.countplot(x='loan_status', data=train)
plt.title("Target Distribution (Loan Status)")
plt.show()

sns.countplot(x='cb_person_default_on_file', hue='loan_status', data=train)
plt.title("Credit History vs Loan Status")
plt.show()

sns.countplot(x='loan_intent', hue='loan_status', data=train)
plt.title("Loan Intent vs Loan Status")
plt.show()

sns.boxplot(x='loan_status', y='person_income', data=train)
plt.title("Applicant Income vs Loan Status")
plt.show()

# 5. Split features & target
X = train.drop('loan_status', axis=1)
y = train['loan_status']

# 6. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test[X.columns])  # keep column order same

# 8. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Evaluate Model
y_val_pred = model.predict(X_val_scaled)
print("=== Classification Metrics ===")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

# 10. Feature Importance Visualization
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8,6), color='skyblue')
plt.title("Feature Importance")
plt.show()

# 11. Example Prediction for a Single Row
example_row = X.iloc[0:1]  # keep as DataFrame to avoid warning
example_row_scaled = scaler.transform(example_row)
example_pred = model.predict(example_row_scaled)
print("Example prediction for first row:", example_pred)

# 12. Predict on Test Set and Save Submission
test_pred = model.predict(X_test_scaled)
submission['loan_status'] = test_pred
submission.to_csv('loan_submission.csv', index=False)
print("Submission saved as 'loan_submission.csv'")

# 13. Optional: Predict on a new dataset
# new_data = pd.read_csv('new_loan_data.csv')
# new_data = new_data[X.columns]  # ensure same columns/order
# for col in new_data.select_dtypes(include='number').columns:
#     new_data[col] = new_data[col].fillna(train[col].median())
# for col in new_data.select_dtypes(include='object').columns:
#     new_data[col] = new_data[col].fillna(train[col].mode()[0])
# for col in new_data.select_dtypes(include='object').columns:
#     new_data[col] = le.transform(new_data[col])
# new_data_scaled = scaler.transform(new_data)
# new_predictions = model.predict(new_data_scaled)
# print("Predictions on new dataset:", new_predictions)


