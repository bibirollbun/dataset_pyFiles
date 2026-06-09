# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Load training data
train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")

# Categorical columns
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# Separate features and target
X = train_data.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_data['satisfaction']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
val_preds = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy (XGBoost): {accuracy:.2f}")


# Confusion Matrix
cm = confusion_matrix(y_val, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoders['satisfaction'].classes_,
            yticklabels=label_encoders['satisfaction'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (XGBoost)')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_val, val_preds, target_names=label_encoders['satisfaction'].classes_))


# Feature Importance
importances = xgb_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()



# Load test data
test_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")
# Encode categorical columns in test set
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    if col in label_encoders:
        test_data[col] = label_encoders[col].transform(test_data[col])

# Prepare test features
X_test = test_data.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Make predictions
test_data['satisfaction'] = xgb_model.predict(X_test)
test_data['satisfaction'] = label_encoders['satisfaction'].inverse_transform(test_data['satisfaction'])

# Save submission
test_data.rename(columns={'id': 'ID'}, inplace=True)
test_data[['ID', 'satisfaction']].to_csv("submission.csv", index=False)


test_data[['ID', 'satisfaction']].head()

