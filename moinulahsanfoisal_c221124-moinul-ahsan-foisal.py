# Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Warnings
import warnings
warnings.filterwarnings('ignore')



# Load train and test datasets
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Preview data
train_df.head()



# Drop any unnamed index columns
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Check for nulls
print("Train Nulls:\n", train_df.isnull().sum())
print("\nTest Nulls:\n", test_df.isnull().sum())

# Drop/Fill nulls
train_df.dropna(inplace=True)
test_df.fillna(method='ffill', inplace=True)

# Label encode categorical features
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

le = LabelEncoder()
for col in categorical_cols:
    if col in train_df.columns:
        train_df[col] = le.fit_transform(train_df[col])
    if col in test_df.columns and col != 'satisfaction':
        test_df[col] = le.transform(test_df[col])

# Define features and target
X = train_df.drop(['id', 'satisfaction'], axis=1)
y = train_df['satisfaction']

# Prepare test set
X_test = test_df.drop('id', axis=1)

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Define base learners
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lr = LogisticRegression()

# Combine into Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('RandomForest', rf), ('GradientBoosting', gb), ('LogisticRegression', lr)],
    voting='hard'  # or 'soft' for probabilistic averaging
)



# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Predict on validation set
y_pred = ensemble_model.predict(X_val)

# Evaluation
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# Predict on test dataset
test_preds = ensemble_model.predict(X_test)

# Decode predictions to original labels
reverse_map = {0: 'neutral or dissatisfied', 1: 'satisfied'}
decoded_preds = pd.Series(test_preds).map(reverse_map)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'satisfaction': decoded_preds
})

# Save CSV
submission.to_csv("submission.csv", index=False)

submission.head()



!pip install shap



# Re-train Gradient Boosting on unscaled data for SHAP interpretation
# Use original column names for readability
X_original = train_df.drop(['id', 'satisfaction'], axis=1)
y_original = train_df['satisfaction']

# Split again using original data
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# Refit model (for SHAP)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_orig, y_train_orig)



import shap

# TreeExplainer supports GradientBoostingClassifier
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_train_orig)



# Plot SHAP summary
shap.summary_plot(shap_values, X_train_orig)



rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lr = LogisticRegression()

ensemble_model = VotingClassifier(
    estimators=[('RandomForest', rf), ('GradientBoosting', gb), ('LogisticRegression', lr)],
    voting='hard'
)



ensemble_model.fit(X_train, y_train)



# Check if rf was trained
rf.fit(X_train, y_train)  # re-train if necessary

# Get feature importances
importances = rf.feature_importances_
feature_names = test_df.drop('id', axis=1).columns

# Create DataFrame for plotting
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()



# Re-train Gradient Boosting model if not trained
gb.fit(X_train, y_train)

# Get importances
gb_importances = gb.feature_importances_

# Plot
gb_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=gb_importances_df)
plt.title("Feature Importances - Gradient Boosting")
plt.tight_layout()
plt.show()



# Predict on test set
final_preds = ensemble_model.predict(X_test)

# Convert predictions back to original labels
label_map = {0: 'neutral or dissatisfied', 1: 'satisfied'}
final_preds_decoded = pd.Series(final_preds).map(label_map)

# ✅ Rename `id` to `ID` exactly
submission = pd.DataFrame({
    'ID': test_df['id'],  # this must be capitalized
    'satisfaction': final_preds_decoded
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

# Preview
submission.head()



submission.to_csv("/kaggle/working/submission.csv", index=False)



# Predict on test data
final_preds = ensemble_model.predict(X_test)

# Map numeric predictions to original labels
label_map = {0: 'neutral or dissatisfied', 1: 'satisfied'}
final_preds_decoded = pd.Series(final_preds).map(label_map)

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': test_df['id'],  # IMPORTANT: Must be exactly 'ID'
    'satisfaction': final_preds_decoded
})

# ✅ Save to Kaggle working directory
submission.to_csv("/kaggle/working/submission.csv", index=False)

# Preview output
submission.head()


