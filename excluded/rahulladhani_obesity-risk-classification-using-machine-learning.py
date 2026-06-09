# Data handling
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")



# Load datasets with personalised variable names
rl_train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
rl_test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
rl_sub_template = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')



## Check Shape & Columns
print("Train shape:", rl_train.shape)
print("Test shape:", rl_test.shape)

print("\nTrain columns:")
print(rl_train.columns.tolist())


# First few rows
rl_train.head()


# General data info
rl_train.info()


# Missing value check
rl_train.isnull().sum()



# Target class distribution
rl_train['NObeyesdad'].value_counts(normalize=True)


import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size
plt.figure(figsize=(10, 6))
sns.countplot(data=rl_train, x='NObeyesdad', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Obesity Risk Categories')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(data=rl_train, x='NObeyesdad', y='Age', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Age Distribution by Obesity Category')
plt.xlabel('NObeyesdad')
plt.ylabel('Age')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(data=rl_train, x='NObeyesdad', hue='Gender', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Obesity Category Distribution by Gender')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(data=rl_train, x='NObeyesdad', hue='family_history_with_overweight', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Obesity Category Distribution by Family History')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.legend(title='Family History')
plt.tight_layout()
plt.show()



# Select only numeric columns
numeric_cols = rl_train.select_dtypes(include='number')

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.violinplot(data=rl_train, x='NObeyesdad', y='FCVC', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Vegetable Consumption (FCVC) Distribution by Obesity Category')
plt.xlabel('NObeyesdad')
plt.ylabel('FCVC (Frequency)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=rl_train, x='NObeyesdad', hue='MTRANS', order=rl_train['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Obesity Category by Mode of Transport')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.legend(title='Mode of Transport')
plt.tight_layout()
plt.show()



import missingno as msno

# Visualise missing data
msno.matrix(rl_train)
plt.title('Missing Values Heatmap')
plt.show()



# Identify categorical features
categorical_cols = rl_train.select_dtypes(include='object').columns.tolist()

# Drop target from that list (we encode it separately)
categorical_cols.remove('NObeyesdad')

print("Categorical Features:", categorical_cols)



## Step 1: Tag the data so we can combine and split cleanly later
rl_train['source'] = 'train'
rl_test['source'] = 'test'

# Combine both datasets
combined = pd.concat([rl_train, rl_test], axis=0)



# One-hot encode the 8 categorical features
combined_encoded = pd.get_dummies(combined, columns=categorical_cols)


# Separate back the encoded train and test data
rl_train_encoded = combined_encoded[combined_encoded['source'] == 'train'].drop(['source'], axis=1)
rl_test_encoded = combined_encoded[combined_encoded['source'] == 'test'].drop(['source', 'NObeyesdad'], axis=1)



rl_train_encoded.dtypes.value_counts()


rl_train_encoded.dtypes[rl_train_encoded.dtypes == 'object']


from sklearn.preprocessing import LabelEncoder

# Create and apply the encoder
target_encoder = LabelEncoder()
rl_train_encoded['target'] = target_encoder.fit_transform(rl_train_encoded['NObeyesdad'])

# Optional: drop the original text column now
rl_train_encoded = rl_train_encoded.drop(columns=['NObeyesdad'])

# Preview the classes
print("Encoded target classes:")
for i, label in enumerate(target_encoder.classes_):
    print(f"{i}: {label}")



from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = rl_train_encoded.drop(columns=['id', 'target'], errors='ignore')
y = rl_train_encoded['target']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialise the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on validation set
y_pred = rf_model.predict(X_val)

# Evaluate
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=target_encoder.classes_))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Create a base Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up the grid search
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Show the best parameters and accuracy score
print("âœ… Best Hyperparameters:", grid_search.best_params_)
print(f"âœ… Best Cross-Validated Accuracy: {grid_search.best_score_:.4f}")



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create the XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softmax',  # For multi-class classification
    num_class=7,                # Number of target classes
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on validation set
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate
accuracy = accuracy_score(y_val, y_pred_xgb)
print(f"âœ… XGBoost Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred_xgb, target_names=target_encoder.classes_))



from xgboost import plot_importance
import matplotlib.pyplot as plt

# Plot top 20 important features
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=20, importance_type='gain')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# Generate confusion matrix
cm = confusion_matrix(y_val, y_pred_xgb)
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()



# Drop 'id' before prediction
rl_test_encoded_fixed = rl_test_encoded.drop(columns=['id'])

# Predict using XGBoost
test_preds = xgb_model.predict(rl_test_encoded_fixed)

# Decode to original class names
test_preds_labels = target_encoder.inverse_transform(test_preds)

# Create submission file
submission = pd.DataFrame({
    'id': rl_test['id'],  # keep original IDs for Kaggle
    'NObeyesdad': test_preds_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved: submission.csv")
submission.head()


