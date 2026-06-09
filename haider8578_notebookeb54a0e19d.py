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


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Load the data
train_df = pd.read_csv('/kaggle/input/south-african-opportunity-prediction-challenge/train_sample.csv')
test_df = pd.read_csv('/kaggle/input/south-african-opportunity-prediction-challenge/test_sample.csv')

# Display basic information
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:")
print(train_df.columns.tolist())
print("\nFirst few rows of training data:")
print(train_df.head())


# Distribution of Progress
plt.figure(figsize=(10, 6))
sns.countplot(x='Progress', data=train_df)
plt.title('Distribution of Progress Values')
plt.xlabel('Progress Level')
plt.ylabel('Count')
plt.show()

# Relationship between Progress and Successful
plt.figure(figsize=(10, 6))
sns.countplot(x='Progress', hue='Successful', data=train_df)
plt.title('Progress vs Successful')
plt.xlabel('Progress Level')
plt.ylabel('Count')
plt.legend(title='Successful')
plt.show()

# Progress level statistics
print("Progress value distribution:")
print(train_df['Progress'].value_counts().sort_index())
print("\nSuccess rate by Progress level:")
print(train_df.groupby('Progress')['Successful'].mean())


# Gender distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=train_df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Race distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Race', data=train_df)
plt.title('Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Progress by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Progress', hue='Gender', data=train_df)
plt.title('Progress by Gender')
plt.xlabel('Progress Level')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

# Progress by Race
plt.figure(figsize=(12, 6))
sns.countplot(x='Progress', hue='Race', data=train_df)
plt.title('Progress by Race')
plt.xlabel('Progress Level')
plt.ylabel('Count')
plt.legend(title='Race')
plt.show()


# Aggregate distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Aggregate'], bins=20, kde=True)
plt.title('Aggregate Score Distribution')
plt.xlabel('Aggregate Score')
plt.ylabel('Count')
plt.show()

# Relationship between Aggregate and Progress
plt.figure(figsize=(10, 6))
sns.boxplot(x='Progress', y='Aggregate', data=train_df)
plt.title('Aggregate Score vs Progress')
plt.xlabel('Progress Level')
plt.ylabel('Aggregate Score')
plt.show()

# Top institutions
plt.figure(figsize=(12, 8))
top_institutions = train_df['Institution'].value_counts().nlargest(10)
sns.barplot(x=top_institutions.values, y=top_institutions.index)
plt.title('Top 10 Institutions')
plt.xlabel('Count')
plt.ylabel('Institution')
plt.show()

# Top qualifications
plt.figure(figsize=(12, 8))
top_qualifications = train_df['Qualification'].value_counts().nlargest(10)
sns.barplot(x=top_qualifications.values, y=top_qualifications.index)
plt.title('Top 10 Qualifications')
plt.xlabel('Count')
plt.ylabel('Qualification')
plt.show()


# Check for missing values
print("Missing values in training data:")
print(train_df.isnull().sum())

# For Disciplines, fill missing with 'None'
train_df['Disciplines'].fillna('None', inplace=True)
test_df['Disciplines'].fillna('None', inplace=True)

# Verify no missing values remain
print("\nMissing values after handling:")
print(train_df.isnull().sum())


# Create label encoders for categorical variables
categorical_features = ['Gender', 'Race', 'Institution', 'Qualification', 'Industry', 'Company']

# Dictionary to store label encoders
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    train_df[feature + '_encoded'] = le.fit_transform(train_df[feature])
    
    # For test set, handle unseen categories
    test_df[feature + '_encoded'] = test_df[feature].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Store the encoder
    label_encoders[feature] = le

print("Categorical features encoded successfully!")
print("Training data shape after encoding:", train_df.shape)


# Split disciplines into multiple binary features
all_disciplines = set()
for disciplines in train_df['Disciplines']:
    if disciplines != 'None':
        for discipline in disciplines.split(', '):
            all_disciplines.add(discipline)

print(f"Found {len(all_disciplines)} unique disciplines")

# Create binary features for each discipline
for discipline in all_disciplines:
    train_df['Discipline_' + discipline] = train_df['Disciplines'].apply(lambda x: 1 if discipline in x else 0)
    test_df['Discipline_' + discipline] = test_df['Disciplines'].apply(lambda x: 1 if discipline in x else 0)

print("Discipline binary features created successfully!")
print("Training data shape after discipline encoding:", train_df.shape)


# Count of disciplines
train_df['Discipline_Count'] = train_df['Disciplines'].apply(lambda x: 0 if x == 'None' else len(x.split(', ')))
test_df['Discipline_Count'] = test_df['Disciplines'].apply(lambda x: 0 if x == 'None' else len(x.split(', ')))

# Age groups
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0, 20, 25, 30, 35, 100], labels=['<20', '20-25', '25-30', '30-35', '35+'])
test_df['Age_Group'] = pd.cut(test_df['Age'], bins=[0, 20, 25, 30, 35, 100], labels=['<20', '20-25', '25-30', '30-35', '35+'])

# Encode age groups
age_group_le = LabelEncoder()
train_df['Age_Group_encoded'] = age_group_le.fit_transform(train_df['Age_Group'])
test_df['Age_Group_encoded'] = age_group_le.transform(test_df['Age_Group'])

# Aggregate score groups
train_df['Aggregate_Group'] = pd.cut(train_df['Aggregate'], bins=[0, 50, 60, 70, 80, 90, 100], labels=['0-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
test_df['Aggregate_Group'] = pd.cut(test_df['Aggregate'], bins=[0, 50, 60, 70, 80, 90, 100], labels=['0-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Encode aggregate groups
aggregate_group_le = LabelEncoder()
train_df['Aggregate_Group_encoded'] = aggregate_group_le.fit_transform(train_df['Aggregate_Group'])
test_df['Aggregate_Group_encoded'] = aggregate_group_le.transform(test_df['Aggregate_Group'])

print("Additional features created successfully!")


# Define feature columns
feature_columns = [
    'Age', 'Aggregate', 'Discipline_Count',
    'Gender_encoded', 'Race_encoded', 'Institution_encoded', 
    'Qualification_encoded', 'Industry_encoded', 'Company_encoded',
    'Age_Group_encoded', 'Aggregate_Group_encoded'
]

# Add discipline binary features
discipline_columns = [col for col in train_df.columns if col.startswith('Discipline_')]
feature_columns.extend(discipline_columns)

# Prepare training data
X_train = train_df[feature_columns]
y_train = train_df['Progress']

# Prepare test data
X_test = test_df[feature_columns]

print(f"Number of features: {len(feature_columns)}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")
print("\nTraining split Progress distribution:")
print(y_train_split.value_counts(normalize=True).sort_index())
print("\nValidation split Progress distribution:")
print(y_val_split.value_counts(normalize=True).sort_index())


# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
    'SVM': SVC(random_state=42)
}

# Train and evaluate each model
model_results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_split, y_train_split)
    
    # Predict on validation set
    y_pred = model.predict(X_val_split)
    
    # Evaluate
    accuracy = accuracy_score(y_val_split, y_pred)
    model_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_val_split, y_pred))


# Compare model accuracies
plt.figure(figsize=(12, 6))
sns.barplot(x=list(model_results.keys()), y=list(model_results.values()))
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Plot confusion matrix for the best model
best_model_name = max(model_results, key=model_results.get)
best_model = models[best_model_name]
y_pred = best_model.predict(X_val_split)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_split, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_val_split.unique()), yticklabels=sorted(y_val_split.unique()))
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Let's assume Random Forest performed best
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_split, y_train_split)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use the best model
best_model = grid_search.best_estimator_


# Predict on validation set
y_val_pred = best_model.predict(X_val_split)

# Calculate metrics
accuracy = accuracy_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val_split, y_val_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_split, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_val_split.unique()), yticklabels=sorted(y_val_split.unique()))
plt.title('Confusion Matrix - Best Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Get feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot top 20 feature importances
plt.figure(figsize=(12, 8))
plt.title('Top 20 Feature Importances')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_columns[i] for i in indices[:20]], rotation=90)
plt.xlim([-1, 20])
plt.tight_layout()
plt.show()

# Print top 10 features
print("Top 10 Important Features:")
for i in range(10):
    print(f"{i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")


# Train the best model on the full training data
print("Training the best model on the full training data...")
best_model.fit(X_train, y_train)
print("Model training completed!")


# Make predictions on test set
print("Making predictions on the test set...")
test_predictions = best_model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Progress': test_predictions
})

# Display first few rows of submission
print("Submission preview:")
print(submission.head())
print("\nSubmission Progress distribution:")
print(submission['Progress'].value_counts().sort_index())


# Save submission to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully as 'submission.csv'")


# Verify submission format
print(f"Submission shape: {submission.shape}")
print("\nSubmission columns:")
print(submission.columns.tolist())
print("\nFirst few rows of submission:")
print(submission.head())

# Check if all Progress values are valid (1-5)
print(f"\nUnique Progress values in submission: {sorted(submission['Progress'].unique())}")

# Check for missing values
print(f"\nMissing values in submission: {submission.isnull().sum().sum()}")

