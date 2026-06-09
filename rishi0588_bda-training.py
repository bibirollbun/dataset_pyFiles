import pandas as pd

# Load the dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Display dataset info
print("Training Data Info:")
print(train_data.info())

print("\nTest Data Info:")
print(test_data.info())

# Display first few rows
print("\nTraining Data Sample:")
print(train_data.head())



# Count missing values in each column
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())

print("\nMissing Values in Test Data:")
print(test_data.isnull().sum())



import seaborn as sns
import matplotlib.pyplot as plt

# Countplot for the target variable
sns.countplot(x=train_data['rainfall'])
plt.title("Target Variable Distribution")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = train_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()



# Plot histograms for numerical columns
train_data.hist(figsize=(12, 10), bins=30)
plt.suptitle("Feature Distributions")
plt.show()



# Boxplot for numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_data.drop(columns=['rainfall']))
plt.xticks(rotation=90)
plt.title("Boxplot of Features to Identify Outliers")
plt.show()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop missing values in test set
test_data = test_data.dropna()

# Store test IDs before modification
test_ids = test_data['id'].values

# Handle missing values in train_data
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Feature Engineering
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# One-hot encoding for categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# Align test_data with X to ensure consistency
X, test_data = X.align(test_data, join="left", axis=1, fill_value=0)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train best model
best_rf = grid_search.best_estimator_

# Validate the model
y_pred = best_rf.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')
print(classification_report(y_val, y_pred))

# Confusion matrix
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Predict on test set
test_predictions = best_rf.predict(test_data)

# Ensure correct submission format
submission = pd.DataFrame({
    'Id': test_ids,
    'RainTomorrow': test_predictions
})
submission.to_csv('submission_final.csv', index=False)
print("Submission file saved as 'submission_final.csv'")





