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


import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced visualizations
import re  # For handling regular expressions


# Read the training data
df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')



# Check the number of rows and columns
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


# Display general dataset information
df.info()


# View the first few records
df.head()


df.describe().T  # Transposed summary for better readability


# Check for missing values
print(df.isnull().sum())

# Check unique values for categorical variables
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")



plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()



sns.countplot(x=df['efs'], palette="Set2")
plt.title("Event-Free Survival (efs) Distribution")
plt.xlabel("EFS Outcome (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()



# Histograms
df.hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

# Boxplot for numerical variables
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.show()



categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], palette="coolwarm")
    plt.title(f"Count of {col}")
    plt.show()



# Dropping the "ID" column as it is not a useful feature for model training
df1 = df.drop(['ID'], axis=1)


# Display the column names of the modified dataset
df1.columns


# Extracting the target variable (efs: Event-Free Survival outcome)
target = df1['efs']


# Display the first few values of the target variable
target.head()


# Creating feature matrix (X) by dropping the target variable "efs" and "efs_time"
X = df1.drop(['efs', 'efs_time'], axis=1)

# Display the first five rows of the feature matrix
X.head()


# Convert categorical variables into dummy/one-hot encoded variables
X = pd.get_dummies(X, drop_first=False)

# Renaming columns to remove special characters (LightGBM does not support them)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in X.columns}
new_n_list = list(new_names.values())

# Handling duplicate column names (LightGBM does not support duplicate features)
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col 
             for i, (col, new_col) in enumerate(new_names.items())}

# Renaming the feature matrix columns with cleaned names
X = X.rename(columns=new_names)



# Display the first five rows after renaming
X.head()


# Display dataset information after processing
X.info()


# Print column names of the processed dataset
X.columns


# Import necessary machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc  # For model evaluation
from lightgbm import LGBMClassifier  # Import LightGBM classifier
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning


# Splitting data into training and testing sets (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.10, random_state=101)



# Define a hyperparameter grid for tuning LightGBM
param_grid = {
    "objective": ["binary"],  # Binary classification task
    "boosting_type": ["gbdt"],  # Gradient Boosting Decision Tree method
    'metric': ['auc'],  # AUC as the evaluation metric
    "random_state": [42],  # Fixing randomness for reproducibility
    'learning_rate': [0.003],  # Learning rate for boosting
    'n_estimators': [1000],  # Number of boosting iterations
    'num_leaves': [31],  # Number of leaves in each tree
    'max_depth': [20],  # Maximum depth of the tree
    'verbosity': [-1],  # Suppress LightGBM logs
    'error_score': ["raise"]  # Raise an error if an issue occurs
}

# Initialize the LightGBM classifier
lgbm_model = LGBMClassifier()

# Perform grid search cross-validation to find the best hyperparameters
grid = GridSearchCV(lgbm_model, param_grid, cv=5)

# Fit the model on training data
grid.fit(X_train, y_train)

# Make predictions on the test set
grid_pred = grid.predict(X_test)

# Print confusion matrix to evaluate classification performance
print(confusion_matrix(y_test, grid_pred))

# Print classification report with precision, recall, and F1-score
print(classification_report(y_test, grid_pred))

# Display the best hyperparameters found by grid search
grid.best_params_



# Read the test dataset
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Drop the "ID" column as it is not needed for prediction
test_df1 = test_df.drop(['ID'], axis=1)

# Convert categorical variables into dummy/one-hot encoded variables
test_df1 = pd.get_dummies(test_df1, drop_first=False)

# Rename test dataset columns to remove special characters (like in training data)
new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in test_df1.columns}
new_n_list = list(new_names.values())

# Handling duplicate column names
new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col 
             for i, (col, new_col) in enumerate(new_names.items())}

# Apply the cleaned column names to the test dataset
test_df1 = test_df1.rename(columns=new_names)

# Display test dataset information after processing
test_df1.info()


# Ensure all columns from the training feature matrix exist in the test dataset
for col in X.columns:
    if col in test_df1.columns:
        pass  # If column exists, do nothing
    else:
        test_df1[col] = np.nan  # If column is missing, add it with NaN values


# Display first few rows of the processed test dataset
test_df1.head()


# Preparing the submission file
submission = pd.DataFrame()  # Create an empty DataFrame
submission['ID'] = test_df['ID']  # Add ID column from the original test dataset

# Predict probabilities for the positive class (efs = 1) using the trained model
submission['prediction'] = grid.predict_proba(test_df1)[:, 1]

# Define the file name for submission
file_name = 'submission.csv'

# Save the submission file as a CSV
submission.to_csv(file_name, index=False)

# Display the final submission DataFrame
submission

