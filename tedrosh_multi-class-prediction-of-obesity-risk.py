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


# 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


# 2: Exploratory Data Analysis
# 2.1 Basic Data Overview
print("Training Data Shape:", train.shape)
print("\nFirst 5 rows:")
display(train.head())

print("\nData Types:")
print(train.dtypes)

print("\nSummary Statistics:")
display(train.describe(include='all'))

# 2.2 Missing Values Analysis
print("\nMissing Values:")
print(train.isnull().sum())

# 2.3 Target Variable Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='NObeyesdad', data=train, palette='viridis')
plt.title('Class Distribution of Obesity Risk Levels')
plt.xticks(rotation=45)
plt.show()

# 2.4 Numerical Features Analysis
numerical = train.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
train[numerical].hist(bins=20, layout=(3, 4), figsize=(15, 10))
plt.tight_layout()
plt.suptitle('Numerical Feature Distributions', y=1.02)
plt.show()

# 2.5 Categorical Features Analysis
categorical = train.select_dtypes(include=['object']).columns.drop('NObeyesdad')
plt.figure(figsize=(15, 15))
for i, col in enumerate(categorical, 1):
    plt.subplot(4, 3, i)
    sns.countplot(x=col, data=train, palette='Set2')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.suptitle('Categorical Feature Distributions', y=1.02)
plt.show()

# 2.6 Correlation Analysis
plt.figure(figsize=(12, 8))
corr_matrix = train.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()



# 3: Prepare Data
X = train.drop('NObeyesdad', axis=1)  # Features
y = train['NObeyesdad']               # Target variable
X_test = test.copy()                  # Test data for final prediction




# 4: Split Training Data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)


# 5: Identify Feature Types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()  # Fixed tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# 6: Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),     # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Encode categories
    ]
)


# 7: Initialize Models
models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000  # Fixed parameter (original had 'mr')
    ),
    'LDA': LinearDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale', 
        decision_function_shape='ovr'
    )
}



# 8: Train and Validate Models
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"{name} Validation Accuracy: {acc:.4f}")



# 9: Submission

for name, model in models.items():
    pipeline.fit(X, y)
    preds = pipeline.predict(X_test)
    submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': preds})
    submission.to_csv(f'submission_{name.lower().replace(" ", "_")}.csv', index=False)

print("\nAll submissions generated successfully!")  



import os

# Define paths
source_path = '/kaggle/working/submission_svm.csv'
target_path = '/kaggle/working/submission.csv'

# Check if source file exists
if os.path.exists(source_path):
    # Remove existing submission.csv if needed
    if os.path.exists(target_path):
        os.remove(target_path)
    
    # Rename the file
    os.rename(source_path, target_path)
    print(f"✅ Renamed '{source_path}' to '{target_path}'")
else:
    print(f"❌ Error: {source_path} does not exist")

# Verify
print("\nCurrent files in /kaggle/working:")
print(os.listdir('/kaggle/working'))

