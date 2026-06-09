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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Analyze target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train)
plt.title('Personality Distribution')
plt.show()
print(train['Personality'].value_counts(normalize=True))


# Check missing values
print("Train missing values:\n", train.isnull().sum())
print("Test missing values:\n", test.isnull().sum())


# Analyze numerical features
numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('id')
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# Analyze categorical features
categorical_cols = train.select_dtypes(include=['object']).columns.drop('Personality')
for col in categorical_cols:
    print(f"{col} value counts:\n", train[col].value_counts(dropna=False), "\n")


# Define features and target
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# Preprocessing pipeline
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# Model pipelines
model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])



# Train models
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)


# Evaluate models
y_pred_lr = model_lr.predict(X_val)
y_pred_rf = model_rf.predict(X_val)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))
print(confusion_matrix(y_val, y_pred_lr))

print("Random Forest:")
print("Accuracy:", accuracy_score(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))
print(confusion_matrix(y_val, y_pred_rf))


# Predict on test set
X_test = test.drop('id', axis=1)
y_test_pred = model_rf.predict(X_test)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)



# Create submission
submission = pd.DataFrame({'id': test['id'], 'Personality': y_test_pred_labels})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")

