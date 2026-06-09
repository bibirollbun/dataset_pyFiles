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


pip show scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier


df_train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


df_train.head(30)


df_train.isnull().sum()


df_train.describe()


df_train.info()


df_train.shape


df_train.corr(numeric_only=True)


df_train=df_train.drop('id', axis=1)
id_test=df_test['id']
df_test=df_test.drop('id', axis=1)


df_train = df_train.drop(columns=['person_age', 'cb_person_cred_hist_length'], axis=1)
df_test = df_test.drop(columns=['person_age', 'cb_person_cred_hist_length'], axis=1)


# Distribution of loan status (approved or not)
sns.countplot(x='loan_status', data=df_train)
plt.title('Loan Approval Status Distribution')
plt.show()


# Scatterplot to show relationship between loan amount and income
sns.scatterplot(x='person_income', y='loan_amnt', data=df_train)
plt.title('Loan Amount vs. Applicant Income')
plt.show()


# Boxplot to visualize loan amount by employment length
sns.boxplot(x='person_emp_length', y='loan_amnt', data=df_train)
plt.title('Loan Amount by Employment Length')
plt.show()



# Scatterplot to show relationship between loan amount and interest rate
sns.scatterplot(x='loan_int_rate', y='loan_amnt', data=df_train)
plt.title('Loan Amount vs. Interest Rate')
plt.show()



# Countplot to show loan status by home ownership type
sns.countplot(x='person_home_ownership', hue='loan_status', data=df_train)
plt.title('Loan Status by Home Ownership')
plt.show()


# Distribution of loan amounts
sns.histplot(df_train['loan_amnt'], bins=30, kde=True)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()


# Countplot to show the distribution of loan grades
sns.countplot(x='loan_grade', data=df_train)
plt.title('Loan Grade Distribution')
plt.show()


# Correlation heatmap to visualize correlations between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Boxplot to show loan amount distribution by loan grade
sns.boxplot(x='loan_grade', y='loan_amnt', data=df_train)
plt.title('Loan Amount by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Loan Amount')
plt.show()



df_train.columns


# Create a new feature for income to loan amount ratio in the train and test datasets
df_train['income_to_loan_ratio'] = df_train['person_income'] / df_train['loan_amnt']
df_test['income_to_loan_ratio'] = df_test['person_income'] / df_test['loan_amnt']


# Loan Amount per Year of Employment in train and test datasets
df_train['loan_per_emp_year'] = df_train['loan_amnt'] / (df_train['person_emp_length'] + 1)
df_test['loan_per_emp_year'] = df_test['loan_amnt'] / (df_test['person_emp_length'] + 1)


df_train.head()


preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']),
        ('label', OrdinalEncoder(), ['loan_grade']),
        ('scaler', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64']))
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

X = df_train.drop(columns=['loan_status']) 
y = df_train['loan_status']

X_preprocessed = pipeline.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Original class distribution:")
print(y_train.value_counts())

print("\nResampled class distribution:")
print(pd.Series(y_resampled).value_counts())


# Train a Random Forest model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# Initialize the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

# Fit the model on the resampled data
gb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_gb))
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb)}')


# Save the trained Gradient Boosting model
joblib.dump(gb_model, 'gradient_boosting_model.pkl')

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')




